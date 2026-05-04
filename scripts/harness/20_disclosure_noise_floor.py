# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Within-condition JSD noise floor for the disclosure pilot's gating.

Question: are the cross-condition JSDs we measured (HP=0.334, LP=0.637,
NB=0.566 bits) large compared to the JSD you'd see between two seed-
subsamples of the *same* condition?

Two anchors:

  1. Claude split-half (internal). For each (category, condition) in
     ``data/claude_disclosure_pilot.jsonl``, randomly split the 15 rows
     into halves (8/7), compute JSD between them, repeat. Reports
     median + 95% interval. Most direct: tells us how much Claude's
     own emission varies under fixed condition at temp=1.0 with N=15.

  2. v3 gemma/qwen cross-seed (external). For each (category, model) in
     v3's emotional_raw, restricted to the same 5 pilot prompt IDs
     (hp01-05 / lp01-05 / nb01-05), draw two independent N=15 sub-
     samples (5 prompts × 3 seeds each), compute JSD, repeat. Reports
     median + 95% interval. Tells us what "stable kaomoji distribution
     under repeated sampling" looks like on similar-scale data on a
     local LM with known calibration.

Decision rule: the pilot's `framed`-vs-`direct` JSD is "real" if it
exceeds the upper end of both noise-floor intervals. If it's inside
either, the apparent shift may be sampling noise rather than a
condition effect.

Usage:
  python scripts/harness/20_disclosure_noise_floor.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.config import DATA_DIR


PILOT_PATH = DATA_DIR / "claude_disclosure_pilot.jsonl"
V3_PATHS = {
    "gemma": DATA_DIR / "emotional_raw.jsonl",
    "qwen": DATA_DIR / "qwen_emotional_raw.jsonl",
}
PILOT_QUADRANTS = ("HP", "LP", "NB")
PILOT_PROMPT_IDS = {
    "HP": [f"hp0{i}" for i in range(1, 6)],
    "LP": [f"lp0{i}" for i in range(1, 6)],
    "NB": [f"nb0{i}" for i in range(1, 6)],
}
N_BOOTSTRAP = 1000
RNG_SEED = 0


def _jsd(p_counts: Counter, q_counts: Counter) -> float:
    """Jensen-Shannon divergence in bits. Mirrors 19_claude_disclosure_pilot."""
    keys = set(p_counts) | set(q_counts)
    if not keys:
        return 0.0
    n_p = sum(p_counts.values())
    n_q = sum(q_counts.values())
    if n_p == 0 or n_q == 0:
        return 0.0
    p = {k: p_counts.get(k, 0) / n_p for k in keys}
    q = {k: q_counts.get(k, 0) / n_q for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl(a, b):
        total = 0.0
        for k in a:
            if a[k] > 0 and b[k] > 0:
                total += a[k] * math.log2(a[k] / b[k])
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _ci(values: list[float], lo: float = 0.025, hi: float = 0.975) -> tuple[float, float, float]:
    """median + 95% percentile interval."""
    if not values:
        return (0.0, 0.0, 0.0)
    s = sorted(values)
    n = len(s)
    return (
        s[n // 2],
        s[max(0, int(n * lo))],
        s[min(n - 1, int(n * hi))],
    )


def _claude_pilot_rows() -> list[dict]:
    rows: list[dict] = []
    with PILOT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    return rows


def _v3_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    return rows


def _split_half_jsd(rows: list[dict], rng: random.Random) -> list[float]:
    """Bootstrap N_BOOTSTRAP split-half JSDs over the row pool's first_word."""
    fws = [canonicalize_kaomoji(r.get("first_word") or "") or "" for r in rows]
    n = len(fws)
    half = n // 2
    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = list(range(n))
        rng.shuffle(idx)
        a = Counter(fws[i] for i in idx[:half])
        b = Counter(fws[i] for i in idx[half:half * 2])
        out.append(_jsd(a, b))
    return out


def _v3_subsample_jsd(rows: list[dict], pilot_ids: list[str], rng: random.Random) -> list[float]:
    """Bootstrap N=15 v3 subsamples (5 prompts × 3 seeds) twice independently,
    JSD between them. ``rows`` already filtered to pilot quadrant + model."""
    by_prompt: dict[str, list[str]] = {}
    for r in rows:
        pid = r.get("prompt_id", "")
        if pid not in pilot_ids:
            continue
        fw = canonicalize_kaomoji(r.get("first_word") or "") or ""
        by_prompt.setdefault(pid, []).append(fw)

    available = [pid for pid in pilot_ids if len(by_prompt.get(pid, [])) >= 6]
    if len(available) < 5:
        return []

    out: list[float] = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.sample(available, 5)
        a_fws: list[str] = []
        b_fws: list[str] = []
        for pid in chosen:
            seeds_pool = list(by_prompt[pid])
            rng.shuffle(seeds_pool)
            a_fws.extend(seeds_pool[:3])
            b_fws.extend(seeds_pool[3:6])
        out.append(_jsd(Counter(a_fws), Counter(b_fws)))
    return out


def main() -> None:
    if not PILOT_PATH.exists():
        raise SystemExit(f"missing {PILOT_PATH} — run 19_claude_disclosure_pilot.py first")

    rng = random.Random(RNG_SEED)

    cross_cond_jsd: dict[str, float] = {}
    pilot_rows = _claude_pilot_rows()
    by_cat_cond: dict[tuple[str, str], list[dict]] = {}
    for r in pilot_rows:
        by_cat_cond.setdefault((r["quadrant"], r["condition"]), []).append(r)
    for cat in PILOT_QUADRANTS:
        a = Counter(canonicalize_kaomoji(r.get("first_word") or "") or ""
                    for r in by_cat_cond.get((cat, "direct"), []))
        b = Counter(canonicalize_kaomoji(r.get("first_word") or "") or ""
                    for r in by_cat_cond.get((cat, "framed"), []))
        cross_cond_jsd[cat] = _jsd(a, b)

    print(f"bootstrap N={N_BOOTSTRAP}\n")

    print("=" * 88)
    print("ANCHOR 1: Claude split-half (internal)")
    print("for each (category, condition), randomly split N=15 into 8/7, compute JSD")
    print("=" * 88)
    print(f"{'cat':<4} {'cond':<6} {'median':>8} {'2.5%':>8} {'97.5%':>8}  "
          f"{'cross-cond JSD':>16} {'verdict':>10}")
    claude_floor: dict[str, float] = {}
    for cat in PILOT_QUADRANTS:
        cell_uppers: list[float] = []
        for cond in ("direct", "framed"):
            rows = by_cat_cond.get((cat, cond), [])
            jsds = _split_half_jsd(rows, rng)
            med, lo, hi = _ci(jsds)
            cell_uppers.append(hi)
            cross = cross_cond_jsd[cat]
            verdict = "ABOVE" if cross > hi else "inside"
            print(f"{cat:<4} {cond:<6} {med:>8.4f} {lo:>8.4f} {hi:>8.4f}  "
                  f"{cross:>16.4f} {verdict:>10}")
        claude_floor[cat] = max(cell_uppers)
        print(f"  -> claude internal floor for {cat} (max of direct/framed 97.5%): {claude_floor[cat]:.4f}")

    print()
    print("=" * 88)
    print("ANCHOR 2: v3 cross-seed (external; gemma + qwen on same 5 pilot prompts)")
    print("for each (category, model), draw two independent 5-prompt × 3-seed samples, JSD")
    print("=" * 88)
    print(f"{'cat':<4} {'model':<8} {'median':>8} {'2.5%':>8} {'97.5%':>8}  "
          f"{'cross-cond JSD':>16} {'verdict':>10}")
    v3_floor: dict[str, float] = {}
    for cat in PILOT_QUADRANTS:
        cell_uppers: list[float] = []
        for model_name, path in V3_PATHS.items():
            if not path.exists():
                print(f"{cat:<4} {model_name:<8}  (no v3 data at {path})")
                continue
            rows = _v3_rows(path)
            rows = [r for r in rows if r.get("prompt_id", "")[:2].lower()
                    == cat.lower()[:2]]
            jsds = _v3_subsample_jsd(rows, PILOT_PROMPT_IDS[cat], rng)
            if not jsds:
                print(f"{cat:<4} {model_name:<8}  (insufficient v3 data)")
                continue
            med, lo, hi = _ci(jsds)
            cell_uppers.append(hi)
            cross = cross_cond_jsd[cat]
            verdict = "ABOVE" if cross > hi else "inside"
            print(f"{cat:<4} {model_name:<8} {med:>8.4f} {lo:>8.4f} {hi:>8.4f}  "
                  f"{cross:>16.4f} {verdict:>10}")
        if cell_uppers:
            v3_floor[cat] = max(cell_uppers)
            print(f"  -> v3 external floor for {cat} (max across gemma/qwen 97.5%): {v3_floor[cat]:.4f}")

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"{'cat':<4} {'cross-cond':>10} {'claude floor':>14} {'v3 floor':>10}  "
          f"{'verdict':>30}")
    for cat in PILOT_QUADRANTS:
        cross = cross_cond_jsd[cat]
        cf = claude_floor.get(cat, float("nan"))
        vf = v3_floor.get(cat, float("nan"))
        if cross > max(cf, vf):
            v = "REAL effect (above both floors)"
        elif cross > cf or cross > vf:
            v = "MARGINAL (above one floor only)"
        else:
            v = "noise (inside both floors)"
        print(f"{cat:<4} {cross:>10.4f} {cf:>14.4f} {vf:>10.4f}  {v:>30}")


if __name__ == "__main__":
    main()
