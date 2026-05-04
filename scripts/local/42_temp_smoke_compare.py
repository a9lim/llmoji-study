"""Temperature smoke comparison — T=1.0 pilot vs T=0.7 v3 main marginals.

Implements the gates from docs/2026-05-03-temp-smoke.md:
  Gate A: top-5 face-set Jaccard (per quadrant)
  Gate B: Δentropy (per quadrant)
  Gate C: JSD(T=1.0 marginal, T=0.7 seed=0 marginal) / cross-seed-JSD floor

Per-quadrant decision per model:
  Path A (rerun): fails any of {jaccard < 0.6, Δentropy > 0.5 nats,
    JSD ratio > 1.5}
  Path B (long tail): inside path-A bounds but Δentropy > 0 OR
    JSD ratio > 1.0 in some quadrant
  Path C (no signal): everything quiet

Inputs:
  data/{short}_temp1_pilot.jsonl  — T=1.0, 1 seed × 120 prompts
  data/{short}_emotional_raw.jsonl — T=0.7, 8 seeds × 120 prompts (v3 main)

Output:
  data/temp_smoke_verdict.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS

MODELS = ["gemma", "qwen"]
QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]

JACCARD_FAIL = 0.6
DELTA_H_FAIL = 0.5
JSD_RATIO_FAIL = 1.5
JSD_RATIO_LONG_TAIL = 1.0


def _quadrant_for(prompt_id: str) -> str:
    """Map prompt_id → 6-quadrant label using HN-D/HN-S split."""
    p = next((p for p in EMOTIONAL_PROMPTS if p.id == prompt_id), None)
    if p is None:
        return ""
    if p.quadrant != "HN":
        return p.quadrant
    return "HN-D" if p.pad_dominance == +1 else "HN-S"


def _load_marginal(
    path: Path, *, seed_filter: int | None = None
) -> dict[str, Counter]:
    """Read jsonl rows → {quadrant: Counter(face: n)} marginal."""
    if not path.exists():
        sys.exit(f"missing input: {path}")
    by_q: dict[str, Counter] = defaultdict(Counter)
    n_total = 0
    n_with_face = 0
    for line in path.open():
        r = json.loads(line)
        if r.get("error"):
            continue
        if seed_filter is not None and int(r.get("seed", 0)) != seed_filter:
            continue
        n_total += 1
        face = r.get("first_word") or ""
        if not face:
            continue
        n_with_face += 1
        q = _quadrant_for(r.get("prompt_id", ""))
        if not q:
            continue
        by_q[q][face] += 1
    print(f"  {path.name}: {n_total} rows, {n_with_face} emit-rows "
          f"({100*n_with_face/max(n_total,1):.0f}%)")
    return dict(by_q)


def _all_seed_marginals(path: Path) -> dict[int, dict[str, Counter]]:
    """Load v3 main, partitioned by seed. {seed: {quadrant: Counter}}."""
    by_seed: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for line in path.open():
        r = json.loads(line)
        if r.get("error"):
            continue
        face = r.get("first_word") or ""
        if not face:
            continue
        s = int(r.get("seed", 0))
        q = _quadrant_for(r.get("prompt_id", ""))
        if not q:
            continue
        by_seed[s][q][face] += 1
    return {s: dict(d) for s, d in by_seed.items()}


def _entropy(c: Counter) -> float:
    """Shannon entropy in nats."""
    n = sum(c.values())
    if n == 0:
        return 0.0
    return float(-sum(
        (cnt / n) * np.log(cnt / n) for cnt in c.values() if cnt > 0
    ))


def _jsd(a: Counter, b: Counter) -> float:
    """Jensen-Shannon divergence in nats over the union of faces."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    na = sum(a.values()) or 1
    nb = sum(b.values()) or 1
    pa = np.array([a.get(k, 0) / na for k in keys])
    pb = np.array([b.get(k, 0) / nb for k in keys])
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _top_k_jaccard(a: Counter, b: Counter, k: int = 5) -> float:
    top_a = {f for f, _ in a.most_common(k)}
    top_b = {f for f, _ in b.most_common(k)}
    if not top_a and not top_b:
        return 1.0
    inter = top_a & top_b
    union = top_a | top_b
    return len(inter) / len(union) if union else 1.0


def _path_classify(jaccard: float, dh: float, jsd_ratio: float) -> str:
    if jaccard < JACCARD_FAIL or dh > DELTA_H_FAIL or jsd_ratio > JSD_RATIO_FAIL:
        return "A (rerun)"
    if dh > 0 or jsd_ratio > JSD_RATIO_LONG_TAIL:
        return "B (long tail)"
    return "C (no signal)"


def _analyze_model(short: str) -> tuple[list[dict], str]:
    M = MODEL_REGISTRY[short]
    pilot_path = DATA_DIR / f"{short}_temp1_pilot.jsonl"
    # T=0.7 baseline lives at _temp0.7-suffixed paths after the 2026-05-03
    # rename — the canonical M.emotional_data_path is now reserved for the
    # incoming T=1.0 v3 main rerun.
    main_path = M.emotional_data_path.with_name(
        M.emotional_data_path.name.replace("_emotional_raw.jsonl", "_emotional_raw_temp0.7.jsonl")
    )
    if main_path == M.emotional_data_path:
        # gemma's path is "emotional_raw.jsonl" (no short prefix);
        # rename target is "emotional_raw_temp0.7.jsonl".
        main_path = M.emotional_data_path.with_name("emotional_raw_temp0.7.jsonl")
    print(f"\n=== {short} ===")
    print(f"  pilot (T=1.0): {pilot_path}")
    print(f"  main  (T=0.7): {main_path}")

    t1 = _load_marginal(pilot_path)
    seed_marginals = _all_seed_marginals(main_path)
    if 0 not in seed_marginals:
        sys.exit(f"{main_path} has no seed=0 rows")
    t07_s0 = seed_marginals[0]
    other_seeds = sorted(s for s in seed_marginals if s != 0)

    rows = []
    overall_path = "C (no signal)"
    for q in QUADRANTS:
        a = t1.get(q, Counter())
        b = t07_s0.get(q, Counter())
        if not a or not b:
            rows.append({
                "model": short, "quadrant": q,
                "n_t1": sum(a.values()), "n_t07_s0": sum(b.values()),
                "jaccard": float("nan"), "h_t1": float("nan"),
                "h_t07": float("nan"), "delta_h": float("nan"),
                "jsd": float("nan"), "jsd_seed_floor": float("nan"),
                "jsd_ratio": float("nan"), "path": "skip (insufficient)",
            })
            continue

        jacc = _top_k_jaccard(a, b, k=5)
        h_t1 = _entropy(a)
        h_t07 = _entropy(b)
        dh = h_t1 - h_t07

        jsd_temp = _jsd(a, b)
        floor_vals = []
        for s in other_seeds:
            c = seed_marginals[s].get(q, Counter())
            if c:
                floor_vals.append(_jsd(b, c))
        jsd_floor = float(np.mean(floor_vals)) if floor_vals else 1e-9
        jsd_floor = max(jsd_floor, 1e-9)
        ratio = jsd_temp / jsd_floor

        path = _path_classify(jacc, dh, ratio)
        # Track worst path across quadrants for the model summary.
        if path.startswith("A"):
            overall_path = "A (rerun)"
        elif path.startswith("B") and overall_path != "A (rerun)":
            overall_path = "B (long tail)"

        rows.append({
            "model": short, "quadrant": q,
            "n_t1": sum(a.values()), "n_t07_s0": sum(b.values()),
            "jaccard": jacc, "h_t1": h_t1, "h_t07": h_t07,
            "delta_h": dh, "jsd": jsd_temp,
            "jsd_seed_floor": jsd_floor, "jsd_ratio": ratio,
            "path": path,
        })
    return rows, overall_path


def _format_top(c: Counter, k: int = 5) -> str:
    top = c.most_common(k)
    return ", ".join(f"{f}({n})" for f, n in top)


def main() -> None:
    all_rows: list[dict] = []
    overall: dict[str, str] = {}
    for m in MODELS:
        rows, model_path = _analyze_model(m)
        all_rows.extend(rows)
        overall[m] = model_path

    out = DATA_DIR / "temp_smoke_verdict.md"
    lines: list[str] = []
    lines.append("# Temperature smoke verdict — T=0.7 → T=1.0")
    lines.append("")
    lines.append("Pre-registered gates from `docs/2026-05-03-temp-smoke.md`. "
                 "Per-quadrant per-model classification:")
    lines.append("")
    lines.append("| model | quadrant | n_t1 | n_t07_s0 | top5 jaccard | "
                 "Δentropy (nats) | JSD | JSD_seed_floor | JSD ratio | path |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in all_rows:
        lines.append(
            f"| {r['model']} | {r['quadrant']} | {r['n_t1']} | "
            f"{r['n_t07_s0']} | {r['jaccard']:.2f} | "
            f"{r['delta_h']:+.2f} | {r['jsd']:.3f} | "
            f"{r['jsd_seed_floor']:.3f} | {r['jsd_ratio']:.2f} | "
            f"{r['path']} |"
        )
    lines.append("")
    lines.append("## Overall verdict per model")
    lines.append("")
    for m, path in overall.items():
        lines.append(f"- **{m}**: {path}")
    lines.append("")
    lines.append("Rules: ")
    lines.append(f"- Path A (rerun): jaccard<{JACCARD_FAIL} OR Δentropy>{DELTA_H_FAIL} OR JSD ratio>{JSD_RATIO_FAIL}")
    lines.append(f"- Path B (long tail): Δentropy>0 OR JSD ratio>{JSD_RATIO_LONG_TAIL} (and not A)")
    lines.append("- Path C (no signal): otherwise")
    lines.append("")
    lines.append("## Per-quadrant top-5 faces (T=0.7 seed=0 vs T=1.0)")
    for m in MODELS:
        M = MODEL_REGISTRY[m]
        pilot_path = DATA_DIR / f"{m}_temp1_pilot.jsonl"
        # T=0.7 baseline lives at _temp0.7-suffixed paths after the 2026-05-03
        # rename — the canonical M.emotional_data_path is now reserved for the
        # incoming T=1.0 v3 main rerun.
        main_path = M.emotional_data_path.with_name(
            M.emotional_data_path.name.replace(
                "_emotional_raw.jsonl", "_emotional_raw_temp0.7.jsonl"
            )
        )
        if main_path == M.emotional_data_path:
            # gemma's path is "emotional_raw.jsonl" (no short prefix);
            # rename target is "emotional_raw_temp0.7.jsonl".
            main_path = M.emotional_data_path.with_name(
                "emotional_raw_temp0.7.jsonl"
            )
        if not pilot_path.exists() or not main_path.exists():
            continue
        t1 = _load_marginal(pilot_path)
        t07_s0 = _all_seed_marginals(main_path).get(0, {})
        lines.append("")
        lines.append(f"### {m}")
        lines.append("")
        lines.append("| quadrant | T=0.7 seed=0 top-5 | T=1.0 top-5 |")
        lines.append("|---|---|---|")
        for q in QUADRANTS:
            a = t1.get(q, Counter())
            b = t07_s0.get(q, Counter())
            lines.append(f"| {q} | {_format_top(b)} | {_format_top(a)} |")

    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    print()
    for m, path in overall.items():
        print(f"  {m}: {path}")


if __name__ == "__main__":
    main()
