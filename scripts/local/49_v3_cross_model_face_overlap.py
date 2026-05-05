# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false, reportCallIssue=false
"""How many faces are emitted by each v3 model, and do their quadrants agree?

Per-model emitted-face vocab from v3 (split-HN). Reads JSONL directly —
no hidden-state load needed since face overlap depends only on
canonicalized first_word + quadrant.
Optionally includes Claude data from the groundtruth pilot
(scripts/harness/23_claude_groundtruth_pilot.py).

Reports:
  - per-model emitted vocab sizes
  - per-model "only" sets, all pairwise (a ∩ b excluding others) sets,
    all-N intersection, and union
  - on the all-N intersection: per-face × per-model emission distribution
    + modal quadrant + JSD pairwise + an "all agree" tally
  - top-K most-divergent shared faces by mean pairwise JSD
  - per-quadrant agreement breakdown (when all models agree)

Output:
  data/v3_cross_model_face_overlap.tsv
    — one row per face in any model's vocab; cols include per-model
      emission counts, per-quadrant breakdowns, modal quadrant per
      model, and pairwise JSD on the all-N intersection subset

Usage:
  # Default: 5 v3 main models, no Claude
  python scripts/local/49_v3_cross_model_face_overlap.py
  # Explicit model set + include Claude
  python scripts/local/49_v3_cross_model_face_overlap.py \\
      --models gemma,qwen,ministral,gpt_oss_20b,granite \\
      --include-claude
  # Subset (e.g. preview while v3 chain is still running)
  python scripts/local/49_v3_cross_model_face_overlap.py \\
      --models gemma --include-claude
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.claude_gt import CLAUDE_RUNS_DIR, find_run_files
from llmoji_study.config import DATA_DIR, MODEL_REGISTRY


# Default v3 main lineup post-2026-05-03 vocab-pilot expansion.
DEFAULT_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]

CLAUDE_KEY = "claude"  # Conventional name for the Claude column in outputs.


def _emit_dist_local(model: str) -> dict[str, dict[str, int]] | None:
    """{face: {quadrant: count}} for each face emitted by `model` in v3.
    Returns None if v3 data isn't on disk or the data file is empty.

    Reads the JSONL directly (no hidden-state load needed); face-overlap
    only depends on first_word + quadrant."""
    M = MODEL_REGISTRY[model]
    if not M.emotional_data_path.exists():
        print(f"  [{model}] no v3 data at {M.emotional_data_path}; skipping")
        return None
    rows: list[dict] = []
    with M.emotional_data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
    if not rows:
        print(f"  [{model}] empty JSONL; skipping")
        return None
    # Derive 6-way quadrant from prompt_id (rule-3 split: hn01-20 = HN-D,
    # hn21-40 = HN-S; lp/hp/ln/nb prefix → that quadrant).
    out: dict[str, dict[str, int]] = {}
    n_kept = 0
    for r in rows:
        pid = (r.get("prompt_id") or "").lower()
        fw_raw = r.get("first_word") or ""
        fw = canonicalize_kaomoji(fw_raw) or ""
        if not (isinstance(fw, str) and len(fw) > 0 and fw.startswith("(")):
            continue
        if pid.startswith("hp"):
            q = "HP"
        elif pid.startswith("lp"):
            q = "LP"
        elif pid.startswith("nb"):
            q = "NB"
        elif pid.startswith("ln"):
            q = "LN"
        elif pid.startswith("hn"):
            try:
                n = int(pid[2:])
            except ValueError:
                continue
            q = "HN-D" if n <= 20 else "HN-S"
        else:
            continue
        if q not in QUADRANT_ORDER:
            continue
        entry = out.setdefault(fw, {qq: 0 for qq in QUADRANT_ORDER})
        entry[q] += 1
        n_kept += 1
    if not out:
        print(f"  [{model}] no kaomoji-bearing rows; skipping")
        return None
    return out


def _emit_dist_claude(claude_runs_dir: Path) -> dict[str, dict[str, int]] | None:
    """Same shape as _emit_dist_local, but for Claude groundtruth runs.
    Unions over every run-N.jsonl in ``claude_runs_dir``. Quadrant is
    already 6-way (no split-HN reconstruction needed). Faces
    canonicalized to match local-model first_word column."""
    runs = find_run_files(claude_runs_dir)
    if not runs:
        print(f"  [{CLAUDE_KEY}] no run-*.jsonl in {claude_runs_dir}; skipping")
        return None
    out: dict[str, dict[str, int]] = {}
    n_rows = 0
    for _idx, jsonl_path in runs:
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                q = r.get("quadrant", "")
                if q not in QUADRANT_ORDER:
                    continue
                fw_raw = r.get("first_word") or ""
                fw = canonicalize_kaomoji(fw_raw) or ""
                if not fw or not fw.startswith("("):
                    continue
                n_rows += 1
                entry = out.setdefault(fw, {qq: 0 for qq in QUADRANT_ORDER})
                entry[q] += 1
    if not out:
        print(f"  [{CLAUDE_KEY}] no kaomoji-bearing rows; skipping")
        return None
    print(f"  [{CLAUDE_KEY}] {n_rows} kaomoji-bearing rows, "
          f"{len(out)} unique faces")
    return out


def _modal(counts: dict[str, int]) -> tuple[str, float]:
    """(modal-quadrant, mode-fraction). Empty → ('', 0)."""
    total = sum(counts.values())
    if total <= 0:
        return ("", 0.0)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return (items[0][0], float(items[0][1]) / total)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. 0 = identical."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], 1e-12, None)))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model keys from MODEL_REGISTRY. "
             f"Default: {','.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--include-claude", action="store_true",
        help=f"Also include Claude data from {CLAUDE_RUNS_DIR.name}/run-*.jsonl.",
    )
    parser.add_argument(
        "--output", default=str(DATA_DIR / "v3_cross_model_face_overlap.tsv"),
        help="Output TSV path. Default: data/v3_cross_model_face_overlap.tsv",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K most-divergent shared faces to print. Default: 10",
    )
    args = parser.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in MODEL_REGISTRY:
            raise SystemExit(
                f"unknown model {m!r}; known: {sorted(MODEL_REGISTRY)}"
            )

    print(f"requested models: {requested}; include_claude: {args.include_claude}")
    print()

    per_source: dict[str, dict[str, dict[str, int]]] = {}
    for m in requested:
        d = _emit_dist_local(m)
        if d is not None:
            per_source[m] = d
            print(f"  [{m}] {len(d)} unique faces emitted in v3")
    if args.include_claude:
        d = _emit_dist_claude(CLAUDE_RUNS_DIR)
        if d is not None:
            per_source[CLAUDE_KEY] = d

    sources = list(per_source.keys())  # Order = registry order, then claude.
    if not sources:
        raise SystemExit("no data found for any requested source")

    print(f"\nactive sources ({len(sources)}): {sources}")
    print("\n--- intersections ---")
    sets = {m: set(per_source[m].keys()) for m in sources}
    for m in sources:
        only = sets[m] - set().union(*(sets[o] for o in sources if o != m))
        print(f"  {m}-only: {len(only)}")
    if len(sources) >= 2:
        for a, b in combinations(sources, 2):
            ab = sets[a] & sets[b]
            others = [s for s in sources if s not in (a, b)]
            ab_only = ab - set().union(*(sets[o] for o in others)) if others else ab
            print(f"  {a} ∩ {b} (excl others): {len(ab_only)}")
    if len(sources) >= 3:
        all_n = set.intersection(*(sets[m] for m in sources))
        print(f"  all-{len(sources)} intersection: {len(all_n)}")
    union = set.union(*(sets[m] for m in sources))
    print(f"  union: {len(union)}")

    # Build the row-per-face TSV.
    rows: list[dict] = []
    for fw in sorted(union):
        rec: dict = {"first_word": fw}
        present_in: list[str] = []
        modal_per: dict[str, str] = {}
        dist_per: dict[str, np.ndarray] = {}
        for m in sources:
            counts = per_source[m].get(fw, {q: 0 for q in QUADRANT_ORDER})
            total = sum(counts.values())
            rec[f"{m}_emit_count"] = total
            for q in QUADRANT_ORDER:
                rec[f"{m}_emit_{q}"] = counts[q]
            modal_q, modal_f = _modal(counts)
            rec[f"{m}_modal"] = modal_q
            rec[f"{m}_modal_frac"] = round(modal_f, 4)
            if total > 0:
                present_in.append(m)
                modal_per[m] = modal_q
                dist_per[m] = np.asarray(
                    [counts[q] for q in QUADRANT_ORDER], dtype=np.float64
                )
        rec["present_in"] = ",".join(present_in)
        rec["n_sources"] = len(present_in)

        # All-source modal agreement (only meaningful for faces present in all).
        if len(present_in) == len(sources):
            modes = [modal_per[m] for m in sources]
            rec["all_modal_agree"] = len(set(modes)) == 1
        else:
            rec["all_modal_agree"] = None

        # Pairwise JSD: filled in only for pairs where both sources emit the
        # face. Pairs missing one side get None.
        for a, b in combinations(sources, 2):
            if a in dist_per and b in dist_per:
                rec[f"jsd_{a}_{b}"] = round(_jsd(dist_per[a], dist_per[b]), 4)
            else:
                rec[f"jsd_{a}_{b}"] = None
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    out_path = Path(args.output)
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"\nwrote {out_path}")

    # Cross-source agreement summary on the all-N intersection.
    inter = df_out[df_out.n_sources == len(sources)].copy()
    print(f"\n--- all-{len(sources)} intersection summary (n={len(inter)}) ---")
    if len(inter) == 0:
        print("  (no faces shared across all sources — try a smaller model set)")
        return

    if "all_modal_agree" in inter.columns:
        agree = int(inter.all_modal_agree.fillna(False).sum())
        print(f"  all sources share modal quadrant: {agree}/{len(inter)} "
              f"({agree/max(len(inter),1):.0%})")

    print("\n  pairwise modal-quadrant agreement (and mean JSD):")
    for a, b in combinations(sources, 2):
        col = f"jsd_{a}_{b}"
        if col not in inter.columns:
            continue
        sub = inter[inter[col].notna()]
        if len(sub) == 0:
            continue
        same = int(((sub[f"{a}_modal"] == sub[f"{b}_modal"]).sum()))
        mean_jsd = float(sub[col].mean())
        print(f"    {a:<12} vs {b:<12}: modal agree {same}/{len(sub)} "
              f"({same/max(len(sub),1):.0%}); mean JSD = {mean_jsd:.3f}")

    # Per-quadrant breakdown — counts faces where ALL sources agree on the
    # given quadrant. Uses the first source's modal column as canonical
    # (they're equal by definition when all_modal_agree=True).
    if any(inter.all_modal_agree.fillna(False)):
        print("\n  per-quadrant agreement (faces where all sources agree):")
        agreed = inter[inter.all_modal_agree.fillna(False)]
        ref_modal_col = f"{sources[0]}_modal"
        for q in QUADRANT_ORDER:
            n = int((agreed[ref_modal_col] == q).sum())
            if n > 0:
                print(f"    {q:<5} {n}")

    # Most-divergent shared faces by mean pairwise JSD across all pairs that
    # have data on this face.
    pair_cols = [f"jsd_{a}_{b}" for a, b in combinations(sources, 2)]
    inter["mean_jsd"] = inter[pair_cols].mean(axis=1, skipna=True)
    worst = inter.sort_values(by="mean_jsd", ascending=False).head(args.top_k)
    print(f"\n  top-{args.top_k} most-divergent shared faces (by mean pairwise JSD):")
    for _, r in worst.iterrows():
        modes = " / ".join(
            f"{m}={r[f'{m}_modal']}({int(r[f'{m}_emit_count'])})"
            for m in sources
        )
        print(f"    {r.first_word}  jsd={r.mean_jsd:.3f}  {modes}")


if __name__ == "__main__":
    main()
