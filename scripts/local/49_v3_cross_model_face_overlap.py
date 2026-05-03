# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""How many faces are emitted by all 3 v3 models, and do their quadrants agree?

Per-model emitted-face vocab from v3 (h_first @ preferred_layer, split-HN).
Reports:
  - per-model emitted vocab sizes
  - all 6 set intersections (single, pairs, triple)
  - on the triple intersection: per-face × per-model emission distribution
    + modal quadrant + JSD pairwise + an "all 3 agree" tally

Output:
  data/v3_cross_model_face_overlap.tsv
    — one row per face in any model's vocab; cols include per-model
      emission counts, per-quadrant breakdowns, modal quadrant per
      model, and pairwise JSD on the triple-intersection subset
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import load_emotional_features


MODELS = ["gemma", "qwen", "ministral"]
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _emit_dist_by_face(model: str) -> dict[str, dict[str, int]]:
    """{face: {quadrant: count}} for each face emitted in v3 by `model`."""
    M = MODEL_REGISTRY[model]
    df, _ = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_first",
        layer=M.preferred_layer, split_hn=True,
    )
    df = df[df["quadrant"].isin(QUADRANT_ORDER)]
    df = df[df["first_word"].astype(str).str.startswith("(")]
    out: dict[str, dict[str, int]] = {}
    for fw, sub in df.groupby("first_word"):
        counts = {q: 0 for q in QUADRANT_ORDER}
        for q, n in sub["quadrant"].value_counts().items():
            counts[str(q)] = int(n)
        out[str(fw)] = counts
    return out


def _modal(counts: dict[str, int]) -> tuple[str, float]:
    """(modal-quadrant, mode-fraction). Empty → ('', 0)."""
    total = sum(counts.values())
    if total <= 0:
        return ("", 0.0)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return (items[0][0], float(items[0][1]) / total)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (base e). 0 = identical."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], 1e-12, None)))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main() -> None:
    per_model: dict[str, dict[str, dict[str, int]]] = {}
    for m in MODELS:
        per_model[m] = _emit_dist_by_face(m)
        print(f"{m}: {len(per_model[m])} unique faces emitted in v3")

    sets = {m: set(per_model[m].keys()) for m in MODELS}
    print("\n--- intersections ---")
    for m in MODELS:
        only = sets[m] - set().union(*(sets[o] for o in MODELS if o != m))
        print(f"  {m}-only: {len(only)}")
    for a, b in combinations(MODELS, 2):
        ab = sets[a] & sets[b]
        ab_only = ab - sets[[m for m in MODELS if m not in (a, b)][0]]
        print(f"  {a} ∩ {b} (excl 3rd): {len(ab_only)}")
    triple = sets[MODELS[0]] & sets[MODELS[1]] & sets[MODELS[2]]
    print(f"  all 3 intersection: {len(triple)}")
    union = sets[MODELS[0]] | sets[MODELS[1]] | sets[MODELS[2]]
    print(f"  union: {len(union)}")

    # Build the row-per-face TSV (every face in any model's vocab).
    rows: list[dict] = []
    for fw in sorted(union):
        rec: dict = {"first_word": fw}
        present_in: list[str] = []
        modal_per_model: dict[str, str] = {}
        dist_per_model: dict[str, np.ndarray] = {}
        for m in MODELS:
            counts = per_model[m].get(fw, {q: 0 for q in QUADRANT_ORDER})
            total = sum(counts.values())
            rec[f"{m}_emit_count"] = total
            for q in QUADRANT_ORDER:
                rec[f"{m}_emit_{q}"] = counts[q]
            modal_q, modal_f = _modal(counts)
            rec[f"{m}_modal"] = modal_q
            rec[f"{m}_modal_frac"] = round(modal_f, 4)
            if total > 0:
                present_in.append(m)
                modal_per_model[m] = modal_q
                dist_per_model[m] = np.asarray(
                    [counts[q] for q in QUADRANT_ORDER], dtype=np.float64
                )
        rec["present_in"] = ",".join(present_in)
        rec["n_models"] = len(present_in)

        if len(present_in) == 3:
            modes = [modal_per_model[m] for m in MODELS]
            rec["all3_modal_agree"] = len(set(modes)) == 1
            for a, b in combinations(MODELS, 2):
                rec[f"jsd_{a}_{b}"] = round(
                    _jsd(dist_per_model[a], dist_per_model[b]), 4
                )
        else:
            rec["all3_modal_agree"] = None
            for a, b in combinations(MODELS, 2):
                rec[f"jsd_{a}_{b}"] = None
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    out_path = DATA_DIR / "v3_cross_model_face_overlap.tsv"
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"\nwrote {out_path}")

    # Triple-intersection summary
    triple_df = df_out[df_out.n_models == 3].copy()
    print(f"\n--- triple-intersection summary (n={len(triple_df)}) ---")
    if len(triple_df) > 0:
        agree = int(triple_df.all3_modal_agree.sum())
        print(f"  all 3 share modal quadrant: {agree}/{len(triple_df)} "
              f"({agree/len(triple_df):.0%})")
        for a, b in combinations(MODELS, 2):
            same = int(((triple_df[f"{a}_modal"] == triple_df[f"{b}_modal"]).sum()))
            mean_jsd = float(triple_df[f"jsd_{a}_{b}"].mean())
            print(f"  {a} vs {b}: modal agree {same}/{len(triple_df)} "
                  f"({same/len(triple_df):.0%}); mean JSD = {mean_jsd:.3f}")

        # Per-quadrant agreement breakdown
        print("\n  modal-agreement breakdown (when all 3 agree):")
        if agree > 0:
            agreed = triple_df[triple_df.all3_modal_agree]
            for q in QUADRANT_ORDER:
                n = int((agreed.gemma_modal == q).sum())
                if n > 0:
                    print(f"    {q}: {n}")

        # Show top disagreeing faces by mean pairwise JSD
        triple_df["mean_jsd"] = triple_df[
            [f"jsd_{a}_{b}" for a, b in combinations(MODELS, 2)]
        ].mean(axis=1)
        worst = triple_df.sort_values(by="mean_jsd", ascending=False).head(10)
        print("\n  top-10 most-divergent shared faces (by mean pairwise JSD):")
        for _, r in worst.iterrows():
            modes = " / ".join(f"{m}={r[f'{m}_modal']}({int(r[f'{m}_emit_count'])})"
                               for m in MODELS)
            print(f"    {r.first_word}  jsd={r.mean_jsd:.3f}  {modes}")


if __name__ == "__main__":
    main()
