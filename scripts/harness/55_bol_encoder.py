"""Bag-of-lexicon (BoL) → face_likelihood-shaped encoder summary TSV.

Converts the per-canonical-kaomoji BoL parquet from script 62 into a
face_likelihood TSV that the ensemble pipeline (52/53/54) auto-
discovers as another encoder named ``bol``.

The synthesizer's structured commit *is* a 6-d quadrant distribution
per face: sum each face's BoL mass on the lexicon's 19 circumplex
slots, partition by Russell quadrant (HP/LP/HN-D/HN-S/LN/NB), then
optionally smooth and L1-normalize. No model call. No projection.

Output schema mirrors ``data/harness/face_likelihood_<enc>_summary.tsv``
(see ``scripts/harness/50_face_likelihood.py:_write_face_likelihood_tsv``)
so the BoL "encoder" plugs into ``face_likelihood_discovery`` like any
other Anthropic-judge or local-LM-head encoder. The differentiator vs
a real likelihood encoder: the ``mean_log_prob_<q>`` columns are
``log(softmax_<q>)`` (no token-derived signal exists), and
``n_face_tokens=0`` flags the row's origin so 53's top-k pooler can
treat BoL specially if needed.

**Independence note for the ensemble.** The BoL encoder is *not*
independent of the harness Haiku face_likelihood encoder — both
trace back to the same Haiku synthesis pass. Treat it as related-
encoder when reporting ensemble results that combine BoL with
``haiku``. BoL vs ``opus`` (introspection arm) and BoL vs the local
LM-head panel (gemma/qwen/ministral/etc) are clean comparisons.

Usage:
    python scripts/harness/55_bol_encoder.py
    python scripts/harness/55_bol_encoder.py --smooth 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from llmoji_study.claude_faces import load_bol_parquet
from llmoji_study.config import (
    CLAUDE_FACES_LEXICON_BAG_PATH,
    DATA_DIR,
)
from llmoji_study.lexicon import (
    QUADRANTS,
    bol_to_quadrant_distribution,
)


# Canonical face union (cross-corpus emit counts) used by every other
# face_likelihood TSV — keeps the empirical columns consistent across
# encoders.
FACE_UNION_PATH = DATA_DIR / "v3_face_union.parquet"
OUT_PATH = DATA_DIR / "harness" / "face_likelihood_bol_summary.tsv"

# `n_face_tokens=0` is the BoL-encoder sentinel value. Real-encoder rows
# (Anthropic judges) carry 1 ('introspective signal, not token-derived');
# local LM-head encoders carry the actual face-token count. 0 marks
# "the encoder doesn't see the kaomoji literally — it's reading the
# synthesizer's structured commit on what the kaomoji means".
BOL_N_FACE_TOKENS = 0


def _build_union_lookup(union_df: pd.DataFrame) -> dict[str, dict]:
    """Same shape as ``50_face_likelihood._build_face_union_lookup`` —
    inlined here to keep this script standalone and avoid importing
    from a sibling script's body."""
    out: dict[str, dict] = {}
    for _, row in union_df.iterrows():
        emit_counts = {q: int(row[f"total_emit_{q}"]) for q in QUADRANTS}
        total = sum(emit_counts.values())
        empirical = max(emit_counts, key=lambda q: emit_counts[q]) if total > 0 else ""
        out[row["first_word"]] = {
            "is_claude": bool(row["is_claude"]),
            "total_emit_count": total,
            "empirical_majority_quadrant": empirical,
            **{f"total_emit_{q}": emit_counts[q] for q in QUADRANTS},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smooth", type=float, default=0.0,
        help=(
            "Dirichlet-like uniform prior added to per-quadrant counts "
            "before normalization. 0 keeps the synthesizer's commit "
            "literal (often hard one-hot for faces with a single primary "
            "pick); 0.05-0.1 rounds edges so JSD-vs-GT isn't dominated "
            "by sparse-row over-confidence. (default: 0.0)"
        ),
    )
    ap.add_argument(
        "--out", type=Path, default=OUT_PATH,
        help=f"output TSV path (default: {OUT_PATH})",
    )
    args = ap.parse_args()

    if not CLAUDE_FACES_LEXICON_BAG_PATH.exists():
        sys.exit(
            f"no BoL parquet at {CLAUDE_FACES_LEXICON_BAG_PATH}; "
            "run scripts/harness/62_corpus_lexicon.py first"
        )
    if not FACE_UNION_PATH.exists():
        sys.exit(
            f"face union not found at {FACE_UNION_PATH}; "
            "run scripts/40_face_union.py first"
        )

    print(f"loading BoL vectors from {CLAUDE_FACES_LEXICON_BAG_PATH.name}...")
    fw, _n, n_v2, B = load_bol_parquet(CLAUDE_FACES_LEXICON_BAG_PATH)
    print(f"  {len(fw)} canonical kaomoji")

    print(f"loading face union from {FACE_UNION_PATH.name}...")
    union_df = pd.read_parquet(FACE_UNION_PATH)
    union_lookup = _build_union_lookup(union_df)
    print(f"  {len(union_lookup)} faces in canonical union")

    # Per-face: BoL → 6-d quadrant distribution → softmax/log-prob row.
    out_rows: list[dict] = []
    n_in_union = 0
    n_no_circumplex = 0
    n_argmax_match = 0
    for i, face in enumerate(fw):
        dist = bol_to_quadrant_distribution(B[i], smooth=args.smooth)
        if dist.sum() <= 0:
            # All extension picks; skip — adding a uniform row would
            # bias the ensemble toward "no information" cells without
            # the encoder having said anything.
            n_no_circumplex += 1
            continue
        soft_dict = {q: float(dist[j]) for j, q in enumerate(QUADRANTS)}
        log_probs = {
            q: math.log(soft_dict[q]) if soft_dict[q] > 0 else math.log(1e-12)
            for q in QUADRANTS
        }
        argmax_q = max(QUADRANTS, key=lambda q: soft_dict[q])
        max_soft = soft_dict[argmax_q]

        union = union_lookup.get(face, {
            "is_claude": False,
            "total_emit_count": 0,
            "empirical_majority_quadrant": "",
            **{f"total_emit_{q}": 0 for q in QUADRANTS},
        })
        if face in union_lookup:
            n_in_union += 1
        emp = union["empirical_majority_quadrant"]
        if emp and argmax_q == emp:
            n_argmax_match += 1

        out_row: dict = {"first_word": face}
        for q in QUADRANTS:
            out_row[f"n_prompts_{q}"] = 0
        for q in QUADRANTS:
            out_row[f"mean_log_prob_{q}"] = log_probs[q]
        for q in QUADRANTS:
            out_row[f"softmax_{q}"] = round(soft_dict[q], 6)
        out_row["predicted_quadrant"] = argmax_q
        out_row["max_softmax"] = round(max_soft, 6)
        out_row["n_face_tokens"] = BOL_N_FACE_TOKENS
        out_row["is_claude"] = union["is_claude"]
        out_row["total_emit_count"] = union["total_emit_count"]
        out_row["empirical_majority_quadrant"] = emp
        for q in QUADRANTS:
            out_row[f"total_emit_{q}"] = union[f"total_emit_{q}"]
        out_row["argmax_matches_empirical"] = bool(emp and argmax_q == emp)
        # BoL-specific provenance — n of v2 description rows that fed
        # the bag. Useful to filter down-rank by sparsity later.
        out_row["n_v2_descs"] = int(n_v2[i])
        out_rows.append(out_row)

    print(f"  {n_in_union}/{len(fw)} faces matched into face union")
    if n_no_circumplex:
        print(
            f"  skipped {n_no_circumplex} faces with no circumplex "
            "commitment (extension-only picks)"
        )
    print(
        f"  argmax-vs-empirical-modal agreement: "
        f"{n_argmax_match}/{len(out_rows)} = "
        f"{n_argmax_match / max(len(out_rows), 1):.1%}"
    )

    df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(
        f"wrote BoL encoder TSV → {args.out}  "
        f"({len(df)} faces × {len(df.columns)} cols)"
    )


if __name__ == "__main__":
    main()
