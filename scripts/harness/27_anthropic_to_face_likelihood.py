# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Convert Anthropic-judge introspective JSONL → face_likelihood TSV.

Bridges scripts 24's introspective judgment output (per-face 6-way
likelihood softmax from a Claude judge — Haiku or Opus) into the
``face_likelihood_<m>_summary.tsv`` schema that scripts 53-56's
ensemble pipeline auto-discovers.

The face_likelihood pipeline (scripts 50-56) was originally designed
for local HF models that expose per-token logits. Anthropic API
models can't produce those, but they CAN produce direct
natural-language quadrant introspection via script 24 — and the
resulting per-face softmax is the same downstream object the
ensemble consumes. This converter writes that softmax into the
columns the ensemble expects, with ``n_prompts_*`` zeroed (since no
prompt-based likelihood was computed) and ``mean_log_prob_*`` as
``log(softmax)``.

Schema for v4 judgment JSONL: each row has
``<short>_lik_HP/LP/HN-D/HN-S/LN/NB`` (six raw likelihoods, may not
sum to 1) plus ``<short>_quadrant`` (derived argmax) +
``<short>_top_likelihood``. Likelihoods are normalized + smoothed
here via ``llmoji_study.jsd.normalize`` (eps=1e-6) so they're
ensemble-compatible.

Empirical columns (``total_emit_*``, ``is_claude``,
``empirical_majority_quadrant``) come from
``data/v3_face_union.parquet`` so the output TSV's behavior-modal
matches what the rest of the pipeline uses.

Usage:
  python scripts/harness/27_anthropic_to_face_likelihood.py --short haiku
  python scripts/harness/27_anthropic_to_face_likelihood.py --short opus
  python scripts/harness/27_anthropic_to_face_likelihood.py --short haiku \\
      --judgment data/haiku_face_quadrant_judgment.jsonl \\
      --output    data/face_likelihood_haiku_summary.tsv

Output:
  data/face_likelihood_<short>_summary.tsv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from llmoji_study.config import DATA_DIR
from llmoji_study.jsd import normalize

QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
FACE_UNION_PATH = DATA_DIR / "v3_face_union.parquet"


def _read_judgment_rows(path: Path, short: str) -> list[dict]:
    """Load successful (non-error) judgment rows; require the
    new-schema ``<short>_lik_*`` fields. Skips error rows + legacy
    v1 rows (``<short>_conf_*``)."""
    if not path.exists():
        sys.exit(f"missing {path}")
    out: list[dict] = []
    new_key = f"{short}_lik_HP"
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            if new_key not in r:
                continue
            out.append(r)
    return out


def _build_face_union_lookup(union_df: pd.DataFrame) -> dict[str, dict]:
    """``{face: {is_claude, total_emit_*, empirical_majority_quadrant,
    total_emit_count}}`` from the canonical face union."""
    out: dict[str, dict] = {}
    for _, row in union_df.iterrows():
        emit_counts = {q: int(row[f"total_emit_{q}"]) for q in QUADRANTS}
        total_emit_count = sum(emit_counts.values())
        if total_emit_count > 0:
            empirical_modal = max(emit_counts, key=lambda q: emit_counts[q])
        else:
            empirical_modal = ""
        out[row["first_word"]] = {
            "is_claude": bool(row["is_claude"]),
            "total_emit_count": total_emit_count,
            "empirical_majority_quadrant": empirical_modal,
            **{f"total_emit_{q}": emit_counts[q] for q in QUADRANTS},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True,
                    help="judge shortname (haiku, opus, ...) — used to "
                         "find the JSONL fields and name the output TSV")
    ap.add_argument("--judgment",
                    help="path to <short>_face_quadrant_judgment.jsonl "
                         "(default: data/<short>_face_quadrant_judgment.jsonl)")
    ap.add_argument("--output",
                    help="path to write face_likelihood_<short>_summary.tsv "
                         "(default: data/face_likelihood_<short>_summary.tsv)")
    args = ap.parse_args()

    short = args.short
    judgment_path = Path(args.judgment) if args.judgment else (
        DATA_DIR / f"{short}_face_quadrant_judgment.jsonl"
    )
    output_path = Path(args.output) if args.output else (
        DATA_DIR / f"face_likelihood_{short}_summary.tsv"
    )

    print(f"reading {judgment_path}")
    rows = _read_judgment_rows(judgment_path, short)
    print(f"  {len(rows)} successful judgment rows")

    if not FACE_UNION_PATH.exists():
        sys.exit(f"missing {FACE_UNION_PATH}; run scripts/local/45_build_face_union.py")
    print(f"reading {FACE_UNION_PATH}")
    union_df = pd.read_parquet(FACE_UNION_PATH)
    union_lookup = _build_face_union_lookup(union_df)
    print(f"  {len(union_lookup)} faces in face union")

    out_rows: list[dict] = []
    n_in_union = 0
    n_argmax_match = 0
    for r in rows:
        face = r["face"]
        # Normalize + smooth likelihoods via the project's standard JSD
        # normalize helper (matches what scripts 53-56 expect downstream).
        raw = {q: float(r.get(f"{short}_lik_{q}", 0.0)) for q in QUADRANTS}
        soft = normalize(raw, vocab=QUADRANTS)  # list aligned to QUADRANTS
        soft_dict = {q: soft[i] for i, q in enumerate(QUADRANTS)}
        # log_prob = log(softmax). Smoothed eps from normalize() means
        # no -inf, but clamp defensively.
        log_probs = {
            q: math.log(soft_dict[q]) if soft_dict[q] > 0 else math.log(1e-12)
            for q in QUADRANTS
        }
        argmax_q = max(QUADRANTS, key=lambda q: soft_dict[q])
        max_soft = soft_dict[argmax_q]

        union = union_lookup.get(face, {
            "is_claude": False,
            "total_emit_count": 0.0,
            "empirical_majority_quadrant": "",
            **{f"total_emit_{q}": 0 for q in QUADRANTS},
        })
        if face in union_lookup:
            n_in_union += 1
        emp = union["empirical_majority_quadrant"]
        if emp and argmax_q == emp:
            n_argmax_match += 1

        # Schema mirrors data/face_likelihood_haiku_summary.tsv (and the
        # other per-encoder summaries that 53-56 auto-discover). The
        # n_prompts_* columns are zero — this signal didn't come from a
        # prompted generation, it came from direct introspection.
        out_row: dict = {"first_word": face}
        for q in QUADRANTS:
            out_row[f"n_prompts_{q}"] = 0
        for q in QUADRANTS:
            out_row[f"mean_log_prob_{q}"] = log_probs[q]
        for q in QUADRANTS:
            out_row[f"softmax_{q}"] = round(soft_dict[q], 6)
        out_row["predicted_quadrant"] = argmax_q
        out_row["max_softmax"] = round(max_soft, 6)
        out_row["n_face_tokens"] = 1  # introspective signal, not token-derived
        out_row["is_claude"] = union["is_claude"]
        out_row["total_emit_count"] = union["total_emit_count"]
        out_row["empirical_majority_quadrant"] = emp
        for q in QUADRANTS:
            out_row[f"total_emit_{q}"] = union[f"total_emit_{q}"]
        out_row["argmax_matches_empirical"] = bool(emp and argmax_q == emp)
        out_rows.append(out_row)

    print(f"  {n_in_union}/{len(rows)} faces matched into face union")
    print(f"  argmax-vs-empirical-modal agreement: "
          f"{n_argmax_match}/{len(rows)} = "
          f"{n_argmax_match / max(len(rows), 1):.1%}")

    df = pd.DataFrame(out_rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nwrote {output_path}  ({len(df)} faces × {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
