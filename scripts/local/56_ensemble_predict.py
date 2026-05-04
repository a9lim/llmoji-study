#!/usr/bin/env python3
"""Aggregate winning ensemble's per-face predictions across all faces.

Takes a list of encoders (the winning subset from script 53), reads
each encoder's per-face summary TSV, computes weighted-vote and
strict-majority per face on the FULL face union (not just the GT
subset), and writes a single per-face prediction TSV with confidence.

Reading:
    For each face f in the union of all encoders' summaries:
        votes[q] = Σ_{enc in subset} softmax_enc(f, q)  # full softmax sum
        ensemble_pred(f) = argmax_q votes[q]
        ensemble_conf(f) = votes[ensemble_pred(f)] / sum(votes)
        majority(f) = quadrant predicted by ≥⌈n/2⌉ encoders, else None

Output (one row per face) includes:
    - per-encoder pred + softmax (parallel columns)
    - ensemble_pred (weighted argmax)
    - ensemble_conf (normalized confidence)
    - majority (or empty if 1-1-1...)
    - vote_strength (e.g. "3-1")
    - empirical (when available — for reference)

Usage:
    python scripts/local/56_ensemble_predict.py \\
        --models gemma,qwen,ministral,llama32_3b
    python scripts/local/56_ensemble_predict.py \\
        --models gemma,qwen,ministral,llama32_3b --prefer-pilot

Outputs:
    data/face_likelihood_ensemble_predict.tsv  — per-face row
    data/face_likelihood_ensemble_predict.md   — summary stats
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _load(model: str, prefer_pilot: bool) -> tuple[pd.DataFrame, str]:
    suffixes = ("_pilot", "") if prefer_pilot else ("", "_pilot")
    for suf in suffixes:
        p = DATA_DIR / f"face_likelihood_{model}{suf}_summary.tsv"
        if p.exists():
            return pd.read_csv(p, sep="\t", keep_default_na=False,
                                na_values=[""]), suf or "full"
    sys.exit(f"missing summary for {model} (tried pilot + full)")


def _strict_majority(preds: list[str]) -> str | None:
    c = Counter(preds)
    top, n = c.most_common(1)[0]
    n_models = len(preds)
    if n >= (n_models // 2 + 1):
        return top
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated encoder names (the winning subset)")
    ap.add_argument("--prefer-pilot", action="store_true",
                    help="prefer pilot summaries over full (default: full first)")
    ap.add_argument("--claude-gt", action="store_true",
                    help="evaluate ensemble accuracy on Claude pilot modal-"
                         "quadrant (only faces Claude actually emits) instead "
                         "of pooled empirical_majority_quadrant")
    ap.add_argument("--claude-gt-floor", type=int, default=1,
                    help="min Claude emits in modal quadrant to include "
                         "face (default 1; raise to 2-3 for sharper labels)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        sys.exit("--models needs at least 2 encoders")

    print(f"loading {len(models)} encoder summaries: {models}")
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for m in models:
        df, src = _load(m, args.prefer_pilot)
        frames[m] = df.set_index("first_word")
        sources[m] = src
        print(f"  {m:20s} {src:5s}  ({len(df)} faces)")

    union = sorted(set().union(*[set(df.index) for df in frames.values()]))
    overlap = sorted(set.intersection(*[set(df.index) for df in frames.values()]))
    print(f"\nunion: {len(union)} faces")
    print(f"overlap (all encoders): {len(overlap)} faces")

    # For empirical/metadata, prefer whichever source has the most rows
    # (typically gemma full).
    base_name = max(frames, key=lambda m: len(frames[m]))
    base = frames[base_name]

    rows = []
    for f in union:
        per_enc_pred: dict[str, str] = {}
        per_enc_conf: dict[str, float] = {}
        per_enc_softmax: dict[str, dict[str, float]] = {}
        for m in models:
            if f not in frames[m].index:
                continue
            r = frames[m].loc[f]
            per_enc_pred[m] = str(r["predicted_quadrant"])
            per_enc_conf[m] = float(r.get("max_softmax", 0.0))
            per_enc_softmax[m] = {q: float(r.get(f"softmax_{q}", 0.0))
                                   for q in QUADRANTS}

        # Sum FULL softmax distribution across encoders (not just argmax conf).
        # This is the cleanest "weighted vote" — each encoder contributes a
        # probability vector, we sum and renormalize.
        votes = {q: sum(per_enc_softmax[m].get(q, 0.0)
                        for m in per_enc_softmax) for q in QUADRANTS}
        total = sum(votes.values())
        if total > 0:
            ensemble_conf = {q: votes[q] / total for q in QUADRANTS}
        else:
            ensemble_conf = {q: 1.0 / len(QUADRANTS) for q in QUADRANTS}
        ensemble_pred = max(ensemble_conf, key=lambda k: ensemble_conf[k])
        ensemble_max_conf = ensemble_conf[ensemble_pred]

        preds_list = list(per_enc_pred.values())
        maj = _strict_majority(preds_list) if preds_list else None
        c = Counter(preds_list)
        strength = "-".join(str(n) for n in sorted(c.values(), reverse=True))

        # Empirical metadata from base (if present)
        emp = ""
        emit = 0
        is_claude = False
        if f in base.index:
            br = base.loc[f]
            emp = str(br.get("empirical_majority_quadrant") or "")
            emit = int(br.get("total_emit_count", 0) or 0)
            is_claude = bool(int(br.get("is_claude", 0) or 0))

        row = {
            "first_word": f,
            "ensemble_pred": ensemble_pred,
            "ensemble_conf": round(ensemble_max_conf, 4),
            "majority_pred": maj or "",
            "vote_strength": strength,
            "n_encoders_voting": len(preds_list),
            "empirical": emp,
            "total_emit_count": emit,
            "is_claude": is_claude,
        }
        for q in QUADRANTS:
            row[f"ensemble_p_{q}"] = round(ensemble_conf[q], 4)
        for m in models:
            row[f"{m}_pred"] = per_enc_pred.get(m, "")
            row[f"{m}_conf"] = round(per_enc_conf.get(m, float("nan")), 3)
        rows.append(row)

    df = pd.DataFrame(rows)

    # If --claude-gt, swap the empirical column with Claude pilot modal labels.
    if args.claude_gt:
        from llmoji_study.claude_gt import load_claude_gt
        cgt = load_claude_gt(floor=args.claude_gt_floor)
        df["empirical"] = df["first_word"].map(
            lambda f: cgt.get(f, ("", 0))[0]
        )
        df["claude_modal_n"] = df["first_word"].map(
            lambda f: cgt.get(f, ("", 0))[1]
        ).astype(int)
        # Override total_emit_count for the GT-mask path below.
        df["total_emit_count"] = df["claude_modal_n"]
        print(f"Claude GT: {int((df['empirical'] != '').sum())} faces with "
              f"modal_n ≥ {args.claude_gt_floor}")

    suffix = "_claude_gt" if args.claude_gt else ""
    out_tsv = DATA_DIR / f"face_likelihood_ensemble_predict{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}  ({len(df)} faces)")

    # Summary stats.
    n_total = len(df)
    n_overlap = int((df["n_encoders_voting"] == len(models)).sum())
    n_partial = n_total - n_overlap
    pred_dist = df["ensemble_pred"].value_counts().to_dict()
    n_with_emp = int((df["empirical"] != "").sum())

    # Accuracy on faces with empirical. For pooled GT (default) require
    # total_emit_count >= 3; for --claude-gt the floor is already applied
    # in load_claude_gt(), so any non-empty empirical counts.
    if args.claude_gt:
        df_gt = df[df["empirical"] != ""]
    else:
        df_gt = df[(df["total_emit_count"] >= 3) & (df["empirical"] != "")]
    if len(df_gt) > 0:
        n_match = int((df_gt["ensemble_pred"] == df_gt["empirical"]).sum())
        acc_gt = n_match / len(df_gt)
    else:
        n_match = 0
        acc_gt = 0.0

    lines: list[str] = []
    lines.append("# Ensemble per-face predictions")
    lines.append("")
    lines.append(f"**Encoders:** {', '.join(models)} "
                 f"(sources: {sources})")
    lines.append(f"**Faces predicted:** {n_total}")
    lines.append(f"  - Full overlap (all encoders predict): {n_overlap}")
    lines.append(f"  - Partial overlap (subset of encoders): {n_partial}")
    lines.append(f"  - With empirical metadata: {n_with_emp}")
    lines.append("")
    lines.append("## Aggregate validation against empirical "
                 f"(total_emit_count ≥ 3, n={len(df_gt)})")
    lines.append("")
    if len(df_gt) > 0:
        lines.append(f"- Ensemble accuracy: **{acc_gt:.1%}** "
                     f"({n_match}/{len(df_gt)})")
        lines.append("")
        # Per-quadrant accuracy
        lines.append("| empirical | n | correct | accuracy |")
        lines.append("|---|---:|---:|---:|")
        for q in QUADRANTS:
            sub = df_gt[df_gt["empirical"] == q]
            n = len(sub)
            if n == 0:
                continue
            nc = int((sub["ensemble_pred"] == q).sum())
            lines.append(f"| {q} | {n} | {nc} | {nc/n:.1%} |")
        lines.append("")
    lines.append("## Predicted quadrant distribution (all faces)")
    lines.append("")
    lines.append("| quadrant | n | share |")
    lines.append("|---|---:|---:|")
    for q in QUADRANTS:
        n = pred_dist.get(q, 0)
        lines.append(f"| {q} | {n} | {n/n_total:.1%} |")
    lines.append("")

    out_md = DATA_DIR / f"face_likelihood_ensemble_predict{suffix}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    print()
    if len(df_gt) > 0:
        print(f"ENSEMBLE accuracy on GT subset: {acc_gt:.1%} "
              f"({n_match}/{len(df_gt)})")


if __name__ == "__main__":
    main()
