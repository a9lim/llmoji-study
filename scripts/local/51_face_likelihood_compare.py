#!/usr/bin/env python3
"""Compare gemma vs qwen face_likelihood predictions.

Methodically identify (1) where the two encoders disagree on
predicted_quadrant, (2) which one matches v3 empirical-emission
majority more often when they disagree, (3) which faces both get
wrong (or agree-but-wrong, which is its own category — likelihood
signal vs sampling-frequency signal).

Inputs (from script 50):
    data/face_likelihood_{gemma,qwen}{,_pilot}_summary.tsv

Outputs:
    data/face_likelihood_gemma_vs_qwen{,_pilot}.tsv  — per-face row
    data/face_likelihood_gemma_vs_qwen{,_pilot}.md   — categorical breakdown

The TSV is the sortable artifact; the markdown is the human-readable
summary with disagreement matrix + concrete face listings per category.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _load_summary(model: str, pilot: bool) -> pd.DataFrame:
    suffix = "_pilot" if pilot else ""
    p = DATA_DIR / f"face_likelihood_{model}{suffix}_summary.tsv"
    if not p.exists():
        sys.exit(f"missing: {p}\nrun: python scripts/local/50_face_likelihood.py "
                 f"--model {model}")
    return pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare gemma vs qwen face_likelihood predictions.")
    ap.add_argument("--pilot", action="store_true",
                    help="compare *_pilot_summary.tsv instead of full")
    ap.add_argument("--ground-truth-floor", type=int, default=3,
                    help="min v3 emit count to treat empirical_majority as ground truth")
    args = ap.parse_args()

    gemma = _load_summary("gemma", args.pilot)
    qwen = _load_summary("qwen", args.pilot)

    print(f"gemma summary: {len(gemma)} rows")
    print(f"qwen summary:  {len(qwen)} rows")

    overlap = sorted(set(gemma["first_word"]) & set(qwen["first_word"]))
    only_gemma = sorted(set(gemma["first_word"]) - set(qwen["first_word"]))
    only_qwen = sorted(set(qwen["first_word"]) - set(gemma["first_word"]))
    print(f"overlap:       {len(overlap)} faces")
    if only_gemma or only_qwen:
        print(f"  gemma-only:  {len(only_gemma)}")
        print(f"  qwen-only:   {len(only_qwen)}")

    g = gemma.set_index("first_word")
    q = qwen.set_index("first_word")

    rows = []
    for f in overlap:
        gr, qr = g.loc[f], q.loc[f]
        emp = str(gr.get("empirical_majority_quadrant") or "")
        emit = int(gr.get("total_emit_count", 0) or 0)
        has_gt = bool(emp) and emit >= args.ground_truth_floor
        gpred, qpred = gr["predicted_quadrant"], qr["predicted_quadrant"]
        rows.append({
            "first_word": f,
            "gemma_pred": gpred,
            "gemma_softmax": float(gr.get("max_softmax", float("nan"))),
            "qwen_pred": qpred,
            "qwen_softmax": float(qr.get("max_softmax", float("nan"))),
            "empirical": emp,
            "total_emit_count": emit,
            "is_claude": bool(int(gr.get("is_claude", 0) or 0)),
            "has_ground_truth": has_gt,
            "agree": gpred == qpred,
            "gemma_matches_empirical": has_gt and gpred == emp,
            "qwen_matches_empirical": has_gt and qpred == emp,
        })
    df = pd.DataFrame(rows)

    suffix = "_pilot" if args.pilot else ""
    out_tsv = DATA_DIR / f"face_likelihood_gemma_vs_qwen{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    # --- Categorical breakdown ---
    n_total = len(df)
    n_agree = int(df["agree"].sum())
    n_disagree = n_total - n_agree
    df_gt = df[df["has_ground_truth"]]
    df_gt_agree = df_gt[df_gt["agree"]]
    df_gt_dis = df_gt[~df_gt["agree"]]

    both_agree_correct = int(df_gt_agree["gemma_matches_empirical"].sum())
    both_agree_wrong = len(df_gt_agree) - both_agree_correct
    gemma_only = int((df_gt_dis["gemma_matches_empirical"]
                      & ~df_gt_dis["qwen_matches_empirical"]).sum())
    qwen_only = int((df_gt_dis["qwen_matches_empirical"]
                     & ~df_gt_dis["gemma_matches_empirical"]).sum())
    both_wrong = int((~df_gt_dis["gemma_matches_empirical"]
                      & ~df_gt_dis["qwen_matches_empirical"]).sum())

    lines: list[str] = []
    lines.append("# Face_likelihood — gemma vs qwen disagreement audit")
    lines.append("")
    lines.append(f"**Source:** `face_likelihood_gemma{suffix}_summary.tsv`, "
                 f"`face_likelihood_qwen{suffix}_summary.tsv`")
    lines.append(f"**Faces compared:** {n_total} (overlap of both encoders' face union)")
    lines.append(f"**Ground-truth floor:** ≥{args.ground_truth_floor} v3 emissions "
                 "for empirical majority to count as ground truth")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Agree on quadrant: **{n_agree}/{n_total}** "
                 f"({n_agree/n_total*100:.1f}%)")
    lines.append(f"- Disagree on quadrant: **{n_disagree}/{n_total}** "
                 f"({n_disagree/n_total*100:.1f}%)")
    lines.append(f"- Faces with ground truth (≥{args.ground_truth_floor} emits): "
                 f"{len(df_gt)}")
    lines.append("")
    lines.append("## On faces with empirical ground truth")
    lines.append("")
    lines.append("| outcome | count | share |")
    lines.append("|---|---:|---:|")
    if len(df_gt) > 0:
        lines.append(f"| both agree, both correct | {both_agree_correct} "
                     f"| {both_agree_correct/len(df_gt)*100:.1f}% |")
        lines.append(f"| both agree, both wrong | {both_agree_wrong} "
                     f"| {both_agree_wrong/len(df_gt)*100:.1f}% |")
        lines.append(f"| disagree, gemma correct only | {gemma_only} "
                     f"| {gemma_only/len(df_gt)*100:.1f}% |")
        lines.append(f"| disagree, qwen correct only | {qwen_only} "
                     f"| {qwen_only/len(df_gt)*100:.1f}% |")
        lines.append(f"| disagree, both wrong | {both_wrong} "
                     f"| {both_wrong/len(df_gt)*100:.1f}% |")
    lines.append("")
    if (gemma_only + qwen_only) > 0:
        wr_g = gemma_only / (gemma_only + qwen_only)
        lines.append(f"**Disagreement winrate** (faces where exactly one is correct): "
                     f"gemma {gemma_only}/{gemma_only+qwen_only} = {wr_g*100:.1f}%, "
                     f"qwen {qwen_only}/{gemma_only+qwen_only} = {(1-wr_g)*100:.1f}%.")
        lines.append("")
    if both_wrong > 0:
        lines.append(f"**Both-wrong rate on disagreements:** "
                     f"{both_wrong}/{len(df_gt_dis)} = "
                     f"{both_wrong/len(df_gt_dis)*100:.1f}%. "
                     f"Where both miss empirical and emit different predictions, "
                     f"likelihood may be reading intrinsic affect against gemma's "
                     f"sampling-frequency-weighted majority.")
        lines.append("")

    # Disagreement matrix.
    lines.append("## Disagreement matrix (gemma_pred × qwen_pred)")
    lines.append("")
    lines.append("Rows = gemma's prediction, cols = qwen's. Diagonal = agreement.")
    lines.append("")
    pivot = (df.assign(n=1)
               .pivot_table(index="gemma_pred", columns="qwen_pred",
                            values="n", aggfunc="sum", fill_value=0))
    pivot = pivot.reindex(index=QUADRANTS, columns=QUADRANTS, fill_value=0)
    lines.append("| gemma\\qwen | " + " | ".join(QUADRANTS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(QUADRANTS)) + "|")
    for qg in QUADRANTS:
        cells = " | ".join(str(int(pivot.loc[qg, qq])) for qq in QUADRANTS)
        lines.append(f"| **{qg}** | {cells} |")
    lines.append("")

    # Disagreement table by empirical quadrant (for the GT subset).
    if len(df_gt_dis) > 0:
        lines.append("## Disagreements grouped by empirical quadrant")
        lines.append("")
        lines.append("| empirical | n_disagreements | gemma_correct | qwen_correct | both_wrong |")
        lines.append("|---|---:|---:|---:|---:|")
        for qe in QUADRANTS:
            sub = df_gt_dis[df_gt_dis["empirical"] == qe]
            n = len(sub)
            ng = int(sub["gemma_matches_empirical"].sum())
            nq = int(sub["qwen_matches_empirical"].sum())
            nbw = n - ng - nq
            lines.append(f"| {qe} | {n} | {ng} | {nq} | {nbw} |")
        lines.append("")

    # Concrete per-disagreement listings.
    def _listing(title: str, sub: pd.DataFrame, *, head: int | None = None) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if len(sub) == 0:
            lines.append("(none)")
            lines.append("")
            return
        sub = sub.sort_values("total_emit_count", ascending=False)
        if head is not None:
            lines.append(f"Showing top {head} by total v3 emission count "
                         f"(of {len(sub)} total).")
            lines.append("")
            sub = sub.head(head)
        lines.append("| face | gemma | softmax | qwen | softmax | empirical | emits | claude |")
        lines.append("|---|---|---:|---|---:|---|---:|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['first_word']}` "
                f"| {r['gemma_pred']} | {r['gemma_softmax']:.3f} "
                f"| {r['qwen_pred']} | {r['qwen_softmax']:.3f} "
                f"| {r['empirical'] or '—'} | {r['total_emit_count']} "
                f"| {'Y' if r['is_claude'] else 'N'} |"
            )
        lines.append("")

    _listing("Disagreements where gemma matches empirical, qwen doesn't",
             df_gt_dis[df_gt_dis["gemma_matches_empirical"]
                       & ~df_gt_dis["qwen_matches_empirical"]])
    _listing("Disagreements where qwen matches empirical, gemma doesn't",
             df_gt_dis[df_gt_dis["qwen_matches_empirical"]
                       & ~df_gt_dis["gemma_matches_empirical"]])
    _listing("Disagreements where neither matches empirical",
             df_gt_dis[~df_gt_dis["gemma_matches_empirical"]
                       & ~df_gt_dis["qwen_matches_empirical"]])
    _listing("Both agree but both wrong "
             "(likelihood signal vs sampling-frequency signal)",
             df_gt_agree[~df_gt_agree["gemma_matches_empirical"]])

    no_gt_dis = df[~df["has_ground_truth"] & ~df["agree"]].copy()
    if len(no_gt_dis) > 0:
        no_gt_dis["combined_softmax"] = (no_gt_dis["gemma_softmax"]
                                         + no_gt_dis["qwen_softmax"]) / 2
        lines.append("## Disagreements on faces without ground truth")
        lines.append("")
        lines.append(f"**{len(no_gt_dis)}** faces (mostly claude-only or low-emit). "
                     "These are the cells where the cross-model bridge has no "
                     "v3-empirical anchor — the disagreement is itself the signal "
                     "to inspect manually. Top 30 by mean(gemma_softmax, qwen_softmax) "
                     "(highest joint confidence):")
        lines.append("")
        top = no_gt_dis.sort_values("combined_softmax", ascending=False).head(30)
        lines.append("| face | gemma | softmax | qwen | softmax | claude |")
        lines.append("|---|---|---:|---|---:|---|")
        for _, r in top.iterrows():
            lines.append(
                f"| `{r['first_word']}` "
                f"| {r['gemma_pred']} | {r['gemma_softmax']:.3f} "
                f"| {r['qwen_pred']} | {r['qwen_softmax']:.3f} "
                f"| {'Y' if r['is_claude'] else 'N'} |"
            )
        lines.append("")

    out_md = DATA_DIR / f"face_likelihood_gemma_vs_qwen{suffix}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    print()
    print(f"agree:    {n_agree}/{n_total} ({n_agree/n_total*100:.1f}%)")
    print(f"disagree: {n_disagree}/{n_total}")
    if len(df_gt) > 0:
        print(f"on {len(df_gt)} faces with empirical (≥{args.ground_truth_floor} emits):")
        print(f"  both agree+correct: {both_agree_correct}")
        print(f"  both agree+wrong:   {both_agree_wrong}")
        print(f"  gemma correct only: {gemma_only}")
        print(f"  qwen correct only:  {qwen_only}")
        print(f"  both wrong:         {both_wrong}")


if __name__ == "__main__":
    main()
