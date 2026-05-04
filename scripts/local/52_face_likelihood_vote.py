#!/usr/bin/env python3
"""N-way voting comparison across face_likelihood encoders.

Combines per-encoder predictions from script 50 into a single
voting prediction per face, and reports:

1. Per-encoder accuracy on the ground-truth subset (faces with
   ``total_emit_count >= --ground-truth-floor``).
2. Voting accuracy under two schemes:
   - Strict majority: pick the prediction held by ≥2 encoders;
     abstain on 1-1-1 (no majority).
   - Confidence-weighted: sum ``max_softmax`` over encoders
     predicting each candidate; argmax (breaks 1-1-1 ties via
     joint confidence).
3. Tie-breaker analysis on the gemma↔qwen disagreement set
   (the 25 cases from script 51 where the two original encoders
   split): how does the third encoder break them, and how many
   does it correctly resolve?
4. Pairwise agreement matrix between encoders.

Inputs (from script 50):
    data/face_likelihood_<m>{,_pilot}_summary.tsv  per --models

Outputs:
    data/face_likelihood_vote_<key>.tsv  — per-face row with all
        predictions, votes, correctness flags
    data/face_likelihood_vote_<key>.md   — full categorical report

The key is ``-`` joined --models plus ``_pilot`` suffix if applicable
(e.g. ``gemma-qwen-ministral_pilot``).

Usage:
    python scripts/local/52_face_likelihood_vote.py \\
        --models gemma,qwen,ministral
    python scripts/local/52_face_likelihood_vote.py \\
        --models gemma,qwen,ministral --pilot
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _parse_model_spec(spec: str, default_pilot: bool) -> tuple[str, bool]:
    """Parse 'gemma' or 'gemma:full' or 'gemma:pilot' → (name, pilot)."""
    if ":" in spec:
        name, mode = spec.split(":", 1)
        if mode not in {"full", "pilot"}:
            sys.exit(f"invalid mode '{mode}' in {spec!r}; expected 'full' or 'pilot'")
        return name, (mode == "pilot")
    return spec, default_pilot


def _load_summary(model: str, pilot: bool) -> pd.DataFrame:
    suffix = "_pilot" if pilot else ""
    p = DATA_DIR / f"face_likelihood_{model}{suffix}_summary.tsv"
    if not p.exists():
        sys.exit(f"missing: {p}\nrun: python scripts/local/50_face_likelihood.py "
                 f"--model {model}")
    return pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])


def _strict_majority(preds: list[str]) -> str | None:
    """Most common prediction; None if there's no majority (all-distinct)."""
    c = Counter(preds)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return None


def _confidence_weighted(preds: list[str], confs: list[float]) -> str:
    """Argmax over Σ confidence per candidate. Always returns a value."""
    weight: dict[str, float] = {}
    for p, c in zip(preds, confs):
        weight[p] = weight.get(p, 0.0) + c
    return max(weight, key=lambda k: weight[k])


def _vote_strength(preds: list[str]) -> str:
    """3-0 / 2-1 / 1-1-1 (or 2-0 / 1-1 / 1 for fewer encoders)."""
    c = Counter(preds)
    counts = sorted(c.values(), reverse=True)
    return "-".join(str(n) for n in counts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="N-way voting across face_likelihood encoders"
    )
    ap.add_argument(
        "--models", required=True,
        help="comma-separated encoder specs. Each spec is either 'NAME' (uses "
             "--pilot default) or 'NAME:full' / 'NAME:pilot' (overrides). "
             "e.g. 'gemma:full,qwen:full,ministral:pilot'.",
    )
    ap.add_argument("--pilot", action="store_true",
                    help="default mode for unannotated specs (default: full)")
    ap.add_argument("--ground-truth-floor", type=int, default=3,
                    help="min v3 emit count to treat empirical as ground truth")
    args = ap.parse_args()

    raw = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(raw) < 2:
        sys.exit("--models needs at least 2 encoders")
    parsed = [_parse_model_spec(s, args.pilot) for s in raw]
    models = [name for name, _ in parsed]
    is_pilot = {name: pilot for name, pilot in parsed}
    key = "-".join(f"{name}{'_pilot' if pilot else ''}" for name, pilot in parsed)

    # Load each summary, restrict to a single index.
    summaries = {m: _load_summary(m, is_pilot[m]).set_index("first_word")
                 for m in models}
    sizes = {m: len(s) for m, s in summaries.items()}
    print("per-encoder summary sizes:", sizes)
    overlap = sorted(set.intersection(*[set(s.index) for s in summaries.values()]))
    print(f"overlap across all {len(models)} encoders: {len(overlap)} faces")

    # Use the first model's empirical / total_emit metadata (it's encoder-
    # invariant, sourced from face_h_first_*.parquet which all encoders
    # populate identically).
    base = summaries[models[0]]

    rows = []
    for f in overlap:
        preds = [str(summaries[m].loc[f, "predicted_quadrant"]) for m in models]
        confs = [float(summaries[m].loc[f, "max_softmax"]) for m in models]
        emp = str(base.loc[f].get("empirical_majority_quadrant") or "")
        emit = int(base.loc[f].get("total_emit_count", 0) or 0)
        has_gt = bool(emp) and emit >= args.ground_truth_floor

        majority = _strict_majority(preds)
        weighted = _confidence_weighted(preds, confs)
        strength = _vote_strength(preds)

        row: dict = {
            "first_word": f,
            "empirical": emp,
            "total_emit_count": emit,
            "is_claude": bool(int(base.loc[f].get("is_claude", 0) or 0)),
            "has_ground_truth": has_gt,
            "vote_majority": majority if majority else "",
            "vote_weighted": weighted,
            "vote_strength": strength,
        }
        for m, p, c in zip(models, preds, confs):
            row[f"{m}_pred"] = p
            row[f"{m}_softmax"] = c
            row[f"{m}_correct"] = (has_gt and p == emp)
        row["majority_correct"] = (has_gt and majority == emp) if majority else None
        row["weighted_correct"] = has_gt and weighted == emp
        rows.append(row)

    df = pd.DataFrame(rows)

    out_tsv = DATA_DIR / f"face_likelihood_vote_{key}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    df_gt = df[df["has_ground_truth"]].copy()
    n_gt = len(df_gt)

    # Per-encoder accuracy.
    per_model_acc = {m: int(df_gt[f"{m}_correct"].sum()) for m in models}
    # Voting accuracy.
    n_majority_resolves = int((df_gt["vote_majority"] != "").sum())
    n_majority_correct = int(df_gt["majority_correct"].fillna(False).sum())
    n_weighted_correct = int(df_gt["weighted_correct"].sum())

    # Pairwise agreement.
    pair_agree: dict[tuple[str, str], int] = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            pair_agree[(m1, m2)] = int((df[f"{m1}_pred"] == df[f"{m2}_pred"]).sum())
    pair_agree_gt: dict[tuple[str, str], int] = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            pair_agree_gt[(m1, m2)] = int(
                (df_gt[f"{m1}_pred"] == df_gt[f"{m2}_pred"]).sum()
            )

    # Vote-strength distribution on GT.
    strength_dist = df_gt["vote_strength"].value_counts().to_dict()

    # Tie-breaker analysis: when exactly one pair agreed (2-1 strength on
    # 3-encoder runs), the dissenter is the model that broke ranks. For
    # 3-encoder runs we report this as "the other model breaks the tie".
    lines: list[str] = []
    lines.append(f"# Face_likelihood — {len(models)}-way vote ({key})")
    lines.append("")
    lines.append(f"**Encoders:** {', '.join(models)}")
    lines.append(f"**Ground-truth floor:** ≥{args.ground_truth_floor} v3 emissions")
    lines.append(f"**Faces compared:** {len(overlap)} (overlap)")
    lines.append(f"**Faces with ground truth:** {n_gt}")
    lines.append("")

    lines.append("## Per-encoder accuracy on GT subset")
    lines.append("")
    lines.append("| encoder | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for m in models:
        lines.append(f"| {m} | {per_model_acc[m]} | {n_gt} | "
                     f"{per_model_acc[m]/n_gt*100:.1f}% |")
    lines.append("")

    lines.append("## Voting accuracy on GT subset")
    lines.append("")
    lines.append("| scheme | correct | denom | accuracy | notes |")
    lines.append("|---|---:|---:|---:|---|")
    if n_majority_resolves > 0:
        lines.append(f"| strict majority (≥2) | {n_majority_correct} "
                     f"| {n_majority_resolves} | "
                     f"{n_majority_correct/n_majority_resolves*100:.1f}% "
                     f"| abstains on {n_gt - n_majority_resolves} all-distinct |")
    lines.append(f"| confidence-weighted | {n_weighted_correct} "
                 f"| {n_gt} | {n_weighted_correct/n_gt*100:.1f}% "
                 f"| argmax on Σ softmax |")
    lines.append("")

    lines.append("## Vote strength distribution on GT subset")
    lines.append("")
    lines.append("| strength | n | share |")
    lines.append("|---|---:|---:|")
    for s, n in sorted(strength_dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {s} | {n} | {n/n_gt*100:.1f}% |")
    lines.append("")

    lines.append("## Pairwise agreement matrix (whole overlap)")
    lines.append("")
    lines.append("| pair | agree | total | rate |")
    lines.append("|---|---:|---:|---:|")
    for pair, n in pair_agree.items():
        lines.append(f"| {pair[0]} ↔ {pair[1]} | {n} | {len(df)} | "
                     f"{n/len(df)*100:.1f}% |")
    lines.append("")
    lines.append("## Pairwise agreement matrix (GT subset)")
    lines.append("")
    lines.append("| pair | agree | total | rate |")
    lines.append("|---|---:|---:|---:|")
    for pair, n in pair_agree_gt.items():
        lines.append(f"| {pair[0]} ↔ {pair[1]} | {n} | {n_gt} | "
                     f"{n/n_gt*100:.1f}% |")
    lines.append("")

    # Tie-breaker analysis: focus on the case the user asked about.
    # When models[0] != models[1], does models[2] (and beyond) break the tie?
    if len(models) >= 3:
        m1, m2, m3 = models[0], models[1], models[2]
        split = df_gt[df_gt[f"{m1}_pred"] != df_gt[f"{m2}_pred"]].copy()
        n_split = len(split)
        if n_split > 0:
            lines.append(f"## Tie-breaker analysis: {m1} ↔ {m2} disagreements "
                         f"(n={n_split}) — does {m3} break them?")
            lines.append("")
            agrees_m1 = int((split[f"{m3}_pred"] == split[f"{m1}_pred"]).sum())
            agrees_m2 = int((split[f"{m3}_pred"] == split[f"{m2}_pred"]).sum())
            agrees_neither = n_split - agrees_m1 - agrees_m2
            lines.append(f"- {m3} sides with **{m1}**: {agrees_m1}/{n_split} "
                         f"({agrees_m1/n_split*100:.1f}%)")
            lines.append(f"- {m3} sides with **{m2}**: {agrees_m2}/{n_split} "
                         f"({agrees_m2/n_split*100:.1f}%)")
            lines.append(f"- {m3} **dissents from both**: {agrees_neither}/{n_split} "
                         f"({agrees_neither/n_split*100:.1f}%)")
            lines.append("")
            # Of the 2-1 ties, how many does majority correctly resolve?
            two_one_mask = ((split[f"{m3}_pred"] == split[f"{m1}_pred"])
                            | (split[f"{m3}_pred"] == split[f"{m2}_pred"]))
            two_one = split[two_one_mask]
            n_two_one = len(two_one)
            n_resolved_correct = int(two_one["majority_correct"].fillna(False).sum())
            if n_two_one > 0:
                lines.append(f"On the {n_two_one} cases where {m3} sided with one "
                             f"of them (2-1 majority), majority-vote was correct on "
                             f"**{n_resolved_correct}/{n_two_one} = "
                             f"{n_resolved_correct/n_two_one*100:.1f}%**.")
                lines.append("")
            # All-distinct cases (1-1-1)
            allsplit = split[(split[f"{m3}_pred"] != split[f"{m1}_pred"])
                             & (split[f"{m3}_pred"] != split[f"{m2}_pred"])]
            if len(allsplit) > 0:
                lines.append(f"### {len(allsplit)} cases all 3 disagree (1-1-1)")
                lines.append("")
                lines.append(f"| face | {m1} | {m2} | {m3} | empirical | emits |")
                lines.append("|---|---|---|---|---|---:|")
                for _, r in allsplit.sort_values(
                        "total_emit_count", ascending=False).iterrows():
                    lines.append(
                        f"| `{r['first_word']}` "
                        f"| {r[f'{m1}_pred']} | {r[f'{m2}_pred']} "
                        f"| {r[f'{m3}_pred']} "
                        f"| {r['empirical'] or '—'} | {r['total_emit_count']} |"
                    )
                lines.append("")

    # Per-quadrant accuracy.
    lines.append("## Per-empirical-quadrant accuracy")
    lines.append("")
    head = ["empirical", "n"] + models + ["majority", "weighted"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---:"] * len(head)) + "|")
    for q in QUADRANTS:
        sub = df_gt[df_gt["empirical"] == q]
        n = len(sub)
        if n == 0:
            continue
        cells = [q, str(n)]
        for m in models:
            cells.append(f"{int(sub[f'{m}_correct'].sum())}/{n}")
        n_maj = int(sub["majority_correct"].fillna(False).sum())
        n_maj_denom = int((sub["vote_majority"] != "").sum())
        cells.append(f"{n_maj}/{n_maj_denom}" if n_maj_denom > 0 else "—")
        cells.append(f"{int(sub['weighted_correct'].sum())}/{n}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Faces only weighted-vote got right (vs no encoder alone).
    won_by_voting = df_gt[df_gt["weighted_correct"]
                          & ~df_gt[[f"{m}_correct" for m in models]].any(axis=1)]
    if len(won_by_voting) > 0:
        lines.append("## Faces where weighted-vote was correct but no single "
                     "encoder was")
        lines.append("")
        lines.append("(Should be empty by definition — voting picks a value "
                     "predicted by at least one encoder. Listed here as a sanity "
                     "check.)")
        lines.append("")

    # Faces where every encoder was wrong (unrecoverable).
    all_wrong = df_gt[~df_gt[[f"{m}_correct" for m in models]].any(axis=1)]
    if len(all_wrong) > 0:
        lines.append(f"## Faces where every encoder missed empirical (n={len(all_wrong)})")
        lines.append("")
        lines.append("These can't be recovered by any vote scheme; they bound "
                     "the ceiling on cross-encoder agreement with v3 sampling.")
        lines.append("")
        cols = ["face"] + [f"{m}_pred" for m in models] + ["empirical", "emits"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in all_wrong.sort_values("total_emit_count",
                                          ascending=False).iterrows():
            cells = [f"`{r['first_word']}`"]
            cells += [r[f"{m}_pred"] for m in models]
            cells += [r["empirical"] or "—", str(r["total_emit_count"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out_md = DATA_DIR / f"face_likelihood_vote_{key}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    print()
    for m in models:
        print(f"  {m:10s}: {per_model_acc[m]}/{n_gt} "
              f"({per_model_acc[m]/n_gt*100:.1f}%)")
    print(f"  weighted   : {n_weighted_correct}/{n_gt} "
          f"({n_weighted_correct/n_gt*100:.1f}%)")
    if n_majority_resolves > 0:
        print(f"  majority   : {n_majority_correct}/{n_majority_resolves} "
              f"({n_majority_correct/n_majority_resolves*100:.1f}%)  "
              f"abstain on {n_gt - n_majority_resolves}")


if __name__ == "__main__":
    main()
