#!/usr/bin/env python3
"""Exhaustive subset search over face_likelihood encoders.

Auto-discovers every ``face_likelihood_<m>{,_pilot}_summary.tsv`` in
DATA_DIR, takes the inner-join overlap on faces, and evaluates every
non-empty subset of encoders under both voting schemes (weighted +
strict majority) on the ground-truth subset (faces with
``total_emit_count >= --ground-truth-floor``).

Reports:

1. Top-K subsets by weighted-vote accuracy (default K=20).
2. Top-K subsets by strict-majority accuracy (denominator excludes
   abstentions, so k=2 with 50% accuracy on 4 cases reads as 2/4).
3. Per-size best subset (singleton, pair, triple, ..., all).
4. Headline: which subset wins overall, by how much it beats the
   best single encoder.

Inputs (auto-discovered):
    data/face_likelihood_<m>_summary.tsv
    data/face_likelihood_<m>_pilot_summary.tsv

Outputs:
    data/face_likelihood_subset_search.tsv  — every subset, rank-ordered
    data/face_likelihood_subset_search.md   — top-K + per-size + headline

Usage:
    python scripts/local/53_face_likelihood_subset_search.py
    python scripts/local/53_face_likelihood_subset_search.py --top-k 30
    python scripts/local/53_face_likelihood_subset_search.py --exclude glm47_flash
    python scripts/local/53_face_likelihood_subset_search.py --prefer-full

By default we pick pilot if both pilot and full TSVs exist for a model
(since most overnight runs are pilot — keeps the inner-join wider).
``--prefer-full`` flips the priority.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _discover(prefer_full: bool, exclude: set[str]) -> dict[str, tuple[str, bool]]:
    """Map model_short → (path, is_pilot). Picks pilot/full per --prefer-full."""
    pat = re.compile(r"^face_likelihood_(?P<m>.+?)(?P<pilot>_pilot)?_summary\.tsv$")
    found: dict[str, dict[bool, str]] = {}
    for p in sorted(DATA_DIR.glob("face_likelihood_*_summary.tsv")):
        m = pat.match(p.name)
        if not m:
            continue
        # Skip our own / vote-output TSVs.
        if m.group("m").startswith(("vote_", "gemma_vs_qwen", "gemma-")):
            continue
        if m.group("m") in exclude:
            continue
        found.setdefault(m.group("m"), {})[bool(m.group("pilot"))] = str(p)
    out: dict[str, tuple[str, bool]] = {}
    for name, by_pilot in found.items():
        if prefer_full:
            order = [False, True]
        else:
            order = [True, False]
        for is_pilot in order:
            if is_pilot in by_pilot:
                out[name] = (by_pilot[is_pilot], is_pilot)
                break
    return out


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def _strict_majority(preds: tuple[str, ...]) -> str | None:
    c = Counter(preds)
    top, n = c.most_common(1)[0]
    n_models = len(preds)
    if n >= (n_models // 2 + 1):
        return top
    return None


def _confidence_weighted(preds: tuple[str, ...], confs: tuple[float, ...]) -> str:
    weight: dict[str, float] = {}
    for p, c in zip(preds, confs):
        weight[p] = weight.get(p, 0.0) + c
    return max(weight, key=lambda k: weight[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--top-k", type=int, default=20,
                    help="report top-K subsets per ranking (default 20)")
    ap.add_argument("--ground-truth-floor", type=int, default=3,
                    help="min v3 emit count to treat empirical as GT")
    ap.add_argument("--exclude", default="",
                    help="comma-separated encoder names to skip")
    ap.add_argument("--prefer-full", action="store_true",
                    help="prefer full over pilot when both exist (default: pilot first)")
    ap.add_argument("--min-models", type=int, default=1,
                    help="minimum subset size to consider (default 1)")
    ap.add_argument("--claude-gt", action="store_true",
                    help="use Claude pilot modal-quadrant as GT (only "
                         "evaluate on faces Claude actually emits) instead "
                         "of pooled empirical_majority_quadrant")
    ap.add_argument("--claude-gt-floor", type=int, default=1,
                    help="min Claude emits in modal quadrant to include "
                         "face (default 1; raise to 2-3 for sharper labels)")
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    discovered = _discover(args.prefer_full, exclude)
    if len(discovered) < 1:
        raise SystemExit(
            f"no face_likelihood_*_summary.tsv found in {DATA_DIR} "
            f"(after excluding {sorted(exclude)})"
        )
    print(f"discovered {len(discovered)} encoders:")
    for name, (path, is_pilot) in discovered.items():
        print(f"  {name:20s} {'(pilot)' if is_pilot else '(full) '} ← {path}")

    summaries = {name: _load(path).set_index("first_word")
                 for name, (path, _) in discovered.items()}
    overlap = sorted(set.intersection(*[set(s.index) for s in summaries.values()]))
    print(f"\noverlap across all encoders: {len(overlap)} faces")

    # Build a single tall dataframe: per-face per-encoder pred + confidence.
    base = next(iter(summaries.values()))
    base = base.loc[overlap]  # type: ignore[assignment]
    # Per-face metadata
    if args.claude_gt:
        from llmoji_study.claude_gt import load_claude_gt
        cgt = load_claude_gt(floor=args.claude_gt_floor)
        emp_col = pd.Series(
            [cgt.get(f, ("", 0))[0] for f in overlap], index=overlap
        )
        emit_col = pd.Series(
            [cgt.get(f, ("", 0))[1] for f in overlap], index=overlap, dtype=int
        )
        has_gt = pd.Series([f in cgt for f in overlap], index=overlap)
        print(f"Claude GT subset (modal_n ≥ {args.claude_gt_floor}): "
              f"{int(has_gt.sum())} faces")
    else:
        emp_col = base["empirical_majority_quadrant"].astype(str).fillna("")
        emit_col = base["total_emit_count"].fillna(0).astype(int)
        has_gt = (emp_col != "") & (emit_col >= args.ground_truth_floor)
        print(f"GT subset (≥{args.ground_truth_floor} emits): "
              f"{int(has_gt.sum())} faces")

    # Pre-extract per-encoder preds + softmax for the GT subset.
    gt_idx = [f for f, ok in zip(overlap, has_gt) if ok]
    emp_gt = [emp_col[f] for f in gt_idx]
    n_gt = len(gt_idx)

    encoders = sorted(discovered)
    pred_by_enc: dict[str, list[str]] = {}
    conf_by_enc: dict[str, list[float]] = {}
    for e in encoders:
        s = summaries[e]
        pred_by_enc[e] = [str(s.loc[f, "predicted_quadrant"]) for f in gt_idx]
        conf_by_enc[e] = [float(s.loc[f, "max_softmax"]) for f in gt_idx]

    # Per-encoder solo accuracy + Cohen's kappa vs empirical.
    # Kappa corrects accuracy for chance: a model that always predicts the
    # majority class gets credit only above its base rate. With 6 quadrants
    # and unequal class priors, kappa is a more honest single-encoder score.
    solo_acc = {e: sum(p == emp for p, emp in zip(pred_by_enc[e], emp_gt)) / n_gt
                for e in encoders}
    solo_kappa = {e: cohen_kappa_score(emp_gt, pred_by_enc[e],
                                       labels=QUADRANTS)
                  for e in encoders}

    # Iterate every non-empty subset, eval both vote schemes.
    # Track full per-face vote predictions per subset so we can compute
    # Cohen's kappa vs empirical (chance-corrected agreement). Kappa uses
    # the FULL n_gt (we treat majority abstentions by carrying the
    # weighted-vote pick into a "kappa-on-resolved-only" stream — see
    # majority_kappa_resolved which uses the resolved subset only).
    rows = []
    for r in range(args.min_models, len(encoders) + 1):
        for combo in combinations(encoders, r):
            n_correct_w = 0
            n_correct_m = 0
            n_resolved = 0
            weighted_preds: list[str] = []
            majority_preds_resolved: list[str] = []
            empirical_resolved: list[str] = []
            for i, emp in enumerate(emp_gt):
                preds = tuple(pred_by_enc[e][i] for e in combo)
                confs = tuple(conf_by_enc[e][i] for e in combo)
                w = _confidence_weighted(preds, confs)
                weighted_preds.append(w)
                if w == emp:
                    n_correct_w += 1
                m = _strict_majority(preds)
                if m is not None:
                    n_resolved += 1
                    majority_preds_resolved.append(m)
                    empirical_resolved.append(emp)
                    if m == emp:
                        n_correct_m += 1
            weighted_kappa = cohen_kappa_score(emp_gt, weighted_preds,
                                               labels=QUADRANTS)
            if n_resolved > 0:
                majority_kappa = cohen_kappa_score(empirical_resolved,
                                                   majority_preds_resolved,
                                                   labels=QUADRANTS)
            else:
                majority_kappa = float("nan")
            rows.append({
                "size": r,
                "encoders": ",".join(combo),
                "weighted_correct": n_correct_w,
                "weighted_acc": n_correct_w / n_gt,
                "weighted_kappa": weighted_kappa,
                "majority_correct": n_correct_m,
                "majority_resolved": n_resolved,
                "majority_acc_resolved": (n_correct_m / n_resolved
                                          if n_resolved > 0 else 0.0),
                "majority_kappa_resolved": majority_kappa,
                "abstain_rate": 1 - n_resolved / n_gt,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("weighted_acc", ascending=False).reset_index(drop=True)

    suffix = "_claude_gt" if args.claude_gt else ""
    out_tsv = DATA_DIR / f"face_likelihood_subset_search{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}  ({len(df)} subsets)")

    # Best subset overall (weighted).
    best = df.iloc[0]
    best_majority = df.sort_values("majority_correct", ascending=False).iloc[0]
    best_solo = max(solo_acc.items(), key=lambda kv: kv[1])
    best_solo_kappa = max(solo_kappa.items(), key=lambda kv: kv[1])
    best_kappa = df.sort_values("weighted_kappa", ascending=False).iloc[0]

    # Pairwise Cohen's kappa across encoders (whole overlap, not just GT).
    # On the full overlap we don't have empirical labels for many faces,
    # but we DO have all encoders' predictions, so kappa-between-encoders
    # is well-defined and tells us how independent the encoders are even
    # outside the GT subset.
    overlap_preds: dict[str, list[str]] = {}
    for e in encoders:
        s = summaries[e]
        overlap_preds[e] = [str(s.loc[f, "predicted_quadrant"]) for f in overlap]
    pair_kappa: dict[tuple[str, str], float] = {}
    for i, e1 in enumerate(encoders):
        for e2 in encoders[i+1:]:
            pair_kappa[(e1, e2)] = cohen_kappa_score(
                overlap_preds[e1], overlap_preds[e2], labels=QUADRANTS,
            )

    # Per-size best.
    per_size = df.sort_values(["size", "weighted_acc"],
                              ascending=[True, False]).groupby("size").first()

    # Build markdown report.
    lines = []
    lines.append("# Face_likelihood — exhaustive subset search")
    lines.append("")
    lines.append(f"**Encoders:** {len(encoders)}  ({', '.join(encoders)})")
    lines.append(f"**Faces (overlap):** {len(overlap)}")
    if args.claude_gt:
        lines.append(f"**GT subset (Claude pilot modal, "
                     f"floor={args.claude_gt_floor}):** {n_gt}")
    else:
        lines.append(f"**GT subset (≥{args.ground_truth_floor} emits, "
                     f"pooled v3+Claude+wild):** {n_gt}")
    lines.append(f"**Subsets evaluated:** {len(df)}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Best single encoder by accuracy: **{best_solo[0]}** at "
                 f"{best_solo[1]:.1%} ({int(best_solo[1] * n_gt)}/{n_gt}); "
                 f"Cohen's κ = {solo_kappa[best_solo[0]]:.3f}")
    if best_solo[0] != best_solo_kappa[0]:
        lines.append(f"- Best single encoder by κ: **{best_solo_kappa[0]}** at "
                     f"κ = {best_solo_kappa[1]:.3f} "
                     f"(accuracy {solo_acc[best_solo_kappa[0]]:.1%})")
    lines.append(f"- Best weighted-vote subset by accuracy: "
                 f"**{{{best['encoders']}}}** at "
                 f"**{best['weighted_acc']:.1%}** "
                 f"({int(best['weighted_correct'])}/{n_gt}) — "
                 f"size {int(best['size'])}, "
                 f"+{(best['weighted_acc'] - best_solo[1]) * 100:.1f}pp over best single; "
                 f"κ = {best['weighted_kappa']:.3f}")
    lines.append(f"- Best weighted-vote subset by κ: "
                 f"**{{{best_kappa['encoders']}}}** at "
                 f"κ = **{best_kappa['weighted_kappa']:.3f}** "
                 f"(accuracy {best_kappa['weighted_acc']:.1%}, "
                 f"size {int(best_kappa['size'])})")
    lines.append(f"- Best strict-majority subset: **{{{best_majority['encoders']}}}** at "
                 f"{best_majority['majority_acc_resolved']:.1%} on "
                 f"{int(best_majority['majority_resolved'])} resolved "
                 f"(abstains on "
                 f"{n_gt - int(best_majority['majority_resolved'])} all-distinct); "
                 f"κ = {best_majority['majority_kappa_resolved']:.3f}")
    lines.append("")
    lines.append("**Reading κ:** Cohen's kappa corrects agreement for "
                 "chance. 0.0 = no signal beyond random, 0.4–0.6 = moderate, "
                 "0.6–0.8 = substantial, >0.8 = near-perfect. Penalizes "
                 "encoders that always predict the majority class — useful "
                 "given GLM's 100%-LN bias. Voting models often have lower "
                 "κ than accuracy because the vote concentrates predictions "
                 "on common quadrants.")
    lines.append("")

    lines.append("## Per-encoder solo accuracy + Cohen's κ vs empirical")
    lines.append("")
    lines.append("| encoder | accuracy | κ |")
    lines.append("|---|---:|---:|")
    for e, a in sorted(solo_acc.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {e} | {a:.1%} ({int(a * n_gt)}/{n_gt}) "
                     f"| {solo_kappa[e]:.3f} |")
    lines.append("")

    lines.append("## Pairwise Cohen's κ across encoders (whole overlap)")
    lines.append("")
    lines.append("Higher κ = more correlated. Useful for ensemble design: "
                 "encoder pairs with low κ make complementary errors and are "
                 "more useful to combine than encoder pairs with high κ.")
    lines.append("")
    lines.append("| pair | κ |")
    lines.append("|---|---:|")
    for (e1, e2), k in sorted(pair_kappa.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {e1} ↔ {e2} | {k:.3f} |")
    lines.append("")

    lines.append(f"## Top {args.top_k} subsets by weighted-vote accuracy")
    lines.append("")
    lines.append("| rank | size | encoders | acc | κ | majority(resolved) | abstain |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for i, r in df.head(args.top_k).iterrows():
        lines.append(
            f"| {int(i) + 1} | {int(r['size'])} | {{{r['encoders']}}} "
            f"| {r['weighted_acc']:.1%} ({int(r['weighted_correct'])}/{n_gt}) "
            f"| {r['weighted_kappa']:.3f} "
            f"| {r['majority_acc_resolved']:.1%} "
            f"({int(r['majority_correct'])}/{int(r['majority_resolved'])}) "
            f"| {r['abstain_rate']:.1%} |"
        )
    lines.append("")

    df_k = df.sort_values("weighted_kappa", ascending=False).reset_index(drop=True)
    lines.append(f"## Top {args.top_k} subsets by weighted-vote Cohen's κ")
    lines.append("")
    lines.append("(Class-imbalanced subsets that ride the empirical "
                 "majority-class base rate score lower here than under raw "
                 "accuracy.)")
    lines.append("")
    lines.append("| rank | size | encoders | κ | acc | majority(resolved) | abstain |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for i, r in df_k.head(args.top_k).iterrows():
        lines.append(
            f"| {int(i) + 1} | {int(r['size'])} | {{{r['encoders']}}} "
            f"| {r['weighted_kappa']:.3f} "
            f"| {r['weighted_acc']:.1%} ({int(r['weighted_correct'])}/{n_gt}) "
            f"| {r['majority_acc_resolved']:.1%} "
            f"({int(r['majority_correct'])}/{int(r['majority_resolved'])}) "
            f"| {r['abstain_rate']:.1%} |"
        )
    lines.append("")

    df_maj = df[df["majority_resolved"] > 0].sort_values(
        ["majority_acc_resolved", "majority_resolved"],
        ascending=[False, False],
    ).reset_index(drop=True)
    lines.append(f"## Top {args.top_k} subsets by strict-majority accuracy")
    lines.append("")
    lines.append("(ties broken by larger n_resolved, i.e. more decisive)")
    lines.append("")
    lines.append("| rank | size | encoders | majority(resolved) | weighted | abstain |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    for i, r in df_maj.head(args.top_k).iterrows():
        lines.append(
            f"| {int(i) + 1} | {int(r['size'])} | {{{r['encoders']}}} "
            f"| {r['majority_acc_resolved']:.1%} "
            f"({int(r['majority_correct'])}/{int(r['majority_resolved'])}) "
            f"| {r['weighted_acc']:.1%} ({int(r['weighted_correct'])}/{n_gt}) "
            f"| {r['abstain_rate']:.1%} |"
        )
    lines.append("")

    lines.append("## Best subset per size (by weighted accuracy)")
    lines.append("")
    lines.append("| size | encoders | acc | κ | majority(resolved) | abstain |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for sz, r in per_size.iterrows():
        lines.append(
            f"| {sz} | {{{r['encoders']}}} "
            f"| {r['weighted_acc']:.1%} ({int(r['weighted_correct'])}/{n_gt}) "
            f"| {r['weighted_kappa']:.3f} "
            f"| {r['majority_acc_resolved']:.1%} "
            f"({int(r['majority_correct'])}/{int(r['majority_resolved'])}) "
            f"| {r['abstain_rate']:.1%} |"
        )
    lines.append("")

    per_size_kappa = df.sort_values(
        ["size", "weighted_kappa"], ascending=[True, False]
    ).groupby("size").first()
    lines.append("## Best subset per size (by κ)")
    lines.append("")
    lines.append("| size | encoders | κ | acc | majority(resolved) | abstain |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for sz, r in per_size_kappa.iterrows():
        lines.append(
            f"| {sz} | {{{r['encoders']}}} "
            f"| {r['weighted_kappa']:.3f} "
            f"| {r['weighted_acc']:.1%} ({int(r['weighted_correct'])}/{n_gt}) "
            f"| {r['majority_acc_resolved']:.1%} "
            f"({int(r['majority_correct'])}/{int(r['majority_resolved'])}) "
            f"| {r['abstain_rate']:.1%} |"
        )
    lines.append("")

    # Inclusion analysis: how often does each encoder appear in the top-K?
    top_set = df.head(args.top_k)
    inclusion: dict[str, int] = {e: 0 for e in encoders}
    for _, r in top_set.iterrows():
        for e in r["encoders"].split(","):
            inclusion[e] += 1
    lines.append(f"## Encoder inclusion frequency in top-{args.top_k} weighted-acc")
    lines.append("")
    lines.append("Heuristic: encoders that appear in nearly all top subsets are "
                 "ensemble-load-bearing; encoders that rarely appear are "
                 "individually weak AND fail to add complementary signal.")
    lines.append("")
    lines.append("| encoder | top-K acc | top-K κ | solo acc | solo κ |")
    lines.append("|---|---:|---:|---:|---:|")
    inclusion_k: dict[str, int] = {e: 0 for e in encoders}
    for _, r in df_k.head(args.top_k).iterrows():
        for e in r["encoders"].split(","):
            inclusion_k[e] += 1
    for e in sorted(inclusion, key=lambda k: -(inclusion[k] + inclusion_k[k])):
        lines.append(f"| {e} | {inclusion[e]}/{args.top_k} "
                     f"| {inclusion_k[e]}/{args.top_k} "
                     f"| {solo_acc[e]:.1%} | {solo_kappa[e]:.3f} |")
    lines.append("")

    out_md = DATA_DIR / f"face_likelihood_subset_search{suffix}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    print()
    print(f"BEST SOLO acc: {best_solo[0]} @ {best_solo[1]:.1%}  "
          f"(κ={solo_kappa[best_solo[0]]:.3f})")
    print(f"BEST SOLO κ:   {best_solo_kappa[0]} @ "
          f"κ={best_solo_kappa[1]:.3f}  ({solo_acc[best_solo_kappa[0]]:.1%})")
    print(f"BEST WEIGHTED ACC: {{{best['encoders']}}}")
    print(f"  size {int(best['size'])}: "
          f"acc={best['weighted_acc']:.1%} "
          f"(+{(best['weighted_acc'] - best_solo[1]) * 100:.1f}pp)  "
          f"κ={best['weighted_kappa']:.3f}")
    print(f"BEST WEIGHTED κ:   {{{best_kappa['encoders']}}}")
    print(f"  size {int(best_kappa['size'])}: "
          f"κ={best_kappa['weighted_kappa']:.3f}  "
          f"acc={best_kappa['weighted_acc']:.1%}")
    print(f"BEST MAJORITY: {{{best_majority['encoders']}}}")
    print(f"  {best_majority['majority_acc_resolved']:.1%} on "
          f"{int(best_majority['majority_resolved'])} resolved  "
          f"κ={best_majority['majority_kappa_resolved']:.3f}")


if __name__ == "__main__":
    main()
