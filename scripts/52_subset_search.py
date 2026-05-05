#!/usr/bin/env python3
"""Exhaustive subset search over face_likelihood encoders — soft metric.

Auto-discovers every ``face_likelihood_<m>{,_pilot}_summary.tsv`` in
DATA_DIR, takes the inner-join overlap on faces, and evaluates every
non-empty subset of encoders by **mean JSD between the ensemble's
per-quadrant distribution and Claude's empirical per-quadrant
distribution.** Hard-accuracy + Cohen's κ are kept as supplementary
informational metrics but are no longer the primary ranking.

Methodology (post-2026-05-04 soft-everywhere shift):

    For each face f in the GT subset:
        gt_dist(f, q)        = empirical Claude (or pooled) emission
                                probability over quadrants
        encoder_dist(e, f, q) = softmax_q the encoder reports
        ensemble_dist(f, q)   = mean over encoders e in the subset of
                                encoder_dist(e, f, q)
        jsd(f)                = JS-divergence(ensemble_dist(f),
                                              gt_dist(f))  in nats

    subset_score = mean over GT faces of jsd(f)   (lower = better)
    similarity   = 1 − jsd_mean / ln 2            (1.0 = identical)

Strict-majority voting is removed — every operation is now a
distribution combine + distribution comparison. Hard predictions
still surface as argmax(ensemble_dist) for the production-shaped
metric, but they don't drive the ranking.

Inputs (auto-discovered, post-2026-05-05 layout):
    data/local/<m>/face_likelihood[_<variant>]_summary.tsv
    data/harness/face_likelihood_<encoder>_summary.tsv
    (the `_pilot` form, where it exists, is treated as a fallback when
    the canonical full file is absent)

Outputs:
    data/local/face_likelihood_subset_search.tsv               (no --claude-gt)
    data/local/face_likelihood_subset_search.md                (no --claude-gt)
    data/face_likelihood_subset_search_claude_gt.tsv           (cross-platform)
    data/face_likelihood_subset_search_claude_gt.md            (cross-platform)

Usage:
    python scripts/52_subset_search.py
    python scripts/52_subset_search.py --top-k 30
    python scripts/52_subset_search.py --claude-gt --claude-gt-floor 3
    python scripts/52_subset_search.py --prefer-full
"""
from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llmoji_study.config import DATA_DIR
from llmoji_study.face_likelihood_discovery import discover_summaries
from llmoji_study.jsd import LN2, js, normalize, similarity

QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")


def _discover(prefer_full: bool, exclude: set[str]) -> dict[str, tuple[str, bool]]:
    """Map encoder_short → (path, is_pilot). Picks pilot/full per ``prefer_full``.

    Wraps the canonical layout discovery in
    ``llmoji_study.face_likelihood_discovery`` and tags pilot-vs-full so
    callers can label the cell origin.
    """
    found = discover_summaries(prefer_full)
    out: dict[str, tuple[str, bool]] = {}
    for name, path in found.items():
        if name in exclude:
            continue
        # `prefer_full=True` resolved this entry from the full file iff
        # the file basename lacks `_pilot_` directly before `_summary`.
        is_pilot = path.endswith("_pilot_summary.tsv")
        out[name] = (path, is_pilot)
    return out


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def _gt_distribution_pooled(s_base: pd.DataFrame,
                            faces: list[str],
                            floor: int) -> tuple[dict[str, dict[str, int]],
                                                 list[str]]:
    """Default-mode GT: pooled v3+Claude+wild emit counts per face.
    Returns (per-face raw counts dict, list of GT-eligible faces).
    """
    counts: dict[str, dict[str, int]] = {}
    eligible: list[str] = []
    for f in faces:
        row = s_base.loc[f]
        total = int(row.get("total_emit_count", 0) or 0)
        if total < floor:
            continue
        d = {q: int(row.get(f"total_emit_{q}", 0) or 0) for q in QUADRANTS}
        if sum(d.values()) == 0:
            continue
        counts[f] = d
        eligible.append(f)
    return counts, eligible


def _gt_distribution_claude(faces: list[str],
                            floor: int) -> tuple[dict[str, dict[str, int]],
                                                 list[str]]:
    """Claude-GT mode: per-face per-quadrant Claude emission counts."""
    from llmoji_study.claude_gt import load_claude_gt_distribution
    cgt = load_claude_gt_distribution(floor=floor)
    counts = {f: cgt[f] for f in faces if f in cgt}
    eligible = [f for f in faces if f in cgt]
    return counts, eligible


def _encoder_dist(s: pd.DataFrame, face: str) -> list[float]:
    """Pull the 6-element softmax distribution for `face` from encoder
    summary `s`. Falls back to uniform if any softmax_<q> is missing."""
    raw = {q: float(s.loc[face, f"softmax_{q}"] or 0.0) for q in QUADRANTS}
    return normalize(raw, QUADRANTS)


def _per_face_jsd(ensemble_dist: list[float],
                  gt_counts: dict[str, int]) -> float:
    """JSD between ensemble distribution and GT distribution for one face.
    GT counts are normalized (with smoothing) here."""
    return js(ensemble_dist, normalize(gt_counts, QUADRANTS))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--top-k", type=int, default=20,
                    help="report top-K subsets per ranking (default 20)")
    ap.add_argument("--ground-truth-floor", type=int, default=3,
                    help="min emit count to include face in GT (default 3)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated encoder names to skip")
    ap.add_argument("--prefer-full", action="store_true",
                    help="prefer full over pilot when both exist (default: pilot first)")
    ap.add_argument("--min-models", type=int, default=1,
                    help="minimum subset size to consider (default 1)")
    ap.add_argument("--claude-gt", action="store_true",
                    help="use Claude empirical per-face per-quadrant "
                         "distribution as GT instead of pooled emit counts")
    ap.add_argument("--claude-gt-floor", type=int, default=3,
                    help="min Claude total emits to include face in GT "
                         "(default 3 — sparse counts are too noisy as "
                         "distribution estimates)")
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    discovered = _discover(args.prefer_full, exclude)
    if not discovered:
        raise SystemExit(
            f"no face_likelihood*_summary.tsv found under {DATA_DIR / 'local'} "
            f"or {DATA_DIR / 'harness'} (after excluding {sorted(exclude)})"
        )
    print(f"discovered {len(discovered)} encoders:")
    for name, (path, is_pilot) in discovered.items():
        print(f"  {name:30s} {'(pilot)' if is_pilot else '(full) '} ← {path}")

    summaries = {name: _load(path).set_index("first_word")
                 for name, (path, _) in discovered.items()}
    overlap = sorted(set.intersection(*[set(s.index) for s in summaries.values()]))
    print(f"\noverlap across all encoders: {len(overlap)} faces")

    # GT distribution per face (raw counts; normalized lazily downstream).
    base = next(iter(summaries.values())).loc[overlap]
    if args.claude_gt:
        gt_counts, gt_faces = _gt_distribution_claude(
            overlap, args.claude_gt_floor,
        )
        print(f"Claude-GT subset (Claude total >= {args.claude_gt_floor}): "
              f"{len(gt_faces)} faces")
    else:
        gt_counts, gt_faces = _gt_distribution_pooled(
            base, overlap, args.ground_truth_floor,
        )
        print(f"Pooled-GT subset (v3+Claude+wild total >= "
              f"{args.ground_truth_floor}): {len(gt_faces)} faces")

    if not gt_faces:
        raise SystemExit("no GT-eligible faces; aborting")

    # GT distributions normalized once (with epsilon-smoothing for stability).
    gt_dists: dict[str, list[float]] = {
        f: normalize(gt_counts[f], QUADRANTS) for f in gt_faces
    }
    # Per-face emit weights for emit-weighted aggregate metrics.
    # Weight = total emit count for that face. Faces Claude uses more
    # contribute proportionally more to the weighted score.
    gt_weights: dict[str, float] = {
        f: float(sum(gt_counts[f].values())) for f in gt_faces
    }
    total_weight = sum(gt_weights.values())
    # GT modal labels (for supplementary accuracy/κ).
    gt_modal: dict[str, str] = {}
    for f in gt_faces:
        c = gt_counts[f]
        gt_modal[f] = max(QUADRANTS, key=lambda q: c.get(q, 0))

    # Per-encoder per-face full softmax + argmax (lazily cached).
    encoders = sorted(discovered)
    enc_dist: dict[str, dict[str, list[float]]] = {e: {} for e in encoders}
    enc_pred: dict[str, dict[str, str]] = {e: {} for e in encoders}
    for e in encoders:
        s = summaries[e]
        for f in gt_faces:
            d = _encoder_dist(s, f)
            enc_dist[e][f] = d
            enc_pred[e][f] = QUADRANTS[max(range(6), key=lambda i: d[i])]

    # Per-encoder solo metrics: mean JSD vs GT (face-uniform AND emit-
    # weighted), accuracy, κ.
    n_gt = len(gt_faces)
    solo_jsd: dict[str, float] = {}            # face-uniform mean
    solo_jsd_weighted: dict[str, float] = {}   # emit-count-weighted mean
    solo_acc: dict[str, float] = {}
    solo_kappa: dict[str, float] = {}
    for e in encoders:
        jsds = [js(enc_dist[e][f], gt_dists[f]) for f in gt_faces]
        solo_jsd[e] = sum(jsds) / n_gt
        solo_jsd_weighted[e] = sum(
            j * gt_weights[f] for j, f in zip(jsds, gt_faces)
        ) / total_weight
        preds = [enc_pred[e][f] for f in gt_faces]
        modal = [gt_modal[f] for f in gt_faces]
        solo_acc[e] = sum(p == m for p, m in zip(preds, modal)) / n_gt
        solo_kappa[e] = cohen_kappa_score(modal, preds, labels=list(QUADRANTS))

    # Iterate every non-empty subset; ensemble = mean of per-encoder dists.
    rows = []
    for r in range(args.min_models, len(encoders) + 1):
        for combo in combinations(encoders, r):
            jsds: list[float] = []
            preds: list[str] = []
            for f in gt_faces:
                # Ensemble distribution = mean of subset's softmax distributions.
                edist = [0.0] * 6
                for e in combo:
                    d = enc_dist[e][f]
                    for i in range(6):
                        edist[i] += d[i]
                edist = [x / len(combo) for x in edist]
                jsds.append(js(edist, gt_dists[f]))
                preds.append(QUADRANTS[max(range(6), key=lambda i: edist[i])])
            mean_jsd = sum(jsds) / n_gt  # face-uniform mean
            mean_jsd_weighted = sum(
                j * gt_weights[f] for j, f in zip(jsds, gt_faces)
            ) / total_weight  # emit-count-weighted mean
            modal = [gt_modal[f] for f in gt_faces]
            n_correct = sum(p == m for p, m in zip(preds, modal))
            kappa = cohen_kappa_score(modal, preds, labels=list(QUADRANTS))
            rows.append({
                "size": r,
                "encoders": ",".join(combo),
                "mean_jsd": mean_jsd,
                "similarity": similarity(mean_jsd),
                "mean_jsd_weighted": mean_jsd_weighted,
                "similarity_weighted": similarity(mean_jsd_weighted),
                "accuracy": n_correct / n_gt,
                "n_correct": n_correct,
                "kappa": kappa,
            })
    df = pd.DataFrame(rows)
    # Primary sort is mean_similarity DESC (== mean_jsd ASC); equivalent
    # numerically but reads more naturally — higher similarity = better.
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)

    suffix = "_claude_gt" if args.claude_gt else ""
    # Cross-platform claude-GT variant lives at data/ root; local cross-model
    # variant lives at data/local/. Same dispatch for the .md sibling below.
    cross_dir = DATA_DIR if args.claude_gt else DATA_DIR / "local"
    out_tsv = cross_dir / f"face_likelihood_subset_search{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}  ({len(df)} subsets)")

    best = df.iloc[0]
    best_solo_sim = max(
        ((e, similarity(j)) for e, j in solo_jsd.items()),
        key=lambda kv: kv[1],
    )
    per_size = (
        df.sort_values(["size", "similarity"], ascending=[True, False])
        .groupby("size").first()
    )

    # Pairwise Cohen's κ across encoders (whole overlap, not just GT) —
    # informational, surfaces independence/complementarity.
    overlap_preds: dict[str, list[str]] = {}
    for e in encoders:
        s = summaries[e]
        overlap_preds[e] = [str(s.loc[f, "predicted_quadrant"]) for f in overlap]
    pair_kappa: dict[tuple[str, str], float] = {}
    for i, e1 in enumerate(encoders):
        for e2 in encoders[i+1:]:
            pair_kappa[(e1, e2)] = cohen_kappa_score(
                overlap_preds[e1], overlap_preds[e2], labels=list(QUADRANTS),
            )

    # Markdown report.
    lines: list[str] = []
    lines.append("# Face_likelihood — exhaustive subset search (soft / JSD)")
    lines.append("")
    lines.append(f"**Encoders:** {len(encoders)}  ({', '.join(encoders)})")
    lines.append(f"**Faces (overlap):** {len(overlap)}")
    if args.claude_gt:
        lines.append(f"**GT subset (Claude empirical, total ≥ "
                     f"{args.claude_gt_floor}):** {n_gt}")
    else:
        lines.append(f"**GT subset (≥{args.ground_truth_floor} pooled "
                     f"emits):** {n_gt}")
    lines.append(f"**Subsets evaluated:** {len(df)}")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(f"**Headline metric: distribution similarity.** For each "
                 f"face the ensemble emits a per-quadrant probability "
                 f"distribution (mean of subset softmaxes); GT is Claude's "
                 f"(or pooled) empirical per-quadrant distribution. We "
                 f"compare distribution-to-distribution via Jensen-Shannon "
                 f"divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] "
                 f"(1.0 = distributions identical, 0.0 = maximally divergent; "
                 f"max JSD ≈ {LN2:.4f}). Argmax accuracy + Cohen's κ are "
                 f"available in the supplementary appendix below — they are "
                 f"the production-shaped reading but lose information at GT-"
                 f"tie boundaries, so they don't drive ranking.")
    lines.append("")
    lines.append("**Two flavors of mean similarity, reported side-by-side:**")
    lines.append("")
    lines.append("- **Face-uniform (`similarity`)** — arithmetic mean of "
                 "per-face JSD across the GT subset. Each face counts "
                 "equally regardless of how often Claude emits it. Reads "
                 "as: \"how well does the ensemble characterize Claude's "
                 "*vocabulary*?\" — sensitive to long-tail failures.")
    lines.append("- **Emit-weighted (`similarity_weighted`)** — weighted "
                 "by per-face Claude emit count. Faces Claude uses more "
                 "contribute proportionally more to the score. Reads as: "
                 "\"how well does the ensemble characterize Claude's "
                 "*emission distribution*?\" — closer to deployment-relevant "
                 "(plugin user encounters faces at frequency, not "
                 "uniformly). Tends to read higher than face-uniform "
                 "because modal faces are easier wins.")
    lines.append("")
    lines.append("Subset ranking below is by **face-uniform similarity** "
                 "(stricter / more honest about coverage). Weighted column "
                 "shown alongside.")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Best single encoder: **{best_solo_sim[0]}** at "
        f"**face-uniform similarity = {best_solo_sim[1]:.3f}** "
        f"(emit-weighted "
        f"{similarity(solo_jsd_weighted[best_solo_sim[0]]):.3f})"
    )
    lines.append(
        f"- Best ensemble subset: **{{{best['encoders']}}}** at "
        f"**face-uniform similarity = {best['similarity']:.3f}** "
        f"(emit-weighted {best['similarity_weighted']:.3f}); "
        f"size {int(best['size'])}; "
        f"Δ vs best solo (face-uniform) = "
        f"+{best['similarity'] - best_solo_sim[1]:.3f}"
    )
    lines.append("")

    lines.append("## Per-encoder solo distribution-similarity")
    lines.append("")
    lines.append("| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |")
    lines.append("|---|---:|---:|---:|")
    for e in sorted(encoders, key=lambda x: -similarity(solo_jsd[x])):
        lines.append(
            f"| {e} | {similarity(solo_jsd[e]):.3f} "
            f"| {similarity(solo_jsd_weighted[e]):.3f} "
            f"| {solo_jsd[e]:.4f} |"
        )
    lines.append("")

    lines.append("## Pairwise Cohen's κ across encoders (whole overlap)")
    lines.append("")
    lines.append("Higher κ = more correlated. Encoder pairs with low κ make "
                 "complementary errors and are more useful to combine.")
    lines.append("")
    lines.append("| pair | κ |")
    lines.append("|---|---:|")
    for (e1, e2), k in sorted(pair_kappa.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {e1} ↔ {e2} | {k:.3f} |")
    lines.append("")

    lines.append(f"## Top {args.top_k} subsets by face-uniform similarity")
    lines.append("")
    lines.append("| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |")
    lines.append("|---:|---:|---|---:|---:|")
    for i, r in df.head(args.top_k).iterrows():
        lines.append(
            f"| {int(i) + 1} | {int(r['size'])} | {{{r['encoders']}}} "
            f"| {r['similarity']:.3f} | {r['similarity_weighted']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-size best subset (by face-uniform similarity)")
    lines.append("")
    lines.append("| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |")
    lines.append("|---:|---|---:|---:|")
    for sz, r in per_size.iterrows():
        lines.append(
            f"| {sz} | {{{r['encoders']}}} "
            f"| {r['similarity']:.3f} | {r['similarity_weighted']:.3f} |"
        )
    lines.append("")

    lines.append("## Supplementary: argmax accuracy + Cohen's κ "
                 "(production-shaped reading)")
    lines.append("")
    lines.append("These metrics treat GT as a one-hot modal label. They "
                 "characterize a deployed plugin that emits a single "
                 "quadrant call, not the distribution-shipping ensemble "
                 "this script ranks. Reported here for legibility against "
                 "older numbers in the project history.")
    lines.append("")
    lines.append("### Per-encoder solo (argmax)")
    lines.append("")
    lines.append("| encoder | accuracy | κ |")
    lines.append("|---|---:|---:|")
    for e in sorted(encoders, key=lambda x: -solo_acc[x]):
        lines.append(
            f"| {e} | {solo_acc[e]:.1%} ({int(solo_acc[e] * n_gt)}/{n_gt}) "
            f"| {solo_kappa[e]:.3f} |"
        )
    lines.append("")
    lines.append("### Top-10 subsets by argmax accuracy")
    lines.append("")
    df_acc = df.sort_values("accuracy", ascending=False).head(10)
    lines.append("| size | encoders | accuracy | κ | similarity |")
    lines.append("|---:|---|---:|---:|---:|")
    for _, r in df_acc.iterrows():
        lines.append(
            f"| {int(r['size'])} | {{{r['encoders']}}} "
            f"| {r['accuracy']:.1%} ({int(r['n_correct'])}/{n_gt}) "
            f"| {r['kappa']:.3f} | {r['similarity']:.3f} |"
        )
    lines.append("")

    out_md = cross_dir / f"face_likelihood_subset_search{suffix}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
