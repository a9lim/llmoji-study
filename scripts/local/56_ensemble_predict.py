#!/usr/bin/env python3
"""Aggregate winning ensemble's per-face predictions across all faces.

Soft-everywhere: emits per-face full ensemble distribution (sum of
per-encoder softmaxes, renormalized), evaluates by mean JSD vs the
empirical Claude (or pooled) per-quadrant distribution, and reports
hard argmax + accuracy + κ as supplementary informational metrics.

Strict-majority voting is removed. The deployable hard label is
``argmax(ensemble_dist)`` — preserved as ``ensemble_pred`` — but the
ranking + evaluation surface is the distribution itself.

Reading:
    For each face f in the union of all encoders' summaries:
        ensemble_dist(f, q) = mean over encoders e in subset of
                              softmax_e(f, q)        (soft vote)
        ensemble_pred(f)    = argmax_q ensemble_dist(f, q)
        ensemble_conf(f)    = ensemble_dist(f, ensemble_pred)
        gt_dist(f, q)       = empirical emission probability over q
                              (Claude-only when --claude-gt; pooled
                              v3+Claude+wild otherwise)
        jsd(f)              = JS-divergence(ensemble_dist(f),
                                            gt_dist(f))  in nats

Output (one row per face) includes:
    - per-encoder pred + max_softmax (parallel columns)
    - ensemble_p_<q> for each quadrant (the full distribution)
    - ensemble_pred (argmax), ensemble_conf (top-1 mass)
    - gt_p_<q> + jsd_vs_gt + similarity (when GT available)
    - empirical_modal + total_emit_count (supplementary)

Usage:
    python scripts/local/56_ensemble_predict.py \\
        --models gemma,qwen,ministral,llama32_3b
    python scripts/local/56_ensemble_predict.py \\
        --models gemma,gpt_oss_20b,granite,ministral,qwen,rinna_jp_3_6b_jpfull \\
        --claude-gt --claude-gt-floor 3

Outputs:
    data/face_likelihood_ensemble_predict{_claude_gt}.tsv  — per-face
    data/face_likelihood_ensemble_predict{_claude_gt}.md   — summary
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llmoji_study.config import DATA_DIR
from llmoji_study.jsd import LN2, js, normalize, similarity

QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")


def _load(model: str, prefer_pilot: bool) -> tuple[pd.DataFrame, str]:
    suffixes = ("_pilot", "") if prefer_pilot else ("", "_pilot")
    for suf in suffixes:
        p = DATA_DIR / f"face_likelihood_{model}{suf}_summary.tsv"
        if p.exists():
            return pd.read_csv(
                p, sep="\t", keep_default_na=False, na_values=[""],
            ), suf or "full"
    sys.exit(f"missing summary for {model} (tried pilot + full)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated encoder names (the winning subset)")
    ap.add_argument("--prefer-pilot", action="store_true",
                    help="prefer pilot summaries over full (default: full first)")
    ap.add_argument("--claude-gt", action="store_true",
                    help="use Claude empirical per-quadrant distribution as "
                         "GT instead of pooled emit-count distribution")
    ap.add_argument("--claude-gt-floor", type=int, default=3,
                    help="min Claude total emits to include face in GT "
                         "(default 3)")
    ap.add_argument("--gt-floor", type=int, default=3,
                    help="min pooled emits to include face in GT (default 3; "
                         "ignored when --claude-gt)")
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
        print(f"  {m:30s} {src:5s}  ({len(df)} faces)")

    union = sorted(set().union(*[set(df.index) for df in frames.values()]))
    print(f"\nunion: {len(union)} faces")

    # Empirical/metadata source: largest available encoder summary.
    base_name = max(frames, key=lambda m: len(frames[m]))
    base = frames[base_name]

    # Per-face GT distribution (raw counts dict). Two paths:
    gt_counts: dict[str, dict[str, int]] = {}
    if args.claude_gt:
        from llmoji_study.claude_gt import load_claude_gt_distribution
        gt_counts = load_claude_gt_distribution(floor=args.claude_gt_floor)
        print(f"Claude-GT: {len(gt_counts)} faces with total ≥ "
              f"{args.claude_gt_floor}")
    else:
        # Pooled mode: read total_emit_<q> columns from base summary.
        for f in union:
            if f not in base.index:
                continue
            br = base.loc[f]
            total = int(br.get("total_emit_count", 0) or 0)
            if total < args.gt_floor:
                continue
            d = {q: int(br.get(f"total_emit_{q}", 0) or 0) for q in QUADRANTS}
            if sum(d.values()) > 0:
                gt_counts[f] = d
        print(f"Pooled-GT: {len(gt_counts)} faces with total ≥ {args.gt_floor}")

    rows: list[dict] = []
    for f in union:
        per_enc_pred: dict[str, str] = {}
        per_enc_conf: dict[str, float] = {}
        per_enc_softmax: dict[str, list[float]] = {}
        for m in models:
            if f not in frames[m].index:
                continue
            r = frames[m].loc[f]
            per_enc_pred[m] = str(r["predicted_quadrant"])
            per_enc_conf[m] = float(r.get("max_softmax", 0.0))
            sm = {q: float(r.get(f"softmax_{q}", 0.0)) for q in QUADRANTS}
            per_enc_softmax[m] = normalize(sm, QUADRANTS)

        if not per_enc_softmax:
            continue  # no encoder has this face; skip

        # Soft vote: mean of per-encoder softmax distributions, then JSD-ready.
        n_voting = len(per_enc_softmax)
        ens = [0.0] * 6
        for m, d in per_enc_softmax.items():
            for i in range(6):
                ens[i] += d[i]
        ens = [x / n_voting for x in ens]
        ens_pred_idx = max(range(6), key=lambda i: ens[i])
        ens_pred = QUADRANTS[ens_pred_idx]
        ens_conf = ens[ens_pred_idx]

        # GT distribution + per-face JSD (if GT exists).
        if f in gt_counts:
            gt_dist = normalize(gt_counts[f], QUADRANTS)
            jsd = js(ens, gt_dist)
            sim = similarity(jsd)
            gt_modal = max(QUADRANTS, key=lambda q: gt_counts[f].get(q, 0))
            gt_total = sum(gt_counts[f].values())
            has_gt = True
        else:
            gt_dist = [0.0] * 6
            jsd = float("nan")
            sim = float("nan")
            gt_modal = ""
            gt_total = 0
            has_gt = False

        # Pooled metadata from base summary.
        emit = 0
        is_claude = False
        if f in base.index:
            br = base.loc[f]
            emit_raw = br.get("total_emit_count", 0)
            try:
                emit = int(emit_raw) if pd.notna(emit_raw) else 0
            except (ValueError, TypeError):
                emit = 0
            is_claude_raw = br.get("is_claude", 0)
            try:
                is_claude = bool(int(is_claude_raw)) if pd.notna(is_claude_raw) else False
            except (ValueError, TypeError):
                is_claude = False

        row = {
            "first_word": f,
            "ensemble_pred": ens_pred,
            "ensemble_conf": round(ens_conf, 4),
            "n_encoders_voting": n_voting,
            "gt_modal": gt_modal,
            "gt_total_emits": gt_total,
            "jsd_vs_gt": round(jsd, 4) if has_gt else float("nan"),
            "similarity": round(sim, 4) if has_gt else float("nan"),
            "argmax_matches_gt_modal": (ens_pred == gt_modal) if has_gt else None,
            "total_emit_count": emit,
            "is_claude": is_claude,
        }
        for i, q in enumerate(QUADRANTS):
            row[f"ensemble_p_{q}"] = round(ens[i], 4)
        for i, q in enumerate(QUADRANTS):
            row[f"gt_p_{q}"] = round(gt_dist[i], 4) if has_gt else float("nan")
        for m in models:
            row[f"{m}_pred"] = per_enc_pred.get(m, "")
            row[f"{m}_conf"] = round(per_enc_conf.get(m, float("nan")), 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    suffix = "_claude_gt" if args.claude_gt else ""
    out_tsv = DATA_DIR / f"face_likelihood_ensemble_predict{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}  ({len(df)} faces)")

    # Aggregate metrics on the GT subset. Two flavors of mean similarity:
    # face-uniform (each face counts equally) and emit-weighted (each
    # face weighted by how often Claude emits it). Reported side-by-side.
    df_gt = df[df["jsd_vs_gt"].notna()].copy()
    n_gt = len(df_gt)
    if n_gt > 0:
        mean_jsd = float(df_gt["jsd_vs_gt"].mean())
        mean_sim = similarity(mean_jsd)
        # Emit-weighted: weight each face's JSD by its total emit count.
        weights = df_gt["gt_total_emits"].astype(float).to_numpy()
        jsd_vals = df_gt["jsd_vs_gt"].astype(float).to_numpy()
        total_w = float(weights.sum())
        if total_w > 0:
            mean_jsd_weighted = float((jsd_vals * weights).sum() / total_w)
        else:
            mean_jsd_weighted = float("nan")
        mean_sim_weighted = similarity(mean_jsd_weighted)
        n_correct = int(df_gt["argmax_matches_gt_modal"].fillna(False).sum())
        acc = n_correct / n_gt
        kappa = cohen_kappa_score(
            df_gt["gt_modal"].tolist(),
            df_gt["ensemble_pred"].tolist(),
            labels=list(QUADRANTS),
        )
    else:
        mean_jsd = float("nan")
        mean_sim = float("nan")
        mean_jsd_weighted = float("nan")
        mean_sim_weighted = float("nan")
        n_correct = 0
        acc = 0.0
        kappa = float("nan")

    lines: list[str] = []
    lines.append("# Ensemble per-face distributions")
    lines.append("")
    lines.append(f"**Encoders:** {', '.join(models)}  "
                 f"(sources: {sources})")
    lines.append(f"**Faces predicted:** {len(df)}")
    lines.append(f"**Faces with GT (for evaluation):** {n_gt}")
    if args.claude_gt:
        lines.append(f"**GT mode:** Claude empirical (total ≥ "
                     f"{args.claude_gt_floor})")
    else:
        lines.append(f"**GT mode:** pooled v3+Claude+wild (total ≥ "
                     f"{args.gt_floor})")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"For each face the ensemble emits a per-quadrant probability "
                 f"distribution (mean of subset softmaxes). GT is Claude's "
                 f"(or pooled) empirical per-quadrant emission distribution. "
                 f"We compare distribution-to-distribution via Jensen-Shannon "
                 f"divergence and report **distribution similarity** = "
                 f"`1 − JSD/ln 2` ∈ [0, 1] (1.0 = identical; max JSD ≈ "
                 f"{LN2:.4f} nats). The deployable output is the *full "
                 f"distribution per face* — \"this face is 56% HP, 23% LP, "
                 f"...\" — not a single hard label.")
    lines.append("")
    if n_gt > 0:
        lines.append("## Headline")
        lines.append("")
        lines.append(
            f"- **Face-uniform mean similarity:** {mean_sim:.3f}  "
            f"(each face counts equally; characterizes Claude's "
            f"*vocabulary*)"
        )
        lines.append(
            f"- **Emit-weighted mean similarity:** {mean_sim_weighted:.3f}  "
            f"(faces weighted by how often Claude emits them; "
            f"characterizes Claude's *emission distribution* — closer to "
            f"deployment-relevance)"
        )
        lines.append(f"  - n_faces evaluated: {n_gt}")
        lines.append(
            f"  - mean JSD: {mean_jsd:.4f} (face-uniform), "
            f"{mean_jsd_weighted:.4f} (emit-weighted) nats"
        )
        lines.append("")
        lines.append("## Per-GT-modal-quadrant breakdown")
        lines.append("")
        lines.append(
            "| GT modal | n | similarity (face-uniform) | "
            "similarity (emit-weighted) |"
        )
        lines.append("|---|---:|---:|---:|")
        for q in QUADRANTS:
            sub = df_gt[df_gt["gt_modal"] == q]
            if len(sub) == 0:
                continue
            sub_jsd = float(sub["jsd_vs_gt"].mean())
            sub_sim = similarity(sub_jsd)
            sub_w = sub["gt_total_emits"].astype(float).to_numpy()
            sub_j = sub["jsd_vs_gt"].astype(float).to_numpy()
            sub_total_w = float(sub_w.sum())
            if sub_total_w > 0:
                sub_jsd_w = float((sub_j * sub_w).sum() / sub_total_w)
                sub_sim_w = similarity(sub_jsd_w)
            else:
                sub_sim_w = float("nan")
            lines.append(
                f"| {q} | {len(sub)} | {sub_sim:.3f} | {sub_sim_w:.3f} |"
            )
        lines.append("")
    lines.append("## Output schema (per-face TSV)")
    lines.append("")
    lines.append("Each row carries:")
    lines.append("")
    lines.append("- `ensemble_p_<q>` for q in {HP, LP, HN-D, HN-S, LN, NB} — "
                 "**the headline output**, the full ensemble distribution.")
    lines.append("- `gt_p_<q>` (when GT exists) — Claude's empirical "
                 "distribution for the same face.")
    lines.append("- `jsd_vs_gt`, `similarity` — per-face evaluation.")
    lines.append("- `<encoder>_pred`, `<encoder>_conf` — per-encoder "
                 "argmax + top-1 mass (for transparency about "
                 "individual contributors).")
    lines.append("- Supplementary: `ensemble_pred` (argmax of distribution), "
                 "`ensemble_conf` (top-1 mass), `argmax_matches_gt_modal` "
                 "(boolean). These are *derived* from the distribution; "
                 "they're the production-shaped reading but not the "
                 "primary output.")
    lines.append("")
    lines.append("## Supplementary metrics (argmax-shaped reading)")
    lines.append("")
    if n_gt > 0:
        lines.append(f"- Hard accuracy (argmax matches GT modal): {acc:.1%} "
                     f"({n_correct}/{n_gt})")
        lines.append(f"- Cohen's κ on argmax: {kappa:.3f}")
        lines.append("")
        lines.append("These characterize a *deployed plugin that emits a "
                     "single quadrant call*. They lose information at GT-tie "
                     "boundaries and aren't the headline.")
        lines.append("")

    out_md = DATA_DIR / f"face_likelihood_ensemble_predict{suffix}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")
    if n_gt > 0:
        print(f"\nENSEMBLE on GT subset (n={n_gt}):")
        print(f"  face-uniform similarity  = {mean_sim:.3f}  "
              f"(mean JSD {mean_jsd:.4f} nats)")
        print(f"  emit-weighted similarity = {mean_sim_weighted:.3f}  "
              f"(mean JSD {mean_jsd_weighted:.4f} nats)")
        print(f"  [supplementary] hard accuracy = {acc:.1%} "
              f"({n_correct}/{n_gt})")
        print(f"  [supplementary] Cohen's κ     = {kappa:.3f}")


if __name__ == "__main__":
    main()
