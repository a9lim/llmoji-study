#!/usr/bin/env python3
"""Top-k per-prompt pooling experiment for face_likelihood.

Currently script 50 aggregates per-face per-quadrant by mean over ALL
prompts in that quadrant (5 in pilot, 20 in full). Top-k pooling uses
the mean of only the top-k MOST-SUPPORTIVE prompts per quadrant —
robust to noisy individual prompts.

For each pilot/full per-cell parquet (face × prompt rows with log_prob):
    score(f, q, k) = mean(top-k log_prob over prompts p in q)
    predicted(f, k) = argmax_q score(f, q, k)

Reports accuracy + Cohen's κ per (encoder, k) on the GT subset.
Compares to baseline (mean-of-all).

Runs on existing per-cell parquets without re-running the model.

Inputs (auto-discovered):
    data/face_likelihood_<m>{,_pilot}.parquet  (per-cell rows)

Outputs:
    data/face_likelihood_topk_pooling.tsv  — per (encoder, k)
    data/face_likelihood_topk_pooling.md   — comparison table

Usage:
    python scripts/local/55_topk_pooling.py
    python scripts/local/55_topk_pooling.py --ks 1,2,3,5,all
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _discover(prefer_full: bool) -> dict[str, tuple[str, bool]]:
    pat = re.compile(r"^face_likelihood_(?P<m>.+?)(?P<pilot>_pilot)?\.parquet$")
    found: dict[str, dict[bool, str]] = {}
    for p in sorted(DATA_DIR.glob("face_likelihood_*.parquet")):
        m = pat.match(p.name)
        if not m:
            continue
        if m.group("m").startswith(("vote_", "gemma_vs_qwen", "gemma-",
                                     "subset_search", "cross_emit",
                                     "topk_pooling", "ensemble_predict")):
            continue
        found.setdefault(m.group("m"), {})[bool(m.group("pilot"))] = str(p)
    out: dict[str, tuple[str, bool]] = {}
    order = [False, True] if prefer_full else [True, False]
    for name, by_pilot in found.items():
        for is_pilot in order:
            if is_pilot in by_pilot:
                out[name] = (by_pilot[is_pilot], is_pilot)
                break
    return out


def _load_face_meta(floor: int) -> pd.DataFrame:
    parq = DATA_DIR / "face_h_first_gemma.parquet"
    if not parq.exists():
        raise SystemExit(f"missing {parq}")
    df = pd.read_parquet(parq)
    df = df[df["total_emit_count"] >= floor].copy()
    emit_cols = [f"total_emit_{q}" for q in QUADRANTS]
    df["empirical"] = df[emit_cols].idxmax(axis=1).str.replace("total_emit_", "")
    return df.set_index("first_word")[["empirical", "total_emit_count"]]


def _evaluate_k(rows: pd.DataFrame, face_meta: pd.DataFrame,
                k: int | None) -> tuple[float, float, int]:
    """For each (face, quadrant) take top-k log_prob mean (or all if k=None).
    Argmax over quadrants → predicted. Score against empirical.
    Returns (accuracy, kappa, n_evaluated).
    """
    # Per (face, quadrant), sort prompts by log_prob desc, take top-k mean.
    if k is None:
        agg = (rows.groupby(["first_word", "quadrant"])["log_prob"]
                   .mean().reset_index())
    else:
        def _topk_mean(s: pd.Series) -> float:
            return s.nlargest(k).mean()
        agg = (rows.groupby(["first_word", "quadrant"])["log_prob"]
                   .apply(_topk_mean).reset_index())
    # Argmax over quadrant per face.
    pivot = agg.pivot(index="first_word", columns="quadrant",
                      values="log_prob").reindex(columns=QUADRANTS)
    pred = pivot.idxmax(axis=1)
    # Restrict to GT subset.
    common = pred.index.intersection(face_meta.index)
    if len(common) == 0:
        return 0.0, float("nan"), 0
    y_pred = [pred[f] for f in common]
    y_emp = [face_meta.loc[f, "empirical"] for f in common]
    n = len(common)
    n_correct = sum(int(p == e) for p, e in zip(y_pred, y_emp))
    try:
        kap = cohen_kappa_score(y_emp, y_pred, labels=QUADRANTS)
    except ValueError:
        kap = float("nan")
    return n_correct / n if n > 0 else 0.0, kap, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,3,5,all",
                    help="comma-sep k values (default 1,2,3,5,all)")
    ap.add_argument("--ground-truth-floor", type=int, default=3)
    ap.add_argument("--prefer-full", action="store_true",
                    help="prefer full per-cell parquets over pilot")
    ap.add_argument("--claude-gt", action="store_true",
                    help="use Claude pilot modal-quadrant as GT instead of "
                         "pooled empirical_majority_quadrant")
    ap.add_argument("--claude-gt-floor", type=int, default=1,
                    help="min Claude emits in modal quadrant to include "
                         "face (default 1)")
    args = ap.parse_args()

    # Parse ks: 'all' → None (mean of all prompts).
    ks: list[int | None] = []
    for tok in args.ks.split(","):
        tok = tok.strip()
        if tok == "all":
            ks.append(None)
        else:
            ks.append(int(tok))

    discovered = _discover(args.prefer_full)
    if not discovered:
        raise SystemExit("no face_likelihood per-cell parquets found")
    print(f"discovered {len(discovered)} encoders:")
    for name, (path, is_pilot) in discovered.items():
        print(f"  {name:20s} {'(pilot)' if is_pilot else '(full) '} ← {path}")

    if args.claude_gt:
        from llmoji_study.claude_gt import load_claude_gt
        cgt = load_claude_gt(floor=args.claude_gt_floor)
        face_meta = pd.DataFrame(
            [(f, q, n) for f, (q, n) in cgt.items()],
            columns=["first_word", "empirical", "total_emit_count"],
        ).set_index("first_word")
        print(f"\nClaude GT subset (modal_n ≥ {args.claude_gt_floor}): "
              f"{len(face_meta)} faces")
    else:
        face_meta = _load_face_meta(args.ground_truth_floor)
        print(f"\nGT subset (≥{args.ground_truth_floor} emits): "
              f"{len(face_meta)} faces")

    # Eval per (encoder, k).
    rows_out = []
    for enc, (path, is_pilot) in discovered.items():
        rows = pd.read_parquet(path)
        # Some pilots cap at fewer prompts/quadrant than full — figure out
        # the effective max k for this encoder.
        max_k = rows.groupby(["first_word", "quadrant"]).size().max()
        for k in ks:
            kk = None if k is None else min(k, max_k)
            label = "all" if k is None else str(k)
            acc, kap, n = _evaluate_k(rows, face_meta, kk)
            rows_out.append({
                "encoder": enc,
                "is_pilot": is_pilot,
                "k": label,
                "k_effective": kk if kk else max_k,
                "n_eval": n,
                "accuracy": acc,
                "kappa": kap,
                "max_k_in_data": max_k,
            })
    df = pd.DataFrame(rows_out)
    suffix = "_claude_gt" if args.claude_gt else ""
    out_tsv = DATA_DIR / f"face_likelihood_topk_pooling{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    # Build markdown report.
    lines: list[str] = []
    lines.append("# Top-k per-prompt pooling — face_likelihood")
    lines.append("")
    lines.append(f"**GT floor:** total_emit_count ≥ {args.ground_truth_floor}")
    lines.append(f"**Source:** {'full' if args.prefer_full else 'pilot'} "
                 "per-cell parquets")
    lines.append("")
    lines.append("Each cell: accuracy / κ. Bold = best k for the encoder. "
                 "**'all' = mean over all prompts** (current default in "
                 "script 50 / 52 / 53).")
    lines.append("")
    pivot_acc = df.pivot(index="encoder", columns="k", values="accuracy")
    pivot_k = df.pivot(index="encoder", columns="k", values="kappa")
    pivot_acc = pivot_acc.reindex(columns=[("all" if k is None else str(k))
                                            for k in ks])
    pivot_k = pivot_k.reindex(columns=[("all" if k is None else str(k))
                                        for k in ks])
    cols = list(pivot_acc.columns)
    lines.append("| encoder | " + " | ".join(f"k={c}" for c in cols)
                 + " | best |")
    lines.append("|---" * (len(cols) + 2) + "|")
    for enc in pivot_acc.index:
        cells = [enc]
        best_acc = -1.0
        best_k = None
        for c in cols:
            a = pivot_acc.loc[enc, c]
            k = pivot_k.loc[enc, c]
            if pd.isna(a):
                cells.append("—")
                continue
            if a > best_acc:
                best_acc = a
                best_k = c
            cells.append(f"{a:.0%} / {k:.2f}")
        cells.append(f"k={best_k} ({best_acc:.0%})")
        # Bold the best cell — re-emit with formatting.
        out_cells = [enc]
        for c in cols:
            a = pivot_acc.loc[enc, c]
            k = pivot_k.loc[enc, c]
            if pd.isna(a):
                out_cells.append("—")
                continue
            cell = f"{a:.0%} / {k:.2f}"
            if c == best_k:
                cell = f"**{cell}**"
            out_cells.append(cell)
        out_cells.append(f"k={best_k}")
        lines.append("| " + " | ".join(out_cells) + " |")
    lines.append("")

    # Per-encoder narrative.
    lines.append("## Best-k per encoder")
    lines.append("")
    for enc in pivot_acc.index:
        baseline = pivot_acc.loc[enc, "all"]
        best = pivot_acc.loc[enc].max()
        delta = (best - baseline) * 100
        best_k = pivot_acc.loc[enc].idxmax()
        if pd.isna(baseline) or pd.isna(best):
            continue
        if delta > 0.5:
            verdict = f"**+{delta:.1f}pp lift** at k={best_k}"
        elif delta < -0.5:
            verdict = f"top-k *hurts* (baseline-all best)"
        else:
            verdict = "no meaningful difference"
        lines.append(f"- **{enc}**: {verdict} "
                     f"(baseline-all {baseline:.0%}, best {best:.0%})")
    lines.append("")

    out_md = DATA_DIR / "face_likelihood_topk_pooling.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
