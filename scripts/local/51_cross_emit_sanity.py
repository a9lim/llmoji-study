#!/usr/bin/env python3
"""Sanity check: does each encoder cross-predict OTHER encoders' kaomoji?

The face_h_first union samples kaomoji from {gemma, qwen, ministral} v3
emissions ∪ claude-faces. If gemma's likelihood test agrees with v3's
empirical-majority on faces that ONLY GEMMA emitted, that's
self-consistent — but uninformative about cross-model affect agreement.
The interesting test: how does QWEN do on gemma-only faces, and vice
versa?

For each encoder × emit-origin partition, report accuracy + Cohen's κ.
Origin partitions on GT subset (total_emit_count ≥ floor):
- gemma_only      : only gemma emitted (≥1)
- qwen_only       : only qwen emitted
- ministral_only  : only ministral emitted
- shared_2        : exactly 2 models emitted
- shared_3        : all 3 models emitted

Reading: if gemma's cross-prediction on qwen_only faces (and vice versa)
stays above ~50%, encoders are converging on shared intrinsic affect.
If it drops below, the empirical signal is too tied to the emitting
model's sampling preference and we need broader v3 coverage / Claude
direct sampling for an unbiased baseline.

Usage:
    python scripts/local/51_cross_emit_sanity.py
    python scripts/local/51_cross_emit_sanity.py --ground-truth-floor 1

Outputs:
    data/face_likelihood_cross_emit_sanity.tsv
    data/face_likelihood_cross_emit_sanity.md
"""

from __future__ import annotations

import argparse
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llmoji_study.jsd import js, normalize, similarity

from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
EMIT_MODELS = ["gemma", "qwen", "ministral"]


def _origin(r: pd.Series) -> str:
    counts = {m: int(r[f"{m}_emit_count"] or 0) for m in EMIT_MODELS}
    nz = [m for m, c in counts.items() if c > 0]
    if len(nz) == 1:
        return f"{nz[0]}_only"
    if len(nz) == 2:
        return "shared_2"
    if len(nz) == 3:
        return "shared_3"
    return "claude_only"


def _discover_summaries(prefer_full: bool) -> dict[str, str]:
    """Walk the post-2026-05-05 layout for face_likelihood summary TSVs.

    Local encoders live at ``data/local/<model>/face_likelihood*_summary.tsv``;
    harness encoders (haiku, opus) at ``data/harness/face_likelihood_*_summary.tsv``.
    Encoder name = ``<model>`` for canonical local files, ``<model>_<variant>``
    for sub-config variants (e.g. ``rinna_jp_3_6b_jpfull``), and ``<encoder>``
    for harness files (e.g. ``haiku``). When a {pilot, full} pair exists
    for the same encoder, picks one per ``prefer_full``.
    """
    local_pat = re.compile(r"^face_likelihood(?:_(?P<variant>.+?))?_summary\.tsv$")
    harness_pat = re.compile(r"^face_likelihood_(?P<enc>.+?)(?P<pilot>_pilot)?_summary\.tsv$")
    found: dict[str, dict[bool, str]] = {}

    # Local: data/local/<model>/face_likelihood[_<variant>]_summary.tsv
    for p in sorted((DATA_DIR / "local").glob("*/face_likelihood*_summary.tsv")):
        m = local_pat.match(p.name)
        if not m:
            continue
        model = p.parent.name
        variant = m.group("variant") or ""
        is_pilot = (variant == "pilot")
        if is_pilot or not variant:
            encoder = model
        else:
            encoder = f"{model}_{variant}"
        if encoder.startswith(("vote_", "gemma_vs_qwen", "gemma-")):
            continue
        found.setdefault(encoder, {})[is_pilot] = str(p)

    # Harness: data/harness/face_likelihood_<encoder>[_pilot]_summary.tsv
    for p in sorted((DATA_DIR / "harness").glob("face_likelihood_*_summary.tsv")):
        m = harness_pat.match(p.name)
        if not m:
            continue
        encoder = m.group("enc")
        is_pilot = bool(m.group("pilot"))
        found.setdefault(encoder, {})[is_pilot] = str(p)

    out: dict[str, str] = {}
    order = [False, True] if prefer_full else [True, False]
    for name, by_pilot in found.items():
        for is_pilot in order:
            if is_pilot in by_pilot:
                out[name] = by_pilot[is_pilot]
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-emit sanity check")
    ap.add_argument("--ground-truth-floor", type=int, default=3,
                    help="min total_emit_count to include in GT (default 3)")
    ap.add_argument("--prefer-full", action="store_true",
                    help="prefer full over pilot when both exist (default: pilot)")
    ap.add_argument("--exclude", default="glm47_flash,deepseek_v2_lite",
                    help="encoders to skip (default: GLM + deepseek — both "
                         "below or near random solo). Pass empty to include all.")
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    # Face origin from the canonical face union. (Pre-2026-05-05 this came
    # from a per-encoder face_h_first parquet; the union is the cross-platform
    # source of truth post-refactor.)
    parq = DATA_DIR / "v3_face_union.parquet"
    if not parq.exists():
        raise SystemExit(f"missing {parq} — run scripts/40_face_union.py first")
    faces = pd.read_parquet(parq)
    if "total_emit_count" not in faces.columns:
        faces["total_emit_count"] = sum(faces[f"total_emit_{q}"] for q in QUADRANTS)
    faces["origin"] = faces.apply(_origin, axis=1)
    gt = faces[faces["total_emit_count"] >= args.ground_truth_floor].copy()

    # Build empirical_majority for each face (already in summary TSVs but
    # we need to map face → empirical regardless of which encoder loaded it).
    emit_cols = [f"total_emit_{q}" for q in QUADRANTS]
    gt["empirical"] = gt[emit_cols].idxmax(axis=1).str.replace("total_emit_", "")
    face_meta = gt.set_index("first_word")[["origin", "empirical",
                                             "total_emit_count"]].copy()

    # Discover encoder summaries.
    summaries = _discover_summaries(args.prefer_full)
    summaries = {k: v for k, v in summaries.items() if k not in exclude}
    if not summaries:
        raise SystemExit("no encoder summaries found")
    print(f"discovered {len(summaries)} encoders: {sorted(summaries)}")

    enc_preds: dict[str, dict[str, str]] = {}
    enc_softmax: dict[str, dict[str, list[float]]] = {}
    for enc, path in summaries.items():
        s = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
        enc_preds[enc] = dict(zip(s["first_word"].astype(str),
                                  s["predicted_quadrant"].astype(str)))
        # Build face → 6-vector softmax for JSD computation.
        sm: dict[str, list[float]] = {}
        for _, row in s.iterrows():
            f = str(row["first_word"])
            d = {q: float(row.get(f"softmax_{q}", 0.0) or 0.0) for q in QUADRANTS}
            sm[f] = normalize(d, QUADRANTS)
        enc_softmax[enc] = sm

    # Per-face empirical distribution (from emit-count columns) for JSD eval.
    face_dist: dict[str, list[float]] = {}
    for f in face_meta.index:
        row = gt[gt["first_word"] == f].iloc[0] if (gt["first_word"] == f).any() else None
        if row is None:
            continue
        d = {q: int(row.get(f"total_emit_{q}", 0) or 0) for q in QUADRANTS}
        if sum(d.values()) > 0:
            face_dist[f] = normalize(d, QUADRANTS)

    # Build per-(encoder, origin) accuracy + kappa.
    rows = []
    origins = ["gemma_only", "qwen_only", "ministral_only",
               "shared_2", "shared_3"]
    for enc in sorted(summaries):
        preds_dict = enc_preds[enc]
        for o in origins:
            sub = face_meta[face_meta["origin"] == o]
            faces_in_enc = [f for f in sub.index if f in preds_dict]
            if not faces_in_enc:
                continue
            y_emp = [face_meta.loc[f, "empirical"] for f in faces_in_enc]
            y_pred = [preds_dict[f] for f in faces_in_enc]
            n = len(faces_in_enc)
            n_correct = sum(int(p == e) for p, e in zip(y_pred, y_emp))
            try:
                k = cohen_kappa_score(y_emp, y_pred, labels=QUADRANTS)
            except ValueError:
                k = float("nan")
            # Per-face JSD between encoder softmax and empirical distribution;
            # mean across faces in this (encoder, origin) cell.
            sm_dict = enc_softmax[enc]
            jsds = [
                js(sm_dict[f], face_dist[f])
                for f in faces_in_enc
                if f in face_dist and f in sm_dict
            ]
            mean_jsd = sum(jsds) / len(jsds) if jsds else float("nan")
            sim = similarity(mean_jsd) if jsds else float("nan")
            rows.append({
                "encoder": enc,
                "origin": o,
                "n": n,
                "similarity": sim,
                "mean_jsd": mean_jsd,
                "n_correct": n_correct,
                "accuracy": n_correct / n if n > 0 else 0.0,
                "kappa": k,
            })
    df = pd.DataFrame(rows)
    out_tsv = DATA_DIR / "local" / "face_likelihood_cross_emit_sanity.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    # Build markdown report.
    lines: list[str] = []
    lines.append("# Cross-emit sanity check")
    lines.append("")
    lines.append(f"**Ground-truth floor:** total_emit_count ≥ "
                 f"{args.ground_truth_floor}")
    lines.append(f"**Encoders compared:** {', '.join(sorted(summaries))}")
    lines.append("")
    lines.append("## Partition counts (in GT subset)")
    lines.append("")
    lines.append("| origin | n |")
    lines.append("|---|---:|")
    for o in origins:
        n = int((face_meta["origin"] == o).sum())
        lines.append(f"| {o} | {n} |")
    lines.append("")

    lines.append("## Accuracy by encoder × origin")
    lines.append("")
    lines.append("Each cell: accuracy (n_correct/n) | κ.  "
                 "**Bold cells** are cross-prediction (encoder predicting on "
                 "faces only EMITTED by other v3 models).")
    lines.append("")
    pivot_acc = df.pivot(index="encoder", columns="origin",
                         values="accuracy").reindex(columns=origins)
    pivot_n = df.pivot(index="encoder", columns="origin",
                       values="n").reindex(columns=origins)
    pivot_correct = df.pivot(index="encoder", columns="origin",
                             values="n_correct").reindex(columns=origins)
    pivot_k = df.pivot(index="encoder", columns="origin",
                       values="kappa").reindex(columns=origins)

    header = ["encoder"] + origins
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    for enc in pivot_acc.index:
        cells = [str(enc)]
        for o in origins:
            n = pivot_n.loc[enc, o]
            if pd.isna(n) or n == 0:
                cells.append("—")
                continue
            n = int(n)
            nc = int(pivot_correct.loc[enc, o])
            a = pivot_acc.loc[enc, o]
            k = pivot_k.loc[enc, o]
            # Mark cross-prediction cells: encoder NOT in v3-trio cannot
            # cross-predict (every face is "other"); for v3-trio encoders,
            # the cross-prediction cells are the OTHER models' "_only".
            is_cross = False
            if enc in EMIT_MODELS:
                if o.endswith("_only") and not o.startswith(f"{enc}_"):
                    is_cross = True
            cell = f"{a:.0%} ({nc}/{n}) | κ={k:.2f}"
            if is_cross:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Pull out the headline cross-prediction numbers.
    lines.append("## Headline cross-predictions")
    lines.append("")
    pairs = [
        ("gemma", "qwen_only"),
        ("qwen", "gemma_only"),
        ("gemma", "ministral_only"),
        ("ministral", "gemma_only"),
        ("qwen", "ministral_only"),
        ("ministral", "qwen_only"),
    ]
    lines.append("| encoder | origin | accuracy | κ | reading |")
    lines.append("|---|---|---:|---:|---|")
    for enc, o in pairs:
        sub = df[(df["encoder"] == enc) & (df["origin"] == o)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        n = int(r["n"])
        if n < 2:
            reading = f"too few ({n})"
        elif r["accuracy"] >= 0.5:
            reading = "✓ converging"
        elif r["accuracy"] >= 0.3:
            reading = "~ ambiguous"
        else:
            reading = "✗ encoder-specific"
        lines.append(f"| {enc} | {o} | {r['accuracy']:.0%} "
                     f"({int(r['n_correct'])}/{n}) | "
                     f"{r['kappa']:.2f} | {reading} |")
    lines.append("")
    lines.append("**Threshold heuristic** (per user's request): >50% =  "
                 "encoders converge on shared intrinsic affect; <30% = the "
                 "empirical-majority signal is too tied to the emitting "
                 "model's sampling preference (would mean we need broader "
                 "v3 coverage and/or a Claude-direct baseline).")
    lines.append("")

    out_md = DATA_DIR / "local" / "face_likelihood_cross_emit_sanity.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}\n")

    print("HEADLINE CROSS-PREDICTIONS:")
    for enc, o in pairs:
        sub = df[(df["encoder"] == enc) & (df["origin"] == o)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(f"  {enc:10s} on {o:18s}: {r['accuracy']:.0%} "
              f"({int(r['n_correct'])}/{int(r['n'])})  κ={r['kappa']:.2f}")


if __name__ == "__main__":
    main()
