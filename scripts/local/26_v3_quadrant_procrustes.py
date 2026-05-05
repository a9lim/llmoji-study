# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
"""v3 N-model quadrant Procrustes in 3D: cross-model quadrant-geometry
alignment across the v3 lineup, with the rule-3-redesign HN split
active. PCA(3) is fit per model on its full layer-stack hidden state;
per-quadrant centroids form 6-point clouds in PC1×PC2×PC3 space, and
cross-model alignment is computed via 3D orthogonal Procrustes against
a chosen reference model (default gemma).

Output is an interactive HTML with one 3D scene per model + one overlay
scene laid out in an auto-sized grid (e.g. 2×2 for 3 models, 3×2 for 5).
Per-model scenes show that model's centroids in its own PC(1,2,3); the
overlay rotates each non-reference model onto the reference, with
per-quadrant connecting lines making divergence read visually.

3D rotation indeterminacy: PC sign-flips and PC-pair swaps both fall
out as proper rotations in O(3), so a Procrustes rotation that
includes a near-180° axis flip just means some model's PC basis is
oriented oppositely to the reference's — a rigid frame change, not a
divergence finding. The reported total rotation angle (the
axis-angle magnitude of R) is the raw orthogonal-Procrustes value
so the number is honest.

Quadrant labels are HP / LP / HN-D / HN-S / LN / NB (rule-3-redesign
6-category split). Untagged HN rows drop out before PCA.

Models without v3 data on disk are skipped with a warning.
Claude can NOT participate (no hidden states from API-side calls).

Usage:
  # Default — 5 v3 main models, gemma as reference:
  python scripts/local/26_v3_quadrant_procrustes.py
  # Subset (e.g. while v3 chain is still running):
  python scripts/local/26_v3_quadrant_procrustes.py --models gemma,qwen
  # Different reference:
  python scripts/local/26_v3_quadrant_procrustes.py --reference qwen

Output:
  figures/local/fig_v3_quadrant_procrustes_3d.html
  figures/local/v3_quadrant_procrustes_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from llmoji_study.config import FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    load_emotional_features_stack,
)


# Default v3 main lineup post-2026-05-03 vocab-pilot expansion.
DEFAULT_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
DEFAULT_REFERENCE = "gemma"
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT

# Plotly 3D marker symbols. Up to 8 distinct shapes available; we map
# the v3 lineup deterministically and fall back to circles for any
# extras (rare).
_MARKER_POOL = ("circle", "diamond", "square", "cross", "x",
                "circle-open", "diamond-open", "square-open")


def _markers_for(models: list[str]) -> dict[str, str]:
    return {m: _MARKER_POOL[i % len(_MARKER_POOL)] for i, m in enumerate(models)}


def _size_for(symbol: str, base: int = 8) -> float:
    """Per-symbol size scaling — Plotly's ``x`` marker renders much
    thicker than the filled shapes at the same nominal size, so it
    visually dominates. ``cross`` (+) renders close enough to the
    filled shapes' visual weight that it doesn't need adjustment.
    Empirically x=0.5×, cross=1.0× brings the marker family into
    visual balance."""
    if symbol == "x":
        return base * 0.5
    return base


def _load_stack(short: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load layer-stack hidden-state representation for `short`.
    Returns (df with split-HN quadrant column, X (n × n_layers·hidden_dim))."""
    return load_emotional_features_stack(short, split_hn=True)


def _fit_pca3(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    return Y, pca


def _centroids_3d(
    df: pd.DataFrame, Y3: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-quadrant centroids in PC(1,2,3) space."""
    out: dict[str, np.ndarray] = {}
    for q in QUADRANT_ORDER:
        mask = (df["quadrant"] == q).to_numpy()
        if mask.any():
            out[q] = Y3[mask].mean(axis=0)
    return out


def _procrustes_align_3d(
    centroids_a: dict[str, np.ndarray],
    centroids_b: dict[str, np.ndarray],
) -> dict:
    """3D orthogonal Procrustes: rotate B-centroids onto A-centroids
    after centering. Returns aligned B coords + the rotation matrix +
    total axis-angle magnitude in degrees."""
    common = [q for q in QUADRANT_ORDER
              if q in centroids_a and q in centroids_b]
    if len(common) < 3:
        return {"common": common, "rotation_deg": float("nan"),
                "residual": float("nan"),
                "Ca_centered": None, "Cb_aligned": None,
                "R": None}
    Ca = np.asarray([centroids_a[q] for q in common])
    Cb = np.asarray([centroids_b[q] for q in common])
    Ca_c = Ca - Ca.mean(axis=0, keepdims=True)
    Cb_c = Cb - Cb.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(Cb_c, Ca_c)
    Cb_aligned = Cb_c @ R
    norm_a = float(np.linalg.norm(Ca_c))
    norm_b = float(np.linalg.norm(Cb_aligned))
    if norm_b > 0:
        Cb_aligned = Cb_aligned * (norm_a / norm_b)
    residual = float(np.linalg.norm(Cb_aligned - Ca_c))
    # Axis-angle magnitude: cos(θ) = (tr(R) - 1) / 2.
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle = float(np.degrees(np.arccos(cos_theta)))
    return {
        "common": common,
        "rotation_deg": angle,
        "residual": residual,
        "Ca_centered": Ca_c,
        "Cb_aligned": Cb_aligned,
        "R": R,
    }


def _add_per_model_scene(
    fig: go.Figure, scene_idx: int,
    centroids: dict[str, np.ndarray],
    title: str,
) -> None:
    """Add per-quadrant centroid markers + axis lines to one 3D scene.

    Plotly subplot scenes are addressed via ``scene{N}`` — caller
    passes scene_idx so traces get attached to the right panel."""
    scene = "scene" if scene_idx == 1 else f"scene{scene_idx}"
    for q, pt in centroids.items():
        color = QUADRANT_COLORS.get(q, "#666")
        fig.add_trace(
            go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode="markers+text",
                marker=dict(size=8, color=color,
                            line=dict(color="black", width=1)),
                text=[q], textposition="top center",
                textfont=dict(size=10),
                name=f"{title} {q}",
                showlegend=False,
                hovertemplate=(
                    f"{title}<br>{q}<br>"
                    "PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}"
                    "<extra></extra>"
                ),
                scene=scene,
            )
        )


def _add_overlay_scene(
    fig: go.Figure, scene_idx: int,
    common: list[str],
    cents_by_model: dict[str, np.ndarray],
    models: list[str],
    reference: str,
    markers: dict[str, str],
) -> None:
    """Add the aligned overlay: each quadrant gets one marker per
    model + a connecting line through them so deviation reads
    visually. Models drawn in the order given; quadrant label
    anchored on the reference model's marker."""
    scene = "scene" if scene_idx == 1 else f"scene{scene_idx}"

    # Per-quadrant per-model markers; quadrant label anchored on reference.
    for i, q in enumerate(common):
        color = QUADRANT_COLORS.get(q, "#666")
        for m in models:
            pt = cents_by_model[m][i]
            text = [q] if m == reference else [""]
            fig.add_trace(
                go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode="markers+text",
                    marker=dict(size=_size_for(markers[m]), color=color,
                                symbol=markers[m],
                                line=dict(color="black", width=1)),
                    text=text, textposition="top center",
                    textfont=dict(size=11),
                    name=q,
                    showlegend=False,
                    hovertemplate=(
                        f"{m}<br>{q}<br>"
                        "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}"
                        "<extra></extra>"
                    ),
                    scene=scene,
                )
            )

    # Marker-shape legend (one entry per model, neutral color).
    for m in models:
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=_size_for(markers[m]), color="#444",
                            symbol=markers[m],
                            line=dict(color="black", width=1)),
                name=f"{m}",
                showlegend=True,
                scene=scene,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model keys. Default: {','.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--reference", default=DEFAULT_REFERENCE,
        help=f"Reference model for Procrustes alignment. "
             f"Default: {DEFAULT_REFERENCE}",
    )
    args = parser.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in MODEL_REGISTRY:
            raise SystemExit(
                f"unknown model {m!r}; known: {sorted(MODEL_REGISTRY)}"
            )
    if args.reference not in requested:
        raise SystemExit(
            f"reference {args.reference!r} must be in --models {requested}"
        )

    print(f"requested models: {requested}; reference: {args.reference}")

    # Load each requested model; skip those with no v3 data.
    loaded: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for m in requested:
        M = MODEL_REGISTRY[m]
        if not M.emotional_data_path.exists():
            print(f"  [{m}] no v3 data at {M.emotional_data_path}; skipping")
            continue
        try:
            df, X = _load_stack(m)
        except Exception as e:
            print(f"  [{m}] load failed: {e}; skipping")
            continue
        loaded[m] = (df, X)
        print(f"  [{m}] layer-stack, {len(df)} rows after split-mode filter, X{X.shape}")

    if args.reference not in loaded:
        raise SystemExit(
            f"reference {args.reference!r} has no v3 data on disk; cannot align"
        )
    if len(loaded) < 2:
        raise SystemExit(
            f"need at least 2 models with v3 data; got {list(loaded)}"
        )

    models = list(loaded.keys())  # final ordered list
    reference = args.reference
    markers = _markers_for(models)

    per_model: dict[str, dict] = {}
    for m in models:
        df, X = loaded[m]
        Y3, pca = _fit_pca3(X)
        per_model[m] = {
            "df": df,
            "Y3": Y3,
            "pca": pca,
            "n": int(len(df)),
            "centroids_3d": _centroids_3d(df, Y3),
        }

    # Procrustes against the reference.
    ref = per_model[reference]["centroids_3d"]
    common = [q for q in QUADRANT_ORDER
              if all(q in per_model[m]["centroids_3d"] for m in models)]
    if len(common) < 3:
        raise SystemExit(
            f"need ≥3 common quadrants across models; got {common}"
        )
    aligned: dict[str, np.ndarray] = {}
    pair_summaries: dict[str, tuple[float, float]] = {reference: (0.0, 0.0)}
    Cg = np.asarray([ref[q] for q in common])
    Cg_c = Cg - Cg.mean(axis=0, keepdims=True)
    aligned[reference] = Cg_c
    procrustes_R: dict[str, np.ndarray | None] = {reference: np.eye(3)}
    for m in models:
        if m == reference:
            continue
        result = _procrustes_align_3d(ref, per_model[m]["centroids_3d"])
        aligned[m] = result["Cb_aligned"]
        pair_summaries[m] = (
            float(result["rotation_deg"]),
            float(result["residual"]),
        )
        procrustes_R[m] = result["R"]

    # Auto-grid: N per-model scenes + 1 overlay = N+1 panels.
    n_panels = len(models) + 1
    cols = max(2, math.ceil(math.sqrt(n_panels)))
    rows = math.ceil(n_panels / cols)

    # Per-panel titles: model L{layer} + variance ratios for per-model
    # panels; rotation/residual summary for the overlay.
    titles = []
    for m in models:
        evr = per_model[m]["pca"].explained_variance_ratio_
        titles.append(
            f"{m} (layer-stack)<br>"
            f"<sub>PC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}% · "
            f"PC3 {evr[2]*100:.1f}% · n={per_model[m]['n']}</sub>"
        )
    overlay_subs = []
    for m in models:
        if m == reference:
            continue
        rot, res = pair_summaries[m]
        overlay_subs.append(f"{m}→{reference} {rot:.1f}° (resid {res:.2f})")
    titles.append(
        f"all overlaid on {reference}<br><sub>" +
        " · ".join(overlay_subs) + "</sub>"
    )
    # Pad with empty titles if grid has more cells than panels.
    while len(titles) < rows * cols:
        titles.append("")

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles,
        horizontal_spacing=0.04, vertical_spacing=0.06,
    )

    # Per-model scenes (1..N) + overlay (N+1). Plotly numbers scenes
    # row-major starting at 1.
    for idx, m in enumerate(models, start=1):
        _add_per_model_scene(fig, idx, per_model[m]["centroids_3d"], m)
    _add_overlay_scene(
        fig, len(models) + 1, common, aligned, models, reference, markers
    )

    # Common scene styling: equal-aspect cube + same axis labels.
    common_axes = dict(
        xaxis=dict(title="PC1", showbackground=True, backgroundcolor="#f8f8f8"),
        yaxis=dict(title="PC2", showbackground=True, backgroundcolor="#f8f8f8"),
        zaxis=dict(title="PC3", showbackground=True, backgroundcolor="#f8f8f8"),
        aspectmode="cube",
    )
    overlay_axes = dict(
        xaxis=dict(title="aligned PC1", showbackground=True, backgroundcolor="#f8f8f8"),
        yaxis=dict(title="aligned PC2", showbackground=True, backgroundcolor="#f8f8f8"),
        zaxis=dict(title="aligned PC3", showbackground=True, backgroundcolor="#f8f8f8"),
        aspectmode="cube",
    )
    layout_updates: dict = {}
    for i in range(1, len(models) + 1):
        key = "scene" if i == 1 else f"scene{i}"
        layout_updates[key] = common_axes
    overlay_key = "scene" if len(models) + 1 == 1 else f"scene{len(models) + 1}"
    layout_updates[overlay_key] = overlay_axes
    fig.update_layout(
        **layout_updates,
        title=dict(
            text=(f"v3 cross-model quadrant geometry in 3D — "
                  f"PC(1,2,3) per model, Procrustes-aligned overlay "
                  f"to {reference} (HN split: HN-D anger / HN-S fear)"),
            x=0.5, xanchor="center",
        ),
        legend=dict(font=dict(size=10), itemsizing="constant"),
        margin=dict(l=10, r=10, t=80, b=10),
        height=max(800, 500 * rows), width=max(1200, 500 * cols),
    )

    out_dir = FIGURES_DIR / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "fig_v3_quadrant_procrustes_3d.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\nwrote {out_html}")

    # Remove now-superseded artifacts so the directory stays honest.
    for stale in ("fig_v3_triplet_procrustes_3d.html",
                  "fig_v3_triplet_procrustes_pc12.png",
                  "fig_v3_triplet_procrustes_pc13.png",
                  "fig_v3_triplet_procrustes_pc23.png",
                  "v3_triplet_procrustes_summary.json"):
        old = out_dir / stale
        if old.exists():
            old.unlink()
            print(f"  removed superseded {old.name}")

    summary = {
        "reference_model": reference,
        "common_quadrants": common,
        "rep": "h_first.layer-stack",
        "models": {
            m: {
                "n_rows": per_model[m]["n"],
                "explained_variance_ratio_pc123": [
                    float(v) for v in per_model[m]["pca"].explained_variance_ratio_
                ],
                "centroids_3d": {
                    q: per_model[m]["centroids_3d"][q].tolist()
                    for q in per_model[m]["centroids_3d"]
                },
                "procrustes_rotation_deg": pair_summaries[m][0],
                "procrustes_residual": pair_summaries[m][1],
                "procrustes_R": (
                    procrustes_R[m].tolist()  # type: ignore[union-attr]
                    if procrustes_R[m] is not None else None
                ),
            }
            for m in models
        },
    }
    out_json = out_dir / "v3_quadrant_procrustes_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    print(f"\n=== 3D Procrustes alignment to {reference} ===")
    print(f"{'model':14s} {'rot°':>8s} {'residual':>10s}")
    for m in models:
        rot, res = pair_summaries[m]
        print(f"{m:14s} {rot:8.2f} {res:10.3f}")


if __name__ == "__main__":
    main()
