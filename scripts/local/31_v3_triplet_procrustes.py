# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
"""v3 triplet Procrustes in 3D: cross-model quadrant-geometry alignment
across gemma / qwen / ministral, with the rule-3-redesign HN split
active. PCA(3) is fit per model on its preferred-layer hidden state;
per-quadrant centroids form 6-point clouds in PC1×PC2×PC3 space, and
the cross-model alignment is computed via 3D orthogonal Procrustes
(qwen→gemma, ministral→gemma).

Output is a single interactive HTML with a 2×2 grid of 3D scenes:

  top-left      gemma centroids in its own PC(1,2,3)
  top-right     qwen centroids in its own PC(1,2,3)
  bottom-left   ministral centroids in its own PC(1,2,3)
  bottom-right  Procrustes overlay: gemma centered (○), qwen
                rotated to fit gemma (◇), ministral rotated to fit
                gemma (□). Same per-quadrant color across the three
                marker styles; lines connect same-quadrant centroids
                across models so divergence reads visually in 3D.

3D rotation indeterminacy: PC sign-flips and PC-pair swaps both fall
out as proper rotations in O(3), so a Procrustes rotation that
includes a near-180° axis flip just means some model's PC basis is
oriented oppositely to gemma's — a rigid frame change, not a
divergence finding. The reported total rotation angle (the
axis-angle magnitude of R) is the raw orthogonal-Procrustes value
so the number is honest.

Quadrant labels are HP / LP / HN-D / HN-S / LN / NB (rule-3-redesign
6-category split). Untagged HN rows drop out before PCA.

Output:
  figures/local/cross_model/fig_v3_triplet_procrustes_3d.html
  figures/local/cross_model/v3_triplet_procrustes_summary.json
"""

from __future__ import annotations

import json
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
    load_emotional_features_all_layers,
)


MODELS = ("gemma", "qwen", "ministral")
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT
REFERENCE_MODEL = "gemma"

# Plotly's 3D markers: small but visually-distinct triplet.
MARKERS_3D = {
    "gemma":     "circle",
    "qwen":      "diamond",
    "ministral": "square",
}


def _load_at_preferred_layer(short: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    M = MODEL_REGISTRY[short]
    df, X3, layer_idxs = load_emotional_features_all_layers(short, split_hn=True)
    layer = M.preferred_layer if M.preferred_layer is not None else max(layer_idxs)
    li = layer_idxs.index(layer)
    X = X3[:, li, :]
    return df, X, layer


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
) -> None:
    """Add the aligned overlay: each quadrant gets one marker per
    model + a connecting line through them so deviation reads
    visually."""
    scene = "scene" if scene_idx == 1 else f"scene{scene_idx}"

    # Per-quadrant connecting lines (gemma → qwen → ministral).
    for i, q in enumerate(common):
        color = QUADRANT_COLORS.get(q, "#666")
        xs = [cents_by_model["gemma"][i, 0],
              cents_by_model["qwen"][i, 0],
              cents_by_model["ministral"][i, 0]]
        ys = [cents_by_model["gemma"][i, 1],
              cents_by_model["qwen"][i, 1],
              cents_by_model["ministral"][i, 1]]
        zs = [cents_by_model["gemma"][i, 2],
              cents_by_model["qwen"][i, 2],
              cents_by_model["ministral"][i, 2]]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=4, dash="dash"),
                opacity=0.6,
                showlegend=False, hoverinfo="skip",
                scene=scene,
            )
        )

    # Per-quadrant per-model markers; quadrant label anchored on gemma.
    for i, q in enumerate(common):
        color = QUADRANT_COLORS.get(q, "#666")
        for m in MODELS:
            pt = cents_by_model[m][i]
            text = [q] if m == "gemma" else [""]
            fig.add_trace(
                go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode="markers+text",
                    marker=dict(size=8, color=color,
                                symbol=MARKERS_3D[m],
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
    for m in MODELS:
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color="#444",
                            symbol=MARKERS_3D[m],
                            line=dict(color="black", width=1)),
                name=f"{m}",
                showlegend=True,
                scene=scene,
            )
        )


def main() -> None:
    print("Loading all three models at their preferred layers...")
    loaded: dict[str, tuple[pd.DataFrame, np.ndarray, int]] = {}
    for m in MODELS:
        df, X, layer = _load_at_preferred_layer(m)
        loaded[m] = (df, X, layer)
        print(f"  {m}: L{layer}, {len(df)} rows after split-mode filter, X{X.shape}")

    per_model: dict[str, dict] = {}
    for m in MODELS:
        df, X, layer = loaded[m]
        Y3, pca = _fit_pca3(X)
        per_model[m] = {
            "df": df,
            "Y3": Y3,
            "pca": pca,
            "layer": layer,
            "n": int(len(df)),
            "centroids_3d": _centroids_3d(df, Y3),
        }

    # Procrustes against gemma.
    ref = per_model[REFERENCE_MODEL]["centroids_3d"]
    common = [q for q in QUADRANT_ORDER
              if all(q in per_model[m]["centroids_3d"] for m in MODELS)]
    aligned: dict[str, np.ndarray] = {}
    pair_summaries: dict[str, tuple[float, float]] = {REFERENCE_MODEL: (0.0, 0.0)}
    Cg = np.asarray([ref[q] for q in common])
    Cg_c = Cg - Cg.mean(axis=0, keepdims=True)
    aligned[REFERENCE_MODEL] = Cg_c
    procrustes_R: dict[str, np.ndarray | None] = {REFERENCE_MODEL: np.eye(3)}
    for m in MODELS:
        if m == REFERENCE_MODEL:
            continue
        result = _procrustes_align_3d(ref, per_model[m]["centroids_3d"])
        aligned[m] = result["Cb_aligned"]
        pair_summaries[m] = (
            float(result["rotation_deg"]),
            float(result["residual"]),
        )
        procrustes_R[m] = result["R"]

    # Build the 2×2 figure with 3D scenes in each cell.
    titles = []
    for m in MODELS:
        evr = per_model[m]["pca"].explained_variance_ratio_
        titles.append(
            f"{m} L{per_model[m]['layer']}<br>"
            f"<sub>PC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}% · "
            f"PC3 {evr[2]*100:.1f}% · n={per_model[m]['n']}</sub>"
        )
    qwen_rot, qwen_res = pair_summaries["qwen"]
    min_rot, min_res = pair_summaries["ministral"]
    titles.append(
        f"all three overlaid on gemma<br>"
        f"<sub>qwen→gemma {qwen_rot:.1f}° (resid {qwen_res:.2f}) · "
        f"ministral→gemma {min_rot:.1f}° (resid {min_res:.2f})</sub>"
    )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles,
        horizontal_spacing=0.04, vertical_spacing=0.06,
    )

    # Per-model scenes (1, 2, 3) + overlay (4).
    _add_per_model_scene(fig, 1, per_model["gemma"]["centroids_3d"], "gemma")
    _add_per_model_scene(fig, 2, per_model["qwen"]["centroids_3d"], "qwen")
    _add_per_model_scene(fig, 3, per_model["ministral"]["centroids_3d"], "ministral")
    _add_overlay_scene(fig, 4, common, aligned)

    # Common scene styling: equal-aspect cube + same axis labels.
    common_axes = dict(
        xaxis=dict(title="PC1", showbackground=True, backgroundcolor="#f8f8f8"),
        yaxis=dict(title="PC2", showbackground=True, backgroundcolor="#f8f8f8"),
        zaxis=dict(title="PC3", showbackground=True, backgroundcolor="#f8f8f8"),
        aspectmode="cube",
    )
    fig.update_layout(
        scene=common_axes,
        scene2=common_axes,
        scene3=common_axes,
        scene4=dict(
            xaxis=dict(title="aligned PC1", showbackground=True, backgroundcolor="#f8f8f8"),
            yaxis=dict(title="aligned PC2", showbackground=True, backgroundcolor="#f8f8f8"),
            zaxis=dict(title="aligned PC3", showbackground=True, backgroundcolor="#f8f8f8"),
            aspectmode="cube",
        ),
        title=dict(
            text=("v3 cross-model quadrant geometry in 3D — "
                  "PC(1,2,3) per model, Procrustes-aligned overlay "
                  "(HN split: HN-D anger / HN-S fear)"),
            x=0.5, xanchor="center",
        ),
        legend=dict(font=dict(size=10), itemsizing="constant"),
        margin=dict(l=10, r=10, t=80, b=10),
        height=1000, width=1400,
    )

    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "fig_v3_triplet_procrustes_3d.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\nwrote {out_html}")

    # Remove the now-superseded 2D PNGs so the directory stays honest.
    for slug in ("pc12", "pc13", "pc23"):
        old = out_dir / f"fig_v3_triplet_procrustes_{slug}.png"
        if old.exists():
            old.unlink()
            print(f"  removed superseded {old.name}")

    summary = {
        "reference_model": REFERENCE_MODEL,
        "common_quadrants": common,
        "models": {
            m: {
                "layer": per_model[m]["layer"],
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
            for m in MODELS
        },
    }
    out_json = out_dir / "v3_triplet_procrustes_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    print("\n=== 3D Procrustes alignment to gemma ===")
    print(f"{'model':12s} {'rot°':>8s} {'residual':>10s}")
    for m in MODELS:
        rot, res = pair_summaries[m]
        print(f"{m:12s} {rot:8.2f} {res:10.3f}")


if __name__ == "__main__":
    main()
