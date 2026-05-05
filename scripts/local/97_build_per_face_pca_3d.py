"""Build per-face 3D PCA scenes for the v3 main lineup, one scene per
model, packaged as a single multi-scene plotly HTML.

For each model in ``MODELS``, this loads the layer-stacked
``h_first`` representation, computes per-face mean centroids, fits
PCA(3) on those centroids, and renders the per-face points colored by
RGB-blend of QUADRANT_COLORS weighted by per-face emission counts.
Marker size scales log-proportional to total emission count so heavy
faces dominate the read.

Output is a 1×N plotly figure with one 3D scene per model. The
companion wrapper ``98_wrap_blog_3d_html.py`` extracts each scene
on-demand and exposes a model toggle for the blog post.

Usage:
    .venv/bin/python scripts/local/97_build_per_face_pca_3d.py
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
from sklearn.decomposition import PCA

from llmoji_study.config import FIGURES_DIR
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    load_emotional_features_stack,
)


# Match the v3 main lineup used by 26_v3_quadrant_procrustes.py.
MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
# Show every face the model emits — singletons included. PCA centroids
# from n=1 are noisy but the user prefers full coverage over per-face
# stability for the blog visualization.
MIN_FACE_COUNT = 1
OUT_HTML = FIGURES_DIR / "local" / "fig_v3_per_face_pca_3d.html"
OUT_META = OUT_HTML.with_suffix(".meta.json")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_blend(quadrant_counts: dict[str, int]) -> str:
    """RGB-linear blend of QUADRANT_COLORS weighted by per-quadrant
    emission counts. Returns ``#RRGGBB`` for plotly marker color."""
    total = sum(quadrant_counts.values())
    if total <= 0:
        return "#909090"
    r = g = b = 0.0
    for q, n in quadrant_counts.items():
        if n <= 0 or q not in QUADRANT_COLORS:
            continue
        cr, cg, cb = _hex_to_rgb(QUADRANT_COLORS[q])
        w = n / total
        r += cr * w
        g += cg * w
        b += cb * w
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def _per_face_centroids(
    df: pd.DataFrame, X: np.ndarray
) -> tuple[list[str], np.ndarray, list[dict[str, int]], list[int]]:
    """Per-face mean hidden-state vector + per-face quadrant breakdown.

    Returns:
        faces: list of face strings (canonical), in stable order.
        means: (n_faces, hidden_dim) centroid matrix.
        breakdowns: list of {quadrant: count} per face.
        totals: list of total emission counts per face.
    """
    rows_per_face: dict[str, list[int]] = {}
    for i, face in enumerate(df["first_word"].astype(str).tolist()):
        rows_per_face.setdefault(face, []).append(i)
    faces: list[str] = []
    means_list: list[np.ndarray] = []
    breakdowns: list[dict[str, int]] = []
    totals: list[int] = []
    quadrants = df["quadrant"].astype(str).tolist()
    for face, idxs in rows_per_face.items():
        if len(idxs) < MIN_FACE_COUNT:
            continue
        faces.append(face)
        sub = X[idxs]
        means_list.append(sub.mean(axis=0))
        bd: dict[str, int] = {}
        for i in idxs:
            q = quadrants[i]
            bd[q] = bd.get(q, 0) + 1
        breakdowns.append(bd)
        totals.append(len(idxs))
    return faces, np.array(means_list), breakdowns, totals


def _build_scene(
    fig: go.Figure, scene_idx: int,
    faces: list[str], coords: np.ndarray,
    breakdowns: list[dict[str, int]], totals: list[int],
) -> None:
    scene = "scene" if scene_idx == 1 else f"scene{scene_idx}"
    colors = [_rgb_blend(b) for b in breakdowns]
    sizes = [max(6, min(28, 6 + 4 * np.log2(max(1, t)))) for t in totals]
    hovertexts = []
    for face, b, t in zip(faces, breakdowns, totals):
        parts = [f"<b>{face}</b>  (n={t})"]
        ordered = sorted(b.items(), key=lambda kv: -kv[1])
        parts.append(", ".join(f"{q}={n}" for q, n in ordered))
        hovertexts.append("<br>".join(parts))
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="black", width=0.5),
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hovertexts,
            text=faces,
            name="",
            showlegend=False,
            scene=scene,
        )
    )


def main() -> None:
    fig = make_subplots(
        rows=1, cols=len(MODELS),
        specs=[[{"type": "scene"} for _ in MODELS]],
    )
    meta: dict[str, str] = {}
    for i, m in enumerate(MODELS, start=1):
        try:
            df, X = load_emotional_features_stack(m, split_hn=True)
        except FileNotFoundError as e:
            print(f"  [{m}] missing data — skipping ({e})")
            meta[m] = f"{m} — data missing"
            continue
        if len(df) == 0:
            print(f"  [{m}] empty after filter — skipping")
            meta[m] = f"{m} — empty"
            continue
        faces, means, breakdowns, totals = _per_face_centroids(df, X)
        if len(faces) < 2:
            print(f"  [{m}] only {len(faces)} faces ≥ {MIN_FACE_COUNT} — skipping")
            meta[m] = f"{m} — fewer than 2 faces ≥ {MIN_FACE_COUNT}"
            continue
        pca = PCA(n_components=3)
        coords = pca.fit_transform(means)
        _build_scene(fig, i, faces, coords, breakdowns, totals)
        evr = pca.explained_variance_ratio_
        meta[m] = (
            f"{len(faces)} faces · "
            f"PC1+2+3 = {(evr[0] + evr[1] + evr[2]) * 100:.1f}% · "
            f"layer-stack"
        )
        print(f"  [{m}] {meta[m]}")

    # Common scene styling.
    for i in range(1, len(MODELS) + 1):
        scene_key = "scene" if i == 1 else f"scene{i}"
        fig.update_layout(**{
            scene_key: dict(
                xaxis=dict(title=dict(text="PC1"), backgroundcolor="#f8f8f8", showbackground=True),
                yaxis=dict(title=dict(text="PC2"), backgroundcolor="#f8f8f8", showbackground=True),
                zaxis=dict(title=dict(text="PC3"), backgroundcolor="#f8f8f8", showbackground=True),
                aspectmode="cube",
            )
        })
    fig.update_layout(
        height=800, width=1500,
        title=dict(
            text="v3 per-face PCA(3) on layer-stack h_first, per model",
            x=0.5, xanchor="center",
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", full_html=True)
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT_HTML}")
    print(f"wrote {OUT_META}")


if __name__ == "__main__":
    main()
