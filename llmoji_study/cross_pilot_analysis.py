# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
"""Cross-pilot pooled analysis in hidden-state space.

Pools v1/v2 (``pilot_raw.jsonl`` + ``data/hidden/v1v2/``) with v3
(``emotional_raw.jsonl`` + ``data/hidden/v3/``) into a single
(metadata, hidden-state feature matrix) pool. Each row contributes its
``h_last`` at the highest captured probe layer by default; configurable
via ``which=`` and ``layer=``.

Replaces the pre-refactor probe-vector version, which operated on
5-dim probe scores that collapsed to ~1 valence direction (PC1 = 89%
variance). Hidden-state cosine at ~4096-dim preserves the full
activation signature.

Plotting conventions match the pre-refactor module:
  - ``plot_pooled_cosine_heatmap`` — per-(kaomoji, source) cosine
    heatmap with hierarchical clustering.
  - ``plot_pooled_pca_scatter``    — PC1 vs PC2 of the same
    per-(kaomoji, source) means.
  - ``pooled_summary_table``       — per-(kaomoji, source) counts +
    taxonomy labels (drops the 5 probe columns; hidden-state means
    aren't interpretable per-dim).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hidden_state_analysis import (
    cosine_similarity_matrix,
    group_mean_vectors,
    load_hidden_features,
)


# v3 prompt_id prefix → pooled source name.
_V3_QUADRANT_SOURCE = {
    "HP": "v3_HP", "LP": "v3_LP", "HN": "v3_HN", "LN": "v3_LN", "NB": "v3_NB",
}

# Reuse the kaomoji-start-char filter from emotional_analysis.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")

# source-color palette. Distinct hues per source family.
SOURCE_COLORS = {
    "baseline":         "#888888",
    "kaomoji_prompted": "#444444",
    "steered_happy":    "#e08a1f",
    "steered_sad":      "#1f5fa8",
    "steered_angry":    "#b93128",
    "steered_calm":     "#2f8860",
    "v3_HP":            "#e6b260",
    "v3_LP":            "#b28c3d",
    "v3_HN":            "#d06c5a",
    "v3_LN":            "#5f7ca8",
    "v3_NB":            "#6e6e6e",
}


def load_pooled_features(
    v1_v2_jsonl: str | Path,
    v3_jsonl: str | Path,
    data_dir: Path,
    *,
    v1_v2_experiment: str = "v1v2",
    v3_experiment: str = "v3",
    which: str = "h_first",
    layer: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Union v1/v2 + v3 JSONL, load per-row hidden-state sidecars, and
    tag each row with a ``source`` derived from v1/v2 ``condition`` or
    v3 quadrant. Filters to kaomoji-bearing rows (first_word starts
    with a kaomoji-ish glyph). Returns (metadata df, hidden-state
    matrix) aligned row-wise.
    """
    df12, X12 = load_hidden_features(
        v1_v2_jsonl, data_dir, v1_v2_experiment,
        which=which, layer=layer,
    )
    df3, X3 = load_hidden_features(
        v3_jsonl, data_dir, v3_experiment,
        which=which, layer=layer,
    )

    if len(df12):
        df12 = df12.assign(source=df12["condition"])
    if len(df3):
        quad = df3["prompt_id"].str[:2].str.upper()
        df3 = df3.assign(source=quad.map(_V3_QUADRANT_SOURCE))

    # Guard: if either side empty, return the non-empty side alone.
    if len(df12) == 0 and len(df3) == 0:
        return pd.DataFrame(), np.zeros((0, 0), dtype=np.float32)
    if len(df12) == 0:
        df, X = df3, X3
    elif len(df3) == 0:
        df, X = df12, X12
    else:
        common_cols = [c for c in df12.columns if c in df3.columns]
        df = pd.concat([df12[common_cols], df3[common_cols]], ignore_index=True)
        X = np.concatenate([X12, X3], axis=0)

    # Filter to kaomoji-bearing rows: first_word starts with a bracket-
    # family glyph. Same filter as the pre-refactor module.
    mask = (
        df["first_word"].astype(str).str.len().to_numpy() > 0
    )
    first_char_ok = np.asarray([
        (s[:1] in KAOMOJI_START_CHARS) if isinstance(s, str) else False
        for s in df["first_word"]
    ])
    keep = mask & first_char_ok
    return df.loc[keep].reset_index(drop=True), X[keep]


def plot_pooled_cosine_heatmap(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_count: int = 3,
    center: bool = True,
) -> None:
    """Per-(kaomoji, source) mean hidden-state vector, pairwise cosine
    similarity with hierarchical-clustering row order. Row tick labels
    are colored by source."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from .emotional_analysis import _use_cjk_font

    _use_cjk_font()

    keys_df, M, counts = group_mean_vectors(
        df, X, ["first_word", "source"], min_count=min_count,
    )
    if len(keys_df) < 3:
        print(f"  [pooled heatmap] only {len(keys_df)} (kaomoji, source) "
              f"with n≥{min_count}; skipping")
        return

    sim = cosine_similarity_matrix(M, center=center)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]

    ordered_keys = keys_df.iloc[order].reset_index(drop=True)
    ordered_counts = counts.iloc[order].to_numpy()
    labels = [
        f"{km}  [{src}]  n={c}"
        for km, src, c in zip(
            ordered_keys["first_word"], ordered_keys["source"], ordered_counts,
        )
    ]
    row_colors = [SOURCE_COLORS.get(src, "#666") for src in ordered_keys["source"]]

    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * n + 5), max(9, 0.28 * n + 4)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered_keys["first_word"].tolist(),
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    centering_note = "grand-mean centered; " if center else "uncentered; "
    ax.set_title(
        f"Pooled per-(kaomoji, source) HIDDEN-STATE cosine similarity\n"
        f"({centering_note}n ≥ {min_count}; {n} rows; v1/v2 + v3)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(1.15, 0.0), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_pca_scatter(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_count: int = 3,
) -> dict[str, Any]:
    """PC1 vs PC2 scatter of per-(kaomoji, source) mean hidden-state
    vectors. Fits PCA on the per-group means (not on row-level data)
    — treats each (kaomoji, source) cell as one point for clustering
    purposes. Returns the explained-variance spectrum dict."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA
    from .emotional_analysis import _use_cjk_font

    _use_cjk_font()

    keys_df, M, counts = group_mean_vectors(
        df, X, ["first_word", "source"], min_count=min_count,
    )
    if len(keys_df) < 3:
        print(f"  [pooled PCA] only {len(keys_df)} tuples; skipping")
        return {}

    n_comp = min(5, M.shape[0], M.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(M)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 9))
    for i in range(len(keys_df)):
        km = keys_df["first_word"].iloc[i]
        src = keys_df["source"].iloc[i]
        color = SOURCE_COLORS.get(src, "#666")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=60,
                   edgecolor="#444", linewidth=0.8, alpha=0.85, zorder=3)
        ax.annotate(km, (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.75,
                    xytext=(4, 4), textcoords="offset points", zorder=4)

    ax.axhline(0, color="#ccc", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PC1  ({var[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1] * 100:.1f}% var)")
    ax.set_title(
        f"Pooled (kaomoji, source) hidden-state means — PC1 vs PC2\n"
        f"(n ≥ {min_count}; {len(keys_df)} points; fill=source)"
    )
    source_legend = [Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()]
    ax.legend(handles=source_legend, loc="best", frameon=False, fontsize=8,
              title="source (fill)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_points": len(keys_df),
        "explained_variance_ratio": var.tolist(),
        "singular_values": pca.singular_values_.tolist(),
    }


def pooled_summary_table(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    min_count: int = 3,
) -> pd.DataFrame:
    """Per-(kaomoji, source) count + within-group cosine-to-mean
    consistency. Drops the 5 probe-mean columns the probe-based version
    had (per-dim hidden-state means aren't interpretable; the
    consistency column is the summary stat)."""
    from .hidden_state_analysis import cosine_to_mean

    rows: list[dict[str, Any]] = []
    if len(df) == 0:
        return pd.DataFrame(rows)

    for (km, src), g in df.groupby(["first_word", "source"]):
        if len(g) < min_count:
            continue
        idxs = g.index.to_numpy()
        vecs = X[idxs]
        sims = cosine_to_mean(vecs)
        rows.append({
            "first_word": km,
            "source": src,
            "n": int(len(g)),
            "median_within_consistency": float(np.median(sims)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["source", "median_within_consistency"], ascending=[True, False],
        ).reset_index(drop=True)
    return out
