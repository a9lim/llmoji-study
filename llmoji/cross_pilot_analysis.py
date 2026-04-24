# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
"""Pooled analysis across v1/v2 (pilot_raw.jsonl) and v3
(emotional_raw.jsonl). Both datasets store `probe_scores_t0` as the
whole-generation aggregate (under stateless=True — see CLAUDE.md
gotcha), so pooling on that column is apples-to-apples.

Grouping: per-(first_word, source) tuples. 'source' distinguishes
v1/v2 condition arms from v3 quadrants, so we can read whether the
same kaomoji carries the same probe signature across steered and
naturalistic regimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PROBES

# v3 prompt_id prefixes → pooled source name.
_V3_QUADRANT_SOURCE = {"HP": "v3_HP", "LP": "v3_LP", "HN": "v3_HN", "LN": "v3_LN"}

# Reuse the kaomoji-start-char filter from emotional_analysis.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")

# source-color palette. Distinct hues per source; grouped visually by
# family (baselines neutral, happy-pole warm, sad-pole cool, angry
# reds, calm greens, v3 quadrants mid-saturation).
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
}


def load_pooled_rows(v1_v2_path: str, v3_path: str) -> pd.DataFrame:
    """Union v1/v2 and v3 JSONL with a 'source' column identifying arm
    (v1/v2) or quadrant (v3). Explodes probe_scores_t0 into per-probe
    columns. Drops rows whose first_word doesn't look like a kaomoji
    and rows whose probe vector is NaN (rare capture-fallback)."""
    v12: pd.DataFrame = pd.read_json(v1_v2_path, lines=True)
    v12 = v12.assign(source=v12["condition"])

    v3: pd.DataFrame = pd.read_json(v3_path, lines=True)
    quad = v3["prompt_id"].str[:2].str.upper()
    v3 = v3.assign(source=quad.map(_V3_QUADRANT_SOURCE))
    # v3 has probe_scores_tlast; we don't use it for pooling.
    if "probe_scores_tlast" in v3.columns:
        v3 = v3.drop(columns=["probe_scores_tlast"])

    common = [
        "source", "prompt_id", "seed", "prompt_text",
        "text", "first_word", "kaomoji", "kaomoji_label",
        "probe_scores_t0",
    ]
    df = pd.concat([v12[common], v3[common]], ignore_index=True)

    # Explode probe_scores_t0 into per-probe columns.
    stacked = np.asarray(df["probe_scores_t0"].tolist(), dtype=float)
    for i, probe in enumerate(PROBES):
        df[f"t0_{probe}"] = stacked[:, i]
    df = df.drop(columns=["probe_scores_t0"])

    # Drop rows with NaN in any probe column (rare capture-fallback).
    probe_cols = [f"t0_{p}" for p in PROBES]
    df = df.dropna(subset=probe_cols)

    # Filter to kaomoji-bearing rows (same logic as emotional_analysis).
    df = df[df["first_word"].astype(str).str.len() > 0]
    df = df[df["first_word"].astype(str).str[0].isin(KAOMOJI_START_CHARS)]
    return df.reset_index(drop=True)


def grouped_kaomoji_source_means(
    df: pd.DataFrame, *, min_count: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Group by (first_word, source), require n >= min_count, return
    (mean-probe-vector DataFrame, count Series). Index is a MultiIndex
    of (first_word, source)."""
    cols = [f"t0_{p}" for p in PROBES]
    grouped = df.groupby(["first_word", "source"])[cols].mean()
    counts = df.groupby(["first_word", "source"]).size()
    keep = counts[counts >= min_count].index
    grouped = grouped.loc[grouped.index.isin(keep)]
    counts = counts.loc[grouped.index]
    return grouped, counts


def plot_pooled_cosine_heatmap(
    df: pd.DataFrame, out_path: str, *, min_count: int = 3,
) -> None:
    """Per-(kaomoji, source) mean probe-vector cosine similarity, with
    hierarchical-clustering row order. Row tick labels are colored by
    source."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .emotional_analysis import _use_cjk_font

    _use_cjk_font()

    grouped, counts = grouped_kaomoji_source_means(df, min_count=min_count)
    if len(grouped) < 3:
        print(f"  [pooled heatmap] only {len(grouped)} (kaomoji, source) with n≥{min_count}; skipping")
        return

    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]

    idx = grouped.index.to_list()
    ordered_idx = [idx[i] for i in order]
    ordered_counts = [int(counts.loc[k]) for k in ordered_idx]
    labels = [f"{km}  [{src}]  n={c}"
              for (km, src), c in zip(ordered_idx, ordered_counts)]
    row_colors = [SOURCE_COLORS.get(src, "#666") for _, src in ordered_idx]

    n = len(ordered_idx)
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * n + 5), max(9, 0.28 * n + 4)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([km for km, _ in ordered_idx],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(
        f"Pooled per-(kaomoji, source) probe-vector cosine similarity\n"
        f"(n ≥ {min_count}; {n} rows; v1/v2 + v3)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(1.15, 0.0), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pooled_summary_table(
    df: pd.DataFrame, *, min_count: int = 3,
) -> pd.DataFrame:
    """Per-(kaomoji, source) summary: n, taxonomy label, and the
    5-probe mean vector."""
    from .taxonomy import TAXONOMY
    cols = [f"t0_{p}" for p in PROBES]
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(["first_word", "source"]):
        km, src = key  # type: ignore[misc]  # pandas groupby-list key is a tuple at runtime
        if len(g) < min_count:
            continue
        means = g[cols].mean().to_dict()
        rows.append({
            "first_word": km,
            "source": src,
            "n": int(len(g)),
            "taxonomy_label": int(TAXONOMY.get(str(km), 0)),
            **{p: float(means[f"t0_{p}"]) for p in PROBES},
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["first_word", "source"]).reset_index(drop=True)
    return out
