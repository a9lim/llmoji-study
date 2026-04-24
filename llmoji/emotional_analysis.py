# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
"""Analysis for the emotional-battery experiment.

Three figures, all operating on final-token probe vectors
(``probe_scores_tlast``) from ``data/emotional_raw.jsonl``:

  - Figure A: per-kaomoji mean vector, pairwise cosine heatmap (the
    v1 Fig 3 analog, computed at the final token instead of token 0).
  - Figure B: within-kaomoji cosine-to-mean distribution, with a
    shuffled-subset null band. The core probative figure: does the same
    kaomoji reliably land in the same probe-space region, more tightly
    than random same-size subsets?
  - Figure C: (kaomoji × quadrant) cosine alignment to quadrant
    aggregates. Does the same kaomoji carry different final-token
    signatures under different Russell quadrants?

Grouping key is ``first_word`` with a kaomoji-prefix-glyph filter,
matching analysis.plot_kaomoji_heatmap. This surfaces every observed
bracket-form, not just taxonomy-registered ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Row filter: first character must be one of these opening brackets or
# common kaomoji-prefix glyphs. Matches analysis.plot_kaomoji_heatmap.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def load_rows(path: str) -> pd.DataFrame:
    """Load emotional_raw.jsonl, explode probe vectors, attach quadrant."""
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    # Explode both probe vectors into per-probe columns.
    for prefix, src in (("t0", "probe_scores_t0"), ("tlast", "probe_scores_tlast")):
        stacked = np.asarray(df[src].tolist(), dtype=float)
        for i, probe in enumerate(PROBES):
            df[f"{prefix}_{probe}"] = stacked[:, i]
        df = df.drop(columns=[src])
    # Derive quadrant from prompt_id prefix ("hp01" -> "HP").
    df["quadrant"] = df["prompt_id"].str[:2].str.upper()
    return df


def tlast_matrix(df: pd.DataFrame) -> np.ndarray:
    """5-axis final-token probe matrix in canonical PROBES order."""
    from .config import PROBES
    cols = [f"tlast_{p}" for p in PROBES]
    return df[cols].to_numpy()


def _use_cjk_font() -> None:
    """Force a matplotlib font that renders the Japanese-bracket kaomoji
    glyphs. Copied from analysis._use_cjk_font to keep this module
    standalone; the two copies should be kept consistent."""
    import matplotlib
    import matplotlib.font_manager as fm
    preferred = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Maru Gothic ProN",
        "Apple Color Emoji", "Noto Sans CJK JP", "Yu Gothic", "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return


def _kaomoji_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose first_word starts with a kaomoji-ish glyph and has
    no NaN in the tlast probe columns."""
    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    sub = df.dropna(subset=tlast_cols).copy()
    sub = sub[sub["first_word"].str.len() > 0]
    sub = sub[sub["first_word"].str[0].isin(KAOMOJI_START_CHARS)]
    return sub


def _grouped_means(sub: pd.DataFrame, *, min_count: int) -> tuple[pd.DataFrame, pd.Series]:
    """Group surviving rows by first_word, keep kaomoji with count >=
    min_count, return (per-kaomoji mean tlast probe matrix, counts)."""
    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    grouped = sub.groupby("first_word")[tlast_cols].mean()
    counts = sub.groupby("first_word").size()
    keep = counts[counts >= min_count].index
    grouped = grouped.loc[grouped.index.isin(keep)]
    counts = counts.loc[grouped.index]
    return grouped, counts


def plot_kaomoji_cosine_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
) -> None:
    """Figure A: per-kaomoji mean final-token probe vector, pairwise
    cosine similarity with hierarchical-clustering row order. Mirrors
    analysis.plot_kaomoji_heatmap but on tlast columns and with
    emotional-experiment title/context."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        print("  [Fig A] no kaomoji rows; skipping")
        return
    grouped, counts = _grouped_means(sub, min_count=min_count)
    if len(grouped) < 3:
        print(f"  [Fig A] only {len(grouped)} kaomoji with n≥{min_count}; skipping")
        return

    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]
    labels = grouped.index.to_numpy()[order]
    label_counts = counts.loc[labels].to_numpy()

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in labels]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * n + 4), max(7, 0.28 * n + 3)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    y_labels = [f"{k}  n={c}" for k, c in zip(labels, label_counts)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(
        f"Figure A: per-kaomoji final-token probe-vector cosine similarity  "
        f"(n ≥ {min_count}; {n} kaomoji)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [
        Patch(color=pole_color[+1], label="taxonomy: happy"),
        Patch(color=pole_color[-1], label="taxonomy: sad"),
        Patch(color=pole_color[0], label="other / unlabeled"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left", bbox_to_anchor=(1.15, 0.0),
        frameon=False, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
