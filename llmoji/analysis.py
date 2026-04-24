# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Analysis helpers for the pilot.

Single-number summary: ARI of k-means (k=2) on the 5-axis probe vector
against pre-registered kaomoji pole labels, restricted to the
kaomoji_prompted unsteered arm with kaomoji_label != 0 (drop 'other'
bucket for this analysis).

Figures:
  - Fig 1a: 2D scatter of (happy.sad_t0, warm.clinical_t0) colored by
    kaomoji pole. No reduction — directly interpretable axes.
  - Fig 1b: PCA(2) of 5-axis probe vector colored by kaomoji pole.
  - Fig 2:  Kaomoji-frequency bars across the four arms. The causal plot.
  - Fig 3:  Per-kaomoji mean probe vector, cosine-similarity heatmap
    with hierarchical clustering (dendrogram-ordered).
  - Fig 4:  Confusion: k-means cluster assignment vs kaomoji pole label.

Decision rules evaluated:
  (1) kaomoji distribution non-degenerate?
  (2) monotonic shift sad-steer < unsteered < happy-steer?
  (3) probe/pole correlation in unsteered arm?

Pyright pragmas above silence pandas-stub noise on `df[bool_mask]`-style
narrowing — research-code trade-off for readable pandas idioms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AxisVerdict:
    """Decision-rule evaluation for one axis (e.g. happy.sad or
    angry.calm). Rule 2 compares the two intervention arms against the
    unsteered arm; callers supply arm names and the positive-pole
    label in ``evaluate_axis``."""
    axis: str
    positive_arm: str                  # e.g. "steered_happy"
    negative_arm: str                  # e.g. "steered_sad"
    rule1_non_degenerate: bool
    rule1_unique_kaomoji: int
    rule2_monotonic_shift: bool
    rule2_positive_fraction_by_condition: dict[str, float]
    rule3_spearman_rho: float
    rule3_spearman_p: float
    kmeans_ari: float

    def summary(self) -> str:
        lines = [
            f"  [axis: {self.axis}]",
            f"  Rule 1 (distribution non-degenerate): {'PASS' if self.rule1_non_degenerate else 'FAIL'}"
            f"  [{self.rule1_unique_kaomoji} labeled kaomoji in unsteered arm]",
            f"  Rule 2 (monotonic steering shift): {'PASS' if self.rule2_monotonic_shift else 'FAIL'}",
            "    positive-pole fraction by condition:",
            *[f"      {k}: {v:.3f}"
              for k, v in self.rule2_positive_fraction_by_condition.items()],
            f"  Rule 3 (probe/pole Spearman): rho={self.rule3_spearman_rho:+.3f}"
            f"  p={self.rule3_spearman_p:.3g}"
            f"  {'PASS' if abs(self.rule3_spearman_rho) > 0.2 else 'FAIL'}",
            f"  k-means ARI (k=2 vs pre-registered pole): {self.kmeans_ari:+.3f}",
        ]
        return "\n".join(lines)


# Back-compat alias so older scripts importing PilotVerdict still work.
PilotVerdict = AxisVerdict


def load_rows(path: str) -> pd.DataFrame:
    """Load the JSONL pilot output into a DataFrame, exploding the
    probe_scores_t0 list into per-probe columns."""
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    # Explode the probe_scores_t0 list column into one column per probe.
    stacked = np.asarray(df["probe_scores_t0"].tolist(), dtype=float)
    for i, probe in enumerate(PROBES):
        df[f"t0_{probe}"] = stacked[:, i]
    df = df.drop(columns=["probe_scores_t0"])
    return df


def probe_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the 5-axis feature matrix in canonical PROBES order."""
    from .config import PROBES
    cols = [f"t0_{p}" for p in PROBES]
    return df[cols].to_numpy()


def _by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    return df.loc[df["condition"] == condition].copy()


def _axis_label(axis: str, first_word: object) -> int:
    """Look up a per-axis pole label for a first_word value. Tolerates
    non-string inputs (NaN rows, etc.) by returning the unmarked label."""
    from .taxonomy import label_on
    if not isinstance(first_word, str) or not first_word:
        return 0
    return label_on(axis, first_word)


def _add_axis_label_column(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Return df with a ``label_<axis>`` column added (pole ∈ {-1, 0, +1})."""
    col = f"label_{axis}"
    if col not in df.columns:
        df = df.copy()
        df[col] = df["first_word"].map(lambda w: _axis_label(axis, w))
    return df


def positive_fraction(df: pd.DataFrame, label_col: str) -> float:
    """Fraction of ``label_col`` == +1 among rows where label ≠ 0."""
    non_other: pd.DataFrame = df.loc[df[label_col] != 0]
    if len(non_other) == 0:
        return float("nan")
    return float((non_other[label_col] == +1).mean())


_AXIS_ARMS: dict[str, tuple[str, str]] = {
    # axis → (positive-pole arm name, negative-pole arm name)
    "happy.sad":  ("steered_happy", "steered_sad"),
    "angry.calm": ("steered_angry", "steered_calm"),
}


def evaluate_axis(df: pd.DataFrame, axis: str) -> AxisVerdict:
    """Decision-rule evaluation on a single axis. Pole labels come from
    ``taxonomy.label_on(axis, first_word)``, not the JSONL's baked-in
    ``kaomoji_label`` (which tracks happy.sad only). ``df`` is expected
    to contain exactly one row per (condition, prompt_id, seed)."""
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from .config import CONDITIONS

    pos_arm, neg_arm = _AXIS_ARMS[axis]
    label_col = f"label_{axis}"
    df = _add_axis_label_column(df, axis)
    unsteered = _by_condition(df, "kaomoji_prompted")

    # --- Rule 1: non-degenerate distribution in unsteered arm ---
    labels = unsteered[label_col]
    labeled = unsteered.loc[labels != 0, "first_word"]
    unique_kaomoji = int(labeled.nunique())
    rule1 = (
        unique_kaomoji >= 3
        and bool((labels == +1).any())
        and bool((labels == -1).any())
    )

    # --- Rule 2: monotonic positive-pole fraction across conditions ---
    frac_by_cond = {
        c: positive_fraction(_by_condition(df, c), label_col) for c in CONDITIONS
    }
    rule2 = (
        frac_by_cond.get(neg_arm, float("nan"))
        < frac_by_cond.get("kaomoji_prompted", float("nan"))
        < frac_by_cond.get(pos_arm, float("nan"))
    )

    # --- Rule 3: probe/pole Spearman on unsteered arm ---
    non_other: pd.DataFrame = unsteered.loc[unsteered[label_col] != 0].dropna(
        subset=[f"t0_{axis}"]
    )
    if len(non_other) >= 8:
        sp = spearmanr(non_other[f"t0_{axis}"], non_other[label_col])
        sp_arr = np.asarray(tuple(sp), dtype=float)
        rho = float(sp_arr[0])
        p = float(sp_arr[1])
    else:
        rho = float("nan")
        p = float("nan")

    # --- k-means ARI on the 5-axis probe vector ---
    X = probe_matrix(non_other)
    y = non_other[label_col].to_numpy()
    if len(X) >= 8 and not np.isnan(X).any():
        km = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
        ari = float(adjusted_rand_score(y, km.labels_))
    else:
        ari = float("nan")

    return AxisVerdict(
        axis=axis,
        positive_arm=pos_arm,
        negative_arm=neg_arm,
        rule1_non_degenerate=bool(rule1),
        rule1_unique_kaomoji=int(unique_kaomoji),
        rule2_monotonic_shift=bool(rule2),
        rule2_positive_fraction_by_condition=frac_by_cond,
        rule3_spearman_rho=rho,
        rule3_spearman_p=p,
        kmeans_ari=ari,
    )


def evaluate(df: pd.DataFrame) -> AxisVerdict:
    """Back-compat single-axis evaluation — always happy.sad."""
    return evaluate_axis(df, "happy.sad")


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _use_cjk_font() -> None:
    """Point matplotlib at a CJK-capable font so kaomoji render
    properly. Falls through silently if none is installed — plots then
    show tofu boxes for Japanese glyphs but everything else renders."""
    import matplotlib
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Noto Sans CJK JP", "Hiragino Sans", "Arial Unicode MS",
                      "Apple SD Gothic Neo", "Hiragino Sans GB", "PingFang SC"):
        if candidate in avail:
            matplotlib.rcParams["font.family"] = [candidate, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def _setup_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_axis_scatter(df: pd.DataFrame, out_path: str) -> None:
    """Fig 1a: 2D scatter of (happy.sad, warm.clinical) on the unsteered
    arm, colored by kaomoji pole. If pole separates on axis 0 and not
    elsewhere, that's axis-aligned signal."""
    import matplotlib.pyplot as plt

    sub = df[(df["condition"] == "kaomoji_prompted") & (df["kaomoji_label"] != 0)]
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, color, name in [(+1, "#d94", "happy"), (-1, "#4a7", "sad")]:
        s = sub[sub["kaomoji_label"] == label]
        ax.scatter(
            s["t0_happy.sad"], s["t0_warm.clinical"],
            c=color, alpha=0.7, label=name, s=36, edgecolor="white", linewidth=0.5,
        )
    ax.axhline(0, color="#999", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#999", linewidth=0.6, zorder=0)
    _setup_axes(ax, "probe scores @ token 0 (unsteered)",
                "happy.sad", "warm.clinical")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(df: pd.DataFrame, out_path: str) -> None:
    """Fig 1b: PCA(2) of the 5-axis probe vector, unsteered arm only,
    colored by kaomoji pole."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    sub = df[(df["condition"] == "kaomoji_prompted") & (df["kaomoji_label"] != 0)]
    X = probe_matrix(sub)
    if len(X) < 4 or np.isnan(X).any():
        return

    pca = PCA(n_components=2, random_state=0).fit(X)
    Z = pca.transform(X)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, color, name in [(+1, "#d94", "happy"), (-1, "#4a7", "sad")]:
        mask = sub["kaomoji_label"].to_numpy() == label
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.7, label=name,
                   s=36, edgecolor="white", linewidth=0.5)
    ev = pca.explained_variance_ratio_
    _setup_axes(
        ax,
        "PCA(2) of 5-axis probe vector (unsteered)",
        f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})",
    )
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_condition_bars(df: pd.DataFrame, out_path: str) -> None:
    """Fig 2: the causal plot. Kaomoji-pole breakdown by condition."""
    import matplotlib.pyplot as plt
    from .config import CONDITIONS

    counts = (
        df.groupby(["condition", "kaomoji_label"]).size().unstack(fill_value=0)
    )
    # ensure all three label columns exist
    for label in (+1, -1, 0):
        if label not in counts.columns:
            counts[label] = 0
    counts = counts.loc[[c for c in CONDITIONS if c in counts.index]]
    totals = counts.sum(axis=1)
    frac = counts.div(totals, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = np.zeros(len(frac))
    palette = {+1: "#d94", -1: "#4a7", 0: "#bbb"}
    names = {+1: "happy", -1: "sad", 0: "other"}
    for label in (+1, -1, 0):
        vals = frac[label].to_numpy()
        ax.bar(frac.index, vals, bottom=bottoms, color=palette[label],
               label=names[label], edgecolor="white", linewidth=0.8)
        bottoms = bottoms + vals
    ax.set_ylim(0, 1)
    _setup_axes(ax, "kaomoji pole by condition", "", "fraction")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kaomoji_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
) -> None:
    """Fig 3: per-kaomoji mean probe vector, cosine-similarity heatmap
    with hierarchical clustering (dendrogram-ordered).

    Groups by the raw ``first_word`` so every bracket-form output the
    model emitted gets a row, not just pre-registered taxonomy entries.
    Row labels are colored by taxonomy pole (happy / sad / other).
    Leading tokens that aren't bracket-form (plain English words like
    ``"I"`` or markdown bolding like ``"**Congratulations!**"``) are
    excluded by requiring the first character to be an opening bracket
    or one of the common kaomoji-prefix glyphs.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .config import PROBES
    from .taxonomy import TAXONOMY

    probe_cols = [f"t0_{p}" for p in PROBES]

    # pool across all conditions — we want a per-kaomoji representation
    # summary, not a condition-stratified one.
    sub = df.dropna(subset=probe_cols).copy()
    # keep first_words that look like kaomoji: start with an opening
    # bracket or one of the common kaomoji-prefix glyphs. This sweeps
    # up every observed paren-form, not just taxonomy-registered ones.
    kaomoji_starts = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")
    sub = sub[sub["first_word"].str.len() > 0]
    sub = sub[sub["first_word"].str[0].isin(kaomoji_starts)]
    if len(sub) == 0:
        return

    grouped = sub.groupby("first_word")[probe_cols].mean()
    counts = sub.groupby("first_word").size()
    keep = counts[counts >= min_count].index
    grouped = grouped.loc[grouped.index.isin(keep)]
    if len(grouped) < 3:
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

    # pole-color each label per the current taxonomy: happy=warm,
    # sad=cool, other=gray. Observed-but-unlabeled variants show gray
    # so the reader can see which kaomoji the taxonomy didn't cover.
    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in labels]

    # size the figure to the number of rows; kaomoji labels are long
    # so x-axis needs more horizontal breathing room than y-axis.
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * n + 4), max(7, 0.28 * n + 3)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # show n= on the y-axis after each kaomoji so readers can weigh the
    # noise on each row.
    y_labels = [f"{k}  n={c}" for k, c in zip(labels, label_counts)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(
        f"per-kaomoji probe-vector cosine similarity  "
        f"(n ≥ {min_count} observations; {n} kaomoji shown)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    # small legend showing the row-color → pole mapping
    from matplotlib.patches import Patch
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


def plot_cluster_confusion(df: pd.DataFrame, out_path: str) -> None:
    """Fig 4: confusion of k-means cluster vs pre-registered kaomoji
    pole, on the unsteered non-other subset."""
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    sub = (
        df[(df["condition"] == "kaomoji_prompted") & (df["kaomoji_label"] != 0)]
        .copy()
    )
    X = probe_matrix(sub)
    if len(X) < 4 or np.isnan(X).any():
        return

    km = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    sub["cluster"] = km.labels_

    conf = (
        sub.groupby(["kaomoji_label", "cluster"]).size().unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(conf.to_numpy(), cmap="Blues")
    ax.set_xticks(range(conf.shape[1]))
    ax.set_yticks(range(conf.shape[0]))
    ax.set_xticklabels([f"cluster {c}" for c in conf.columns])
    ax.set_yticklabels([("happy" if y == +1 else "sad") for y in conf.index])
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, str(int(conf.iloc[i, j])),
                    ha="center", va="center", color="black")
    ax.set_title("k-means vs pre-registered pole")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def all_figures(df: pd.DataFrame, figures_dir: str) -> None:
    import os
    _use_cjk_font()
    os.makedirs(figures_dir, exist_ok=True)
    plot_axis_scatter(df, os.path.join(figures_dir, "fig1a_axis_scatter.png"))
    plot_pca_scatter(df, os.path.join(figures_dir, "fig1b_pca_scatter.png"))
    plot_condition_bars(df, os.path.join(figures_dir, "fig2_condition_bars.png"))
    plot_kaomoji_heatmap(df, os.path.join(figures_dir, "fig3_kaomoji_heatmap.png"))
    plot_cluster_confusion(df, os.path.join(figures_dir, "fig4_cluster_confusion.png"))
