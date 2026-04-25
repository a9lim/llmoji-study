# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
"""Analysis for the v3 emotional-battery experiment.

Hidden-state-first: the per-kaomoji cosine heatmap (Fig A), the
within-kaomoji consistency plot (Fig B), the kaomoji × quadrant
alignment (Fig C), the v3 PCA scatter, and the per-kaomoji summary
table all operate on hidden-state features loaded from per-row .npz
sidecars via ``llmoji.hidden_state_analysis.load_hidden_features``.
They pass (metadata DataFrame, hidden-state matrix) aligned row-wise.

Kept as probe-specific (these answer questions about probe structure
that the hidden-state replacements don't): ``compute_probe_correlations``
/ ``plot_probe_correlation_matrix`` — these still need the probe-score
columns from the JSONL. Loaded via the separate ``load_rows`` helper.

Kept as non-probe, non-hidden-state: ``prompt_kaomoji_matrix`` /
``plot_prompt_kaomoji_matrix`` — these are emission counts, not
feature-space analyses.

Centering: default centering ON for all cosine heatmaps — same
reasoning as the probe-based version (shared response-baseline
direction dominates uncentered cosine), amplified in 4096-dim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Row filter: first character must be one of these opening brackets or
# common kaomoji-prefix glyphs. Matches the v1/v2 analysis convention.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


# Russell-quadrant palette + ordering. Shared with scripts/17_v3_face_scatters.py
# so per-face plots use a consistent colour scheme.
QUADRANT_ORDER = ["HP", "LP", "HN", "LN", "NB"]
QUADRANT_COLORS = {
    "HP": "#d62728",  # red — high-arousal positive
    "LP": "#2ca02c",  # green — low-arousal positive
    "HN": "#ff7f0e",  # orange — high-arousal negative
    "LN": "#1f77b4",  # blue — low-arousal negative
    "NB": "#7f7f7f",  # gray — neutral baseline
}


def per_face_dominant_quadrant(df: pd.DataFrame) -> dict[str, str]:
    """For each first_word, return its dominant emission quadrant —
    the quadrant it appears in most. Ties broken by QUADRANT_ORDER
    position (earlier wins)."""
    from collections import Counter
    out: dict[str, str] = {}
    for fw, sub in df.groupby("first_word"):
        counts = Counter(sub["quadrant"].tolist())
        max_count = max(counts.values())
        candidates = [q for q in QUADRANT_ORDER if counts.get(q, 0) == max_count]
        out[str(fw)] = candidates[0] if candidates else "NB"
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_rows(path: str) -> pd.DataFrame:
    """Load v3 emotional_raw.jsonl, explode probe vectors into per-
    probe columns (for probe-correlation analysis), attach quadrant.
    Hidden-state analysis uses ``load_emotional_features`` instead."""
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    for prefix, src in (("t0", "probe_scores_t0"), ("tlast", "probe_scores_tlast")):
        if src in df.columns:
            stacked = np.asarray(df[src].tolist(), dtype=float)
            for i, probe in enumerate(PROBES):
                df[f"{prefix}_{probe}"] = stacked[:, i]
            df = df.drop(columns=[src])
    df["quadrant"] = df["prompt_id"].str[:2].str.upper()
    return df


def load_v1v2_neutral_baseline(path: str) -> pd.DataFrame:
    """v1/v2 kaomoji_prompted rows with neutral-valence prompts, as
    probe-column DataFrame. Used by the probe-correlation plots for a
    cross-experiment neutral comparator."""
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    sub = df[
        (df["condition"] == "kaomoji_prompted")
        & (df["prompt_valence"] == 0)
    ].copy()
    stacked = np.asarray(sub["probe_scores_t0"].tolist(), dtype=float)
    for i, probe in enumerate(PROBES):
        sub[f"t0_{probe}"] = stacked[:, i]
        sub[f"tlast_{probe}"] = stacked[:, i]
    sub["quadrant"] = "NB"
    return sub


def load_emotional_features(
    jsonl_path: str | Path,
    data_dir: Path,
    *,
    experiment: str = "v3",
    which: str = "h_last",
    layer: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load v3 JSONL + its hidden-state sidecars. Filters to kaomoji-
    bearing rows by first_word prefix. Attaches a ``quadrant`` column
    (HP/LP/HN/LN/NB) derived from prompt_id.

    Returns (metadata df, (n_rows, hidden_dim) feature matrix) aligned
    row-wise. Downstream plot functions take this pair directly.
    """
    from .hidden_state_analysis import load_hidden_features
    df, X = load_hidden_features(
        jsonl_path, data_dir, experiment,
        which=which, layer=layer,
    )
    if len(df) == 0:
        return df, X
    from .taxonomy import canonicalize_kaomoji
    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word_raw=df["first_word"],
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and len(s) > 0 and s[0] in KAOMOJI_START_CHARS
        for s in df["first_word"]
    ])
    return df.loc[mask].reset_index(drop=True), X[mask]


def load_v1v2_neutral_baseline_features(
    jsonl_path: str | Path,
    data_dir: Path,
    *,
    experiment: str = "v1v2",
    which: str = "h_last",
    layer: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """v1/v2 neutral-valence kaomoji_prompted rows as hidden-state
    features. Used for adding a cross-experiment neutral baseline to
    the v3 PCA scatter."""
    from .hidden_state_analysis import load_hidden_features
    df, X = load_hidden_features(
        jsonl_path, data_dir, experiment,
        which=which, layer=layer,
    )
    if len(df) == 0:
        return df, X
    from .taxonomy import canonicalize_kaomoji
    mask = (
        (df["condition"] == "kaomoji_prompted")
        & (df["prompt_valence"] == 0)
    ).to_numpy()
    df_nb = df.loc[mask].assign(quadrant="NB").reset_index(drop=True)
    df_nb = df_nb.assign(
        first_word_raw=df_nb["first_word"],
        first_word=df_nb["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    X_nb = X[mask]
    # Also apply the kaomoji first-char filter for consistency with v3.
    kao_mask = np.asarray([
        isinstance(s, str) and len(s) > 0 and s[0] in KAOMOJI_START_CHARS
        for s in df_nb["first_word"]
    ])
    return df_nb.loc[kao_mask].reset_index(drop=True), X_nb[kao_mask]


# ---------------------------------------------------------------------------
# Font / utility
# ---------------------------------------------------------------------------


def _use_cjk_font() -> None:
    """Configure matplotlib font-family as a fallback *chain* covering
    the kaomoji character set — see CLAUDE.md font-fallback gotcha."""
    import matplotlib
    import matplotlib.font_manager as fm
    chain = [
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
        "DejaVu Serif",
        "Tahoma",
        "Noto Sans Canadian Aboriginal",
        "Heiti TC",
        "Hiragino Sans", "Apple Symbols",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


# ---------------------------------------------------------------------------
# Figure A: per-kaomoji cosine heatmap (hidden-state)
# ---------------------------------------------------------------------------


def plot_kaomoji_cosine_heatmap(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_count: int = 0,
    center: bool = True,
) -> None:
    """Figure A: per-kaomoji mean hidden-state vector, pairwise cosine
    similarity with hierarchical-clustering row order. Row/column
    labels coloured by dominant emission quadrant (HP/LP/HN/LN/NB)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from .hidden_state_analysis import cosine_similarity_matrix, group_mean_vectors

    _use_cjk_font()

    if len(df) == 0:
        print("  [Fig A] no rows; skipping")
        return

    keys_df, M, counts = group_mean_vectors(
        df, X, "first_word", min_count=min_count,
    )
    if len(keys_df) < 3:
        print(f"  [Fig A] only {len(keys_df)} kaomoji with n≥{min_count}; skipping")
        return

    sim = cosine_similarity_matrix(M, center=center)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]
    labels = keys_df["first_word"].iloc[order].to_numpy()
    label_counts = counts.iloc[order].to_numpy()

    quadrant_for = per_face_dominant_quadrant(df)
    label_quadrants = [quadrant_for.get(str(k), "NB") for k in labels]
    row_colors = [QUADRANT_COLORS[q] for q in label_quadrants]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * n + 4), max(7, 0.28 * n + 3)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    y_labels = [f"{k}  n={c}" for k, c in zip(labels, label_counts)]
    ax.set_xticklabels(labels.tolist(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    centering_note = "grand-mean centered; " if center else "uncentered; "
    filter_note = "" if min_count <= 1 else f"n ≥ {min_count}; "
    ax.set_title(
        f"Figure A: per-kaomoji HIDDEN-STATE cosine similarity  "
        f"({centering_note}{filter_note}{n} kaomoji)\n"
        "rows/cols hierarchically clustered; tick colour = dominant emission quadrant"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [
        Patch(color=QUADRANT_COLORS[q], label=q) for q in QUADRANT_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left", bbox_to_anchor=(1.15, 0.0),
        frameon=False, fontsize=8, title="dominant quadrant",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: within-kaomoji consistency (hidden-state)
# ---------------------------------------------------------------------------


def plot_within_kaomoji_consistency(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_count: int = 3,
    null_iters: int = 500,
    null_seed: int = 0,
) -> None:
    """Figure B: for each kaomoji with n >= min_count, distribution of
    cosine(row_vector, kaomoji_mean_vector) in hidden-state space,
    with a shuffled-subset null band. Rows below the null are
    signatures tighter than random same-size subsets."""
    import matplotlib.pyplot as plt
    from .hidden_state_analysis import cosine_to_mean
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    if len(df) == 0:
        print("  [Fig B] no rows; skipping")
        return

    per_kaomoji: list[tuple[str, np.ndarray, int]] = []
    pool = X  # all kaomoji-bearing rows (caller already filtered)
    for km, group in df.groupby("first_word"):
        if len(group) < min_count:
            continue
        idxs = group.index.to_numpy()
        vecs = X[idxs]
        sims = cosine_to_mean(vecs)
        per_kaomoji.append((str(km), sims, len(group)))
    if len(per_kaomoji) < 3:
        print(f"  [Fig B] only {len(per_kaomoji)} kaomoji with n≥{min_count}; skipping")
        return

    per_kaomoji.sort(key=lambda t: float(np.median(t[1])), reverse=False)

    # Null band: per distinct group size, draw null_iters random
    # subsets of that size from the full kaomoji pool, compute each
    # subset's cosine-to-mean median/IQR.
    rng = np.random.default_rng(null_seed)
    sizes = sorted({t[2] for t in per_kaomoji})
    null_median: dict[int, float] = {}
    null_q25: dict[int, float] = {}
    null_q75: dict[int, float] = {}
    N = len(pool)
    for size in sizes:
        medians = np.empty(null_iters)
        for j in range(null_iters):
            idx = rng.choice(N, size=size, replace=False)
            sims = cosine_to_mean(pool[idx])
            medians[j] = float(np.median(sims))
        null_median[size] = float(np.median(medians))
        null_q25[size] = float(np.quantile(medians, 0.25))
        null_q75[size] = float(np.quantile(medians, 0.75))

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}

    n = len(per_kaomoji)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * n + 2)))

    for y, entry in enumerate(per_kaomoji):
        size = entry[2]
        ax.fill_betweenx(
            [y - 0.4, y + 0.4],
            null_q25[size], null_q75[size],
            color="#cccccc", alpha=0.6, linewidth=0,
        )
        ax.plot(
            [null_median[size], null_median[size]],
            [y - 0.4, y + 0.4],
            color="#888888", linewidth=1,
        )

    for y, entry in enumerate(per_kaomoji):
        km, sims, _ = entry
        color = pole_color.get(TAXONOMY.get(km, 0), "#666")
        jitter = (rng.random(len(sims)) - 0.5) * 0.3
        ax.scatter(sims, np.full(len(sims), y) + jitter, s=14, color=color, alpha=0.7)
        ax.plot(
            [float(np.median(sims))] * 2,
            [y - 0.4, y + 0.4],
            color=color, linewidth=2,
        )

    y_labels = [f"{km}  n={size}" for km, _, size in per_kaomoji]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, (km, _, _) in zip(ax.get_yticklabels(), per_kaomoji):
        tick.set_color(pole_color.get(TAXONOMY.get(km, 0), "#666"))
    ax.set_xlabel("cosine(row, kaomoji mean) — hidden-state space")
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_title(
        f"Figure B: within-kaomoji HIDDEN-STATE consistency vs shuffled null\n"
        f"(n ≥ {min_count}; null = {null_iters} random same-size subsets)"
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=pole_color[+1], label="taxonomy: happy"),
        Patch(color=pole_color[-1], label="taxonomy: sad"),
        Patch(color=pole_color[0], label="other / unlabeled"),
        Patch(color="#cccccc", label="null band (IQR)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C: kaomoji × quadrant alignment (hidden-state)
# ---------------------------------------------------------------------------


def plot_kaomoji_quadrant_alignment(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_count: int = 3,
    min_per_cell: int = 2,
    center: bool = True,
) -> None:
    """Figure C: for each (kaomoji × quadrant) cell with n >=
    min_per_cell, cosine(cell_mean_hidden, quadrant_aggregate_hidden).
    Centered cosine — both cells and quadrant aggregates centered
    against the same hidden-state pool mean."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from .hidden_state_analysis import cosine_similarity_matrix, group_mean_vectors
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    if len(df) == 0:
        print("  [Fig C] no rows; skipping")
        return

    keys_df, grouped, counts = group_mean_vectors(
        df, X, "first_word", min_count=min_count,
    )
    if len(keys_df) < 3:
        print(f"  [Fig C] only {len(keys_df)} kaomoji with n≥{min_count}; skipping")
        return

    # Common centering reference: the pool mean.
    pool_mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float32)

    quadrants = sorted(df["quadrant"].unique().tolist())
    q_means: dict[str, np.ndarray] = {}
    for q in quadrants:
        q_mask = (df["quadrant"] == q).to_numpy()
        if not q_mask.any():
            q_means[q] = np.full(X.shape[1], np.nan)
        else:
            q_means[q] = X[q_mask].mean(axis=0) - pool_mean

    kms = keys_df["first_word"].tolist()
    cell_sim = np.full((len(kms), len(quadrants)), np.nan)
    cell_n = np.zeros((len(kms), len(quadrants)), dtype=int)
    for i, km in enumerate(kms):
        for j, q in enumerate(quadrants):
            cell_mask = (
                (df["first_word"] == km) & (df["quadrant"] == q)
            ).to_numpy()
            cell_n[i, j] = int(cell_mask.sum())
            if cell_n[i, j] < min_per_cell:
                continue
            cell_mean = X[cell_mask].mean(axis=0) - pool_mean
            qm = q_means[q]
            if np.isnan(qm).any():
                continue
            # Cosine between centered cell and centered quadrant aggregate.
            denom = float(np.linalg.norm(cell_mean) * np.linalg.norm(qm))
            if denom <= 0:
                continue
            cell_sim[i, j] = float(cell_mean @ qm / denom)

    # Row ordering: cluster kaomoji means (same recipe as Fig A).
    sim = cosine_similarity_matrix(grouped, center=center)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    kms_ordered = [kms[i] for i in order]
    cell_sim = cell_sim[order, :]
    cell_n = cell_n[order, :]
    row_counts = counts.iloc[order].to_numpy()

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in kms_ordered]

    n = len(kms_ordered)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(quadrants) + 3),
                                    max(4, 0.28 * n + 2)))
    im = ax.imshow(cell_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(quadrants)))
    ax.set_yticks(range(n))
    ax.set_xticklabels(quadrants)
    y_labels = [f"{k}  n={c}" for k, c in zip(kms_ordered, row_counts)]
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    for i in range(n):
        for j in range(len(quadrants)):
            count = int(cell_n[i, j])
            if count < min_per_cell:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#eeeeee",
                        hatch="///", edgecolor="#bbbbbb", linewidth=0,
                    )
                )
            ax.text(j, i, str(count), ha="center", va="center", fontsize=7,
                    color="#333" if count >= min_per_cell else "#888")

    centering_note = "pool-mean centered; " if center else "uncentered; "
    ax.set_title(
        f"Figure C: kaomoji × quadrant HIDDEN-STATE alignment\n"
        f"({centering_note}hatched = n<{min_per_cell} observations)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# v3 PCA scatter with Russell-quadrant coloring (hidden-state)
# ---------------------------------------------------------------------------


def plot_v3_pca_valence_arousal(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: str,
    *,
    min_per_cell: int = 2,
    baseline_df: "pd.DataFrame | None" = None,
    baseline_X: "np.ndarray | None" = None,
) -> dict[str, Any]:
    """PCA on v3 row-level hidden-state vectors; project per-(kaomoji,
    quadrant) means through the fitted axes; scatter with Russell-
    circumplex palette (HP orange, LP green, HN red, LN blue, NB gray).

    Fit on combined pool (v3 + optional baseline) so axes reflect all
    plotted data."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA

    _use_cjk_font()

    if len(df) == 0:
        print("  [v3 PCA] no rows; skipping")
        return {}

    # Combine with baseline if provided.
    if baseline_df is not None and baseline_X is not None and len(baseline_df) > 0:
        common_cols = [c for c in df.columns if c in baseline_df.columns]
        df_all = pd.concat(
            [df[common_cols], baseline_df[common_cols]], ignore_index=True,
        )
        X_all = np.concatenate([X, baseline_X], axis=0)
    else:
        df_all = df
        X_all = X

    # Fit PCA on row-level hidden states.
    n_comp = min(5, X_all.shape[0], X_all.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_all)
    var = pca.explained_variance_ratio_

    # Per-(kaomoji, quadrant) means, projected.
    groups: list[tuple[str, str, int]] = []
    group_vecs: list[np.ndarray] = []
    for key, g in df_all.groupby(["first_word", "quadrant"]):
        km, q = key  # type: ignore[misc]
        if len(g) < min_per_cell:
            continue
        idxs = g.index.to_numpy()
        groups.append((str(km), str(q), int(len(g))))
        group_vecs.append(X_all[idxs].mean(axis=0))
    if len(groups) < 3:
        print(f"  [v3 PCA] only {len(groups)} cells with n≥{min_per_cell}; skipping")
        return {}

    coords = pca.transform(np.asarray(group_vecs, dtype=np.float32))

    quadrant_color = {
        "HP": "#e9a01f", "LP": "#4a8a5a",
        "HN": "#c9372d", "LN": "#3d68a8",
        "NB": "#888888",
    }

    fig, ax = plt.subplots(figsize=(11, 9))

    for (km, q, n), pt in zip(groups, coords):
        c = quadrant_color.get(q, "#666")
        ax.scatter(pt[0], pt[1], c=c, s=40 + n * 4,
                   edgecolor="black", linewidth=0.4, alpha=0.78, zorder=3)
        if n >= 3:
            ax.annotate(km, (pt[0], pt[1]), fontsize=6, alpha=0.85,
                        xytext=(5, 5), textcoords="offset points", zorder=4)

    # Per-quadrant centroid stars + within-quadrant / between-centroid stats.
    centroids: dict[str, list[float]] = {}
    within_std: dict[str, list[float]] = {}
    all_quads = sorted({g[1] for g in groups})
    for q_name in all_quads:
        mask = np.array([g[1] == q_name for g in groups])
        if not mask.any():
            continue
        sub_coords = coords[mask][:, :2]
        centroid = sub_coords.mean(axis=0)
        centroids[q_name] = centroid.tolist()
        within_std[q_name] = (
            sub_coords.std(axis=0, ddof=0).tolist()
            if mask.sum() > 1 else [0.0, 0.0]
        )
        c = quadrant_color.get(q_name, "#666")
        ax.plot(centroid[0], centroid[1], marker="*", markersize=28,
                color=c, markeredgecolor="black", markeredgewidth=1.6, zorder=5)
        ax.annotate(f"  {q_name}", (centroid[0], centroid[1]),
                    fontsize=10, fontweight="bold", zorder=6,
                    xytext=(12, 0), textcoords="offset points", va="center")

    ax.axhline(0, color="#ccc", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PC1  ({var[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1] * 100:.1f}% var)")
    ax.set_title(
        f"v3 HIDDEN-STATE PCA — {len(df_all)} rows, {len(groups)} "
        f"(kaomoji, quadrant) cells projected (n ≥ {min_per_cell})"
    )

    legend_labels = [
        ("HP", "HP (high arousal, positive)"),
        ("LP", "LP (low arousal, positive)"),
        ("HN", "HN (high arousal, negative)"),
        ("LN", "LN (low arousal, negative)"),
        ("NB", "NB (neutral baseline)"),
    ]
    legend_handles = [
        Patch(color=quadrant_color[k], label=lbl)
        for k, lbl in legend_labels if k in quadrant_color
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False,
              fontsize=9, title="Russell quadrant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Separation diagnostics.
    centroid_arr = np.array([centroids[q] for q in centroids])
    between_pc1 = float(centroid_arr[:, 0].std(ddof=0)) if len(centroid_arr) else 0.0
    between_pc2 = float(centroid_arr[:, 1].std(ddof=0)) if len(centroid_arr) else 0.0

    return {
        "n_rows_fit": int(len(df_all)),
        "n_cells_plotted": len(groups),
        "explained_variance_ratio": var.tolist(),
        "components": pca.components_[:2].tolist(),
        "quadrant_centroids_pc12": centroids,
        "within_quadrant_std_pc12": within_std,
        "between_centroid_std_pc1": between_pc1,
        "between_centroid_std_pc2": between_pc2,
    }


# ---------------------------------------------------------------------------
# Probe-specific (kept; operates on probe-score columns)
# ---------------------------------------------------------------------------


def compute_probe_correlations(
    df: pd.DataFrame, *, timestep: str = "t0",
) -> dict[str, Any]:
    """Full pairwise Pearson + Spearman correlation between probe
    scores at the given timestep. Returns overall + per-quadrant.
    Inputs: df must have ``t0_<probe>`` / ``tlast_<probe>`` columns
    from ``load_rows``. Run on v3 unsteered data only — steering
    would shift probes and confound the collapse reading."""
    from scipy.stats import pearsonr, spearmanr
    from .config import PROBES

    cols = [f"{timestep}_{p}" for p in PROBES]
    out: dict[str, Any] = {"probes": list(PROBES), "by_subset": {}}

    def pair_stats(sub: pd.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n < 3:
            return {"n": n, "pearson": None, "spearman": None}
        vals = sub[cols].to_numpy()
        p = np.full((len(PROBES), len(PROBES)), np.nan)
        s = np.full((len(PROBES), len(PROBES)), np.nan)
        for i in range(len(PROBES)):
            for j in range(len(PROBES)):
                if i == j:
                    p[i, j] = 1.0
                    s[i, j] = 1.0
                else:
                    p[i, j] = float(pearsonr(vals[:, i], vals[:, j])[0])
                    s[i, j] = float(spearmanr(vals[:, i], vals[:, j])[0])
        return {"n": int(n), "pearson": p.tolist(), "spearman": s.tolist()}

    out["by_subset"]["all"] = pair_stats(df)
    for q in ("HP", "LP", "HN", "LN", "NB"):
        out["by_subset"][q] = pair_stats(df[df["quadrant"] == q])
    return out


def plot_probe_correlation_matrix(
    df: pd.DataFrame, out_path: str, *,
    method: str = "pearson", timestep: str = "t0",
) -> None:
    """Multi-panel: overall probe-correlation matrix + one per quadrant."""
    import matplotlib.pyplot as plt
    from .config import PROBES

    _use_cjk_font()
    stats = compute_probe_correlations(df, timestep=timestep)
    panels = [
        ("all", "all v3 rows"), ("HP", "HP"), ("LP", "LP"),
        ("HN", "HN"), ("LN", "LN"), ("NB", "NB"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
    im = None
    for ax, (key, title) in zip(axes, panels):
        sub = stats["by_subset"].get(key, {"n": 0, "pearson": None, "spearman": None})
        mat = sub.get(method)
        if mat is None:
            ax.text(0.5, 0.5, f"n={sub['n']}\n(too few)", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{title}  n={sub['n']}")
            continue
        arr = np.asarray(mat)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(PROBES)))
        ax.set_yticks(range(len(PROBES)))
        ax.set_xticklabels(PROBES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(PROBES, fontsize=7)
        for i in range(len(PROBES)):
            for j in range(len(PROBES)):
                ax.text(j, i, f"{arr[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(arr[i, j]) > 0.5 else "#333")
        ax.set_title(f"{title}  n={sub['n']}")
    fig.suptitle(
        f"v3 probe-probe {method} correlations "
        f"(t0 = whole-generation aggregate under stateless)", fontsize=11,
    )
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.7, label=f"{method} r")
        cb.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Emission-count matrix (kept; not probe-based, not hidden-state-based)
# ---------------------------------------------------------------------------


def prompt_kaomoji_matrix(
    df: pd.DataFrame, *, top_k: int = 12, min_prompt_emissions: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(N-prompt × top-K kaomoji) emission-count matrix, rows ordered
    by quadrant (HP, LP, HN, LN, NB). Returns (matrix, row_meta)."""
    kao_rows = df[df["first_word"].apply(
        lambda s: isinstance(s, str) and len(s) > 0 and s[0] in KAOMOJI_START_CHARS
    )].copy()
    if len(kao_rows) == 0:
        return pd.DataFrame(), pd.DataFrame()

    top = kao_rows["first_word"].value_counts().head(top_k).index.tolist()
    prompts = df[["prompt_id", "quadrant", "prompt_text"]].drop_duplicates("prompt_id")
    q_order = {"HP": 0, "LP": 1, "HN": 2, "LN": 3, "NB": 4}
    prompts = prompts.assign(_qord=prompts["quadrant"].map(q_order))
    prompts = prompts.sort_values(["_qord", "prompt_id"]).drop(columns=["_qord"])

    mat = pd.DataFrame(
        0, index=prompts["prompt_id"].tolist(), columns=top, dtype=int,
    )
    for pid, group in kao_rows.groupby("prompt_id"):
        if pid not in mat.index:
            continue
        counts = group["first_word"].value_counts()
        for km in top:
            mat.at[pid, km] = int(counts.get(km, 0))

    prompts = prompts.assign(
        total_emissions=[int(mat.loc[pid].sum()) for pid in prompts["prompt_id"]]
    )
    if min_prompt_emissions > 0:
        keep = prompts[prompts["total_emissions"] >= min_prompt_emissions]["prompt_id"]
        mat = mat.loc[keep]
        prompts = prompts[prompts["prompt_id"].isin(keep)]
    return mat, prompts.reset_index(drop=True)


def plot_prompt_kaomoji_matrix(
    df: pd.DataFrame, out_path: str, *, top_k: int = 12,
) -> None:
    """Heatmap of prompt-level emission counts, rows grouped by
    quadrant with horizontal dividers between quadrants."""
    import matplotlib.pyplot as plt

    _use_cjk_font()

    mat, meta = prompt_kaomoji_matrix(df, top_k=top_k)
    if mat.empty:
        print("  [prompt matrix] no rows; skipping")
        return

    quad_colors = {
        "HP": "#e9a01f", "LP": "#4a8a5a",
        "HN": "#c9372d", "LN": "#3d68a8", "NB": "#888888",
    }
    row_colors = [quad_colors.get(q, "#666") for q in meta["quadrant"]]

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(mat.columns) + 4),
                                    max(8, 0.18 * len(mat) + 3)))
    im = ax.imshow(mat.to_numpy(), cmap="magma", aspect="auto",
                   vmin=0, vmax=8)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    y_labels = [f"{pid}  [{q}]  {txt[:40]}"
                for pid, q, txt in zip(meta["prompt_id"],
                                       meta["quadrant"],
                                       meta["prompt_text"])]
    ax.set_yticklabels(y_labels, fontsize=6)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    prev_q = None
    for i, q in enumerate(meta["quadrant"]):
        if prev_q is not None and q != prev_q:
            ax.axhline(i - 0.5, color="#333", linewidth=1)
        prev_q = q

    for i in range(len(mat)):
        for j in range(len(mat.columns)):
            v = int(mat.iat[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v >= 4 else "#333", fontsize=6)

    ax.set_title(
        f"v3 prompt × kaomoji emission counts "
        f"(top-{top_k} kaomoji; rows by quadrant)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label="count out of 8 seeds")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table (hidden-state)
# ---------------------------------------------------------------------------


def summary_table(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    min_count: int = 3,
) -> pd.DataFrame:
    """Per-kaomoji summary with hidden-state cosine-to-mean consistency.

    Columns: first_word, n, taxonomy_label, median_within_consistency,
    dominant_quadrant, HP_n, LP_n, HN_n, LN_n, NB_n.
    """
    from .hidden_state_analysis import cosine_to_mean
    from .taxonomy import TAXONOMY

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "first_word", "n", "taxonomy_label", "median_within_consistency",
            "dominant_quadrant", "HP_n", "LP_n", "HN_n", "LN_n", "NB_n",
        ])

    rows: list[dict[str, Any]] = []
    for km, group in df.groupby("first_word"):
        if len(group) < min_count:
            continue
        idxs = group.index.to_numpy()
        vecs = X[idxs]
        sims = cosine_to_mean(vecs)
        q_counts = group["quadrant"].value_counts()
        dominant = str(q_counts.idxmax()) if len(q_counts) else ""
        rows.append({
            "first_word": km,
            "n": int(len(group)),
            "taxonomy_label": int(TAXONOMY.get(str(km), 0)),
            "median_within_consistency": float(np.median(sims)),
            "dominant_quadrant": dominant,
            "HP_n": int(q_counts.get("HP", 0)),
            "LP_n": int(q_counts.get("LP", 0)),
            "HN_n": int(q_counts.get("HN", 0)),
            "LN_n": int(q_counts.get("LN", 0)),
            "NB_n": int(q_counts.get("NB", 0)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("median_within_consistency", ascending=False).reset_index(drop=True)
    return out
