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
    """Configure matplotlib font-family as a fallback *chain* covering
    the kaomoji character set. matplotlib 3.6+ does per-glyph fallback
    through this list. The chain below hits 100% of the 90ish non-ASCII
    characters seen across Claude's + gemma's kaomoji output on this
    machine — intersection of Arial Unicode MS (primary), DejaVu Sans +
    Serif (IPA + misc), Tahoma (˵ ˶), Noto Sans Canadian Aboriginal
    (ᗒ ᗕ ᗜ ᗨ), Heiti TC (꒳). Copied to analysis._use_cjk_font and
    scripts/09_claude_faces_plot.py — keep consistent."""
    import matplotlib
    import matplotlib.font_manager as fm
    chain = [
        "Noto Sans CJK JP",  # primary for JP brackets and kana
        "Arial Unicode MS",  # 82% broad Unicode coverage
        "DejaVu Sans",
        "DejaVu Serif",      # covers ᴥ (bear cheek)
        "Tahoma",            # covers ˵ ˶
        "Noto Sans Canadian Aboriginal",  # covers ᗒ ᗕ ᗜ ᗨ
        "Heiti TC",          # covers ꒳
        "Hiragino Sans", "Apple Symbols",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


def _probe_cols(timestep: str) -> list[str]:
    """Probe column list for a timestep prefix ('t0' or 'tlast')."""
    from .config import PROBES
    return [f"{timestep}_{p}" for p in PROBES]


def _kaomoji_rows(df: pd.DataFrame, *, timestep: str = "tlast") -> pd.DataFrame:
    """Rows whose first_word starts with a kaomoji-ish glyph and has
    no NaN in the selected-timestep probe columns."""
    cols = _probe_cols(timestep)
    sub = df.dropna(subset=cols).copy()
    sub = sub[sub["first_word"].str.len() > 0]
    sub = sub[sub["first_word"].str[0].isin(KAOMOJI_START_CHARS)]
    return sub


def _grouped_means(
    sub: pd.DataFrame, *, min_count: int, timestep: str = "tlast",
) -> tuple[pd.DataFrame, pd.Series]:
    """Group surviving rows by first_word, keep kaomoji with count >=
    min_count, return (per-kaomoji mean probe matrix, counts)."""
    cols = _probe_cols(timestep)
    grouped = sub.groupby("first_word")[cols].mean()
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
    timestep: str = "tlast",
) -> None:
    """Figure A: per-kaomoji mean probe vector (at the given timestep),
    pairwise cosine similarity with hierarchical-clustering row order.
    Mirrors analysis.plot_kaomoji_heatmap with emotional-experiment
    title/context. Pass timestep='t0' for first-token state, 'tlast'
    for final-token state."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df, timestep=timestep)
    if len(sub) == 0:
        print(f"  [Fig A {timestep}] no kaomoji rows; skipping")
        return
    grouped, counts = _grouped_means(sub, min_count=min_count, timestep=timestep)
    if len(grouped) < 3:
        print(f"  [Fig A {timestep}] only {len(grouped)} kaomoji with n≥{min_count}; skipping")
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
    ts_label = "first-token (t=0)" if timestep == "t0" else "final-token (t=last)"
    ax.set_title(
        f"Figure A: per-kaomoji {ts_label} probe-vector cosine similarity  "
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


def _cosine_to_mean(vectors: np.ndarray) -> np.ndarray:
    """For each row in `vectors`, its cosine similarity to the mean
    across rows. Handles zero-norm edge case by returning zeros there."""
    if len(vectors) == 0:
        return np.zeros(0)
    mean = vectors.mean(axis=0, keepdims=True)
    mean_norm = np.linalg.norm(mean, axis=1)
    row_norms = np.linalg.norm(vectors, axis=1)
    denom = row_norms * mean_norm
    dots = (vectors * mean).sum(axis=1)
    out = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom > 0)
    return out


def plot_within_kaomoji_consistency(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
    null_iters: int = 500,
    null_seed: int = 0,
    timestep: str = "tlast",
) -> None:
    """Figure B: for each kaomoji with n >= min_count, the distribution
    of cosine(row_vector, kaomoji_mean_vector) across its occurrences.
    Plotted as a horizontal strip chart with per-kaomoji median markers,
    ordered by median (top = tightest). A shaded band behind the strip
    shows the median ± IQR of null subsets (random same-size samples
    from the full kaomoji-bearing pool), interpolated over the
    observed-counts range.

    Interpretation: rows below the null band are real within-kaomoji
    signatures; rows inside the band are indistinguishable from random
    and don't support the 'kaomoji tracks state' hypothesis. Pass
    timestep='t0' for first-token state, 'tlast' for final-token.
    """
    import matplotlib.pyplot as plt
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df, timestep=timestep)
    if len(sub) == 0:
        print(f"  [Fig B {timestep}] no kaomoji rows; skipping")
        return

    cols = _probe_cols(timestep)
    pool = sub[cols].to_numpy()  # all kaomoji-bearing rows

    per_kaomoji: list[tuple[str, np.ndarray, int]] = []
    for km, group in sub.groupby("first_word"):
        if len(group) < min_count:
            continue
        vecs = group[cols].to_numpy()
        sims = _cosine_to_mean(vecs)
        per_kaomoji.append((str(km), sims, len(group)))
    if len(per_kaomoji) < 3:
        print(f"  [Fig B {timestep}] only {len(per_kaomoji)} kaomoji with n≥{min_count}; skipping")
        return

    # sort by median consistency, descending (tightest on top when
    # plotted bottom-to-top later)
    per_kaomoji.sort(key=lambda t: float(np.median(t[1])), reverse=False)

    # Null band: for each distinct group size present in the data,
    # draw `null_iters` random subsets of that size from the full pool
    # and compute each subset's cosine-to-its-own-mean distribution.
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
            sims = _cosine_to_mean(pool[idx])
            medians[j] = float(np.median(sims))
        null_median[size] = float(np.median(medians))
        null_q25[size] = float(np.quantile(medians, 0.25))
        null_q75[size] = float(np.quantile(medians, 0.75))

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}

    n = len(per_kaomoji)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * n + 2)))

    # draw null band as per-row shaded spans (each row's null is sized
    # to that row's n, so the band is stepped rather than continuous)
    for y, (km, _sims, size) in enumerate(per_kaomoji):
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

    # scatter observed per-row cosines + median tick per row
    for y, (km, sims, _size) in enumerate(per_kaomoji):
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
    ax.set_xlabel("cosine(row, kaomoji mean)")
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ts_label = "first-token (t=0)" if timestep == "t0" else "final-token (t=last)"
    ax.set_title(
        f"Figure B: within-kaomoji {ts_label} consistency vs shuffled null\n"
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


def plot_kaomoji_quadrant_alignment(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
    min_per_cell: int = 2,
    timestep: str = "tlast",
) -> None:
    """Figure C: for each kaomoji × quadrant cell with >= min_per_cell
    observations, the cosine similarity between the cell's mean probe
    vector (at the given timestep) and each of the four quadrant-
    aggregate means (averaged across all kaomoji rows in that quadrant).

    Heatmap rows are kaomoji with overall n >= min_count, ordered by
    the row-clustering from Figure A (computed here independently).
    Cells with < min_per_cell observations are shown as hatched/blank.
    Sample counts annotated in cells.

    Interpretation: if row ``(｡◕‿◕｡)`` looks red in HP and LP columns
    but blue in HN and LN, valence-context is written into its final-
    token signature. If the row is uniform, the signature is
    context-invariant.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df, timestep=timestep)
    if len(sub) == 0:
        print(f"  [Fig C {timestep}] no kaomoji rows; skipping")
        return

    from .config import PROBES
    cols = _probe_cols(timestep)
    grouped, counts = _grouped_means(sub, min_count=min_count, timestep=timestep)
    if len(grouped) < 3:
        print(f"  [Fig C {timestep}] only {len(grouped)} kaomoji with n≥{min_count}; skipping")
        return

    # Quadrant aggregates: mean probe vector per quadrant across all
    # kaomoji-bearing rows (not per-kaomoji-then-mean).
    quadrants = ["HP", "LP", "HN", "LN"]
    q_means: dict[str, np.ndarray] = {}
    for q in quadrants:
        q_rows = sub[sub["quadrant"] == q]
        if len(q_rows) == 0:
            q_means[q] = np.full(len(PROBES), np.nan)
        else:
            q_means[q] = q_rows[cols].to_numpy().mean(axis=0)

    # Per-(kaomoji, quadrant) mean and count.
    kms = list(grouped.index)
    cell_sim = np.full((len(kms), len(quadrants)), np.nan)
    cell_n = np.zeros((len(kms), len(quadrants)), dtype=int)
    for i, km in enumerate(kms):
        for j, q in enumerate(quadrants):
            cell_rows = sub[(sub["first_word"] == km) & (sub["quadrant"] == q)]
            cell_n[i, j] = len(cell_rows)
            if len(cell_rows) < min_per_cell:
                continue
            cell_mean = cell_rows[cols].to_numpy().mean(axis=0)
            if np.isnan(q_means[q]).any():
                continue
            a = cell_mean.reshape(1, -1)
            b = q_means[q].reshape(1, -1)
            cell_sim[i, j] = float(cosine_similarity(a, b)[0, 0])

    # Row ordering: cluster kaomoji means (same as Figure A).
    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    kms_ordered = [kms[i] for i in order]
    cell_sim = cell_sim[order, :]
    cell_n = cell_n[order, :]
    row_counts = counts.loc[kms_ordered].to_numpy()

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in kms_ordered]

    n = len(kms_ordered)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * n + 2)))
    im = ax.imshow(cell_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(quadrants)))
    ax.set_yticks(range(n))
    ax.set_xticklabels(quadrants)
    y_labels = [f"{k}  n={c}" for k, c in zip(kms_ordered, row_counts)]
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    # annotate cells with count; blank out sub-min cells with hatching
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

    ts_label = "first-token (t=0)" if timestep == "t0" else "final-token (t=last)"
    ax.set_title(
        f"Figure C: kaomoji × quadrant alignment ({ts_label})\n"
        f"(color = cosine sim; hatched = n<{min_per_cell} observations)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_table(
    df: pd.DataFrame, *, min_count: int = 3, timestep: str = "tlast",
) -> pd.DataFrame:
    """Per-kaomoji summary for the emotional experiment. One row per
    kaomoji with n >= min_count:

      first_word, n, taxonomy_label, median_within_consistency,
      dominant_quadrant, HP_n, LP_n, HN_n, LN_n

    Consistency column is computed on the given timestep's probe
    vectors ('t0' or 'tlast').
    """
    from .taxonomy import TAXONOMY

    cols = _probe_cols(timestep)
    sub = _kaomoji_rows(df, timestep=timestep)
    if len(sub) == 0:
        return pd.DataFrame(columns=[
            "first_word", "n", "taxonomy_label", "median_within_consistency",
            "dominant_quadrant", "HP_n", "LP_n", "HN_n", "LN_n",
        ])

    rows: list[dict[str, Any]] = []
    for km, group in sub.groupby("first_word"):
        if len(group) < min_count:
            continue
        vecs = group[cols].to_numpy()
        sims = _cosine_to_mean(vecs)
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
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("median_within_consistency", ascending=False).reset_index(drop=True)
    return out


def compute_probe_correlations(
    df: pd.DataFrame, *, timestep: str = "t0",
) -> dict[str, Any]:
    """Full pairwise Pearson + Spearman correlation between probe
    scores at the given timestep. Returns overall + per-quadrant.
    Run on v3 unsteered data only — steered data would shift the
    probe distributions and confound the collapse reading.
    """
    from scipy.stats import pearsonr, spearmanr
    from .config import PROBES

    cols = _probe_cols(timestep)
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
    for q in ("HP", "LP", "HN", "LN"):
        out["by_subset"][q] = pair_stats(df[df["quadrant"] == q])
    return out


def plot_probe_correlation_matrix(
    df: pd.DataFrame, out_path: str, *,
    method: str = "pearson", timestep: str = "t0",
) -> None:
    """Plot a 5-panel figure: overall probe correlation matrix + one
    per Russell quadrant. method='pearson' or 'spearman'. Uses t0
    (== whole-generation aggregate under stateless=True) — this is
    the same column v2's valence-collapse claim was derived from."""
    import matplotlib.pyplot as plt
    from .config import PROBES

    _use_cjk_font()
    stats = compute_probe_correlations(df, timestep=timestep)
    panels = [("all", "all v3 rows"), ("HP", "HP"), ("LP", "LP"),
              ("HN", "HN"), ("LN", "LN")]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    im = None
    for ax, (key, title) in zip(axes, panels):
        sub = stats["by_subset"][key]
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
    fig.suptitle(f"v3 probe-probe {method} correlations "
                 f"(t0 = whole-generation aggregate under stateless)",
                 fontsize=11)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.7, label=f"{method} r")
        cb.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prompt_kaomoji_matrix(
    df: pd.DataFrame, *, top_k: int = 12, min_prompt_emissions: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(80-prompt × top-K kaomoji) emission-count matrix. Rows are
    ordered by quadrant (HP, LP, HN, LN) then by prompt_id within
    quadrant. Returns (matrix, row_meta) where row_meta has prompt_id,
    quadrant, prompt_text, total_emissions."""
    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        return pd.DataFrame(), pd.DataFrame()

    top = sub["first_word"].value_counts().head(top_k).index.tolist()
    prompts = df[["prompt_id", "quadrant", "prompt_text"]].drop_duplicates("prompt_id")
    q_order = {"HP": 0, "LP": 1, "HN": 2, "LN": 3}
    prompts = prompts.assign(_qord=prompts["quadrant"].map(q_order))
    prompts = prompts.sort_values(["_qord", "prompt_id"]).drop(columns=["_qord"])

    mat = pd.DataFrame(
        0, index=prompts["prompt_id"].tolist(), columns=top, dtype=int,
    )
    for pid, group in sub.groupby("prompt_id"):
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

    quad_colors = {"HP": "#e6b260", "LP": "#b28c3d",
                   "HN": "#d06c5a", "LN": "#5f7ca8"}
    row_colors = [quad_colors[q] for q in meta["quadrant"]]

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

    # Quadrant divider lines.
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
