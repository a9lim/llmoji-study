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

# Split-quadrant ordering — same five plus HN bisected on PAD dominance
# (HN-D anger/contempt, HN-S fear/anxiety). Used by figures opted into the
# rule-3-redesign view via ``split_hn=True`` on loaders. Untagged HN rows
# (the borderline-mixed prompts hn06/hn15/hn17) drop out of split-mode
# entirely; the categorical column gets pd.NA for them so the analysis
# can either filter or surface them as a small grey bucket as it sees fit.
QUADRANT_ORDER_SPLIT = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
# Canonical Russell-circumplex mapping. Saturated enough that 50/50
# RGB-linear mixes between adjacent-quadrant pairs are still
# recognizable (HN+LN → muted purple, HP+LP → olive, etc.); perceived
# luminance balanced ~L*55–62 across all 5 so weighted mixes don't
# drift in brightness when one quadrant dominates by saturation alone.
# Diagonal "contradictory" mixes (HN+LP "anger meets calm",
# HP+LN "excited meets sad") fall to brown / desaturated gray —
# informatively unusual.
QUADRANT_COLORS = {
    "HP": "#d49b3a",  # gold — high arousal, positive (excitement/joy)
    "LP": "#4aa66a",  # green — low arousal, positive (calm/contentment)
    "HN": "#d44a4a",  # red — high arousal, negative (anger/anxiety)
    "LN": "#4a7ed4",  # blue — low arousal, negative (sadness/depression)
    "NB": "#909090",  # gray — neutral baseline
    # Rule-3-redesign PAD-dominance split (2026-05-01). HN-D inherits HN
    # red so aggregate-HN views stay backward-compatible; HN-S takes a
    # saturation-matched magenta-purple that reads as "negative but
    # submissive" without colliding with LN blue.
    "HN-D": "#d44a4a",  # red — anger / contempt (high PAD dominance)
    "HN-S": "#9d4ad4",  # magenta-purple — fear / anxiety (low PAD dominance)
}

# Subset alias for code that wants only the new sub-quadrant pair.
PAD_SUB_COLORS = {q: QUADRANT_COLORS[q] for q in ("HN-D", "HN-S")}


def _palette_for(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return ``(order, colors)`` matching whatever quadrant labels are
    present in ``df["quadrant"]``. Auto-switches to the rule-3 split
    ordering (HN-D / HN-S in lieu of aggregate HN) when those labels
    appear; the colors dict is shared (it includes both)."""
    if "quadrant" not in df.columns:
        return QUADRANT_ORDER, QUADRANT_COLORS
    quads = set(df["quadrant"].astype(str).unique())
    if "HN-D" in quads or "HN-S" in quads:
        return QUADRANT_ORDER_SPLIT, QUADRANT_COLORS
    return QUADRANT_ORDER, QUADRANT_COLORS


def _hn_split_map() -> dict[str, str]:
    """Return ``{prompt_id: 'HN-D' | 'HN-S'}`` for tagged HN prompts.
    Pulls from the EmotionalPrompt registry — single source of truth.
    Untagged HN prompts (pad_dominance == 0) are absent from the map."""
    from .emotional_prompts import EMOTIONAL_PROMPTS
    return {
        p.id: ("HN-D" if p.pad_dominance > 0 else "HN-S")
        for p in EMOTIONAL_PROMPTS
        if p.quadrant == "HN" and p.pad_dominance != 0
    }


def apply_hn_split(
    df: pd.DataFrame,
    X: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Replace the ``quadrant`` column in-place with HN-split labels
    (HN→HN-D/HN-S) and drop rows with untagged HN prompts.

    For scripts that build their own quadrant column from
    ``prompt_id[:2]`` (rather than going through
    ``load_emotional_features``). Pass the row-aligned feature matrix
    ``X`` to keep df+X aligned across the row drop.

    Returns ``(df, X)`` after the split; X is None if not provided."""
    hn_split = _hn_split_map()
    new_q = df.apply(
        lambda r: (
            hn_split.get(r["prompt_id"], None)
            if r["quadrant"] == "HN"
            else r["quadrant"]
        ),
        axis=1,
    )
    keep = new_q.notna().to_numpy()
    df = df.loc[keep].copy()
    df["quadrant"] = new_q[keep].to_numpy()
    df = df.reset_index(drop=True)
    if X is not None:
        X = X[keep]
    return df, X


def per_face_dominant_quadrant(df: pd.DataFrame) -> dict[str, str]:
    """For each first_word, return its dominant emission quadrant —
    the quadrant it appears in most. Ties broken by quadrant-order
    position (earlier wins). Auto-uses the split palette when df has
    HN-D / HN-S labels."""
    from collections import Counter
    order, _ = _palette_for(df)
    out: dict[str, str] = {}
    for fw, sub in df.groupby("first_word"):
        counts = Counter(sub["quadrant"].tolist())
        max_count = max(counts.values())
        candidates = [q for q in order if counts.get(q, 0) == max_count]
        out[str(fw)] = candidates[0] if candidates else "NB"
    return out


def per_face_quadrant_weights(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """For each first_word, return a dict mapping quadrant -> normalized
    emission weight (sum to 1 across the active quadrant set).

    A face emitted in 21 LN rows + 20 HN rows + 0 elsewhere yields
    ``{"LN": 0.512, "HN": 0.488, "HP": 0, "LP": 0, "NB": 0}``.
    With ``split_hn=True`` data, the dict is keyed on the 6-category
    split palette instead. Faces with zero total emissions return
    all-zero weights (caller should guard).
    """
    from collections import Counter
    order, _ = _palette_for(df)
    out: dict[str, dict[str, float]] = {}
    for fw, sub in df.groupby("first_word"):
        counts = Counter(sub["quadrant"].tolist())
        total = sum(counts.values())
        if total == 0:
            out[str(fw)] = {q: 0.0 for q in order}
            continue
        out[str(fw)] = {
            q: counts.get(q, 0) / total for q in order
        }
    return out


def mix_quadrant_color(
    weights: dict[str, float],
    colors: dict[str, str] | None = None,
) -> tuple[float, float, float]:
    """Linear-RGB mix of quadrant colors weighted by ``weights``.

    Weights are expected to sum to 1.0 across the active quadrant set
    (`per_face_quadrant_weights` produces them). The default ``colors``
    is the classic 5-quadrant palette; pass ``QUADRANT_COLORS_SPLIT``
    when working with HN-D/HN-S labelled data.

    A face that's 100% one quadrant returns that quadrant's pure
    color; a 50/50 split returns the RGB midpoint; an even split
    across the full active set returns the palette centroid (close to
    mid-gray for the balanced case).
    """
    if colors is None:
        colors = QUADRANT_COLORS
    from matplotlib.colors import to_rgb
    r = g = b = 0.0
    for q, w in weights.items():
        if w <= 0 or q not in colors:
            continue
        qr, qg, qb = to_rgb(colors[q])
        r += w * qr
        g += w * qg
        b += w * qb
    return (r, g, b)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_rows(path: str) -> pd.DataFrame:
    """Load v3 emotional_raw.jsonl, explode probe vectors into per-
    probe columns (for probe-correlation analysis), attach quadrant.
    Hidden-state analysis uses ``load_emotional_features`` instead.

    Unpacks both core probe lists (``probe_scores_t0/tlast``,
    indexed by ``PROBES`` order) and extension probe dicts
    (``extension_probe_scores_t0/tlast``, dict-keyed by name) into
    ``t0_<probe>`` / ``tlast_<probe>`` columns. The
    ``extension_probe_names`` returned by
    :func:`available_extension_probes` lists which extension columns
    were found.
    """
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    # Core probes — list-indexed by PROBES order.
    for prefix, src in (("t0", "probe_scores_t0"), ("tlast", "probe_scores_tlast")):
        if src in df.columns:
            stacked = np.asarray(df[src].tolist(), dtype=float)
            for i, probe in enumerate(PROBES):
                df[f"{prefix}_{probe}"] = stacked[:, i]
            df = df.drop(columns=[src])
    # Extension probes — dict-keyed; column union over all rows.
    for prefix, src in (("t0", "extension_probe_scores_t0"),
                         ("tlast", "extension_probe_scores_tlast")):
        if src not in df.columns:
            continue
        keys: set[str] = set()
        for d in df[src]:
            if isinstance(d, dict):
                keys.update(d.keys())
        for k in sorted(keys):
            df[f"{prefix}_{k}"] = [
                (d.get(k) if isinstance(d, dict) else float("nan"))
                for d in df[src]
            ]
        df = df.drop(columns=[src])
    df["quadrant"] = df["prompt_id"].str[:2].str.upper()
    return df


def available_extension_probes(df: pd.DataFrame) -> list[str]:
    """Names of extension probes whose ``t0_<name>`` (or ``tlast_<name>``)
    columns were unpacked by :func:`load_rows`. Useful for callers that
    want to iterate the extension subset without hardcoding it."""
    from .config import PROBES
    core = set(PROBES)
    ext: set[str] = set()
    for col in df.columns:
        if not isinstance(col, str):
            continue
        for prefix in ("t0_", "tlast_"):
            if col.startswith(prefix):
                name = col[len(prefix):]
                if name not in core:
                    ext.add(name)
                break
    return sorted(ext)


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
    which: str = "h_first",
    layer: int | None = None,
    split_hn: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load v3 JSONL + its hidden-state sidecars. Filters to kaomoji-
    bearing rows by first_word prefix. Attaches a ``quadrant`` column
    (HP/LP/HN/LN/NB) derived from prompt_id.

    With ``split_hn=True``, also overwrites the ``quadrant`` column
    using the rule-3-redesign HN split (HN→HN-D/HN-S, with the 3
    untagged borderline prompts dropped from the result entirely).
    Use this for figures that should show the dominance split as
    a first-class category.

    The ``$LLMOJI_WHICH`` environment variable (if set to one of
    ``h_first`` / ``h_last`` / ``h_mean``) overrides the ``which``
    keyword argument. Used by 2026-05-02's project-wide h_first
    sweep — set it once, run any script, and every hidden-state
    aggregation respects it without per-script edits.

    Returns (metadata df, (n_rows, hidden_dim) feature matrix) aligned
    row-wise. Downstream plot functions take this pair directly.
    """
    import os
    env_which = os.environ.get("LLMOJI_WHICH")
    if env_which:
        if env_which not in ("h_first", "h_last", "h_mean"):
            raise ValueError(
                f"LLMOJI_WHICH must be h_first|h_last|h_mean, got {env_which!r}"
            )
        which = env_which
    from .hidden_state_analysis import load_hidden_features
    df, X = load_hidden_features(
        jsonl_path, data_dir, experiment,
        which=which, layer=layer,
    )
    if len(df) == 0:
        return df, X
    from llmoji.taxonomy import canonicalize_kaomoji
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
    df = df.loc[mask].reset_index(drop=True)
    X = X[mask]
    if split_hn:
        hn_split = _hn_split_map()
        new_q = df.apply(
            lambda r: (
                hn_split.get(r["prompt_id"], None)
                if r["quadrant"] == "HN"
                else r["quadrant"]
            ),
            axis=1,
        )
        keep = new_q.notna().to_numpy()
        df = df.loc[keep].copy()
        df["quadrant"] = new_q[keep].to_numpy()
        X = X[keep]
        df = df.reset_index(drop=True)
    return df, X


def load_v1v2_neutral_baseline_features(
    jsonl_path: str | Path,
    data_dir: Path,
    *,
    experiment: str = "v1v2",
    which: str = "h_first",
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
    from llmoji.taxonomy import canonicalize_kaomoji
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
    the kaomoji character set — see CLAUDE.md font-fallback gotcha.

    Also registers the project-local `data/fonts/NotoEmoji-Regular.ttf`
    (monochrome) so SMP emoji glyphs that appear inside some
    Qwen-emitted forms — e.g. `(🌫️🐕✨)`, `(🥺💧)` — render instead of
    showing as missing-glyph rectangles. macOS ships only color-emoji
    TTC which matplotlib's text engine cannot rasterize.
    """
    import matplotlib
    import matplotlib.font_manager as fm
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    emoji_font = repo_root / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    if emoji_font.exists() and "Noto Emoji" not in {f.name for f in fm.fontManager.ttflist}:
        try:
            fm.fontManager.addfont(str(emoji_font))
        except Exception:
            pass
    chain = [
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
        "DejaVu Serif",
        "Tahoma",
        "Noto Sans Canadian Aboriginal",
        "Heiti TC",
        "Hiragino Sans",
        "Apple Symbols",
        "Noto Emoji",
        "Helvetica Neue",  # covers stray punctuation like U+2E1D ⸝
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

    order_qs, palette = _palette_for(df)
    quadrant_for = per_face_dominant_quadrant(df)
    label_quadrants = [quadrant_for.get(str(k), "NB") for k in labels]
    row_colors = [palette.get(q, "#666") for q in label_quadrants]

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
        Patch(color=palette[q], label=q) for q in order_qs if q in palette
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

    # Per-face dot color = dominant Russell-quadrant of that face,
    # which generalizes across models (TAXONOMY-based pole color was
    # gemma-vocab-tied and grayed out everything else).
    order_qs, palette = _palette_for(df)
    quadrant_for = per_face_dominant_quadrant(df)

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
        color = palette.get(quadrant_for.get(km, "NB"), "#666")
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
        tick.set_color(palette.get(quadrant_for.get(km, "NB"), "#666"))
    ax.set_xlabel("cosine(row, kaomoji mean) — hidden-state space")
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_title(
        f"Figure B: within-kaomoji HIDDEN-STATE consistency vs shuffled null\n"
        f"(n ≥ {min_count}; null = {null_iters} random same-size subsets)"
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=palette[q], label=q) for q in order_qs if q in palette
    ] + [Patch(color="#cccccc", label="null band (IQR)")]
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
    against the same hidden-state pool mean.

    Y-axis row labels are tinted by per-face mixed quadrant color
    (RGB blend of `QUADRANT_COLORS` weighted by per-quadrant
    emission count). Cross-quadrant emitters render as visible
    mixes; pure-quadrant faces stay at their endpoint color."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from .hidden_state_analysis import cosine_similarity_matrix, group_mean_vectors

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

    # Row tint: per-face mixed quadrant color (RGB-linear blend of
    # QUADRANT_COLORS weighted by emission distribution). More
    # informative than the old TAXONOMY-pole 3-state coloring —
    # cross-quadrant emitters render visibly mixed, e.g. a face
    # that's 21 LN + 20 HN reads as purple instead of green/orange.
    weights = per_face_quadrant_weights(df)
    row_colors = [
        mix_quadrant_color(weights.get(k, {q: 0.0 for q in QUADRANT_ORDER}))
        for k in kms_ordered
    ]

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

    # Use the global QUADRANT_COLORS (canonical Russell palette).

    fig, ax = plt.subplots(figsize=(11, 9))

    for (km, q, n), pt in zip(groups, coords):
        c = QUADRANT_COLORS.get(q, "#666")
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
        c = QUADRANT_COLORS.get(q_name, "#666")
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

    order_qs, palette = _palette_for(df_all)
    label_text = {
        "HP": "HP (high arousal, positive)",
        "LP": "LP (low arousal, positive)",
        "HN": "HN (high arousal, negative)",
        "LN": "LN (low arousal, negative)",
        "NB": "NB (neutral baseline)",
        "HN-D": "HN-D (anger / contempt)",
        "HN-S": "HN-S (fear / anxiety)",
    }
    legend_handles = [
        Patch(color=palette[k], label=label_text.get(k, k))
        for k in order_qs if k in palette
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
    probes: list[str] | None = None,
) -> dict[str, Any]:
    """Full pairwise Pearson + Spearman correlation between probe
    scores at the given timestep. Returns overall + per-quadrant.
    Inputs: df must have ``t0_<probe>`` / ``tlast_<probe>`` columns
    from ``load_rows`` (which unpacks core + extension into the same
    naming scheme). Pass ``probes=`` to override the default probe
    set; default is core PROBES + every extension probe whose column
    is present on df. Run on v3 unsteered data only — steering would
    shift probes and confound the collapse reading."""
    from scipy.stats import pearsonr, spearmanr
    from .config import PROBES

    use_probes = probes
    if use_probes is None:
        use_probes = list(PROBES) + available_extension_probes(df)
    cols = [f"{timestep}_{p}" for p in use_probes]
    cols = [c for c in cols if c in df.columns]
    use_probes = [c[len(timestep) + 1:] for c in cols]   # keep alignment
    n_p = len(use_probes)
    out: dict[str, Any] = {"probes": use_probes, "by_subset": {}}

    def pair_stats(sub: pd.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n < 3:
            return {"n": n, "pearson": None, "spearman": None}
        vals = sub[cols].to_numpy(dtype=float)
        p = np.full((n_p, n_p), np.nan)
        s = np.full((n_p, n_p), np.nan)
        for i in range(n_p):
            xi = vals[:, i]; mi = ~np.isnan(xi)
            for j in range(n_p):
                if i == j:
                    p[i, j] = 1.0
                    s[i, j] = 1.0
                    continue
                xj = vals[:, j]
                m = mi & ~np.isnan(xj)
                if m.sum() < 3:
                    continue
                p[i, j] = float(pearsonr(xi[m], xj[m])[0])
                s[i, j] = float(spearmanr(xi[m], xj[m])[0])
        return {"n": int(n), "pearson": p.tolist(), "spearman": s.tolist()}

    out["by_subset"]["all"] = pair_stats(df)
    for q in ("HP", "LP", "HN", "LN", "NB"):
        out["by_subset"][q] = pair_stats(df[df["quadrant"] == q])
    return out


def plot_probe_correlation_matrix(
    df: pd.DataFrame, out_path: str, *,
    method: str = "pearson", timestep: str = "t0",
    probes: list[str] | None = None,
) -> None:
    """Multi-panel: overall probe-correlation matrix + one per quadrant.

    Default probe set is core PROBES + every extension probe present
    on df (via :func:`available_extension_probes`). Pass ``probes=``
    to subset, e.g. ``["happy.sad", "fearful.unflinching",
    "angry.calm"]`` for the affect trio."""
    import matplotlib.pyplot as plt
    from .config import PROBES

    _use_cjk_font()
    if probes is None:
        probes = list(PROBES) + available_extension_probes(df)
    stats = compute_probe_correlations(df, timestep=timestep, probes=probes)
    probes_used = stats["probes"]
    n_p = len(probes_used)
    panels = [
        ("all", "all v3 rows"), ("HP", "HP"), ("LP", "LP"),
        ("HN", "HN"), ("LN", "LN"), ("NB", "NB"),
    ]

    panel_w = max(3.5, 0.32 * n_p + 1.5)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(panel_w * len(panels), panel_w + 0.6))
    im = None
    core_set = set(PROBES)
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
        ax.set_xticks(range(n_p))
        ax.set_yticks(range(n_p))
        ax.set_xticklabels(probes_used, rotation=55, ha="right", fontsize=6.5)
        ax.set_yticklabels(probes_used, fontsize=6.5)
        # Mark core/extension boundary if both are present.
        ext_present = any(p not in core_set for p in probes_used)
        if ext_present:
            sep = sum(1 for p in probes_used if p in core_set) - 0.5
            ax.axhline(sep, color="black", linewidth=0.5)
            ax.axvline(sep, color="black", linewidth=0.5)
        for i in range(n_p):
            for j in range(n_p):
                v = arr[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.5 else "#333")
        ax.set_title(f"{title}  n={sub['n']}")
    fig.suptitle(
        f"v3 probe-probe {method} correlations  ({n_p} probes; "
        f"core + extension separated by black line)", fontsize=11,
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

    Columns: first_word, n, median_within_consistency, dominant_quadrant,
    HP_n, LP_n, HN_n, LN_n, NB_n.
    """
    from .hidden_state_analysis import cosine_to_mean

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "first_word", "n", "median_within_consistency",
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
