"""Stacked-bar chart for per-project Russell-quadrant histograms.

Used by ``scripts/66_per_project_quadrants.py`` for all
three resolution modes (``gt-priority`` / ``ensemble`` / ``gt-only``).
One row per project sorted by total emissions descending, plus a
``(global)`` row at the top. Bar segments share-of-known per quadrant;
Russell-circumplex palette from ``emotional_analysis.QUADRANT_COLORS``.
Renders kaomoji-bearing project names safely via the same CJK font
fallback chain the rest of the project's matplotlib output uses.
"""

from __future__ import annotations

from pathlib import Path

from llmoji_study.emotional_analysis import QUADRANT_COLORS

QUADRANTS_SPLIT = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _use_cjk_font() -> None:
    """Same fallback chain as ``scripts/harness/63_corpus_pca.py`` /
    ``llmoji_study/emotional_analysis.py`` — keep these in sync."""
    import matplotlib
    import matplotlib.font_manager as fm

    repo_root = Path(__file__).resolve().parent.parent
    emoji_font = repo_root / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    if emoji_font.exists() and "Noto Emoji" not in {f.name for f in fm.fontManager.ttflist}:
        try:
            fm.fontManager.addfont(str(emoji_font))
        except Exception:
            pass
    chain = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif",
        "Tahoma", "Noto Sans Canadian Aboriginal", "Heiti TC",
        "Hiragino Sans", "Apple Symbols", "Noto Emoji", "Helvetica Neue",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


def plot_per_project_quadrants(
    per_proj: dict[str, dict[str, int]],
    per_proj_total: dict[str, int],
    global_counts: dict[str, int],
    *,
    title: str,
    out_path: Path,
    min_per_project: int = 5,
    subtitle: str | None = None,
) -> Path:
    """Write a horizontal stacked-bar chart of per-project quadrant shares.

    Parameters
    ----------
    per_proj : ``{project: {quadrant: count}}`` of resolved emissions only.
    per_proj_total : ``{project: total_emissions}`` (resolved + unknown).
    global_counts : ``{quadrant: count}`` aggregate across projects.
    title : figure suptitle.
    out_path : destination ``.png``.
    min_per_project : skip projects with fewer than N total emissions.
    subtitle : small caption rendered below the title.

    Returns the resolved ``out_path`` after saving.
    """
    _use_cjk_font()
    import matplotlib.pyplot as plt
    import numpy as np

    projects = [
        p for p in sorted(per_proj_total, key=lambda k: -per_proj_total[k])
        if per_proj_total[p] >= min_per_project
    ]

    rows: list[tuple[str, dict[str, int], int]] = [
        ("(global)", dict(global_counts), sum(per_proj_total.values()))
    ]
    for p in projects:
        rows.append((p, per_proj[p], per_proj_total[p]))

    n_rows = len(rows)
    fig_h = max(2.4, 0.32 * n_rows + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    y_pos = np.arange(n_rows)[::-1]  # global on top
    labels = [r[0] for r in rows]
    n_totals = [r[2] for r in rows]

    quadrants = QUADRANTS_SPLIT
    shares = np.zeros((n_rows, len(quadrants)))
    for i, (_proj, counts, _n_total) in enumerate(rows):
        n_known = sum(counts.get(q, 0) for q in quadrants)
        if n_known == 0:
            continue
        for j, q in enumerate(quadrants):
            shares[i, j] = counts.get(q, 0) / n_known

    left = np.zeros(n_rows)
    for j, q in enumerate(quadrants):
        widths = shares[:, j]
        ax.barh(
            y_pos, widths, left=left, height=0.78,
            color=QUADRANT_COLORS[q], label=q,
            edgecolor="white", linewidth=0.4,
        )
        # Inline labels for segments ≥ 8% (else the text overlaps).
        for i, w in enumerate(widths):
            if w >= 0.08:
                ax.text(
                    left[i] + w / 2, y_pos[i],
                    f"{w*100:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)],
                        fontsize=8)
    ax.set_xlabel("share of resolved emissions", fontsize=9)

    # n-emissions on the right edge.
    for i, n in enumerate(n_totals):
        ax.text(
            1.012, y_pos[i], f"n={n}",
            ha="left", va="center",
            fontsize=8, color="#444",
        )

    ax.set_title(title, fontsize=12, pad=10)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, color="#555",
        )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.6 / fig_h),
        ncol=len(quadrants), frameon=False, fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
