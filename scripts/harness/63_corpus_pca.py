"""Bag-of-lexicon (BoL) PCA chart of Claude's kaomoji vocabulary.

Reads the 48-d per-canonical-kaomoji BoL parquet from script 62
(structured commit over the locked llmoji v2 LEXICON), fits PCA on
those 48 dims, and plots a 2D scatter.

Two panels:
  1. PC1 vs PC2 colored by the **inferred Russell quadrant** from
     the BoL's circumplex slots
     (:func:`llmoji_study.lexicon.bol_modal_quadrant`). This is the
     synthesizer's structured commit on what each face means — no
     encoder, no projection.
  2. PC1 vs PC2 with KMeans(k=15) labeled by the **modal lexicon
     word** of each cluster (deterministic, no Haiku call).

Pre-2026-05-06 this script ran on 384-d MiniLM-encoded prose
descriptions and labeled clusters via t-SNE + HDBSCAN. The new
representation is 8× smaller, fully interpretable, and uses the
synthesizer's structured commit directly.

Output: ``figures/harness/claude_faces_pca.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from llmoji_study.claude_faces import load_bol_parquet
from llmoji_study.config import (
    CLAUDE_FACES_LEXICON_BAG_PATH,
    FIGURES_DIR,
)
from llmoji_study.emotional_analysis import QUADRANT_COLORS
from llmoji_study.lexicon import (
    LEXICON_WORDS,
    QUADRANTS,
    bol_modal_quadrant,
    top_lexicon_words,
)


def _use_cjk_font() -> None:
    """Synced with llmoji_study/per_project_charts.py /
    llmoji_study/emotional_analysis.py — keep these chains in sync."""
    import matplotlib.font_manager as fm
    repo_root = Path(__file__).resolve().parent.parent.parent
    emoji_font = repo_root / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    if emoji_font.exists() and "Noto Emoji" not in {f.name for f in fm.fontManager.ttflist}:
        try:
            fm.fontManager.addfont(str(emoji_font))
        except Exception:
            pass
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans",
        "DejaVu Serif", "Tahoma", "Noto Sans Canadian Aboriginal",
        "Heiti TC", "Noto Emoji", "Helvetica Neue",
    ]


# Color for "no circumplex commitment" faces — extension-only picks,
# rare but possible.
_NO_QUADRANT_COLOR = "#cccccc"


def _scatter_quadrant_panel(
    ax,
    coords: np.ndarray,
    fw: list[str],
    n: np.ndarray,
    quadrants: list[str | None],
    title: str,
    *,
    annotate_top: int = 30,
) -> None:
    """Quadrant-colored panel: dots sized by log frequency, colored by
    BoL-inferred modal Russell quadrant."""
    sizes = np.clip(20 + 80 * np.log1p(n), 20, 600)
    colors = [
        QUADRANT_COLORS.get(q, _NO_QUADRANT_COLOR) if q else _NO_QUADRANT_COLOR
        for q in quadrants
    ]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=sizes, alpha=0.78,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    top_idx = np.argsort(-n)[:annotate_top]
    for i in top_idx:
        ax.annotate(
            fw[i], xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=8, color="#222", zorder=5,
        )

    # Legend
    seen = []
    for q in QUADRANTS:
        if q in quadrants:
            seen.append(q)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=QUADRANT_COLORS[q],
               markeredgecolor="white", label=q)
        for q in seen
    ]
    if any(q is None for q in quadrants):
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=_NO_QUADRANT_COLOR,
                   markeredgecolor="white", label="no circumplex"),
        )
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)

    ax.axhline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=11)


def _scatter_cluster_panel(
    ax,
    coords: np.ndarray,
    fw: list[str],
    n: np.ndarray,
    B: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    annotate_top: int = 30,
) -> None:
    """KMeans-colored panel with deterministic modal-lexicon-word
    cluster labels."""
    unique = sorted(set(int(l) for l in labels))
    cmap = plt.get_cmap("tab20")
    color_for = {u: cmap(i / max(1, len(unique) - 1)) for i, u in enumerate(unique)}

    sizes = np.clip(20 + 80 * np.log1p(n), 20, 600)
    cs = [color_for[int(l)] for l in labels]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cs, s=sizes, alpha=0.78,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    top_idx = np.argsort(-n)[:annotate_top]
    for i in top_idx:
        ax.annotate(
            fw[i], xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=8, color="#222", zorder=5,
        )

    # Cluster center labels: top-2 modal lexicon words per cluster.
    for c in unique:
        mask = labels == c
        if not mask.any():
            continue
        cluster_bol = B[mask].mean(axis=0)
        top = top_lexicon_words(cluster_bol, k=2)
        if not top:
            continue
        lbl = "/".join(w for w, _ in top)
        cx = float(coords[mask, 0].mean())
        cy = float(coords[mask, 1].mean())
        ax.text(
            cx, cy, lbl,
            fontsize=9, fontweight="bold", color="#111",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color_for[c], alpha=0.9),
            zorder=6,
        )

    ax.axhline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=11)


def main() -> None:
    if not CLAUDE_FACES_LEXICON_BAG_PATH.exists():
        print(
            f"no BoL parquet at {CLAUDE_FACES_LEXICON_BAG_PATH}; "
            "run scripts/harness/62_corpus_lexicon.py first"
        )
        sys.exit(1)
    _use_cjk_font()

    print("loading BoL vectors...")
    fw, n, _n_v2, B = load_bol_parquet(CLAUDE_FACES_LEXICON_BAG_PATH)
    assert B.shape[1] == len(LEXICON_WORDS)
    print(f"  {len(fw)} canonical kaomoji, dim={B.shape[1]} (lexicon)")
    print(f"  total emissions across canonical kaomoji: {int(n.sum())}")

    quadrants: list[str | None] = [bol_modal_quadrant(B[i]) for i in range(len(fw))]
    n_no_q = sum(1 for q in quadrants if q is None)
    if n_no_q:
        print(
            f"  {n_no_q} faces have no circumplex commitment "
            "(extension-only picks)"
        )

    print("fitting PCA(2) on 48-d BoL...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(B)
    var = pca.explained_variance_ratio_
    print(
        f"  PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}% "
        f"(top-2 cumulative {sum(var)*100:.1f}%)"
    )
    # Loadings: which lexicon words drive each PC?
    for j, label in enumerate(["PC1", "PC2"]):
        order = np.argsort(-np.abs(pca.components_[j]))
        top = [(LEXICON_WORDS[i], float(pca.components_[j][i])) for i in order[:6]]
        terse = ", ".join(f"{w}{'+' if v > 0 else ''}{v:+.2f}" for w, v in top)
        print(f"  {label} top loadings: {terse}")

    print("clustering KMeans(k=15) on BoL...")
    km = KMeans(n_clusters=15, n_init=10, random_state=0)
    km_labels = km.fit_predict(B)

    print("plotting...")
    harness_dir = FIGURES_DIR / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    _scatter_quadrant_panel(
        axes[0], coords, fw, n, quadrants,
        title=(
            f"Claude faces - BoL PCA, colored by inferred Russell quadrant\n"
            f"{len(fw)} canonical kaomoji; PC1 {var[0]*100:.1f}%, "
            f"PC2 {var[1]*100:.1f}%"
        ),
    )
    _scatter_cluster_panel(
        axes[1], coords, fw, n, B, km_labels,
        title=(
            f"Claude faces - BoL PCA, KMeans(k=15)\n"
            "labels are modal lexicon words per cluster (deterministic)"
        ),
    )
    for ax in axes:
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")

    fig.tight_layout()
    out = harness_dir / "claude_faces_pca.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
