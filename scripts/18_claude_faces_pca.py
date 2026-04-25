"""Eriskii-style PCA chart of Claude's kaomoji vocabulary.

Counterpart to scripts/09_claude_faces_plot.py (t-SNE) and
scripts/16_eriskii_replication.py (axis projections). Reads the
description-based per-kaomoji embeddings (claude_faces_embed_description
.parquet — eriskii's pipeline: each kaomoji embedded by its haiku-
synthesized meaning), canonicalizes near-duplicate forms via
``llmoji.taxonomy.canonicalize_kaomoji`` (NFKC + arm-modifier strip),
fits PCA on the merged per-canonical embeddings, and plots a 2D
scatter with HDBSCAN clusters and frequency-sized markers.

Two panels:
  1. PC1 vs PC2 with HDBSCAN clusters (auto-k) — main figure.
  2. PC1 vs PC2 with KMeans(k=15) — parity with eriskii's published
     panel.

Output: figures/claude_faces_pca.png
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from llmoji.claude_faces import load_embeddings_canonical
from llmoji.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    CLAUDE_KAOMOJI_PATH,
    FIGURES_DIR,
)
from llmoji.taxonomy import canonicalize_kaomoji


def _use_cjk_font() -> None:
    """Synced with llmoji/analysis.py, llmoji/emotional_analysis.py,
    scripts/09_claude_faces_plot.py — keep these four chains in sync."""
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans",
        "DejaVu Serif", "Tahoma", "Noto Sans Canadian Aboriginal",
        "Heiti TC",
    ]


def emission_frequencies(claude_kaomoji_path: Path) -> Counter[str]:
    """Per-canonical-kaomoji emission count from the raw scrape."""
    counter: Counter[str] = Counter()
    rows = pd.read_json(claude_kaomoji_path, lines=True)
    for fw in rows["first_word"]:
        if isinstance(fw, str) and fw:
            counter[canonicalize_kaomoji(fw)] += 1
    return counter


def _hdbscan_labels(E: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    try:
        import hdbscan  # type: ignore
    except ImportError:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(E)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(E)


def _scatter_panel(
    ax,
    coords: np.ndarray,
    fw: list[str],
    n: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    palette: list[str] | None = None,
    annotate_top: int = 30,
) -> None:
    """Draw a single PCA panel: dots sized by log frequency, coloured
    by cluster label (-1 = noise = light gray), top-N labelled."""
    unique = sorted(set(labels.tolist()))
    if palette is None:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i / max(1, len(unique) - 1)) for i in range(len(unique))]
    color_for: dict[int, object] = {}
    pal_iter = iter(palette)
    for u in unique:
        if u == -1:
            color_for[u] = "#bbbbbb"
        else:
            color_for[u] = next(pal_iter)

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
    ax.axhline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#ddd", linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=11)


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(
            f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
            "run scripts/15 first"
        )
        sys.exit(1)
    _use_cjk_font()

    print("loading description embeddings (canonicalized)...")
    fw, n_desc, E = load_embeddings_canonical(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} canonical kaomoji")

    # n in claude_faces_embed_description.parquet is "n_descriptions" — how
    # many per-instance haiku passes contributed to the synthesis. For
    # marker sizing we want emission frequency from the raw scrape.
    print("loading raw emission frequencies...")
    freq = emission_frequencies(CLAUDE_KAOMOJI_PATH)
    n_emit = np.array([freq.get(f, 0) for f in fw], dtype=int)
    if (n_emit == 0).any():
        # Fall back to n_descriptions if some canonical forms aren't in
        # the raw scrape (shouldn't happen, but defensive).
        n_emit = np.where(n_emit > 0, n_emit, n_desc)
    print(f"  total emissions across {len(fw)} canonical kaomoji: {int(n_emit.sum())}")

    print("fitting PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(E)
    var = pca.explained_variance_ratio_
    print(
        f"  PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}% "
        f"(top-2 cumulative {sum(var)*100:.1f}%)"
    )

    print("clustering...")
    hdb_labels = _hdbscan_labels(E, min_cluster_size=5)
    n_hdb = len(set(hdb_labels.tolist()) - {-1})
    n_noise = int((hdb_labels == -1).sum())
    print(f"  HDBSCAN: {n_hdb} clusters, {n_noise} noise points")

    km = KMeans(n_clusters=15, n_init=10, random_state=0)
    km_labels = km.fit_predict(E)
    print("  KMeans: k=15 fit complete")

    print("plotting...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    _scatter_panel(
        axes[0], coords, fw, n_emit, hdb_labels,
        title=(
            f"Claude faces — description-PCA, HDBSCAN\n"
            f"{len(fw)} canonical kaomoji, {n_hdb} clusters + "
            f"{n_noise} noise; PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%"
        ),
    )
    _scatter_panel(
        axes[1], coords, fw, n_emit, km_labels,
        title=(
            f"Claude faces — description-PCA, KMeans (k=15)\n"
            f"parity with eriskii's published panel"
        ),
    )
    for ax in axes:
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")

    fig.tight_layout()
    out = FIGURES_DIR / "claude_faces_pca.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
