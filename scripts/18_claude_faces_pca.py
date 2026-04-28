"""Eriskii-style PCA chart of Claude's kaomoji vocabulary.

Counterpart to ``scripts/16_eriskii_replication.py`` (axis projection +
t-SNE). Reads the description-based per-canonical-kaomoji embeddings
from script 15, fits PCA on those embeddings, and plots a 2D scatter
with HDBSCAN clusters and frequency-sized markers.

Two panels:
  1. PC1 vs PC2 with HDBSCAN clusters (auto-k) — main figure.
  2. PC1 vs PC2 with KMeans(k=15) — parity with eriskii's published
     panel.

Output: ``figures/harness/claude_faces_pca.png``

Pre-2026-04-27 this script also pulled per-canonical raw emission
counts from ``data/claude_kaomoji.jsonl``. Post-refactor that file is
gone — counts come from the parquet's ``n`` column, which is already
``count_total`` from the HF corpus pull.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from llmoji_study.claude_faces import load_embeddings
from llmoji_study.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    FIGURES_DIR,
)


def _use_cjk_font() -> None:
    """Synced with llmoji_study/analysis.py, llmoji_study/emotional_analysis.py,
    scripts/16_eriskii_replication.py — keep these chains in sync."""
    import matplotlib.font_manager as fm
    repo_root = Path(__file__).resolve().parent.parent
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
    """Single PCA panel: dots sized by log frequency, colored by
    cluster label (-1 = noise = light gray), top-N annotated."""
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

    print("loading description embeddings...")
    fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} canonical kaomoji")
    print(f"  total emissions across canonical kaomoji: {int(n.sum())}")

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
    harness_dir = FIGURES_DIR / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    _scatter_panel(
        axes[0], coords, fw, n, hdb_labels,
        title=(
            f"Claude faces — description-PCA, HDBSCAN\n"
            f"{len(fw)} canonical kaomoji, {n_hdb} clusters + "
            f"{n_noise} noise; PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%"
        ),
    )
    _scatter_panel(
        axes[1], coords, fw, n, km_labels,
        title=(
            f"Claude faces — description-PCA, KMeans (k=15)\n"
            f"parity with eriskii's published panel"
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
