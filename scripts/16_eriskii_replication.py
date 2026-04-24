"""Eriskii-replication step 3: analysis + figures.

Sections in build order:
  - axis projection: data/eriskii_axes.tsv +
    figures/eriskii_axis_<name>.png × 21
  - clusters: data/eriskii_clusters.tsv +
    figures/eriskii_clusters_tsne.png
  - per-model: data/eriskii_per_model.tsv +
    figures/eriskii_per_model_axes_{mean,std}.png
  - per-project: data/eriskii_per_project.tsv +
    figures/eriskii_per_project_axes_{mean,std}.png
  - mechanistic bridge: data/eriskii_user_kaomoji_axis_corr.tsv +
    figures/eriskii_user_kaomoji_axis_corr.png
  - narrative writeup: data/eriskii_comparison.md

Usage:
  python scripts/16_eriskii_replication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji.claude_faces import EMBED_MODEL, load_embeddings
from llmoji.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    DATA_DIR,
    ERISKII_AXES,
    ERISKII_AXES_TSV,
    FIGURES_DIR,
)
from llmoji.eriskii import compute_axis_vectors, project_onto_axes
from llmoji.eriskii_prompts import AXIS_ANCHORS


def _use_cjk_font() -> None:
    """Same fallback chain used in analysis.py / emotional_analysis.py /
    09_claude_faces_plot.py — copy here for consistency."""
    import matplotlib
    import matplotlib.font_manager as fm
    chain = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif",
        "Tahoma", "Noto Sans Canadian Aboriginal", "Heiti TC",
        "Hiragino Sans", "Apple Symbols",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


def section_axes(
    fw: list[str],
    n: np.ndarray,
    P: np.ndarray,
) -> pd.DataFrame:
    """Write eriskii_axes.tsv + one ranked-bar figure per axis."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"first_word": fw, "n": n})
    for j, name in enumerate(ERISKII_AXES):
        df[name] = P[:, j]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ERISKII_AXES_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_AXES_TSV}  ({len(df)} kaomoji × {len(ERISKII_AXES)} axes)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(ERISKII_AXES):
        scores = P[:, j]
        order = np.argsort(-scores)
        top = order[:15]
        bot = order[-15:][::-1]
        idxs = list(top) + list(bot)
        labels = [fw[i] for i in idxs]
        vals = [scores[i] for i in idxs]
        counts = [n[i] for i in idxs]

        fig, ax = plt.subplots(figsize=(6, 8))
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        y = np.arange(len(idxs))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.axhline(14.5, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(f"{name} projection (cosine)")
        ax.set_title(f"top-15 / bottom-15 on {name}\n(bar color = emission count)")
        fig.tight_layout()
        out = FIGURES_DIR / f"eriskii_axis_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
    return df


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
              "run scripts/15 first")
        sys.exit(1)
    _use_cjk_font()

    print("loading description embeddings...")
    fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} kaomoji, {E.shape[1]}-dim")

    print("computing axis vectors...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    axes = compute_axis_vectors(embedder, AXIS_ANCHORS)

    print("projecting kaomoji onto axes...")
    P = project_onto_axes(E, axes, ERISKII_AXES)

    print("\n=== Section: axes ===")
    section_axes(fw, n, P)


if __name__ == "__main__":
    main()
