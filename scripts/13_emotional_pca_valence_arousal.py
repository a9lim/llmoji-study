"""v3-only PCA on row probe vectors, per-(kaomoji, quadrant) means
projected into PC1/PC2, plotted with valence + arousal colorings.

Complements scripts/10_cross_pilot_clustering.py — that one pools
v1/v2 + v3 and PC1 ends up dominated by steering shifts. This one
restricts to naturalistic-regime data so the PC axes have a chance
to reflect valence/arousal structure directly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import EMOTIONAL_DATA_PATH, FIGURES_DIR, PROBES
from llmoji.emotional_analysis import (
    load_rows,
    plot_v3_pca_valence_arousal,
)


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} v3 rows")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "fig_v3_pca_valence_arousal.png"
    stats = plot_v3_pca_valence_arousal(df, str(fig_path))
    print(f"\nwrote {fig_path}")
    print(f"fit PCA on {stats.get('n_rows_fit')} rows; "
          f"plotted {stats.get('n_cells_plotted')} (kaomoji, quadrant) cells")

    print("\nPCA explained-variance spectrum:")
    for i, v in enumerate(stats.get("explained_variance_ratio", []), 1):
        print(f"  PC{i}: {v * 100:6.2f}%")

    components = stats.get("components") or []
    if len(components) >= 2:
        print("\nPC1 loadings (probe -> weight on PC1):")
        for name, load in zip(PROBES, components[0]):
            print(f"  {name:>22}  {load:+.3f}")
        print("\nPC2 loadings (probe -> weight on PC2):")
        for name, load in zip(PROBES, components[1]):
            print(f"  {name:>22}  {load:+.3f}")

    centroids = stats.get("quadrant_centroids_pc12") or {}
    if centroids:
        print("\nper-quadrant centroid (PC1, PC2):")
        for q in ("HP", "LP", "HN", "LN"):
            if q in centroids:
                pc1, pc2 = centroids[q]
                print(f"  {q}:  ({pc1:+.3f}, {pc2:+.3f})")


if __name__ == "__main__":
    main()
