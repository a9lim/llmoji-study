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

from llmoji.config import (
    EMOTIONAL_DATA_PATH,
    FIGURES_DIR,
    PILOT_RAW_PATH,
    PROBES,
)
from llmoji.emotional_analysis import (
    load_rows,
    load_v1v2_neutral_baseline,
    plot_v3_pca_valence_arousal,
)


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} v3 rows")

    baseline = None
    if PILOT_RAW_PATH.exists():
        baseline = load_v1v2_neutral_baseline(str(PILOT_RAW_PATH))
        print(f"loaded {len(baseline)} v1/v2 kaomoji_prompted neutral-valence rows")
    else:
        print(f"(no pilot_raw at {PILOT_RAW_PATH}; skipping neutral baseline)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "fig_v3_pca_valence_arousal.png"
    stats = plot_v3_pca_valence_arousal(df, str(fig_path), baseline_df=baseline)
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
    within = stats.get("within_quadrant_std_pc12") or {}
    between_pc1 = stats.get("between_centroid_std_pc1", 0.0)
    between_pc2 = stats.get("between_centroid_std_pc2", 0.0)

    if centroids:
        print("\nper-quadrant centroid (PC1, PC2)  |  within-quadrant std (PC1, PC2):")
        for q in ("HP", "LP", "HN", "LN", "NB"):
            if q in centroids:
                pc1, pc2 = centroids[q]
                s1, s2 = within.get(q, [0.0, 0.0])
                print(f"  {q}:  ({pc1:+.3f}, {pc2:+.3f})   "
                      f"|  ({s1:.3f}, {s2:.3f})")

        # Separation quality: between-centroid spread divided by mean
        # within-quadrant spread. >1 = between-centroid is bigger than
        # the typical within-quadrant scatter = clean separation.
        mean_within_pc1 = sum(v[0] for v in within.values()) / max(1, len(within))
        mean_within_pc2 = sum(v[1] for v in within.values()) / max(1, len(within))
        print(f"\nseparation ratio (between-centroid std / mean within-quadrant std):")
        if mean_within_pc1 > 0:
            print(f"  PC1: {between_pc1 / mean_within_pc1:.2f}  "
                  f"(between {between_pc1:.3f}, mean within {mean_within_pc1:.3f})")
        if mean_within_pc2 > 0:
            print(f"  PC2: {between_pc2 / mean_within_pc2:.2f}  "
                  f"(between {between_pc2:.3f}, mean within {mean_within_pc2:.3f})")


if __name__ == "__main__":
    main()
