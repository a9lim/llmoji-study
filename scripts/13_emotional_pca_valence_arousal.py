"""v3 PCA on hidden-state features, Russell-quadrant-colored scatter
with optional v1/v2 neutral-valence baseline.

Loads v3 per-row hidden-state sidecars from data/hidden/v3/ and the
v1/v2 neutral-valence rows from data/hidden/v1v2/ (if present), then
fits PCA on the combined row-level hidden states and projects
per-(kaomoji, quadrant) means through."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    DATA_DIR,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
    current_model,
)
from llmoji.emotional_analysis import (
    load_emotional_features,
    load_v1v2_neutral_baseline_features,
    plot_v3_pca_valence_arousal,
)


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} python scripts/03_emotional_run.py first")
        return

    print(f"model: {M.short_name}")
    print("loading v3 hidden-state features...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
    )
    print(f"loaded {len(df)} v3 kaomoji-bearing rows; X shape {X.shape}")
    if len(df) == 0:
        print("nothing to plot; the v3 run needs to land hidden-state sidecars first")
        return

    # v1/v2 baseline overlay only applies to the gemma run (PILOT_RAW_PATH
    # is gemma-only — no Qwen/Ministral steering data exists). Quietly skip
    # the overlay for non-gemma models even if a stray PILOT_RAW_PATH file
    # exists.
    baseline_df = baseline_X = None
    if M.short_name == "gemma" and PILOT_RAW_PATH.exists():
        baseline_df, baseline_X = load_v1v2_neutral_baseline_features(
            str(PILOT_RAW_PATH), DATA_DIR,
            experiment=PILOT_EXPERIMENT, which="h_mean",
        )
        print(f"loaded {len(baseline_df)} v1/v2 neutral-valence baseline rows")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = M.figures_dir / "fig_v3_pca_valence_arousal.png"
    stats = plot_v3_pca_valence_arousal(
        df, X, str(fig_path),
        baseline_df=baseline_df, baseline_X=baseline_X,
    )
    print(f"\nwrote {fig_path}")
    print(f"fit PCA on {stats.get('n_rows_fit')} rows; "
          f"plotted {stats.get('n_cells_plotted')} (kaomoji, quadrant) cells")

    print("\nPCA explained-variance spectrum:")
    for i, v in enumerate(stats.get("explained_variance_ratio", []), 1):
        print(f"  PC{i}: {v * 100:6.2f}%")

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
