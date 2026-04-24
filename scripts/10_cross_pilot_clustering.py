"""Cross-pilot pooled clustering driver (hidden-state). Loads
pilot_raw.jsonl (v1/v2) + emotional_raw.jsonl (v3) metadata and their
per-row hidden-state sidecars, pools per-(kaomoji, source) mean
hidden vectors, writes figures + per-row summary TSV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    FIGURES_DIR,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
)
from llmoji.cross_pilot_analysis import (
    load_pooled_features,
    plot_pooled_cosine_heatmap,
    plot_pooled_pca_scatter,
    pooled_summary_table,
)


def main() -> None:
    if not PILOT_RAW_PATH.exists():
        print(f"no data at {PILOT_RAW_PATH}; run scripts/01_pilot_run.py first")
        return
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    print("loading pooled hidden-state features (which=h_last, layer=max)...")
    df, X = load_pooled_features(
        str(PILOT_RAW_PATH), str(EMOTIONAL_DATA_PATH), DATA_DIR,
        v1_v2_experiment=PILOT_EXPERIMENT,
        v3_experiment=EMOTIONAL_EXPERIMENT,
        which="h_last",
    )
    print(f"pooled {len(df)} kaomoji-bearing rows; X shape {X.shape}")
    if len(df) == 0:
        print("nothing to plot; re-runs need to land hidden-state sidecars first")
        return

    print("rows per source:")
    print(df["source"].value_counts().to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = FIGURES_DIR / "fig_pool_cosine.png"
    plot_pooled_cosine_heatmap(df, X, str(fig_path), center=True)
    print(f"\nwrote {fig_path}  (grand-mean centered)")

    fig_uncentered = FIGURES_DIR / "fig_pool_cosine_uncentered.png"
    plot_pooled_cosine_heatmap(df, X, str(fig_uncentered), center=False)
    print(f"wrote {fig_uncentered}  (uncentered; comparison)")

    fig_pca = FIGURES_DIR / "fig_pool_pca.png"
    pca_stats = plot_pooled_pca_scatter(df, X, str(fig_pca))
    print(f"wrote {fig_pca}")
    if pca_stats:
        print("\nPCA explained-variance spectrum (on per-(kaomoji, source) means):")
        for i, v in enumerate(pca_stats.get("explained_variance_ratio", []), 1):
            print(f"  PC{i}: {v * 100:6.2f}%")

    summary = pooled_summary_table(df, X, min_count=3)
    summary_path = DATA_DIR / "pool_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nwrote {summary_path} ({len(summary)} rows)")
    if len(summary):
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
