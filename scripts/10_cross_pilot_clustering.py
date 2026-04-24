"""Cross-pilot pooled clustering driver. Reads pilot_raw.jsonl (v1/v2)
and emotional_raw.jsonl (v3), pools per-(kaomoji, source) aggregate
probe vectors, writes figures/fig_pool_cosine.png and
data/pool_summary.tsv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    FIGURES_DIR,
    PILOT_RAW_PATH,
)
from llmoji.cross_pilot_analysis import (
    grouped_kaomoji_source_means,
    load_pooled_rows,
    plot_pooled_cosine_heatmap,
    pooled_summary_table,
)


def main() -> None:
    if not PILOT_RAW_PATH.exists():
        print(f"no data at {PILOT_RAW_PATH}; run scripts/01_pilot_run.py first")
        return
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_pooled_rows(str(PILOT_RAW_PATH), str(EMOTIONAL_DATA_PATH))
    print(f"pooled {len(df)} kaomoji-bearing rows across v1/v2 + v3")
    print("rows per source:")
    print(df["source"].value_counts().to_string())

    grouped, _ = grouped_kaomoji_source_means(df, min_count=3)
    print(f"\n{len(grouped)} (kaomoji, source) tuples survive n≥3 filter")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = FIGURES_DIR / "fig_pool_cosine.png"
    plot_pooled_cosine_heatmap(df, str(fig_path))
    print(f"\nwrote {fig_path}")

    summary = pooled_summary_table(df, min_count=3)
    summary_path = DATA_DIR / "pool_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"wrote {summary_path} ({len(summary)} rows)")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
