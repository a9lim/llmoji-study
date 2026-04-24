"""v3 prompt × kaomoji emission matrix. Row = prompt (80 of them,
grouped by Russell quadrant); column = top-K kaomoji; cell = emission
count out of 8 seeds. Surfaces within-quadrant variation that the
per-kaomoji summary averages over."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import DATA_DIR, EMOTIONAL_DATA_PATH, FIGURES_DIR
from llmoji.emotional_analysis import (
    load_rows,
    plot_prompt_kaomoji_matrix,
    prompt_kaomoji_matrix,
)


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} v3 rows")

    mat, meta = prompt_kaomoji_matrix(df, top_k=12)
    print(f"built {len(mat)} prompts × {len(mat.columns)} kaomoji matrix")
    print("\nper-quadrant total emissions in top-12 kaomoji:")
    for q in ("HP", "LP", "HN", "LN"):
        q_meta = meta[meta["quadrant"] == q]
        q_mat = mat.loc[q_meta["prompt_id"]]
        print(f"  {q}: prompts={len(q_meta)}  sum={int(q_mat.to_numpy().sum())}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = FIGURES_DIR / "fig_v3_prompt_kaomoji.png"
    plot_prompt_kaomoji_matrix(df, str(fig_path), top_k=12)
    print(f"\nwrote {fig_path}")

    # TSV of the full matrix with prompt text for spreadsheet inspection.
    out = mat.copy()
    out.insert(0, "quadrant", meta.set_index("prompt_id").loc[out.index, "quadrant"])
    out.insert(1, "prompt_text", meta.set_index("prompt_id").loc[out.index, "prompt_text"])
    out.insert(2, "total_emissions",
               meta.set_index("prompt_id").loc[out.index, "total_emissions"])
    tsv_path = DATA_DIR / "v3_prompt_kaomoji_matrix.tsv"
    out.to_csv(tsv_path, sep="\t", index=True, index_label="prompt_id")
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
