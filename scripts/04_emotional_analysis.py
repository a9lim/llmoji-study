"""Emotional-battery analysis driver (hidden-state).

Reads data/emotional_raw.jsonl + per-row hidden-state sidecars from
data/hidden/v3/. Re-labels kaomoji in place via taxonomy.extract,
prints per-quadrant emission stats, writes three hidden-state figures
(Fig A, B, C) + a per-kaomoji summary TSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    EMOTIONAL_SUMMARY_PATH,
    FIGURES_DIR,
)
from llmoji.emotional_analysis import (
    load_emotional_features,
    plot_kaomoji_cosine_heatmap,
    plot_kaomoji_quadrant_alignment,
    plot_within_kaomoji_consistency,
    summary_table,
)
from llmoji.taxonomy import extract


def _relabel_in_place(path: Path) -> None:
    """Re-extract first_word / kaomoji / kaomoji_label via the current
    taxonomy and rewrite the JSONL in place. Cheap; safe to run every
    time the analysis script starts."""
    if not path.exists():
        return
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    out_lines: list[str] = []
    for l in lines:
        r = json.loads(l)
        if "error" in r:
            out_lines.append(l)
            continue
        m = extract(r.get("text", ""))
        r["first_word"] = m.first_word
        r["kaomoji"] = m.kaomoji
        r["kaomoji_label"] = m.label
        out_lines.append(json.dumps(r))
    path.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return
    print(f"re-labeling kaomoji in {EMOTIONAL_DATA_PATH}")
    _relabel_in_place(EMOTIONAL_DATA_PATH)

    print("loading hidden-state features (which=h_mean, layer=max)...")
    df, X = load_emotional_features(
        str(EMOTIONAL_DATA_PATH), DATA_DIR,
        experiment=EMOTIONAL_EXPERIMENT,
        which="h_mean",
    )
    print(f"loaded {len(df)} kaomoji-bearing rows; X shape {X.shape}")
    if len(df) == 0:
        print("nothing to plot; the v3 run needs to land hidden-state sidecars first")
        return

    # Per-quadrant emission summary.
    print("\nper-quadrant kaomoji emission (first-word filter):")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        n = len(q_rows)
        uniq = int(q_rows["first_word"].nunique()) if n else 0
        print(f"  {q}: {n} kaomoji-bearing rows; {uniq} distinct forms")

    # Top kaomoji per quadrant.
    print("\ntop-5 first_words per quadrant (by count):")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        top = q_rows["first_word"].value_counts().head(5)
        print(f"  {q}:")
        for km, c in top.items():
            print(f"    {km}  ({c})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_a = FIGURES_DIR / "fig_emo_a_kaomoji_sim.png"
    fig_b = FIGURES_DIR / "fig_emo_b_kaomoji_consistency.png"
    fig_c = FIGURES_DIR / "fig_emo_c_kaomoji_quadrant.png"

    print("\nwriting figures...")
    plot_kaomoji_cosine_heatmap(df, X, str(fig_a))
    print(f"  wrote {fig_a}")
    plot_within_kaomoji_consistency(df, X, str(fig_b))
    print(f"  wrote {fig_b}")
    plot_kaomoji_quadrant_alignment(df, X, str(fig_c))
    print(f"  wrote {fig_c}")

    summary = summary_table(df, X)
    summary.to_csv(EMOTIONAL_SUMMARY_PATH, sep="\t", index=False)
    print(f"\nwrote per-kaomoji summary to {EMOTIONAL_SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
