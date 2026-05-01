"""Emotional-battery analysis driver (hidden-state).

Reads data/emotional_raw.jsonl + per-row hidden-state sidecars from
data/hidden/v3/. Re-extracts first_word in place via the current
canonicalization, prints per-quadrant emission stats, writes three
hidden-state figures (Fig A, B, C) + a per-kaomoji summary TSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    DATA_DIR,
    current_model,
)
from llmoji_study.emotional_analysis import (
    load_emotional_features,
    plot_kaomoji_cosine_heatmap,
    plot_kaomoji_quadrant_alignment,
    plot_within_kaomoji_consistency,
    summary_table,
)
from llmoji.taxonomy import extract


def _relabel_in_place(path: Path) -> None:
    """Re-extract first_word via the current llmoji.taxonomy.extract
    rules and rewrite the JSONL in place. Cheap; safe to run every
    time the analysis script starts. Drops legacy kaomoji /
    kaomoji_label fields if present (pre-2026-04-30 capture wrote
    those gemma-tuned TAXONOMY columns; new captures don't)."""
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
        # Drop legacy TAXONOMY-tuned fields if present.
        r.pop("kaomoji", None)
        r.pop("kaomoji_label", None)
        out_lines.append(json.dumps(r))
    path.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} python scripts/03_emotional_run.py first")
        return
    print(f"model: {M.short_name}; data: {M.emotional_data_path}")
    print(f"re-labeling kaomoji in {M.emotional_data_path}")
    _relabel_in_place(M.emotional_data_path)

    layer_label = "max" if M.preferred_layer is None else f"L{M.preferred_layer}"
    print(f"loading hidden-state features (which=h_mean, layer={layer_label})...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment,
        which="h_first",
        layer=M.preferred_layer,
        split_hn=True,
    )
    print(f"loaded {len(df)} kaomoji-bearing rows; X shape {X.shape}  (HN split into HN-D/HN-S; untagged HN rows dropped)")
    if len(df) == 0:
        print("nothing to plot; the v3 run needs to land hidden-state sidecars first")
        return

    print("\nper-quadrant kaomoji emission (first-word filter):")
    for q in ("HP", "LP", "HN-D", "HN-S", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        n = len(q_rows)
        uniq = int(q_rows["first_word"].nunique()) if n else 0
        print(f"  {q}: {n} kaomoji-bearing rows; {uniq} distinct forms")

    print("\ntop-5 first_words per quadrant (by count):")
    for q in ("HP", "LP", "HN-D", "HN-S", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        top = q_rows["first_word"].value_counts().head(5)
        print(f"  {q}:")
        for km, c in top.items():
            print(f"    {km}  ({c})")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_a = M.figures_dir / "fig_emo_a_kaomoji_sim.png"
    fig_b = M.figures_dir / "fig_emo_b_kaomoji_consistency.png"
    fig_c = M.figures_dir / "fig_emo_c_kaomoji_quadrant.png"

    print("\nwriting figures...")
    plot_kaomoji_cosine_heatmap(df, X, str(fig_a))
    print(f"  wrote {fig_a}")
    plot_within_kaomoji_consistency(df, X, str(fig_b))
    print(f"  wrote {fig_b}")
    plot_kaomoji_quadrant_alignment(df, X, str(fig_c))
    print(f"  wrote {fig_c}")

    summary = summary_table(df, X)
    summary.to_csv(M.emotional_summary_path, sep="\t", index=False)
    print(f"\nwrote per-kaomoji summary to {M.emotional_summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
