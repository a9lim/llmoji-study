"""Emotional-battery analysis driver.

Reads data/emotional_raw.jsonl, re-labels kaomoji in place via
taxonomy.extract (per CLAUDE.md gotcha — JSONL labels are frozen at
write time), prints per-quadrant emission stats, writes three figures
and a per-kaomoji summary TSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_SUMMARY_PATH,
    FIGURES_DIR,
)
from llmoji.emotional_analysis import (
    load_rows,
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

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} rows")

    # per-quadrant emission summary
    print("\nper-quadrant kaomoji emission:")
    for q in ("HP", "LP", "HN", "LN"):
        q_rows = df[df["quadrant"] == q]
        n = len(q_rows)
        k = int(q_rows["kaomoji"].notna().sum()) if n else 0
        uniq = int(q_rows.dropna(subset=["kaomoji"])["kaomoji"].nunique()) if n else 0
        rate = (k / n) if n else 0.0
        print(f"  {q}: {k}/{n} rows bear a kaomoji ({rate:.0%}); {uniq} distinct forms")

    # top kaomoji per quadrant (up to 5)
    print("\ntop-5 kaomoji per quadrant (by count):")
    for q in ("HP", "LP", "HN", "LN"):
        q_rows = df[(df["quadrant"] == q) & df["kaomoji"].notna()]
        top = q_rows["kaomoji"].value_counts().head(5)
        print(f"  {q}:")
        for km, c in top.items():
            print(f"    {km}  ({c})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_a = FIGURES_DIR / "fig_emo_a_kaomoji_sim_tlast.png"
    fig_b = FIGURES_DIR / "fig_emo_b_kaomoji_consistency_tlast.png"
    fig_c = FIGURES_DIR / "fig_emo_c_kaomoji_quadrant_tlast.png"

    # One set of figures only: under stateless=True, probe_scores_t0
    # and probe_scores_tlast currently hold the same whole-generation
    # aggregate (see CLAUDE.md "stateless=True collapses per_generation"
    # gotcha). Producing a duplicate t=0 set would be misleading until
    # the capture-code fix runs on a fresh pilot. The analysis module's
    # timestep= parameter stays in place for that future run.
    print("\nwriting figures...")
    plot_kaomoji_cosine_heatmap(df, str(fig_a), timestep="tlast")
    print(f"  wrote {fig_a}")
    plot_within_kaomoji_consistency(df, str(fig_b), timestep="tlast")
    print(f"  wrote {fig_b}")
    plot_kaomoji_quadrant_alignment(df, str(fig_c), timestep="tlast")
    print(f"  wrote {fig_c}")

    summary = summary_table(df, timestep="tlast")
    summary.to_csv(EMOTIONAL_SUMMARY_PATH, sep="\t", index=False)
    print(f"\nwrote per-kaomoji summary to {EMOTIONAL_SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
