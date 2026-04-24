"""Analyze the pilot output.

Reads data/pilot_raw.jsonl, evaluates the three pre-registered decision
rules + the k-means ARI summary, and writes figures to figures/.

Usage:
    python scripts/02_pilot_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.analysis import all_figures, evaluate_axis, load_rows
from llmoji.config import FIGURES_DIR, PILOT_RAW_PATH, STEERED_AXES


def main() -> None:
    if not PILOT_RAW_PATH.exists():
        print(f"no pilot data at {PILOT_RAW_PATH}; run scripts/01_pilot_run.py first")
        sys.exit(1)

    df = load_rows(str(PILOT_RAW_PATH))
    total = len(df)
    errors = 0
    if "error" in df.columns:
        errors = int(df["error"].notna().sum())
        df = df[df["error"].isna()] if errors else df
    print(f"loaded {total} rows ({errors} errors, {total - errors} usable)")

    # --- breakdown ---
    print("\n=== breakdown by (condition, kaomoji pole) ===")
    breakdown = (
        df.groupby(["condition", "kaomoji_label"]).size().unstack(fill_value=0)
    )
    print(breakdown)

    # --- decision rules, per axis ---
    print("\n=== decision rules ===")
    verdicts = {a: evaluate_axis(df, a) for a in STEERED_AXES}
    for a, v in verdicts.items():
        print(v.summary())
        print()

    # --- figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_figures(df, str(FIGURES_DIR))
    print(f"figures written to {FIGURES_DIR}")

    # --- go / no-go call (per axis) ---
    print("\n=== verdict ===")
    for a, v in verdicts.items():
        passed_core = v.rule1_non_degenerate and v.rule2_monotonic_shift
        if passed_core:
            tag = "GO — core rules pass on this axis."
        elif v.rule1_non_degenerate and not v.rule2_monotonic_shift:
            tag = "AMBIGUOUS — diverse kaomoji but no monotonic causal shift."
        else:
            tag = "NO-GO — insufficient kaomoji diversity for this axis."
        print(f"  {a}: {tag}")


if __name__ == "__main__":
    main()
