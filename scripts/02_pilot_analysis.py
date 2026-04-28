"""Analyze the pilot output.

Reads data/pilot_raw.jsonl, evaluates the three pre-registered decision
rules + the k-means ARI summary, and writes figures to
figures/local/gemma/ (v1/v2 are gemma-only).

Usage:
    python scripts/02_pilot_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.analysis import all_figures, evaluate_axis, load_rows
from llmoji_study.config import (
    DATA_DIR,
    FIGURES_DIR,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
    STEERED_AXES,
)

PILOT_FIGURES_DIR = FIGURES_DIR / "local" / "gemma"
from llmoji_study.hidden_state_analysis import load_hidden_features


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
    # For hidden-state figures (Fig 1b PCA, Fig 3 cosine), load the
    # per-row hidden-state matrix from sidecars. Fig 1a / 2 / 4 stay
    # probe-based and just use the df columns. `load_hidden_features`
    # reads the JSONL independently and returns its own metadata df
    # aligned with X; we discard that df and use the probe-column df
    # from load_rows for probe-based figures, passing only X forward.
    print("\nloading hidden-state features for Fig 1b + Fig 3...")
    df_hidden, X = load_hidden_features(
        str(PILOT_RAW_PATH), DATA_DIR,
        experiment=PILOT_EXPERIMENT, which="h_last",
    )
    # Align X to df via row_uuid (drops rows without sidecars).
    uuid_to_idx = {u: i for i, u in enumerate(df_hidden["row_uuid"])}
    aligned_idx = [uuid_to_idx.get(u, -1) for u in df.get("row_uuid", [])]
    keep = [i for i in aligned_idx if i >= 0]
    if len(keep) < len(df):
        print(f"  [align] {len(keep)}/{len(df)} rows have hidden-state sidecars")
    df = df.iloc[[k for k, i in enumerate(aligned_idx) if i >= 0]].reset_index(drop=True)
    X = X[keep]

    PILOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_figures(df, X, str(PILOT_FIGURES_DIR))
    print(f"figures written to {PILOT_FIGURES_DIR}")

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
