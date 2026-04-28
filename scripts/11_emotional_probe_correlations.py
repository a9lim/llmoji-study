"""v3 probe-correlation analysis. Replicates v2's valence-collapse
claim on naturalistic unsteered data: does happy.sad × angry.calm
correlate as strongly in v3 as v2 said it should?

Pre-registered reading:
  |ρ(happy.sad, angry.calm)| > 0.7 on all 640 rows → v2 replicates;
  |ρ| < 0.4 → v2's collapse was a steering artifact, naturalistic
             data has richer structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import DATA_DIR, PROBES, current_model
from llmoji_study.emotional_analysis import (
    compute_probe_correlations,
    load_rows,
    plot_probe_correlation_matrix,
)


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} scripts/03_emotional_run.py first")
        return

    df = load_rows(str(M.emotional_data_path))
    print(f"loaded {len(df)} v3 rows ({M.short_name})")

    stats = compute_probe_correlations(df, timestep="t0")
    i_hs = PROBES.index("happy.sad")
    i_ac = PROBES.index("angry.calm")

    print("\nhappy.sad × angry.calm correlation (critical pair):")
    for key in ("all", "HP", "LP", "HN", "LN"):
        sub = stats["by_subset"][key]
        n = sub["n"]
        if sub["pearson"] is None:
            print(f"  {key}: n={n}  (too few)")
            continue
        r = sub["pearson"][i_hs][i_ac]
        rho = sub["spearman"][i_hs][i_ac]
        print(f"  {key}: n={n}  pearson r = {r:+.3f}  spearman rho = {rho:+.3f}")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_p = M.figures_dir / "fig_v3_corr_pearson.png"
    fig_s = M.figures_dir / "fig_v3_corr_spearman.png"
    plot_probe_correlation_matrix(df, str(fig_p), method="pearson")
    print(f"\nwrote {fig_p}")
    plot_probe_correlation_matrix(df, str(fig_s), method="spearman")
    print(f"wrote {fig_s}")

    stats_path = DATA_DIR / "v3_probe_correlations.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
