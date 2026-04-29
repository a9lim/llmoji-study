"""v3 probe-correlation analysis. Replicates v2's valence-collapse
claim on naturalistic unsteered data and extends it to the
post-2026-04-29 affect trio (happy / fearful / angry).

Pre-registered reading on the original 2-probe pair:
  |ρ(happy.sad, angry.calm)| > 0.7 → v2 replicates;
  |ρ| < 0.4 → v2's collapse was a steering artifact, naturalistic
             data has richer structure.

Extended reading on the trio:
  PAD theory predicts fear and anger separate within HN. If
  ρ(fearful.unflinching, angry.calm) on HN-only rows is strongly
  negative, the model's hidden state distinguishes them even when
  the kaomoji vocabulary doesn't. Reported per quadrant.

Outputs two correlation-matrix figures:
  - fig_v3_corr_pearson.png — full PROBES_ALL grid (core + ext),
    one panel per subset (all + 5 quadrants).
  - fig_v3_corr_spearman.png — same, Spearman rho.

Plus a JSON summary with the trio-pair correlations broken out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import DATA_DIR, current_model
from llmoji_study.emotional_analysis import (
    available_extension_probes,
    compute_probe_correlations,
    load_rows,
    plot_probe_correlation_matrix,
)


# Affect trio for the headline correlation table. Order: valence,
# fear, anger.
TRIO = ["happy.sad", "fearful.unflinching", "angry.calm"]
TRIO_PAIRS = [
    ("happy.sad", "fearful.unflinching"),
    ("happy.sad", "angry.calm"),
    ("fearful.unflinching", "angry.calm"),
]


def _safe_pair(stats: dict, key: str, probes: list[str],
               a: str, b: str) -> tuple[float | None, float | None, int]:
    sub = stats["by_subset"].get(key)
    if sub is None or sub.get("pearson") is None:
        return None, None, sub["n"] if sub else 0
    if a not in probes or b not in probes:
        return None, None, sub["n"]
    i = probes.index(a); j = probes.index(b)
    return float(sub["pearson"][i][j]), float(sub["spearman"][i][j]), sub["n"]


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} scripts/03_emotional_run.py first")
        return

    df = load_rows(str(M.emotional_data_path))
    ext = available_extension_probes(df)
    print(f"loaded {len(df)} v3 rows ({M.short_name})")
    print(f"extension probes available: {ext}")

    stats = compute_probe_correlations(df, timestep="t0")
    probes_used = stats["probes"]

    print("\naffect trio correlations (t0):")
    print(f"  {'pair':<48} {'all':>22} {'HP':>10} {'LP':>10} {'HN':>10} {'LN':>10}")
    for a, b in TRIO_PAIRS:
        pair_lab = f"{a}~{b}"
        cells = []
        for key in ("all", "HP", "LP", "HN", "LN"):
            r, rho, n = _safe_pair(stats, key, probes_used, a, b)
            if r is None:
                cells.append(f"n={n} —".rjust(22 if key == "all" else 10))
            elif key == "all":
                cells.append(f"n={n}  r={r:+.3f}  rho={rho:+.3f}".rjust(22))
            else:
                cells.append(f"r={r:+.3f}".rjust(10))
        print(f"  {pair_lab:<48} " + " ".join(cells))

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_p = M.figures_dir / "fig_v3_corr_pearson.png"
    fig_s = M.figures_dir / "fig_v3_corr_spearman.png"
    plot_probe_correlation_matrix(df, str(fig_p), method="pearson")
    print(f"\nwrote {fig_p}")
    plot_probe_correlation_matrix(df, str(fig_s), method="spearman")
    print(f"wrote {fig_s}")

    stats_path = DATA_DIR / "v3_probe_correlations.json"
    # Add a top-level "trio" section so a quick `jq '.trio'` reads
    # exactly the headline numbers without iterating the full matrix.
    trio_stats: dict = {}
    for key in ("all", "HP", "LP", "HN", "LN", "NB"):
        per_pair = {}
        for a, b in TRIO_PAIRS:
            r, rho, n = _safe_pair(stats, key, probes_used, a, b)
            per_pair[f"{a}~{b}"] = {"n": n, "pearson": r, "spearman": rho}
        trio_stats[key] = per_pair
    stats["trio"] = trio_stats
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
