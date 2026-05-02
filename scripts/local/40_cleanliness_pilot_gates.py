"""Cleanliness-pilot gate check — compares new pilot data against
pre-cleanliness backup on the 4 pre-registered gates from
docs/2026-05-03-cleanliness-pilot.md:

  Gate 1 — Russell-quadrant silhouette over PCA(2) of h_first ≥ prior
  Gate 2 — HN-S > HN-D direction on fearful.unflinching (every aggregate)
  Gate 3 — NB centeredness ‖NB - grand‖ improves vs prior
  Gate 4 — HP↔LP arousal separation on PC2 widens vs prior

Loads from canonical paths for the new pilot, and from
``data/*_pre_cleanliness*.jsonl`` + ``data/hidden/v3*_pre_cleanliness/``
for the prior baseline. Fair-N comparisons subsample the prior to
1 seed per prompt (the pilot is 1 seed/cell).

Usage:
    python scripts/local/40_cleanliness_pilot_gates.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    _hn_split_map,
    load_emotional_features,
)


PRIOR_ARCHIVE = DATA_DIR / "archive" / "2026-05-03_pre_cleanliness"


# Map model short_name → (new jsonl path, new experiment, prior jsonl path,
# prior experiment). Prior data lives under PRIOR_ARCHIVE post-rerun
# archive (2026-05-03); the *_pre_cleanliness suffix is preserved on the
# files themselves.
def _paths_for(short: str) -> tuple[Path, str, Path, str]:
    M = MODEL_REGISTRY[short]
    new_jsonl = M.emotional_data_path
    new_exp = M.experiment
    prior_jsonl = PRIOR_ARCHIVE / (
        M.emotional_data_path.stem + "_pre_cleanliness" + M.emotional_data_path.suffix
    )
    prior_exp = M.experiment + "_pre_cleanliness"
    return new_jsonl, new_exp, prior_jsonl, prior_exp


def _load(jsonl: Path, experiment: str, layer: int | None,
          *, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, np.ndarray]:
    """Sidecars resolve under ``data_dir/hidden/<experiment>/<uuid>.npz``;
    pass ``PRIOR_ARCHIVE`` as ``data_dir`` to load the archived prior
    sidecars at ``data/archive/2026-05-03_pre_cleanliness/hidden/...``."""
    df, X = load_emotional_features(
        str(jsonl), data_dir,
        experiment=experiment, which="h_first", layer=layer,
    )
    return df.reset_index(drop=True), X


def _subsample_one_seed(df: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep one row per prompt_id (the lowest-seed surviving row).
    Aligns prior-data N to the pilot's 1-seed/cell shape."""
    if "seed" not in df.columns:
        return df, X
    keep_mask = df.sort_values(["prompt_id", "seed"]).drop_duplicates(
        "prompt_id", keep="first"
    ).index
    sub_df = df.loc[keep_mask].reset_index(drop=True)
    sub_X = X[keep_mask.to_numpy()]
    return sub_df, sub_X


# ---------------------------------------------------------------------------
# Gate 1 — silhouette
# ---------------------------------------------------------------------------


def gate1_silhouette(df: pd.DataFrame, X: np.ndarray) -> float:
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    return float(silhouette_score(Y, df["quadrant"].to_numpy()))


# ---------------------------------------------------------------------------
# Gate 2 — fearful direction
# ---------------------------------------------------------------------------


def _hn_split(df: pd.DataFrame) -> pd.DataFrame:
    """Attach an `hn_split` column ∈ {HN-D, HN-S, None} via the
    rule-3 registry."""
    split = _hn_split_map()
    df = df.copy()
    df["hn_split"] = df["prompt_id"].map(split)
    return df


def _fearful_at(row: pd.Series, agg: str) -> float:
    """Pull fearful.unflinching at {t0, tlast, mean} from either
    schema. New (3-probe) data has fearful in `probe_scores_t0` (idx
    determined by PROBES) and `probe_means["fearful.unflinching"]`.
    Old (5-probe) data has it in `extension_probe_scores_t0` /
    `_tlast` / `extension_probe_means` dicts."""
    from llmoji_study.config import PROBES
    name = "fearful.unflinching"
    if agg == "t0":
        scores = row.get("probe_scores_t0", None)
        if scores and name in PROBES:
            return float(scores[PROBES.index(name)])
        ext = row.get("extension_probe_scores_t0", None)
        if isinstance(ext, dict) and name in ext:
            return float(ext[name])
    elif agg == "tlast":
        scores = row.get("probe_scores_tlast", None)
        if scores and name in PROBES:
            return float(scores[PROBES.index(name)])
        ext = row.get("extension_probe_scores_tlast", None)
        if isinstance(ext, dict) and name in ext:
            return float(ext[name])
    elif agg == "mean":
        means = row.get("probe_means", None)
        if isinstance(means, dict) and name in means:
            return float(means[name])
        ext = row.get("extension_probe_means", None)
        if isinstance(ext, dict) and name in ext:
            return float(ext[name])
    return float("nan")


def gate2_fearful_direction(df: pd.DataFrame) -> dict[str, float]:
    """Return {t0: S-D, tlast: S-D, mean: S-D} for fearful.unflinching.
    Handles both schemas (3-probe new + 5-probe-with-extensions old)."""
    sub = _hn_split(df)
    out: dict[str, float] = {}
    for agg in ("t0", "tlast", "mean"):
        sub_agg = sub.assign(_fearful=sub.apply(lambda r: _fearful_at(r, agg), axis=1))
        d = sub_agg.loc[sub_agg["hn_split"] == "HN-D", "_fearful"].mean()
        s = sub_agg.loc[sub_agg["hn_split"] == "HN-S", "_fearful"].mean()
        out[agg] = float(s - d)
    return out


# ---------------------------------------------------------------------------
# Gates 3, 4 — basis-invariant centroid geometry (full hidden space)
# ---------------------------------------------------------------------------


def gate3_nb_within_scatter(df: pd.DataFrame, X: np.ndarray) -> float:
    """Within-NB scatter: mean ‖row − NB_centroid‖ over NB rows in
    full hidden space. If the cleanliness pass made NB more uniformly
    affectless (less hidden valence variance across NB prompts), this
    number should DROP. Basis-invariant (no PCA), so directly
    comparable across runs."""
    nb_mask = (df["quadrant"] == "NB").to_numpy()
    if nb_mask.sum() == 0:
        return float("nan")
    nb_X = X[nb_mask]
    centroid = nb_X.mean(axis=0)
    return float(np.linalg.norm(nb_X - centroid, axis=1).mean())


def gate4_hp_lp_full_distance(df: pd.DataFrame, X: np.ndarray) -> float:
    """Euclidean distance between HP and LP centroids in FULL hidden
    space. Basis-invariant. The cleanliness pass tightened HP to
    unambiguous high-arousal joy and LP to gentle sensory satisfaction
    — they should pull apart in the affect manifold."""
    hp_mask = (df["quadrant"] == "HP").to_numpy()
    lp_mask = (df["quadrant"] == "LP").to_numpy()
    if hp_mask.sum() == 0 or lp_mask.sum() == 0:
        return float("nan")
    hp_centroid = X[hp_mask].mean(axis=0)
    lp_centroid = X[lp_mask].mean(axis=0)
    return float(np.linalg.norm(hp_centroid - lp_centroid))


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------


def run_model(short: str) -> dict[str, Any]:
    M = MODEL_REGISTRY[short]
    new_jsonl, new_exp, prior_jsonl, prior_exp = _paths_for(short)

    if not new_jsonl.exists():
        print(f"[{short}] no new pilot data at {new_jsonl}; skipping")
        return {}
    if not prior_jsonl.exists():
        print(f"[{short}] no prior backup at {prior_jsonl}; skipping")
        return {}

    print(f"\n=== {short} (h_first @ L{M.preferred_layer}) ===")
    print(f"  new   : {new_jsonl}  ({new_exp})")
    print(f"  prior : {prior_jsonl}  ({prior_exp})")

    df_new, X_new = _load(new_jsonl, new_exp, M.preferred_layer)
    df_prior_full, X_prior_full = _load(
        prior_jsonl, prior_exp, M.preferred_layer, data_dir=PRIOR_ARCHIVE,
    )
    df_prior, X_prior = _subsample_one_seed(df_prior_full, X_prior_full)
    print(f"  new   rows: {len(df_new)},  X {X_new.shape}")
    print(f"  prior rows (1-seed subsample): {len(df_prior)}  "
          f"(of {len(df_prior_full)} full)")

    # Gate 1
    sil_new = gate1_silhouette(df_new, X_new)
    sil_prior = gate1_silhouette(df_prior, X_prior)
    g1_pass = sil_new >= sil_prior

    # Gate 2 — direction-only on every (model, aggregate); use FULL prior for stability
    g2_new = gate2_fearful_direction(df_new)
    g2_prior = gate2_fearful_direction(df_prior_full)
    g2_pass = all(v > 0 for v in g2_new.values() if not np.isnan(v))

    # Gate 3 — within-NB scatter (basis-invariant, full hidden space)
    nb_new = gate3_nb_within_scatter(df_new, X_new)
    nb_prior = gate3_nb_within_scatter(df_prior, X_prior)
    g3_pass = nb_new < nb_prior

    # Gate 4 — HP↔LP centroid distance (basis-invariant, full hidden space)
    hp_lp_new = gate4_hp_lp_full_distance(df_new, X_new)
    hp_lp_prior = gate4_hp_lp_full_distance(df_prior, X_prior)
    g4_pass = hp_lp_new >= hp_lp_prior

    print(f"  Gate 1  silhouette       new={sil_new:+.3f}  prior={sil_prior:+.3f}  "
          f"{'PASS' if g1_pass else 'FAIL'}")
    print(f"  Gate 2  fearful S-D")
    for agg in ("t0", "tlast", "mean"):
        nv = g2_new.get(agg, float("nan"))
        pv = g2_prior.get(agg, float("nan"))
        print(f"           {agg:<6}            new={nv:+.4f}  prior={pv:+.4f}  "
              f"{'PASS' if nv > 0 else 'FAIL'}")
    print(f"  Gate 3  NB within-scatter (full)  new={nb_new:.3f}  prior={nb_prior:.3f}  "
          f"{'PASS (new<prior)' if g3_pass else 'FAIL'}")
    print(f"  Gate 4  HP↔LP centroid dist (full)  new={hp_lp_new:.3f}  prior={hp_lp_prior:.3f}  "
          f"{'PASS (new>=prior)' if g4_pass else 'FAIL'}")

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass])
    print(f"  → {n_pass}/4 gates pass")

    return {
        "model": short,
        "gates": {
            "gate1_silhouette": {"new": sil_new, "prior": sil_prior, "pass": g1_pass},
            "gate2_fearful_dir": {"new": g2_new, "prior": g2_prior, "pass": g2_pass},
            "gate3_nb_within_scatter": {"new": nb_new, "prior": nb_prior, "pass": g3_pass},
            "gate4_hp_lp_full_distance": {"new": hp_lp_new, "prior": hp_lp_prior, "pass": g4_pass},
        },
        "n_pass": n_pass,
    }


def main() -> None:
    summaries: dict[str, dict] = {}
    for short in ("gemma", "qwen", "ministral"):
        s = run_model(short)
        if s:
            summaries[short] = s

    if not summaries:
        print("\nno models had both new + prior data; nothing to compare.")
        return

    print("\n========== CROSS-MODEL VERDICT ==========")
    print(f"  {'model':<10}  {'g1':>4}  {'g2':>4}  {'g3':>4}  {'g4':>4}  {'pass':>5}")
    for short, s in summaries.items():
        g = s["gates"]
        print(f"  {short:<10}  "
              f"{'✓' if g['gate1_silhouette']['pass'] else '✗':>4}  "
              f"{'✓' if g['gate2_fearful_dir']['pass'] else '✗':>4}  "
              f"{'✓' if g['gate3_nb_within_scatter']['pass'] else '✗':>4}  "
              f"{'✓' if g['gate4_hp_lp_full_distance']['pass'] else '✗':>4}  "
              f"{s['n_pass']}/4")

    all_pass = all(s["n_pass"] == 4 for s in summaries.values())
    if all_pass:
        print("\n  ALL GATES PASS on every model → COMMIT FULL N=2880 RERUN")
    else:
        print("\n  one or more gates fail → see per-gate output above; don't escalate compute yet")


if __name__ == "__main__":
    main()
