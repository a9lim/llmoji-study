# pyright: reportAttributeAccessIssue=false
"""Bootstrap + subsample variance estimate for the face_gain metric.

face_gain = face-centroid R² − quadrant-centroid R² in full hidden-state
space (script 25's headline number). Currently reported as a single
in-sample point estimate; we don't know whether a cross-prompt-iteration
delta of +1pp is real or sample noise.

Two estimators:

1. **Prompt-level bootstrap** (n_boot=200): resample the 120 prompt_ids
   with replacement, take all 8 seeds per resampled prompt, recompute
   face/quadrant centroids and the R²s on each bootstrap. Standard
   error on the metric, with a 2.5/97.5 percentile CI.

2. **Subsample-by-N curve** (N ∈ {30, 60, 90, 120} prompts × 50 reps):
   how does the metric stabilize as a function of prompt count? Tells
   us whether 120 is enough or we should plan future pilots wider.

Both bootstraps resample at prompt-level (not row-level) because the
8 seeds of one prompt share the same prompt-side hidden state — they
are not iid. Resampling rows would underestimate variance.

Outputs:
  data/face_gain_variance_<short>.tsv   (per-model summary)
  data/face_gain_variance_bootstrap.tsv (full distribution, all models)
  figures/local/cross_model/fig_face_gain_variance.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmoji_study.config import FIGURES_DIR, DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    _use_cjk_font,
    load_emotional_features_stack,
)


def _r2_by_label(X: np.ndarray, labels: np.ndarray) -> float:
    """1 − SSwithin/SStotal, full hidden-state space."""
    grand = X.mean(axis=0)
    Xc = X - grand
    ss_total = float((Xc * Xc).sum())
    if ss_total <= 0:
        return float("nan")
    ss_within = 0.0
    for label in pd.unique(labels):
        mask = labels == label
        if not mask.any():
            continue
        xs = X[mask]
        mu = xs.mean(axis=0)
        resid = xs - mu
        ss_within += float((resid * resid).sum())
    return 1.0 - ss_within / ss_total


def _face_gain(X: np.ndarray, faces: np.ndarray, quads: np.ndarray) -> tuple[float, float, float]:
    r2_face = _r2_by_label(X, faces)
    r2_quad = _r2_by_label(X, quads)
    return r2_face, r2_quad, r2_face - r2_quad


def _resample_prompts(
    prompt_ids: np.ndarray,
    rng: np.random.Generator,
    n_prompts: int | None = None,
) -> np.ndarray:
    """Return row indices for a prompt-level resample. Sampling is
    with replacement when n_prompts == len(unique); subsample mode is
    without replacement when n_prompts < len(unique)."""
    unique = pd.unique(prompt_ids)
    if n_prompts is None:
        n_prompts = len(unique)
    if n_prompts == len(unique):
        sampled = rng.choice(unique, size=n_prompts, replace=True)
    else:
        sampled = rng.choice(unique, size=n_prompts, replace=False)
    # Build per-prompt row-index lists once for speed.
    by_prompt: dict[str, np.ndarray] = {}
    for pid in unique:
        by_prompt[str(pid)] = np.where(prompt_ids == pid)[0]
    parts = [by_prompt[str(pid)] for pid in sampled]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def _bootstrap_one_model(
    short: str,
    *,
    n_boot: int = 200,
    subsample_ns: tuple[int, ...] = (30, 60, 90, 120),
    subsample_reps: int = 50,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run the two estimators for one model. Returns:

      summary_row  — single row dict for the per-model TSV
      boot_df      — long-form per-bootstrap rows for plotting
      point_est    — observed (non-bootstrap) face_gain on full data
    """
    print(f"\n=== {short} ===")
    df, X = load_emotional_features_stack(short, which="h_first", split_hn=True)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    faces = df["first_word"].astype(str).to_numpy()
    quads = df["quadrant"].astype(str).to_numpy()
    pids = df["prompt_id"].astype(str).to_numpy()
    print(f"  {len(df)} kaomoji-bearing rows; {len(pd.unique(pids))} unique prompts; "
          f"X.shape={X.shape}")

    r2_f0, r2_q0, gain0 = _face_gain(X, faces, quads)
    print(f"  point estimate: r2_face={r2_f0:.4f}  r2_quad={r2_q0:.4f}  "
          f"face_gain={gain0*100:+.2f}pp")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    print(f"  bootstrap (n={n_boot}, prompt-level resample)...")
    boot_gains: list[float] = []
    for b in range(n_boot):
        idx = _resample_prompts(pids, rng)
        Xb = X[idx]
        r2_f, r2_q, gain = _face_gain(Xb, faces[idx], quads[idx])
        boot_gains.append(gain)
        rows.append({"model": short, "kind": "bootstrap", "n_prompts": int(len(pd.unique(pids))),
                     "rep": b, "r2_face": r2_f, "r2_quad": r2_q, "face_gain": gain})
    boot = np.asarray(boot_gains)
    mean_b = float(boot.mean())
    std_b = float(boot.std(ddof=1))
    lo_b = float(np.quantile(boot, 0.025))
    hi_b = float(np.quantile(boot, 0.975))
    print(f"    bootstrap face_gain: {mean_b*100:+.2f}pp ± {std_b*100:.2f}pp   "
          f"(95% CI [{lo_b*100:+.2f}, {hi_b*100:+.2f}])")

    print(f"  subsample-by-N ({subsample_reps} reps each, N ∈ {subsample_ns})...")
    sub_summary: dict[int, tuple[float, float]] = {}
    for N in subsample_ns:
        if N > len(pd.unique(pids)):
            continue
        gains = []
        for r in range(subsample_reps):
            idx = _resample_prompts(pids, rng, n_prompts=N)
            Xs = X[idx]
            _, _, gain = _face_gain(Xs, faces[idx], quads[idx])
            gains.append(gain)
            rows.append({"model": short, "kind": "subsample", "n_prompts": int(N),
                         "rep": r, "r2_face": float("nan"), "r2_quad": float("nan"),
                         "face_gain": gain})
        arr = np.asarray(gains)
        sub_summary[N] = (float(arr.mean()), float(arr.std(ddof=1)))
        print(f"    N={N:3d}: face_gain={arr.mean()*100:+.2f}pp ± {arr.std(ddof=1)*100:.2f}pp")

    summary = {
        "model": short,
        "n_rows": int(len(df)),
        "n_prompts": int(len(pd.unique(pids))),
        "point_face_gain": gain0,
        "boot_mean": mean_b,
        "boot_std": std_b,
        "boot_lo95": lo_b,
        "boot_hi95": hi_b,
    }
    for N, (m, s) in sub_summary.items():
        summary[f"sub{N}_mean"] = m
        summary[f"sub{N}_std"] = s

    return pd.DataFrame([summary]), pd.DataFrame(rows), {"point": gain0}


def _plot_variance_grid(boot_df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    _use_cjk_font()
    if len(boot_df) == 0:
        return
    models = sorted(boot_df["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: bootstrap distributions per model.
    ax = axes[0]
    for i, m in enumerate(models):
        sub = boot_df[(boot_df["model"] == m) & (boot_df["kind"] == "bootstrap")]
        if len(sub) == 0:
            continue
        ax.hist(sub["face_gain"] * 100, bins=30, alpha=0.45, label=m, edgecolor="black", linewidth=0.4)
        # point estimate vline
        pt = float(summary[summary["model"] == m]["point_face_gain"].iloc[0]) * 100
        ax.axvline(pt, color=f"C{i}", linestyle=":", linewidth=1.2)
    ax.set_xlabel("face_gain (percentage points)")
    ax.set_ylabel("count (bootstrap reps)")
    ax.set_title("bootstrap distribution of face_gain (n=200, prompt-level resample)\n"
                 "dotted vline = point estimate on full data")
    ax.legend(loc="best", fontsize=9, frameon=False)

    # Panel B: subsample-by-N curve, mean ± std per N.
    ax = axes[1]
    for i, m in enumerate(models):
        sub = boot_df[(boot_df["model"] == m) & (boot_df["kind"] == "subsample")]
        if len(sub) == 0:
            continue
        agg = sub.groupby("n_prompts")["face_gain"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n_prompts"], agg["mean"] * 100, yerr=agg["std"] * 100,
                    marker="o", capsize=4, label=m, linewidth=1.2)
    ax.set_xlabel("subsample size (prompts)")
    ax.set_ylabel("face_gain (percentage points)")
    ax.set_title("subsample-by-N stability\n(50 reps per N)")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.axhline(0, color="#888", linewidth=0.4)

    fig.suptitle("face_gain variance estimate — sampling noise band", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="gemma,qwen,ministral",
                   help="comma-separated model shorts; default canonical 3")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-sub-reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    boots: list[pd.DataFrame] = []
    for short in args.models.split(","):
        short = short.strip()
        if short not in MODEL_REGISTRY:
            print(f"[skip] {short}: not in MODEL_REGISTRY")
            continue
        if not MODEL_REGISTRY[short].emotional_data_path.exists():
            print(f"[skip] {short}: no v3 data")
            continue
        s_df, b_df, _ = _bootstrap_one_model(
            short, n_boot=args.n_boot,
            subsample_reps=args.n_sub_reps, seed=args.seed,
        )
        if len(s_df):
            summaries.append(s_df)
            boots.append(b_df)

    if not summaries:
        print("nothing produced; bailing")
        return
    summary = pd.concat(summaries, ignore_index=True)
    boot = pd.concat(boots, ignore_index=True)

    sum_path = DATA_DIR / "face_gain_variance.tsv"
    boot_path = DATA_DIR / "face_gain_variance_bootstrap.tsv"
    summary.to_csv(sum_path, sep="\t", index=False, float_format="%.5f")
    boot.to_csv(boot_path, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {sum_path}")
    print(f"wrote {boot_path}")

    fig_path = out_dir / "fig_face_gain_variance.png"
    _plot_variance_grid(boot, summary, fig_path)
    print(f"wrote {fig_path}")

    print("\n--- decision rule reference ---")
    print("for any prompt-iteration delta vs the v2 baseline, treat it as REAL only if")
    print("|delta| > 2 * boot_std (≈ p<0.05 two-sided under approx-normal bootstrap).")
    print(summary[["model", "point_face_gain", "boot_std", "boot_lo95", "boot_hi95"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
