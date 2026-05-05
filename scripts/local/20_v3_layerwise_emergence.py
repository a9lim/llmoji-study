"""v3 layer-wise emergence trajectory.

For each captured probe layer, fit PCA(2) on h_mean and measure how
cleanly the five Russell quadrants separate. Three diagnostics per
layer:

* silhouette score over per-row PC1/PC2 coordinates with quadrant
  labels — single number, sensitive to within-cluster compactness;
* between-centroid std on PC1 and PC2 — separation diagnostic that
  the pre-2026-04-28 deepest-layer-only PCA reported;
* explained-variance fractions for PC1 and PC2 — to track how much of
  the variance the affect axes carry as a function of depth.

Two figures per model: a five-panel layer-vs-metric trajectory and a
strip of PCA scatters at four depth quartiles. With both gemma and
qwen runs available, also writes a cross-model comparison figure
overlaying the silhouette trajectories on a fractional-depth x-axis
(layer / max_layer) so the differing layer counts (gemma 2-57,
qwen 2-61) line up.

Caches the (n_rows, n_layers, hidden_dim) h_mean tensor at
``data/local/cache/v3_<short>_h_mean_all_layers.npz`` per model. Re-run with
``--rebuild`` to force recompute (typed as a flag, not a value).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from llmoji_study.config import FIGURES_DIR, MODEL_REGISTRY, current_model, resolve_model
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    _use_cjk_font,
    load_emotional_features_all_layers,
)
# In split mode (rule 3 redesign), iterate the 6-element ordering so
# HN-D and HN-S each get a centroid + scatter; non-split data shows up
# under aggregate HN, which falls through the split iterator harmlessly
# (the loader's split_hn=True ensures we ARE in split mode here).
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT


def _per_layer_metrics(
    X3: np.ndarray,
    quadrants: np.ndarray,
    layer_idxs: list[int],
) -> pd.DataFrame:
    """One row per layer: PC1+PC2 explained-variance, silhouette over
    quadrant labels in PC1-PC2 space, between-centroid std per PC."""
    rows: list[dict] = []
    quad_strings = quadrants.astype(str)
    unique_q = np.unique(quad_strings)
    can_silhouette = len(unique_q) >= 2
    for li, layer in enumerate(layer_idxs):
        X = X3[:, li, :]
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        if can_silhouette:
            sil = float(silhouette_score(Y, quad_strings, metric="euclidean"))
        else:
            sil = float("nan")
        # Between-centroid std on each PC.
        cent_pc1 = []
        cent_pc2 = []
        for q in unique_q:
            mask = quad_strings == q
            if mask.sum() == 0:
                continue
            cent_pc1.append(float(Y[mask, 0].mean()))
            cent_pc2.append(float(Y[mask, 1].mean()))
        between_pc1 = float(np.std(cent_pc1, ddof=0)) if cent_pc1 else 0.0
        between_pc2 = float(np.std(cent_pc2, ddof=0)) if cent_pc2 else 0.0
        rows.append({
            "layer": int(layer),
            "pc1_var": float(var[0]),
            "pc2_var": float(var[1]),
            "silhouette": sil,
            "between_centroid_std_pc1": between_pc1,
            "between_centroid_std_pc2": between_pc2,
        })
    return pd.DataFrame(rows)


def _plot_per_model_trajectory(
    metrics: pd.DataFrame,
    short_name: str,
    out_path: Path,
) -> None:
    """Five-panel trajectory: silhouette, PC1+PC2 explained var,
    between-centroid std on each PC."""
    _use_cjk_font()
    layers = metrics["layer"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(layers, metrics["silhouette"], color="#222", linewidth=1.4)
    ax.scatter(layers, metrics["silhouette"], s=12, color="#222")
    ax.axhline(0, color="#bbb", linewidth=0.5, zorder=0)
    ax.set_ylabel("silhouette score\n(quadrants in PC1-PC2)")
    ax.set_title(f"{short_name} — quadrant separation by layer")

    ax = axes[0, 1]
    ax.plot(layers, metrics["pc1_var"] * 100, label="PC1", color="#3b6ea5", linewidth=1.4)
    ax.plot(layers, metrics["pc2_var"] * 100, label="PC2", color="#c25a22", linewidth=1.4)
    ax.set_ylabel("explained variance (%)")
    ax.set_title("PCA spectrum (top-2)")
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[1, 0]
    ax.plot(layers, metrics["between_centroid_std_pc1"], color="#3b6ea5", linewidth=1.4)
    ax.scatter(layers, metrics["between_centroid_std_pc1"], s=12, color="#3b6ea5")
    ax.set_xlabel("probe layer")
    ax.set_ylabel("between-centroid std")
    ax.set_title("PC1 quadrant spread")

    ax = axes[1, 1]
    ax.plot(layers, metrics["between_centroid_std_pc2"], color="#c25a22", linewidth=1.4)
    ax.scatter(layers, metrics["between_centroid_std_pc2"], s=12, color="#c25a22")
    ax.set_xlabel("probe layer")
    ax.set_title("PC2 quadrant spread")

    fig.suptitle(
        f"v3 layer-wise emergence — {short_name}: "
        f"layers {layers.min()} → {layers.max()}",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_quartile_pca_strip(
    df: pd.DataFrame,
    X3: np.ndarray,
    layer_idxs: list[int],
    short_name: str,
    out_path: Path,
) -> None:
    """Four PCA scatter panels at the 25/50/75/100% depth-percentile
    layers — visual companion to the metric trajectories."""
    _use_cjk_font()
    quartile_idxs = [
        len(layer_idxs) // 4,
        len(layer_idxs) // 2,
        3 * len(layer_idxs) // 4,
        len(layer_idxs) - 1,
    ]
    quadrants = df["quadrant"].to_numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, li in zip(axes, quartile_idxs):
        layer = layer_idxs[li]
        X = X3[:, li, :]
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        for q in QUADRANT_ORDER:
            mask = quadrants == q
            if not mask.any():
                continue
            ax.scatter(
                Y[mask, 0], Y[mask, 1],
                s=10, color=QUADRANT_COLORS[q], alpha=0.7,
                edgecolor="none", label=q,
            )
        # Centroid stars.
        for q in QUADRANT_ORDER:
            mask = quadrants == q
            if not mask.any():
                continue
            cent = Y[mask].mean(axis=0)
            ax.plot(
                cent[0], cent[1], marker="*", markersize=15,
                color=QUADRANT_COLORS[q],
                markeredgecolor="black", markeredgewidth=0.8, zorder=5,
            )
        ax.set_title(
            f"layer {layer}\n"
            f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)

    axes[-1].legend(loc="best", fontsize=8, frameon=False, title="quadrant")
    fig.suptitle(
        f"v3 PCA at depth quartiles — {short_name} "
        f"({X3.shape[0]} kaomoji-bearing rows)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_comparison(
    per_model: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Overlay silhouette + between-centroid-std trajectories with x-axis
    normalized to fractional depth so models with different layer
    counts (gemma 56, qwen 60) line up. Shows whether the affect
    representation emerges at the same relative depth across
    architectures."""
    _use_cjk_font()
    palette = {"gemma": "#1f77b4", "qwen": "#d62728", "ministral": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for short, metrics in per_model.items():
        layers = metrics["layer"].to_numpy()
        depth = (layers - layers.min()) / (layers.max() - layers.min())
        ax.plot(depth, metrics["silhouette"],
                color=palette.get(short, "#666"), linewidth=1.6, label=short)
        ax.scatter(depth, metrics["silhouette"],
                   s=14, color=palette.get(short, "#666"))
    ax.axhline(0, color="#bbb", linewidth=0.5, zorder=0)
    ax.set_xlabel("fractional depth  (layer normalized to [0, 1])")
    ax.set_ylabel("silhouette score (quadrants in PC1-PC2)")
    ax.set_title("quadrant separation")
    ax.legend(loc="best", fontsize=9, frameon=False)

    ax = axes[1]
    for short, metrics in per_model.items():
        layers = metrics["layer"].to_numpy()
        depth = (layers - layers.min()) / (layers.max() - layers.min())
        ax.plot(depth, metrics["between_centroid_std_pc1"],
                color=palette.get(short, "#666"), linewidth=1.6, label=f"{short} PC1")
    ax.set_xlabel("fractional depth")
    ax.set_ylabel("between-centroid std on PC1")
    ax.set_title("PC1 spread")
    ax.legend(loc="best", fontsize=9, frameon=False)

    ax = axes[2]
    for short, metrics in per_model.items():
        layers = metrics["layer"].to_numpy()
        depth = (layers - layers.min()) / (layers.max() - layers.min())
        ax.plot(depth, metrics["between_centroid_std_pc2"],
                color=palette.get(short, "#666"), linewidth=1.6, label=f"{short} PC2")
    ax.set_xlabel("fractional depth")
    ax.set_ylabel("between-centroid std on PC2")
    ax.set_title("PC2 spread")
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle("v3 layer-wise emergence — cross-model comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rebuild = "--rebuild" in sys.argv

    # Default: process every model with v3 data on disk. Setting
    # LLMOJI_MODEL still works but is no longer required — the script
    # produces both per-model and cross-model figures in one shot.
    candidates = [name for name, M in MODEL_REGISTRY.items()
                  if M.emotional_data_path.exists()]
    env_choice = current_model().short_name
    # If the user explicitly set $LLMOJI_MODEL, restrict to that one.
    import os
    if "LLMOJI_MODEL" in os.environ:
        candidates = [env_choice]

    print(f"models with v3 data: {candidates}")
    if not candidates:
        print("no v3 runs found; nothing to do")
        return

    per_model_metrics: dict[str, pd.DataFrame] = {}
    for short in candidates:
        M = resolve_model(short)
        print(f"\n=== {short} ===")
        print(f"loading multi-layer h_mean tensor (cached at "
              f"data/local/cache/v3_{short}_h_mean_all_layers.npz)...")
        df, X3, layer_idxs = load_emotional_features_all_layers(
            short, split_hn=True, rebuild=rebuild,
        )
        if len(df) == 0:
            print(f"  no kaomoji-bearing rows for {short}; skipping")
            continue
        print(f"  {len(df)} rows × {len(layer_idxs)} layers × {X3.shape[2]}-dim hidden")

        print("  computing per-layer metrics...")
        metrics = _per_layer_metrics(X3, df["quadrant"].to_numpy(), layer_idxs)
        per_model_metrics[short] = metrics

        # Print top-3 layers by silhouette so the headline finding shows up
        # in the run log without needing the figure.
        top = metrics.sort_values("silhouette", ascending=False).head(5)
        print(f"  top-5 layers by silhouette:")
        for _, r in top.iterrows():
            print(f"    L{int(r['layer']):>2}  sil={r['silhouette']:+.3f}  "
                  f"PC1var={r['pc1_var']*100:5.2f}%  PC2var={r['pc2_var']*100:5.2f}%  "
                  f"betw(PC1)={r['between_centroid_std_pc1']:.2f}  "
                  f"betw(PC2)={r['between_centroid_std_pc2']:.2f}")

        # Save per-layer metrics TSV.
        M.figures_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = M.figures_dir / "v3_layerwise_emergence.tsv"
        metrics.to_csv(tsv_path, sep="\t", index=False)
        print(f"  wrote {tsv_path}")

        traj_path = M.figures_dir / "fig_v3_layerwise_emergence.png"
        _plot_per_model_trajectory(metrics, short, traj_path)
        print(f"  wrote {traj_path}")

        strip_path = M.figures_dir / "fig_v3_layerwise_pca_quartiles.png"
        _plot_quartile_pca_strip(df, X3, layer_idxs, short, strip_path)
        print(f"  wrote {strip_path}")

    if len(per_model_metrics) >= 2:
        cross_dir = FIGURES_DIR / "local"
        cross_dir.mkdir(parents=True, exist_ok=True)
        cross_path = cross_dir / "fig_v3_layerwise_emergence_compare.png"
        _plot_cross_model_comparison(per_model_metrics, cross_path)
        print(f"\nwrote {cross_path}")


if __name__ == "__main__":
    main()
