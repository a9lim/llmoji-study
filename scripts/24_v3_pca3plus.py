# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""v3 PC3+ structure + saklas-probe cross-reference.

PC1+PC2 of v3 h_mean absorb valence and arousal (~20% of variance on
gemma, ~23% on qwen). What's in PC3-PC8? This script:

* Fits PCA(8) on v3 h_mean per model.
* Plots PC3 vs PC4 and PC5 vs PC6 colored by quadrant — should NOT
  show clean quadrant separation if PC1+PC2 already absorbed the
  affect signal.
* Computes Pearson correlation of each PC{1..8} with each of the
  five saklas probe scores (happy.sad, angry.calm, confident.uncertain,
  warm.clinical, humorous.serious), at both t0 (whole-generation
  aggregate) and tlast (last-token). Heatmap PC × probe.
* Identifies which PCs carry non-affect signal — large correlations
  with confident.uncertain, warm.clinical, or humorous.serious on PCs
  beyond the first two would surface a hidden non-affect axis the
  model uses for kaomoji choice.

Outputs to figures/local/<short>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from llmoji_study.config import DATA_DIR, PROBES, current_model
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
)


def _attach_probe_columns(df: pd.DataFrame, jsonl_path: Path) -> pd.DataFrame:
    """Re-load JSONL probe-score arrays and merge into df by row_uuid.
    ``load_emotional_features`` strips probe columns by default; we
    pull them back for the cross-reference."""
    raw = pd.read_json(jsonl_path, lines=True)
    # raw has probe_scores_t0 / probe_scores_tlast as length-5 lists,
    # in PROBES order.
    cols = ["row_uuid"]
    if "probe_scores_t0" in raw.columns:
        cols.append("probe_scores_t0")
    if "probe_scores_tlast" in raw.columns:
        cols.append("probe_scores_tlast")
    raw = raw[cols].copy()
    if "probe_scores_t0" in raw.columns:
        t0 = np.asarray(raw["probe_scores_t0"].tolist(), dtype=float)
        for i, p in enumerate(PROBES):
            raw[f"t0_{p}"] = t0[:, i]
        raw = raw.drop(columns=["probe_scores_t0"])
    if "probe_scores_tlast" in raw.columns:
        tl = np.asarray(raw["probe_scores_tlast"].tolist(), dtype=float)
        for i, p in enumerate(PROBES):
            raw[f"tlast_{p}"] = tl[:, i]
        raw = raw.drop(columns=["probe_scores_tlast"])
    out = df.merge(raw, on="row_uuid", how="left")
    return out


def _plot_pca_higher_components(
    coords: np.ndarray,
    var_ratio: np.ndarray,
    quadrants: np.ndarray,
    out_path: Path,
    short_name: str,
) -> None:
    """Three panels: PC1 vs PC2 (reference), PC3 vs PC4, PC5 vs PC6.
    All colored by quadrant. PC1-PC2 should show the Russell
    circumplex; PC3-PC6 should NOT, if affect is fully absorbed up
    front."""
    _use_cjk_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    pairs = [(0, 1), (2, 3), (4, 5)]
    for ax, (i, j) in zip(axes, pairs):
        for q in QUADRANT_ORDER:
            mask = quadrants == q
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, i], coords[mask, j],
                c=QUADRANT_COLORS[q], s=14, alpha=0.7,
                edgecolor="none", label=q,
            )
        for q in QUADRANT_ORDER:
            mask = quadrants == q
            if not mask.any():
                continue
            cent = coords[mask][:, [i, j]].mean(axis=0)
            ax.plot(cent[0], cent[1], marker="*", markersize=18,
                    color=QUADRANT_COLORS[q],
                    markeredgecolor="black", markeredgewidth=0.9, zorder=5)
        ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
        ax.set_xlabel(f"PC{i+1} ({var_ratio[i]*100:.1f}%)")
        ax.set_ylabel(f"PC{j+1} ({var_ratio[j]*100:.1f}%)")
        ax.set_title(f"PC{i+1} vs PC{j+1}")

    axes[-1].legend(loc="best", fontsize=8, frameon=False, title="quadrant")
    fig.suptitle(
        f"v3 PCA higher-order components — {short_name}\n"
        "(PC1-PC2 = affect; PC3+ should NOT show quadrant structure)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pc_probe_correlations(
    pc_probe_t0: np.ndarray,
    pc_probe_tlast: np.ndarray,
    n_components: int,
    out_path: Path,
    short_name: str,
) -> None:
    """PC × probe Pearson correlation heatmap, side-by-side t0 / tlast.
    Strong off-diagonal entries on PC3+ identify non-affect axes the
    model uses (e.g. PC3 lit up on humorous.serious would mean the
    model's third-most-variable internal axis tracks register)."""
    _use_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             sharey=True)

    for ax, mat, title in zip(axes,
                              [pc_probe_t0, pc_probe_tlast],
                              ["t0 (whole-gen mean)", "tlast (final token)"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PROBES)))
        ax.set_xticklabels(PROBES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_components))
        ax.set_yticklabels([f"PC{k+1}" for k in range(n_components)], fontsize=8)
        ax.set_title(title)
        for i in range(n_components):
            for j in range(len(PROBES)):
                v = mat[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.5 else "#222")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"v3 PCA component × saklas probe correlation — {short_name}\n"
        "(which PCs absorb which probe direction?)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no v3 data at {M.emotional_data_path}")
        sys.exit(1)

    layer_label = "max" if M.preferred_layer is None else f"L{M.preferred_layer}"
    print(f"model: {M.short_name}")
    print(f"loading v3 hidden-state features (h_mean, layer={layer_label})...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
        layer=M.preferred_layer,
    )
    print(f"  {len(df)} rows, X {X.shape}")
    df = _attach_probe_columns(df, M.emotional_data_path)
    print(f"  attached probe columns; {len(df)} rows after merge")

    n_components = 8
    print(f"\nfitting PCA(n_components={n_components})...")
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print("  explained-variance spectrum:")
    for k, v in enumerate(var, 1):
        print(f"    PC{k}: {v*100:6.2f}%  (cumulative {var[:k].sum()*100:5.2f}%)")

    M.figures_dir.mkdir(parents=True, exist_ok=True)

    # Higher-component scatters.
    pca_path = M.figures_dir / "fig_v3_pca3plus_quadrants.png"
    _plot_pca_higher_components(
        coords, var, df["quadrant"].to_numpy(),
        pca_path, M.short_name,
    )
    print(f"wrote {pca_path}")

    # PC × probe correlation.
    pc_probe_t0 = np.zeros((n_components, len(PROBES)))
    pc_probe_tlast = np.zeros((n_components, len(PROBES)))
    for k in range(n_components):
        for j, probe in enumerate(PROBES):
            t0_col = f"t0_{probe}"
            tl_col = f"tlast_{probe}"
            if t0_col in df.columns:
                t0 = df[t0_col].to_numpy()
                mask = ~np.isnan(t0)
                if mask.sum() >= 3:
                    pc_probe_t0[k, j] = float(pearsonr(coords[mask, k], t0[mask])[0])
            if tl_col in df.columns:
                tl = df[tl_col].to_numpy()
                mask = ~np.isnan(tl)
                if mask.sum() >= 3:
                    pc_probe_tlast[k, j] = float(pearsonr(coords[mask, k], tl[mask])[0])

    print("\nPC × probe Pearson correlation (t0):")
    print(f"  {'':4} " + " ".join(f"{p[:10]:>10}" for p in PROBES))
    for k in range(n_components):
        row = " ".join(f"{pc_probe_t0[k, j]:+.2f}".rjust(10) for j in range(len(PROBES)))
        print(f"  PC{k+1}: {row}")

    print("\nPC × probe Pearson correlation (tlast):")
    print(f"  {'':4} " + " ".join(f"{p[:10]:>10}" for p in PROBES))
    for k in range(n_components):
        row = " ".join(f"{pc_probe_tlast[k, j]:+.2f}".rjust(10) for j in range(len(PROBES)))
        print(f"  PC{k+1}: {row}")

    corr_path = M.figures_dir / "fig_v3_pca_probe_correlations.png"
    _plot_pc_probe_correlations(
        pc_probe_t0, pc_probe_tlast, n_components,
        corr_path, M.short_name,
    )
    print(f"wrote {corr_path}")

    # Save TSV.
    tsv_rows = []
    for k in range(n_components):
        row = {"component": f"PC{k+1}", "explained_variance_pct": float(var[k] * 100)}
        for j, probe in enumerate(PROBES):
            row[f"t0_{probe}_r"] = float(pc_probe_t0[k, j])
            row[f"tlast_{probe}_r"] = float(pc_probe_tlast[k, j])
        tsv_rows.append(row)
    tsv_path = M.figures_dir / "v3_pca_probe_correlations.tsv"
    pd.DataFrame(tsv_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
