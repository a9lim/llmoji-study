# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""v3 PC × probe correlation cross-reference (probes ALL).

PC1+PC2 of v3 h_mean absorb valence and arousal (~20% on gemma,
~23% on qwen). What's in PC3-PC8?

* Fits PCA(8) on v3 h_mean per model.
* Computes Pearson correlation of each PC{1..8} with each loaded
  probe (5 core PROBES + 3 explicit extension + any auto-discovered
  probes living in `extension_probe_scores_*`),
  at both t0 (state at first generated token) and tlast (final
  generated token). Heatmap PC × probe, side-by-side panels.
* Identifies which PCs carry non-affect signal — large correlations
  with confident.uncertain, warm.clinical, humorous.serious, or
  hallucinating.grounded on PCs beyond the first two surface a
  hidden non-affect axis the model uses for kaomoji choice.

The previous companion figure `fig_v3_pca3plus_quadrants.png`
(per-quadrant PCx scatters at PC1×PC2 / PC1×PC3 / PC2×PC3) was
deleted 2026-04-29; the interactive 3D version
``figures/local/cross_model/fig_v3_extension_3d_pca.html`` covers
the same ground via rotation, and adds a PC3 readout directly.

Outputs to figures/local/<short>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from llmoji_study.config import PROBES, current_model
from llmoji_study.emotional_analysis import (
    _use_cjk_font,
    load_emotional_features_stack,
)


def _attach_probe_columns(df: pd.DataFrame, jsonl_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Re-load JSONL probe scores and merge into df by row_uuid.

    Returns (merged df, probes_in_order). Probes_in_order is
    PROBES + sorted extension probes that actually appear on the
    JSONL — this is the column set both heatmap panels use.
    """
    raw = pd.read_json(jsonl_path, lines=True)
    cols = ["row_uuid"]
    for c in ("probe_scores_t0", "probe_scores_tlast",
              "extension_probe_scores_t0", "extension_probe_scores_tlast"):
        if c in raw.columns:
            cols.append(c)
    raw = raw[cols].copy()

    # Core probes — list-indexed by PROBES order.
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

    # Extension probes — dict-keyed; column union over all rows.
    ext_keys: list[str] = []
    for src_col, prefix in (("extension_probe_scores_t0", "t0"),
                             ("extension_probe_scores_tlast", "tlast")):
        if src_col not in raw.columns:
            continue
        keys: set[str] = set()
        for d in raw[src_col]:
            if isinstance(d, dict):
                keys.update(d.keys())
        for k in sorted(keys):
            raw[f"{prefix}_{k}"] = [
                (d.get(k) if isinstance(d, dict) else None)
                for d in raw[src_col]
            ]
        ext_keys = sorted(set(ext_keys) | keys)
        raw = raw.drop(columns=[src_col])

    out = df.merge(raw, on="row_uuid", how="left")
    probes_all = list(PROBES) + ext_keys
    return out, probes_all


def _plot_pc_probe_correlations(
    pc_probe_t0: np.ndarray,
    pc_probe_tlast: np.ndarray,
    n_components: int,
    probes_all: list[str],
    out_path: Path,
    short_name: str,
) -> None:
    """PC × probe Pearson correlation heatmap, side-by-side t0 / tlast."""
    _use_cjk_font()
    n_probes = len(probes_all)
    # Width scales with probe count so labels stay readable.
    panel_w = max(6.5, 0.55 * n_probes)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w, 4.8 + 0.18 * n_components),
                             sharey=True)

    im = None
    for ax, mat, title in zip(axes,
                              [pc_probe_t0, pc_probe_tlast],
                              ["t0 (h_first)", "tlast (h_last)"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_probes))
        ax.set_xticklabels(probes_all, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(n_components))
        ax.set_yticklabels([f"PC{k+1}" for k in range(n_components)], fontsize=8)
        # Mark the boundary between core PROBES and extension probes.
        sep = len(PROBES) - 0.5
        ax.axvline(sep, color="black", linewidth=0.6)
        ax.set_title(title)
        for i in range(n_components):
            for j in range(n_probes):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > 0.5 else "#222")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"v3 PCA component × probe correlation — {short_name}\n"
        "(left of black line = core PROBES; right = extension; "
        "t0 / tlast scoring at h_first / h_last)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no v3 data at {M.emotional_data_path}")
        sys.exit(1)

    print(f"model: {M.short_name}")
    print("loading v3 hidden-state features (h_first, layer-stack)...")
    df, X = load_emotional_features_stack(
        M.short_name, which="h_first",
    )
    print(f"  {len(df)} rows, X {X.shape} (layer-stack)")
    df, probes_all = _attach_probe_columns(df, M.emotional_data_path)
    print(f"  attached probe columns; {len(df)} rows after merge")
    print(f"  probes considered: {len(probes_all)} "
          f"({len(PROBES)} core + {len(probes_all) - len(PROBES)} extension)")

    n_components = 8
    print(f"\nfitting PCA(n_components={n_components})...")
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print("  explained-variance spectrum:")
    for k, v in enumerate(var, 1):
        print(f"    PC{k}: {v*100:6.2f}%  (cumulative {var[:k].sum()*100:5.2f}%)")

    M.figures_dir.mkdir(parents=True, exist_ok=True)

    pc_probe_t0 = np.full((n_components, len(probes_all)), np.nan)
    pc_probe_tlast = np.full((n_components, len(probes_all)), np.nan)
    for k in range(n_components):
        for j, probe in enumerate(probes_all):
            for src, mat in (("t0", pc_probe_t0), ("tlast", pc_probe_tlast)):
                col = f"{src}_{probe}"
                if col not in df.columns:
                    continue
                vals = df[col].to_numpy(dtype=float)
                mask = ~np.isnan(vals)
                if mask.sum() >= 3:
                    mat[k, j] = float(pearsonr(coords[mask, k], vals[mask])[0])

    print("\nPC × probe Pearson correlation (t0 / h_first):")
    print(f"  {'':4} " + " ".join(f"{p[:10]:>10}" for p in probes_all))
    for k in range(n_components):
        row = " ".join(
            (f"{pc_probe_t0[k, j]:+.2f}" if not np.isnan(pc_probe_t0[k, j]) else "  nan").rjust(10)
            for j in range(len(probes_all))
        )
        print(f"  PC{k+1}: {row}")

    print("\nPC × probe Pearson correlation (tlast / h_last):")
    print(f"  {'':4} " + " ".join(f"{p[:10]:>10}" for p in probes_all))
    for k in range(n_components):
        row = " ".join(
            (f"{pc_probe_tlast[k, j]:+.2f}" if not np.isnan(pc_probe_tlast[k, j]) else "  nan").rjust(10)
            for j in range(len(probes_all))
        )
        print(f"  PC{k+1}: {row}")

    corr_path = M.figures_dir / "fig_v3_pca_probe_correlations.png"
    _plot_pc_probe_correlations(
        pc_probe_t0, pc_probe_tlast, n_components,
        probes_all, corr_path, M.short_name,
    )
    print(f"wrote {corr_path}")

    tsv_rows = []
    for k in range(n_components):
        row = {"component": f"PC{k+1}", "explained_variance_pct": float(var[k] * 100)}
        for j, probe in enumerate(probes_all):
            row[f"t0_{probe}_r"] = float(pc_probe_t0[k, j])
            row[f"tlast_{probe}_r"] = float(pc_probe_tlast[k, j])
        tsv_rows.append(row)
    tsv_path = M.figures_dir / "v3_pca_probe_correlations.tsv"
    pd.DataFrame(tsv_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
