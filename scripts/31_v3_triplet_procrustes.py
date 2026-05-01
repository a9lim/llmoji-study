"""v3 triplet Procrustes: cross-model quadrant-geometry alignment
across gemma / qwen / ministral, with the rule-3-redesign HN split
active. PCA(3) is fit per model on its preferred-layer hidden state;
the three figures below slice that decomposition through different
PC pairs.

Three 2×2 figures:

  fig_v3_triplet_procrustes_pc12.png    PC1 × PC2  (Russell circumplex)
  fig_v3_triplet_procrustes_pc13.png    PC1 × PC3
  fig_v3_triplet_procrustes_pc23.png    PC2 × PC3

Each 2×2 panel:

  top-left      gemma centroids in the active PC plane at L31
  top-right     qwen centroids in the active PC plane at L61
  bottom-left   ministral centroids in the active PC plane at L21
  bottom-right  Procrustes overlay: gemma centered (○), qwen
                rotated to fit gemma (△), ministral rotated to fit
                gemma (□). Same per-quadrant color across the three
                marker styles; lines connect same-quadrant centroids
                across models so divergence reads visually.

PC sign-indeterminacy means rotations near ±180° just indicate that
some model assigns opposite signs to the active PCs vs gemma — a
rigid axis flip, not a divergence finding. The reported rotation is
the raw orthogonal-Procrustes angle so the number is honest.

Quadrant labels are HP / LP / HN-D / HN-S / LN / NB (rule-3-redesign
6-category split). Untagged HN rows (hn06 / hn15 / hn17, three
borderline reads) drop out before PCA.

Output:
  figures/local/cross_model/fig_v3_triplet_procrustes_pc{12,13,23}.png
  figures/local/cross_model/fig_v3_triplet_procrustes.png  (alias for _pc12)
  figures/local/cross_model/v3_triplet_procrustes_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    KAOMOJI_START_CHARS,
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    _hn_split_map,
    _use_cjk_font,
)
from llmoji_study.hidden_state_analysis import load_hidden_features_all_layers

MODELS = ("gemma", "qwen", "ministral")
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT
_HN_SPLIT = _hn_split_map()


def _quad(pid: str) -> str:
    if len(pid) < 2:
        return "??"
    base = pid[:2].upper()
    if base == "HN":
        return _HN_SPLIT.get(pid, "??")
    return base


def _load_at_preferred_layer(short: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Load v3 h_mean at this model's preferred layer, with HN split.
    Drops kaomoji-empty + untagged-HN rows. Returns (df, X) row-aligned."""
    M = MODEL_REGISTRY[short]
    cache = DATA_DIR / "cache" / f"v3_{short}_h_mean_all_layers.npz"
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which="h_first", cache_path=cache,
    )
    layer = M.preferred_layer if M.preferred_layer is not None else max(layer_idxs)
    li = layer_idxs.index(layer)
    X = X3[:, li, :]

    # Kaomoji-start filter + HN-split tagging
    from llmoji.taxonomy import canonicalize_kaomoji
    df = df.assign(
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
        quadrant=df["prompt_id"].map(_quad),
    )
    keep = np.asarray([
        isinstance(fw, str) and len(fw) > 0 and fw[0] in KAOMOJI_START_CHARS
        and q in QUADRANT_ORDER
        for fw, q in zip(df["first_word"], df["quadrant"])
    ])
    df = df.loc[keep].reset_index(drop=True)
    X = X[keep]
    return df, X, layer


def _fit_pca3(df: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Fit PCA(3) on the model's hidden states and return the (n, 3)
    projection plus the fitted PCA object (for its
    `explained_variance_ratio_`)."""
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    return Y, pca


def _centroids_in_plane(
    df: pd.DataFrame, Y3: np.ndarray, pc_pair: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Per-quadrant centroids in the 2D plane spanned by the given PC
    pair (e.g. (0, 1) → PC1 × PC2; (0, 2) → PC1 × PC3)."""
    i, j = pc_pair
    centroids: dict[str, np.ndarray] = {}
    for q in QUADRANT_ORDER:
        mask = (df["quadrant"] == q).to_numpy()
        if mask.any():
            sub = Y3[mask][:, [i, j]]
            centroids[q] = sub.mean(axis=0)
    return centroids


# Back-compat alias: a few callers (and earlier docs) reference the old
# PC1×PC2-specific helper. New code should use _fit_pca3 +
# _centroids_in_plane.
def _centroids_in_pca2(df: pd.DataFrame, X: np.ndarray) -> tuple[
    dict[str, np.ndarray], np.ndarray, PCA
]:
    Y3, pca = _fit_pca3(df, X)
    cents = _centroids_in_plane(df, Y3, (0, 1))
    return cents, Y3[:, :2], pca


def _procrustes_align(
    centroids_a: dict[str, np.ndarray],
    centroids_b: dict[str, np.ndarray],
) -> dict[str, object]:
    """Find rotation R such that B-centroids @ R best fits A-centroids
    after centering. Rescale aligned B to match A's norm."""
    common = [q for q in QUADRANT_ORDER if q in centroids_a and q in centroids_b]
    if len(common) < 2:
        return {"common": common, "rotation_deg": float("nan"),
                "residual": float("nan"), "Cg": None, "Cq_aligned": None}
    Ca = np.asarray([centroids_a[q] for q in common])
    Cb = np.asarray([centroids_b[q] for q in common])
    Ca_c = Ca - Ca.mean(axis=0, keepdims=True)
    Cb_c = Cb - Cb.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(Cb_c, Ca_c)
    Cb_aligned = Cb_c @ R
    norm_a = float(np.linalg.norm(Ca_c))
    norm_b = float(np.linalg.norm(Cb_aligned))
    if norm_b > 0:
        Cb_aligned = Cb_aligned * (norm_a / norm_b)
    residual = float(np.linalg.norm(Cb_aligned - Ca_c))
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {
        "common": common,
        "rotation_deg": angle,
        "residual": residual,
        "Ca_centered": Ca_c,
        "Cb_aligned": Cb_aligned,
    }


def _draw_centroids(ax, centroids: dict[str, np.ndarray], title: str) -> None:
    for q, pt in centroids.items():
        color = QUADRANT_COLORS.get(q, "#666")
        ax.scatter(pt[0], pt[1], c=color, s=200,
                   edgecolor="black", linewidth=0.8)
        ax.annotate(q, (pt[0], pt[1]),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)


REFERENCE_MODEL = "gemma"   # all alignments computed against this model

# Marker shapes for the overlay panel — visually distinct enough to
# read at small sizes, kept consistent with figure conventions
# elsewhere (○ / △ / □ is the standard 3-way differentiator).
MARKERS = {
    "gemma":     "o",   # circle
    "qwen":      "^",   # triangle
    "ministral": "s",   # square
}


def _draw_overlay_triplet(
    ax,
    common: list[str],
    cents_by_model: dict[str, np.ndarray],   # model -> (n_common, 2) coords in shared frame
    pair_summaries: dict[str, tuple[float, float]],  # model -> (rot_deg, residual)
) -> None:
    # Per-quadrant: scatter for each model with its own marker shape;
    # connect the same-quadrant points across the three models with a
    # thin line so deviation reads visually.
    for i, q in enumerate(common):
        color = QUADRANT_COLORS.get(q, "#666")
        # connecting line — gemma → qwen → ministral
        xs = [cents_by_model["gemma"][i, 0],
              cents_by_model["qwen"][i, 0],
              cents_by_model["ministral"][i, 0]]
        ys = [cents_by_model["gemma"][i, 1],
              cents_by_model["qwen"][i, 1],
              cents_by_model["ministral"][i, 1]]
        ax.plot(xs, ys, color=color, linestyle="--",
                linewidth=0.9, alpha=0.55, zorder=1)
        for short in MODELS:
            pt = cents_by_model[short][i]
            ax.scatter(pt[0], pt[1], c=color, s=200,
                       marker=MARKERS[short], edgecolor="black",
                       linewidth=0.8, zorder=3)
        # Quadrant label anchored on gemma's position (the reference)
        gpt = cents_by_model["gemma"][i]
        ax.annotate(q, (gpt[0], gpt[1]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold")

    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.set_xlabel("aligned PC1")
    ax.set_ylabel("aligned PC2")

    # Legend: one entry per model + marker
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=MARKERS[m], color="w",
               markerfacecolor="#444", markeredgecolor="black",
               markersize=10, label=m, markeredgewidth=0.8)
        for m in MODELS
    ]
    ax.legend(handles=legend_handles, loc="best",
              fontsize=9, frameon=False, title="model (marker)")

    # Title with both pair Procrustes summaries
    qwen_rot, qwen_res = pair_summaries["qwen"]
    min_rot, min_res = pair_summaries["ministral"]
    ax.set_title(
        f"all three overlaid on gemma\n"
        f"qwen→gemma: rot {qwen_rot:+.1f}°, residual {qwen_res:.2f}   "
        f"ministral→gemma: rot {min_rot:+.1f}°, residual {min_res:.2f}",
        fontsize=10,
    )


# Three planes through PCA(3): the canonical PC1×PC2 plane (the
# Russell-circumplex view) and two PC3-bearing planes that test
# whether structure orthogonal to the affect plane is shared
# across models.
PC_PAIRS: list[tuple[tuple[int, int], str, str]] = [
    ((0, 1), "pc12", "PC1 × PC2"),
    ((0, 2), "pc13", "PC1 × PC3"),
    ((1, 2), "pc23", "PC2 × PC3"),
]


def _build_figure_for_pair(
    per_model: dict[str, dict],
    pc_pair: tuple[int, int],
    plane_label: str,
    out_path: Path,
) -> dict:
    """Build the 2×2 figure for one PC pair. Returns the per-pair
    summary dict (per-model centroids, rotations, residuals)."""
    pi, pj = pc_pair
    pc_a = pi + 1
    pc_b = pj + 1

    cents_by_model: dict[str, dict[str, np.ndarray]] = {
        m: _centroids_in_plane(per_model[m]["df"], per_model[m]["Y3"], pc_pair)
        for m in MODELS
    }
    common = [q for q in QUADRANT_ORDER
              if all(q in cents_by_model[m] for m in MODELS)]

    ref_cents = cents_by_model[REFERENCE_MODEL]
    aligned: dict[str, np.ndarray] = {}
    pair_summaries: dict[str, tuple[float, float]] = {}
    Cg = np.asarray([ref_cents[q] for q in common])
    Cg_c = Cg - Cg.mean(axis=0, keepdims=True)
    aligned[REFERENCE_MODEL] = Cg_c
    pair_summaries[REFERENCE_MODEL] = (0.0, 0.0)
    for m in MODELS:
        if m == REFERENCE_MODEL:
            continue
        result = _procrustes_align(ref_cents, cents_by_model[m])
        aligned[m] = result["Cb_aligned"]
        pair_summaries[m] = (
            float(result["rotation_deg"]),
            float(result["residual"]),
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panel_for = {
        "gemma":     axes[0, 0],
        "qwen":      axes[0, 1],
        "ministral": axes[1, 0],
    }
    for m in MODELS:
        ax = panel_for[m]
        info = per_model[m]
        evr = info["pca"].explained_variance_ratio_
        _draw_centroids(
            ax, cents_by_model[m],
            f"{m} L{info['layer']}  "
            f"(PC{pc_a} {evr[pi]*100:.1f}%, PC{pc_b} {evr[pj]*100:.1f}%, "
            f"n={info['n']})",
        )
        # axis labels match the active plane
        ax.set_xlabel(f"PC{pc_a}")
        ax.set_ylabel(f"PC{pc_b}")

    _draw_overlay_triplet(axes[1, 1], common, aligned, pair_summaries)
    axes[1, 1].set_xlabel(f"aligned PC{pc_a}")
    axes[1, 1].set_ylabel(f"aligned PC{pc_b}")

    fig.suptitle(
        f"v3 cross-model quadrant geometry — {plane_label} "
        f"(HN split: HN-D anger / HN-S fear)",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "plane": plane_label,
        "pc_indices": list(pc_pair),
        "common_quadrants": common,
        "models": {
            m: {
                "centroids": {q: cents_by_model[m][q].tolist()
                              for q in cents_by_model[m]},
                "procrustes_rotation_deg": pair_summaries[m][0],
                "procrustes_residual": pair_summaries[m][1],
            }
            for m in MODELS
        },
    }


def main() -> None:
    _use_cjk_font()
    print("Loading all three models at their preferred layers...")
    loaded: dict[str, tuple[pd.DataFrame, np.ndarray, int]] = {}
    for m in MODELS:
        df, X, layer = _load_at_preferred_layer(m)
        loaded[m] = (df, X, layer)
        print(f"  {m}: L{layer}, {len(df)} rows after split-mode filter, X{X.shape}")

    # Per-model: fit PCA(3) once. The three PC-pair figures slice this
    # 3-component decomposition, so PC1+2+3 explained-variance is a
    # single per-model property regardless of which plane we plot.
    per_model: dict[str, dict] = {}
    for m in MODELS:
        df, X, layer = loaded[m]
        Y3, pca = _fit_pca3(df, X)
        per_model[m] = {
            "df": df,
            "Y3": Y3,
            "pca": pca,
            "layer": layer,
            "n": len(df),
        }

    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "reference_model": REFERENCE_MODEL,
        "models": {
            m: {
                "layer": per_model[m]["layer"],
                "n_rows": per_model[m]["n"],
                "explained_variance_ratio_pc123": [
                    float(v) for v in per_model[m]["pca"].explained_variance_ratio_
                ],
            } for m in MODELS
        },
        "planes": [],
    }

    print("\n=== Procrustes alignment to gemma per PC-pair ===")
    print(f"{'plane':10s} {'model':12s} {'rot°':>10s} {'residual':>12s}")
    for pc_pair, slug, label in PC_PAIRS:
        out_png = out_dir / f"fig_v3_triplet_procrustes_{slug}.png"
        plane_summary = _build_figure_for_pair(per_model, pc_pair, label, out_png)
        summary["planes"].append(plane_summary)
        print(f"  wrote {out_png}")
        for m in MODELS:
            rot = plane_summary["models"][m]["procrustes_rotation_deg"]
            res = plane_summary["models"][m]["procrustes_residual"]
            print(f"{label:10s} {m:12s} {rot:+10.2f} {res:12.3f}")
        print()

    out_json = out_dir / "v3_triplet_procrustes_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # Back-compat: keep the old unsuffixed PC1×PC2 filename pointing at
    # the new pc12 file. A symlink would be nicer but copying is more
    # portable across filesystems.
    legacy = out_dir / "fig_v3_triplet_procrustes.png"
    pc12_path = out_dir / "fig_v3_triplet_procrustes_pc12.png"
    if pc12_path.exists():
        legacy.write_bytes(pc12_path.read_bytes())
        print(f"refreshed legacy alias {legacy}")


if __name__ == "__main__":
    main()
