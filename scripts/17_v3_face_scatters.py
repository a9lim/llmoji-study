"""v3 face-level scatters + cosine heatmap (descriptive, post-run).

Three figures, all on the v3 data only:
  - figures/fig_v3_face_pca_by_quadrant.png:
    Per-kaomoji mean h_mean (averaged-over-tokens hidden state) ->
    PCA -> 2D scatter colored by the face's dominant emission
    quadrant (HP / LP / HN / LN / NB).
  - figures/fig_v3_face_probe_scatter.png:
    Per-kaomoji mean (happy.sad, angry.calm) using probe_means
    (whole-generation aggregates), same dominant-quadrant coloring.
  - figures/fig_v3_face_cosine_heatmap.png:
    Pairwise cosine similarity between per-face mean h_mean
    vectors, centered (grand mean subtracted), rows/cols sorted by
    dominant quadrant with quadrant boundaries drawn.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    FIGURES_DIR,
)
from llmoji.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
    per_face_dominant_quadrant,
)


def _add_quadrant_legend(ax) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   color=QUADRANT_COLORS[q], label=q)
        for q in QUADRANT_ORDER
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9, title="dominant quadrant")


def plot_face_pca_by_quadrant(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: Path,
    *,
    annotate_top: int = 30,
) -> dict:
    """Per-face mean h_mean -> PCA -> scatter colored by dominant quadrant."""
    df = df.reset_index(drop=True)
    rows = []
    means = []
    for fw, sub in df.groupby("first_word"):
        idxs = sub.index.to_numpy()
        rows.append({"first_word": fw, "n": len(sub)})
        means.append(X[idxs].mean(axis=0))
    fdf = pd.DataFrame(rows)
    M = np.asarray(means)

    pca = PCA(n_components=2)
    Y = pca.fit_transform(M)

    quadrant = per_face_dominant_quadrant(df)
    colors = [QUADRANT_COLORS[quadrant.get(fw, "NB")] for fw in fdf["first_word"]]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    top_idx = np.argsort(-fdf["n"].to_numpy())[:annotate_top]
    for i in top_idx:
        ax.annotate(
            fdf.iloc[i]["first_word"],
            xy=(Y[i, 0], Y[i, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=10, color="#222",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(
        f"v3 per-kaomoji h_mean PCA  ({len(fdf)} kaomoji)\n"
        "colored by dominant emission quadrant"
    )
    _add_quadrant_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_faces": len(fdf),
    }


def plot_face_probe_scatter(
    df: pd.DataFrame,
    out_path: Path,
    *,
    annotate_top: int = 30,
) -> dict:
    """Per-face mean (happy.sad, angry.calm) using probe_means dict."""
    rows = []
    for fw, sub in df.groupby("first_word"):
        # probe_means is a dict per row
        hs = np.array([r.get("happy.sad", float("nan")) for r in sub["probe_means"]])
        ac = np.array([r.get("angry.calm", float("nan")) for r in sub["probe_means"]])
        rows.append({
            "first_word": fw,
            "n": len(sub),
            "happy_sad": float(np.nanmean(hs)),
            "angry_calm": float(np.nanmean(ac)),
        })
    fdf = pd.DataFrame(rows)

    quadrant = per_face_dominant_quadrant(df)
    colors = [QUADRANT_COLORS[quadrant.get(fw, "NB")] for fw in fdf["first_word"]]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        fdf["happy_sad"], fdf["angry_calm"],
        c=colors, s=sizes, alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)

    top_idx = np.argsort(-fdf["n"].to_numpy())[:annotate_top]
    for i in top_idx:
        ax.annotate(
            fdf.iloc[i]["first_word"],
            xy=(fdf.iloc[i]["happy_sad"], fdf.iloc[i]["angry_calm"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=10, color="#222",
        )

    ax.set_xlabel("mean happy.sad probe (whole-gen mean)  ->  positive = happy")
    ax.set_ylabel("mean angry.calm probe (whole-gen mean)  ->  positive = angry")
    ax.set_title(
        f"v3 per-kaomoji probe scatter  ({len(fdf)} kaomoji)\n"
        "saklas bipolar probes (whole-generation means), colored by dominant quadrant"
    )
    _add_quadrant_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    from scipy.stats import pearsonr
    r, p = pearsonr(fdf["happy_sad"].to_numpy(), fdf["angry_calm"].to_numpy())
    return {
        "n_faces": len(fdf),
        "n_emissions": int(fdf["n"].sum()),
        "probe_pair_pearson_r": float(r),
        "probe_pair_p": float(p),
    }


def plot_face_cosine_heatmap(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: Path,
) -> dict:
    """Pairwise cosine similarity between per-face mean h_mean vectors,
    grand-mean centered, rows/cols sorted by dominant quadrant."""
    df = df.reset_index(drop=True)
    rows = []
    means = []
    for fw, sub in df.groupby("first_word"):
        idxs = sub.index.to_numpy()
        rows.append({"first_word": fw, "n": len(sub)})
        means.append(X[idxs].mean(axis=0))
    fdf = pd.DataFrame(rows)
    M = np.asarray(means)

    # Sort by dominant quadrant (HP < LP < HN < LN < NB), then by emission count desc within quadrant
    quadrant = per_face_dominant_quadrant(df)
    fdf["quadrant"] = [quadrant[fw] for fw in fdf["first_word"]]
    fdf["q_order"] = fdf["quadrant"].map({q: i for i, q in enumerate(QUADRANT_ORDER)})
    sort_idx = fdf.sort_values(["q_order", "n"], ascending=[True, False]).index.to_numpy()
    M_sorted = M[sort_idx]
    fdf_sorted = fdf.iloc[sort_idx].reset_index(drop=True)

    # Center on grand mean (anti-baseline-collapse, see CLAUDE.md gotcha)
    grand_mean = M_sorted.mean(axis=0, keepdims=True)
    M_centered = M_sorted - grand_mean

    # L2-normalize each row, then dot product = cosine
    norms = np.linalg.norm(M_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M_n = M_centered / norms
    C = M_n @ M_n.T  # (n_faces, n_faces) cosine matrix

    # Compute quadrant boundaries
    q_changes = []
    for i in range(1, len(fdf_sorted)):
        if fdf_sorted.iloc[i]["quadrant"] != fdf_sorted.iloc[i - 1]["quadrant"]:
            q_changes.append(i)

    fig, ax = plt.subplots(figsize=(13, 12))
    vabs = float(max(abs(C.min()), abs(C.max())))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(len(fdf_sorted)))
    ax.set_yticks(range(len(fdf_sorted)))
    ax.set_xticklabels(fdf_sorted["first_word"], rotation=90, fontsize=8)
    ax.set_yticklabels(fdf_sorted["first_word"], fontsize=8)

    # Tint y-axis labels by quadrant color so the structure is readable.
    for tick, q in zip(ax.get_yticklabels(), fdf_sorted["quadrant"]):
        tick.set_color(QUADRANT_COLORS[q])
    for tick, q in zip(ax.get_xticklabels(), fdf_sorted["quadrant"]):
        tick.set_color(QUADRANT_COLORS[q])

    # Quadrant boundary lines
    for i in q_changes:
        ax.axhline(i - 0.5, color="black", linewidth=0.6, alpha=0.7)
        ax.axvline(i - 0.5, color="black", linewidth=0.6, alpha=0.7)

    fig.colorbar(im, ax=ax, shrink=0.7, label="centered cosine similarity")
    ax.set_title(
        f"v3 per-kaomoji h_mean cosine similarity  ({len(fdf_sorted)} kaomoji)\n"
        "rows/cols sorted by dominant quadrant; grand-mean centered; "
        "tick colors = quadrant"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_faces": len(fdf_sorted)}


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}")
        sys.exit(1)
    _use_cjk_font()

    print("loading v3 hidden-state features (which=h_mean)...")
    df, X = load_emotional_features(
        str(EMOTIONAL_DATA_PATH), DATA_DIR,
        experiment=EMOTIONAL_EXPERIMENT, which="h_mean",
    )
    df = df[df["first_word"].notna() & (df["first_word"] != "")].reset_index(drop=True)
    print(f"  {len(df)} kaomoji-bearing rows; "
          f"{df['first_word'].nunique()} unique faces; X {X.shape}")

    quadrant = per_face_dominant_quadrant(df)
    counts = Counter(quadrant.values())
    print("  faces by dominant quadrant:",
          {q: counts.get(q, 0) for q in QUADRANT_ORDER})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out1 = FIGURES_DIR / "fig_v3_face_pca_by_quadrant.png"
    print("\nplotting per-face PCA by quadrant...")
    s1 = plot_face_pca_by_quadrant(df, X, out1)
    print(f"  wrote {out1}")
    print(f"  PC1 {s1['explained_variance_ratio'][0]*100:.1f}%, "
          f"PC2 {s1['explained_variance_ratio'][1]*100:.1f}%")

    out2 = FIGURES_DIR / "fig_v3_face_probe_scatter.png"
    print("\nplotting per-face probe scatter (probe_means)...")
    s2 = plot_face_probe_scatter(df, out2)
    print(f"  wrote {out2}")
    print(f"  {s2['n_faces']} faces, {s2['n_emissions']} total emissions")
    print(f"  Pearson(mean happy.sad, mean angry.calm) across faces: "
          f"r={s2['probe_pair_pearson_r']:+.3f}, p={s2['probe_pair_p']:.3g}")

    out3 = FIGURES_DIR / "fig_v3_face_cosine_heatmap.png"
    print("\nplotting per-face cosine heatmap (centered)...")
    s3 = plot_face_cosine_heatmap(df, X, out3)
    print(f"  wrote {out3}")
    print(f"  {s3['n_faces']} faces in heatmap")


if __name__ == "__main__":
    main()
