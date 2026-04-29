"""v3 per-face cosine heatmap (descriptive, post-run).

Single figure:

  figures/local/<short>/fig_v3_face_cosine_heatmap.png:
      Pairwise cosine similarity between per-face mean h_mean
      vectors (grand-mean centered), rows/cols sorted by dominant
      quadrant with quadrant boundaries drawn. Tick colors are
      per-face RGB blends of QUADRANT_COLORS, weighted by per-
      quadrant emission counts.

This script used to produce three figures — per-face PCA
(`fig_v3_face_pca_by_quadrant`), per-face probe scatter
(`fig_v3_face_probe_scatter`), and the cosine heatmap. Both
scatter outputs were retired 2026-04-29 once the interactive 3D
HTMLs landed:

  - PCA → ``figures/local/cross_model/fig_v3_extension_3d_pca_per_face.html``
    (PC1 × PC2 × PC3, rotatable, hover for face + counts).
  - Probe trio → ``figures/local/cross_model/fig_v3_extension_3d_probes_per_face.html``
    (happy × fearful × angry, same per-face conventions).

The cosine heatmap stays here — it's a different analysis, not a
probe-direction scatter, and the 3D HTMLs don't cover it.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmoji_study.config import (
    DATA_DIR,
    current_model,
)
from llmoji_study.emotional_analysis import (
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
    mix_quadrant_color,
    per_face_dominant_quadrant,
    per_face_quadrant_weights,
)


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

    quadrant = per_face_dominant_quadrant(df)
    fdf["quadrant"] = [quadrant[fw] for fw in fdf["first_word"]]
    fdf["q_order"] = fdf["quadrant"].map({q: i for i, q in enumerate(QUADRANT_ORDER)})
    sort_idx = fdf.sort_values(["q_order", "n"], ascending=[True, False]).index.to_numpy()
    M_sorted = M[sort_idx]
    fdf_sorted = fdf.iloc[sort_idx].reset_index(drop=True)

    grand_mean = M_sorted.mean(axis=0, keepdims=True)
    M_centered = M_sorted - grand_mean

    norms = np.linalg.norm(M_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M_n = M_centered / norms
    C = M_n @ M_n.T

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

    weights = per_face_quadrant_weights(df)
    for tick, fw in zip(ax.get_yticklabels(), fdf_sorted["first_word"]):
        tick.set_color(mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        ))
    for tick, fw in zip(ax.get_xticklabels(), fdf_sorted["first_word"]):
        tick.set_color(mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        ))

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
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}")
        sys.exit(1)
    _use_cjk_font()

    layer_label = "max" if M.preferred_layer is None else f"L{M.preferred_layer}"
    print(f"model: {M.short_name}")
    print(f"loading v3 hidden-state features (which=h_mean, layer={layer_label})...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
        layer=M.preferred_layer,
    )
    df = df[df["first_word"].notna() & (df["first_word"] != "")].reset_index(drop=True)
    print(f"  {len(df)} kaomoji-bearing rows; "
          f"{df['first_word'].nunique()} unique faces; X {X.shape}")

    quadrant = per_face_dominant_quadrant(df)
    counts = Counter(quadrant.values())
    print("  faces by dominant quadrant:",
          {q: counts.get(q, 0) for q in QUADRANT_ORDER})

    M.figures_dir.mkdir(parents=True, exist_ok=True)

    out = M.figures_dir / "fig_v3_face_cosine_heatmap.png"
    print("\nplotting per-face cosine heatmap (centered)...")
    s = plot_face_cosine_heatmap(df, X, out)
    print(f"  wrote {out}")
    print(f"  {s['n_faces']} faces in heatmap")


if __name__ == "__main__":
    main()
