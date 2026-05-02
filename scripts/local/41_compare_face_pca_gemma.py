"""Side-by-side per-face PCA: gemma new (cleanliness pass) vs prior.

Each face = one point in PC1×PC2 of the per-face mean h_first vectors,
sized by emission count, colored by per-face quadrant blend (RGB mix
of QUADRANT_COLORS weighted by emission counts per quadrant).

Three panels:
  (a) prior, prior's own PCA basis
  (b) new, new's own PCA basis
  (c) both projected into a SHARED basis (PCA fit on combined per-face means)

Saves to figures/local/gemma/fig_v3_face_pca_pre_vs_post_cleanliness.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
    apply_hn_split,
    load_emotional_features,
    mix_quadrant_color,
    per_face_quadrant_weights,
)
from llmoji_study.hidden_state_analysis import group_mean_vectors


PRIOR_ARCHIVE = DATA_DIR / "archive" / "2026-05-03_pre_cleanliness"


def _load(short: str, jsonl: Path, experiment: str, layer: int):
    """Load + apply HN split (so colors mix in HN-D/HN-S, not aggregate HN).
    Routes prior loads to PRIOR_ARCHIVE so sidecars resolve."""
    data_dir = PRIOR_ARCHIVE if "_pre_cleanliness" in str(jsonl) else DATA_DIR
    df, X = load_emotional_features(
        str(jsonl), data_dir,
        experiment=experiment, which="h_first", layer=layer,
        split_hn=True,
    )
    return df.reset_index(drop=True), X


def _per_face_means(df: pd.DataFrame, X: np.ndarray, *, min_n: int = 2):
    """Return (face_strs, mean_matrix, weight_dicts, counts) — face means
    in full hidden space, plus per-face quadrant blend weights for coloring."""
    keys_df, M, counts = group_mean_vectors(df, X, "first_word", min_count=min_n)
    weights_full = per_face_quadrant_weights(df)
    weights = [weights_full.get(f, {q: 0.0 for q in QUADRANT_ORDER}) for f in keys_df["first_word"]]
    return keys_df["first_word"].to_numpy(), M, weights, counts


def _scatter(ax, Y, faces, weights, counts, title: str, *, label_top_n: int = 8):
    colors = [mix_quadrant_color(w) for w in weights]
    sizes = 60 + 6 * np.sqrt(counts.astype(float))
    ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=sizes, edgecolor="black",
               linewidth=0.4, alpha=0.88)
    # Label the top-N most-emitted faces inline
    top_idx = np.argsort(counts)[::-1][:label_top_n]
    for i in top_idx:
        ax.annotate(
            f"{faces[i]} ({int(counts[i])})",
            xy=(Y[i, 0], Y[i, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#222",
        )
    ax.axhline(0, color="#bbb", linewidth=0.5)
    ax.axvline(0, color="#bbb", linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def run_for(short: str):
    _use_cjk_font()
    M = MODEL_REGISTRY[short]
    layer = M.preferred_layer
    new_jsonl = M.emotional_data_path
    prior_jsonl = PRIOR_ARCHIVE / (
        new_jsonl.stem + "_pre_cleanliness" + new_jsonl.suffix
    )
    prior_exp = M.experiment + "_pre_cleanliness"

    print(f"\n=== {short} (h_first @ L{layer}) ===")
    print(f"loading prior {short} ({prior_jsonl})...")
    df_prior, X_prior = _load(short, prior_jsonl, prior_exp, layer)
    print(f"  {len(df_prior)} rows")
    print(f"loading new {short} ({new_jsonl})...")
    df_new, X_new = _load(short, new_jsonl, M.experiment, layer)
    print(f"  {len(df_new)} rows")

    f_prior, M_prior, w_prior, n_prior = _per_face_means(df_prior, X_prior, min_n=2)
    f_new, M_new, w_new, n_new = _per_face_means(df_new, X_new, min_n=2)
    print(f"prior: {len(f_prior)} unique faces (n>=2); "
          f"new: {len(f_new)} unique faces (n>=2)")

    # Face overlap stats
    set_p, set_n = set(f_prior), set(f_new)
    print(f"face overlap: prior∩new = {len(set_p & set_n)}; "
          f"prior-only = {len(set_p - set_n)}; new-only = {len(set_n - set_p)}")

    # PCA panels
    pca_prior = PCA(n_components=2).fit(M_prior)
    pca_new = PCA(n_components=2).fit(M_new)
    Y_prior_own = pca_prior.transform(M_prior)
    Y_new_own = pca_new.transform(M_new)
    print(f"prior PC1+PC2 explained var: "
          f"{pca_prior.explained_variance_ratio_[0]*100:.1f}% / "
          f"{pca_prior.explained_variance_ratio_[1]*100:.1f}%")
    print(f"new   PC1+PC2 explained var: "
          f"{pca_new.explained_variance_ratio_[0]*100:.1f}% / "
          f"{pca_new.explained_variance_ratio_[1]*100:.1f}%")

    # Shared-basis PCA — fit on stacked face means, project both
    M_both = np.vstack([M_prior, M_new])
    pca_shared = PCA(n_components=2).fit(M_both)
    Y_shared = pca_shared.transform(M_both)
    Y_p_shared = Y_shared[:len(M_prior)]
    Y_n_shared = Y_shared[len(M_prior):]
    print(f"shared PC1+PC2 explained var: "
          f"{pca_shared.explained_variance_ratio_[0]*100:.1f}% / "
          f"{pca_shared.explained_variance_ratio_[1]*100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    _scatter(axes[0, 0], Y_prior_own, f_prior, w_prior, n_prior,
             f"(a) PRIOR gemma — own PCA basis  "
             f"(n_faces={len(f_prior)}, var "
             f"{pca_prior.explained_variance_ratio_[0]*100:.1f}/"
             f"{pca_prior.explained_variance_ratio_[1]*100:.1f}%)")
    _scatter(axes[0, 1], Y_new_own, f_new, w_new, n_new,
             f"(b) NEW gemma (cleanliness) — own PCA basis  "
             f"(n_faces={len(f_new)}, var "
             f"{pca_new.explained_variance_ratio_[0]*100:.1f}/"
             f"{pca_new.explained_variance_ratio_[1]*100:.1f}%)")
    _scatter(axes[1, 0], Y_p_shared, f_prior, w_prior, n_prior,
             f"(c) PRIOR — shared PCA basis (fit on prior ∪ new face means, "
             f"var {pca_shared.explained_variance_ratio_[0]*100:.1f}/"
             f"{pca_shared.explained_variance_ratio_[1]*100:.1f}%)")
    _scatter(axes[1, 1], Y_n_shared, f_new, w_new, n_new,
             "(d) NEW — shared PCA basis (same as (c), apples-to-apples)")

    # Match shared-basis axes between (c) and (d) for visual comparison.
    for ax in (axes[1, 0], axes[1, 1]):
        all_y = np.vstack([Y_p_shared, Y_n_shared])
        m, M_ = all_y.min(0), all_y.max(0)
        pad = 0.10 * (M_ - m)
        ax.set_xlim(m[0] - pad[0], M_[0] + pad[0])
        ax.set_ylim(m[1] - pad[1], M_[1] + pad[1])

    # Legend on (a) showing the quadrant palette.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=QUADRANT_COLORS[q],
                   markeredgecolor="black", markersize=8, label=q)
        for q in ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
    ]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=7, frameon=True)

    fig.suptitle(
        f"{short} per-face hidden-state PCA — pre vs post cleanliness pass\n"
        f"h_first @ L{layer}; prior subsampled to 1 seed/cell ({len(df_prior)} rows); "
        f"new ({len(df_new)} rows); face overlap "
        f"prior∩new={len(set_p & set_n)}, prior-only={len(set_p - set_n)}, "
        f"new-only={len(set_n - set_p)}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = M.figures_dir / "fig_v3_face_pca_pre_vs_post_cleanliness.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    for short in ("gemma", "qwen", "ministral"):
        try:
            run_for(short)
        except FileNotFoundError as e:
            print(f"  [{short}] skipping: {e}")


if __name__ == "__main__":
    # Subsample prior to 1 seed/cell so face counts are comparable to
    # the N=1 pilot. Without this, the "prior" face counts dominate
    # by 8× and the panel comparison is misleading.
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Monkey-patch load helper to subsample prior data to 1-seed/cell
    # (the gate-check script does this too, for the same fairness reason).
    _orig_load = _load

    def _load_with_subsample(short, jsonl, experiment, layer):
        df, X = _orig_load(short, jsonl, experiment, layer)
        if "_pre_cleanliness" not in str(jsonl):
            return df, X
        if "seed" not in df.columns:
            return df, X
        keep = df.sort_values(["prompt_id", "seed"]).drop_duplicates(
            "prompt_id", keep="first"
        ).index
        return df.loc[keep].reset_index(drop=True), X[keep.to_numpy()]

    _load = _load_with_subsample
    main()
