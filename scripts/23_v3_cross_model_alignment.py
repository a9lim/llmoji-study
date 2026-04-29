# pyright: reportArgumentType=false, reportCallIssue=false
"""v3 cross-model representational alignment (gemma ↔ qwen).

The v3 runs use the same 100 prompts × 8 seeds on both models, so
rows pair cleanly by (prompt_id, seed). Three measurements:

* Linear CKA between paired h_mean matrices, overall and per-layer
  (using *fractional depth* on the x-axis so gemma's 56 layers and
  qwen's 60 layers line up). Single number per layer; high CKA =
  the two models put paired prompts in geometrically similar
  configurations within their respective hidden spaces.

* Top-k canonical correlations from sklearn's CCA (n_components=10)
  between (X_gemma, X_qwen). Tells us how many independent shared
  affect axes there are — if only the first 1-2 are high, the shared
  representation is low-rank and probably collapses to valence/arousal.
  If 5-10 stay high, deeper non-affect alignment exists.

* Per-quadrant centroids in shared CCA(2) space, plotted as a
  side-by-side gemma vs qwen geometry comparison. Tests whether the
  Russell circumplex has the same geometric shape across the two
  models' internal representations — if HP/LP/HN/LN/NB land in the
  same relative configuration after CCA alignment, the divergent
  probe-geometry finding (gemma r=−0.94 / qwen r=−0.12 between
  happy.sad and angry.calm) is a probe-extraction artifact rather
  than a genuine architectural difference.

* Procrustes alignment of per-quadrant centroids in each model's
  PCA(2) plane — orthogonal best-fit rotation+scale of qwen's
  centroids onto gemma's. Reports the rotation angle and residual.

Outputs to figures/local/cross_model/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from llmoji_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    KAOMOJI_START_CHARS,
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
)
from llmoji_study.hidden_state_analysis import load_hidden_features_all_layers


# ---------------------------------------------------------------------------
# CKA — kernel form via centered Gram matrices (n×n, faster than n×d×n×d)
# ---------------------------------------------------------------------------


def _gram_centered(X: np.ndarray) -> np.ndarray:
    """Centered linear-kernel Gram matrix. K = X X^T then double-center
    via H K H where H = I - 11^T/n. Equivalent to (X - mean) (X - mean)^T,
    which we compute that way directly to avoid a full Gram first."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc @ Xc.T  # (n, n)


def _cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA from two centered Gram matrices.

    HSIC(K, L) = trace(K L) / (n-1)^2 (constants cancel in ratio)
    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    trace(K L) = sum(K * L) elementwise.
    """
    num = float((K * L).sum())
    den = float(np.sqrt((K * K).sum()) * np.sqrt((L * L).sum()))
    return num / den if den > 0 else 0.0


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Convenience: build Grams + return CKA. For grid computation
    callers should precompute Grams once per layer and pass them in.
    Kept as a single-call API for one-off / smoke-test use."""
    return _cka_from_grams(_gram_centered(X), _gram_centered(Y))


__all__ = ["_linear_cka", "_cka_from_grams", "_gram_centered"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_v3_paired() -> tuple[
    pd.DataFrame, np.ndarray, list[int],
    pd.DataFrame, np.ndarray, list[int],
    np.ndarray,
]:
    """Load gemma and qwen v3 multi-layer h_mean tensors and align rows
    by (prompt_id, seed). Applies the kaomoji-start filter to BOTH
    models so a row is kept only if both models emitted a kaomoji-
    bearing first word.

    Returns ``(df_g, X3_g, layers_g, df_q, X3_q, layers_q, kaomoji_mask)``
    where the dataframes / tensors are in matching paired-row order.
    """
    g_cache = DATA_DIR / "cache" / "v3_gemma_h_mean_all_layers.npz"
    q_cache = DATA_DIR / "cache" / "v3_qwen_h_mean_all_layers.npz"
    Mg = MODEL_REGISTRY["gemma"]
    Mq = MODEL_REGISTRY["qwen"]
    print(f"loading gemma v3 (cache: {g_cache.exists()})...")
    df_g, X3_g, layers_g = load_hidden_features_all_layers(
        Mg.emotional_data_path, DATA_DIR, Mg.experiment,
        which="h_mean", cache_path=g_cache,
    )
    print(f"loading qwen v3 (cache: {q_cache.exists()})...")
    df_q, X3_q, layers_q = load_hidden_features_all_layers(
        Mq.emotional_data_path, DATA_DIR, Mq.experiment,
        which="h_mean", cache_path=q_cache,
    )
    print(f"  gemma {X3_g.shape}, qwen {X3_q.shape}")

    # Pair by (prompt_id, seed). Each model has 800 unique pairs.
    df_g = df_g.assign(_pkey=df_g["prompt_id"].astype(str) + "::" + df_g["seed"].astype(str))
    df_q = df_q.assign(_pkey=df_q["prompt_id"].astype(str) + "::" + df_q["seed"].astype(str))

    common = sorted(set(df_g["_pkey"]).intersection(df_q["_pkey"]))
    g_pos = {k: i for i, k in enumerate(df_g["_pkey"])}
    q_pos = {k: i for i, k in enumerate(df_q["_pkey"])}
    g_idx = np.array([g_pos[k] for k in common])
    q_idx = np.array([q_pos[k] for k in common])

    df_g = df_g.iloc[g_idx].reset_index(drop=True)
    df_q = df_q.iloc[q_idx].reset_index(drop=True)
    X3_g = X3_g[g_idx]
    X3_q = X3_q[q_idx]

    # Kaomoji-start filter — keep rows where BOTH first_words are kaomoji.
    from llmoji.taxonomy import canonicalize_kaomoji
    df_g = df_g.assign(
        quadrant=df_g["prompt_id"].str[:2].str.upper(),
        first_word_canon=df_g["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    df_q = df_q.assign(
        quadrant=df_q["prompt_id"].str[:2].str.upper(),
        first_word_canon=df_q["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )

    def is_kao(s) -> bool:
        return isinstance(s, str) and len(s) > 0 and s[0] in KAOMOJI_START_CHARS

    g_mask = df_g["first_word_canon"].map(is_kao).to_numpy()
    q_mask = df_q["first_word_canon"].map(is_kao).to_numpy()
    both = g_mask & q_mask
    print(f"  paired rows: {len(common)}; both-kaomoji: {int(both.sum())}")

    df_g = df_g.loc[both].reset_index(drop=True)
    df_q = df_q.loc[both].reset_index(drop=True)
    X3_g = X3_g[both]
    X3_q = X3_q[both]
    return df_g, X3_g, layers_g, df_q, X3_q, layers_q, both


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_cka_per_layer(
    cka_grid: np.ndarray,
    layers_g: list[int],
    layers_q: list[int],
    out_path: Path,
) -> None:
    """Per-layer CKA heatmap (gemma layer × qwen layer) plus
    fractional-depth diagonal trace. The diagonal is the
    same-fractional-depth alignment trace; the off-diagonals expose
    cross-depth alignment."""
    _use_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    im = ax.imshow(
        cka_grid, cmap="viridis", aspect="auto",
        vmin=0, vmax=float(cka_grid.max()),
        extent=(0, 1, 0, 1), origin="lower",
    )
    ax.set_xlabel("qwen fractional depth")
    ax.set_ylabel("gemma fractional depth")
    ax.set_title("linear CKA across all (gemma, qwen) layer pairs")
    fig.colorbar(im, ax=ax, shrink=0.7, label="CKA")
    ax.plot([0, 1], [0, 1], color="white", linewidth=0.6, linestyle="--",
            alpha=0.7)

    ax = axes[1]
    # Diagonal trace: at each fractional depth, take CKA between the
    # closest gemma layer and closest qwen layer.
    n_g = len(layers_g)
    n_q = len(layers_q)
    n_steps = max(n_g, n_q)
    fracs = np.linspace(0, 1, n_steps)
    diag = []
    for f in fracs:
        gi = min(int(round(f * (n_g - 1))), n_g - 1)
        qi = min(int(round(f * (n_q - 1))), n_q - 1)
        diag.append(cka_grid[gi, qi])
    ax.plot(fracs, diag, color="#222", linewidth=1.6)
    ax.scatter(fracs, diag, s=12, color="#222")
    ax.set_xlabel("fractional depth")
    ax.set_ylabel("CKA at same-depth layer pair")
    ax.set_title("alignment trace along the depth diagonal")
    ax.set_ylim(0, max(0.5, float(max(diag) * 1.1)))

    fig.suptitle("v3 cross-model CKA — gemma ↔ qwen "
                 "(800 paired rows, both kaomoji-bearing)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cca_canonical(
    correlations: np.ndarray,
    out_path: Path,
) -> None:
    """Top-k canonical correlations from CCA. Bar chart with a y-axis
    cap of 1.0; tall trailing bars = many shared affect axes."""
    _use_cjk_font()
    k = len(correlations)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, k + 1), correlations, color="#1f77b4", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    for i, r in enumerate(correlations, 1):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("canonical component")
    ax.set_ylabel("canonical correlation")
    ax.set_xticks(range(1, k + 1))
    ax.set_ylim(0, 1.05)
    ax.set_title(f"v3 cross-model CCA — top-{k} canonical correlations\n"
                 "(at deepest layer of each model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_quadrant_geometry_compare(
    df_g: pd.DataFrame, X_g: np.ndarray,
    df_q: pd.DataFrame, X_q: np.ndarray,
    out_path: Path,
) -> dict:
    """Side-by-side: per-quadrant centroids in each model's PCA(2)
    plane, plus a Procrustes-aligned overlay panel. Returns rotation
    angle and Procrustes residual."""
    _use_cjk_font()
    quadrants = QUADRANT_ORDER

    # Fit per-model PCA(2).
    pg = PCA(n_components=2).fit(X_g)
    pq = PCA(n_components=2).fit(X_q)
    Yg = pg.transform(X_g)
    Yq = pq.transform(X_q)

    cg = {}
    cq = {}
    for q in quadrants:
        gm = (df_g["quadrant"] == q).to_numpy()
        qm = (df_q["quadrant"] == q).to_numpy()
        if gm.any():
            cg[q] = Yg[gm].mean(axis=0)
        if qm.any():
            cq[q] = Yq[qm].mean(axis=0)

    common_qs = [q for q in quadrants if q in cg and q in cq]
    if len(common_qs) < 2:
        print("  [quadrant geometry] not enough shared quadrants; skipping")
        return {}

    Cg = np.asarray([cg[q] for q in common_qs])
    Cq = np.asarray([cq[q] for q in common_qs])

    # Procrustes — find rotation matrix R that best fits Cq @ R to Cg
    # (centered). orthogonal_procrustes returns (R, scale).
    Cg_c = Cg - Cg.mean(axis=0, keepdims=True)
    Cq_c = Cq - Cq.mean(axis=0, keepdims=True)
    R, scale = orthogonal_procrustes(Cq_c, Cg_c)
    Cq_aligned = (Cq_c @ R) * (scale / max(1e-12, float(np.linalg.norm(Cq_c))))
    # The above scale from scipy is sum of singular values; we want a
    # comparable plot, so renormalize Cq_aligned to match Cg_c norm.
    norm_g = float(np.linalg.norm(Cg_c))
    norm_qa = float(np.linalg.norm(Cq_aligned))
    if norm_qa > 0:
        Cq_aligned = Cq_aligned * (norm_g / norm_qa)
    residual = float(np.linalg.norm(Cq_aligned - Cg_c))
    # Rotation angle from the orthogonal R.
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    def draw(ax, coords_dict, title):
        for q, pt in coords_dict.items():
            color = QUADRANT_COLORS.get(q, "#666")
            ax.scatter(pt[0], pt[1], c=color, s=220,
                       edgecolor="black", linewidth=0.8)
            ax.annotate(q, (pt[0], pt[1]),
                        xytext=(8, 5), textcoords="offset points",
                        fontsize=11, fontweight="bold")
        ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)

    draw(axes[0], cg, "gemma quadrant centroids "
                      f"(PC1 {pg.explained_variance_ratio_[0]*100:.1f}%, "
                      f"PC2 {pg.explained_variance_ratio_[1]*100:.1f}%)")
    draw(axes[1], cq, "qwen quadrant centroids "
                      f"(PC1 {pq.explained_variance_ratio_[0]*100:.1f}%, "
                      f"PC2 {pq.explained_variance_ratio_[1]*100:.1f}%)")

    # Aligned overlay: gemma centered + qwen procrustes-aligned.
    ax = axes[2]
    for i, q in enumerate(common_qs):
        color = QUADRANT_COLORS.get(q, "#666")
        ax.scatter(Cg_c[i, 0], Cg_c[i, 1], c=color, s=220, marker="o",
                   edgecolor="black", linewidth=0.8, label=f"{q} gemma" if i == 0 else None)
        ax.scatter(Cq_aligned[i, 0], Cq_aligned[i, 1], c=color, s=220,
                   marker="^", edgecolor="black", linewidth=0.8,
                   label=f"{q} qwen (aligned)" if i == 0 else None)
        ax.plot([Cg_c[i, 0], Cq_aligned[i, 0]],
                [Cg_c[i, 1], Cq_aligned[i, 1]],
                color=color, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.annotate(q, (Cg_c[i, 0], Cg_c[i, 1]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=11, fontweight="bold")
    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.set_xlabel("aligned PC1")
    ax.set_ylabel("aligned PC2")
    ax.set_title(f"Procrustes overlay — rotation {angle:+.1f}°, "
                 f"residual {residual:.2f}\n"
                 "circles = gemma centered; triangles = qwen rotated to fit gemma")

    fig.suptitle("v3 cross-model quadrant geometry — same Russell shape?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "procrustes_rotation_deg": angle,
        "procrustes_residual": residual,
        "common_quadrants": common_qs,
        "gemma_centroids": {q: cg[q].tolist() for q in common_qs},
        "qwen_centroids": {q: cq[q].tolist() for q in common_qs},
    }


def main() -> None:
    if not (
        MODEL_REGISTRY["gemma"].emotional_data_path.exists()
        and MODEL_REGISTRY["qwen"].emotional_data_path.exists()
    ):
        print("need both gemma and qwen v3 runs on disk")
        sys.exit(1)

    df_g, X3_g, layers_g, df_q, X3_q, layers_q, _ = _load_v3_paired()

    # Take h_mean at each model's preferred layer (peak Russell-quadrant
    # silhouette per scripts/21). gemma=L31, qwen=L61. Falls back to
    # deepest if a model has no preferred layer set.
    Mg = MODEL_REGISTRY["gemma"]
    Mq = MODEL_REGISTRY["qwen"]
    g_target = Mg.preferred_layer if Mg.preferred_layer is not None else layers_g[-1]
    q_target = Mq.preferred_layer if Mq.preferred_layer is not None else layers_q[-1]
    g_idx = layers_g.index(g_target)
    q_idx = layers_q.index(q_target)
    Xg_deep = X3_g[:, g_idx, :]
    Xq_deep = X3_q[:, q_idx, :]

    print(f"\ngemma layer L{g_target} (preferred) -> {Xg_deep.shape}")
    print(f"qwen  layer L{q_target} (preferred) -> {Xq_deep.shape}")

    # Per-layer CKA grid via precomputed centered Grams. Each Gram is
    # one (800, 800) matmul against a (800, ~5300) matrix — small in
    # both time and memory. The 3360 pairwise CKA evaluations after
    # that are O(n^2) elementwise products, milliseconds each.
    print(f"\ncomputing per-layer CKA grid "
          f"({len(layers_g)} × {len(layers_q)} = "
          f"{len(layers_g) * len(layers_q)} pairs) via Gram-form...")
    print("  building centered Grams per layer...")
    grams_g = [_gram_centered(X3_g[:, i, :]) for i in range(len(layers_g))]
    grams_q = [_gram_centered(X3_q[:, j, :]) for j in range(len(layers_q))]
    print(f"  Grams ready: {len(grams_g)} gemma + {len(grams_q)} qwen layers")
    cka_grid = np.zeros((len(layers_g), len(layers_q)), dtype=np.float32)
    for i in range(len(layers_g)):
        for j in range(len(layers_q)):
            cka_grid[i, j] = _cka_from_grams(grams_g[i], grams_q[j])
    print(f"  CKA grid: min {cka_grid.min():.3f}, "
          f"max {cka_grid.max():.3f}")
    print(f"  preferred-layer pair (gemma L{g_target} ↔ qwen L{q_target}): "
          f"CKA={cka_grid[g_idx, q_idx]:.3f}")
    print(f"  deepest-deepest (L{layers_g[-1]} ↔ L{layers_q[-1]}): "
          f"CKA={cka_grid[-1, -1]:.3f}")
    # Best-aligned cross-layer pair.
    bi, bj = np.unravel_index(np.argmax(cka_grid), cka_grid.shape)
    print(f"  best alignment: gemma L{layers_g[int(bi)]} ↔ qwen L{layers_q[int(bj)]}  "
          f"CKA={cka_grid[bi, bj]:.3f}")

    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    cka_path = out_dir / "fig_v3_cka_per_layer.png"
    _plot_cka_per_layer(cka_grid, layers_g, layers_q, cka_path)
    print(f"wrote {cka_path}")

    # Save CKA grid as TSV for downstream poking.
    cka_df = pd.DataFrame(
        cka_grid, index=[f"L{l}" for l in layers_g],
        columns=[f"L{l}" for l in layers_q],
    )
    cka_df.to_csv(out_dir / "v3_cka_per_layer.tsv", sep="\t")

    # CCA on deepest layers. Both hidden spaces have rank ≥ n_samples
    # (5376 / 5120 ≫ 800), so in-sample CCA recovers spurious perfect
    # correlations regardless of PCA prefix dim. We need a held-out
    # split for honest numbers: fit CCA on 70% of paired prompts,
    # report correlations on the held-out 30%. Project both to PCA(20)
    # first to keep the canonical components in an interpretable
    # subspace (top-20 PCs cover ~50% of each model's variance).
    rng = np.random.default_rng(0)
    n_pairs = Xg_deep.shape[0]
    perm = rng.permutation(n_pairs)
    n_train = int(0.7 * n_pairs)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    pca_dim = 20
    pg = PCA(n_components=pca_dim).fit(Xg_deep[train_idx])
    pq = PCA(n_components=pca_dim).fit(Xq_deep[train_idx])
    Xg_red_tr = pg.transform(Xg_deep[train_idx])
    Xq_red_tr = pq.transform(Xq_deep[train_idx])
    Xg_red_te = pg.transform(Xg_deep[test_idx])
    Xq_red_te = pq.transform(Xq_deep[test_idx])
    print(f"\nfitting CCA(n_components=10) on PCA({pca_dim})-reduced features, "
          f"{n_train}-row train / {n_pairs - n_train}-row test split "
          f"(gemma PCA var kept: {pg.explained_variance_ratio_.sum()*100:.1f}%, "
          f"qwen: {pq.explained_variance_ratio_.sum()*100:.1f}%)...")
    cca = CCA(n_components=10, max_iter=1000)
    cca.fit(Xg_red_tr, Xq_red_tr)
    Xg_cca_tr, Xq_cca_tr = cca.transform(Xg_red_tr, Xq_red_tr)
    Xg_cca_te, Xq_cca_te = cca.transform(Xg_red_te, Xq_red_te)
    correlations_train = np.array([
        float(np.corrcoef(Xg_cca_tr[:, k], Xq_cca_tr[:, k])[0, 1])
        for k in range(Xg_cca_tr.shape[1])
    ])
    correlations = np.array([
        float(np.corrcoef(Xg_cca_te[:, k], Xq_cca_te[:, k])[0, 1])
        for k in range(Xg_cca_te.shape[1])
    ])
    print(f"  train canonical correlations: {[f'{r:.3f}' for r in correlations_train]}")
    print(f"  test  canonical correlations: {[f'{r:.3f}' for r in correlations]}")

    cca_path = out_dir / "fig_v3_cca_canonical_correlations.png"
    _plot_cca_canonical(correlations, cca_path)
    print(f"wrote {cca_path}")

    # Quadrant geometry comparison + Procrustes alignment.
    geom_path = out_dir / "fig_v3_quadrant_geometry_compare.png"
    geom = _plot_quadrant_geometry_compare(df_g, Xg_deep, df_q, Xq_deep, geom_path)
    print(f"wrote {geom_path}")
    if geom:
        print(f"  Procrustes rotation: {geom['procrustes_rotation_deg']:+.1f}°  "
              f"residual: {geom['procrustes_residual']:.3f}")

    # Save scalar summary.
    summary = {
        "n_paired_rows": int(Xg_deep.shape[0]),
        "gemma_layer": int(g_target),
        "qwen_layer": int(q_target),
        "gemma_deepest_layer": int(layers_g[-1]),
        "qwen_deepest_layer": int(layers_q[-1]),
        "cka_preferred_pair": float(cka_grid[g_idx, q_idx]),
        "cka_deepest_pair": float(cka_grid[-1, -1]),
        "cka_max": float(cka_grid.max()),
        "cka_max_layers": (int(layers_g[bi]), int(layers_q[bj])),
        "cca_top10_correlations": correlations.tolist(),
        **geom,
    }
    import json
    summary_path = out_dir / "v3_cross_model_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
