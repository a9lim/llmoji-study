# pyright: reportArgumentType=false, reportCallIssue=false
"""v3 cross-model representational alignment (parametric pair).

Configurable via --ref / --target; defaults to gemma / qwen for back-
compat with the original 2026-04-XX analysis. Both models in a pair
need their v3 main run on disk + paired prompts/seeds. For all-pairs
coverage across the 5-model v3 lineup, run this script in a loop with
each pair (10 pairs total).

The v3 runs use the same 120 prompts × 8 seeds across all main-lineup
models, so rows pair cleanly by (prompt_id, seed). Two measurements:

* Linear CKA between paired h_first matrices, overall and per-layer
  (using *fractional depth* on the x-axis so models with different
  layer counts line up). Single number per layer; high CKA = the
  two models put paired prompts in geometrically similar
  configurations within their respective hidden spaces.

* Top-k canonical correlations from sklearn's CCA (n_components=10)
  between (X_ref, X_target). Tells us how many independent shared
  affect axes there are — if only the first 1-2 are high, the shared
  representation is low-rank and probably collapses to valence/arousal.
  If 5-10 stay high, deeper non-affect alignment exists.

The per-quadrant Procrustes-alignment view that used to live here was
removed 2026-05-04: script 31 covers the same question (cross-model
quadrant-geometry alignment) in 3D over each model's full row set
(rather than the paired-row subset), generalized to N models, and the
two analyses were producing redundant figures.

Outputs to figures/local/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from llmoji_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    KAOMOJI_START_CHARS,
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


def _load_v3_paired(ref: str = "gemma", target: str = "qwen") -> tuple[
    pd.DataFrame, np.ndarray, list[int],
    pd.DataFrame, np.ndarray, list[int],
    np.ndarray,
]:
    """Load `ref` and `target` v3 multi-layer h_first tensors and align
    rows by (prompt_id, seed). Applies the kaomoji-start filter to BOTH
    models so a row is kept only if both models emitted a kaomoji-
    bearing first word.

    Returns ``(df_g, X3_g, layers_g, df_q, X3_q, layers_q, kaomoji_mask)``
    where the dataframes / tensors are in matching paired-row order.
    Internal variable names retain g/q for back-compat with the original
    gemma/qwen analysis; semantically g=ref, q=target.
    """
    g_cache = DATA_DIR / "local" / "cache" / f"{ref}_h_mean_all_layers.npz"
    q_cache = DATA_DIR / "local" / "cache" / f"{target}_h_mean_all_layers.npz"
    Mg = MODEL_REGISTRY[ref]
    Mq = MODEL_REGISTRY[target]
    print(f"loading {ref} v3 (cache: {g_cache.exists()})...")
    df_g, X3_g, layers_g = load_hidden_features_all_layers(
        Mg.emotional_data_path, DATA_DIR, Mg.experiment,
        which="h_first", cache_path=g_cache,
    )
    print(f"loading {target} v3 (cache: {q_cache.exists()})...")
    df_q, X3_q, layers_q = load_hidden_features_all_layers(
        Mq.emotional_data_path, DATA_DIR, Mq.experiment,
        which="h_first", cache_path=q_cache,
    )
    print(f"  {ref} {X3_g.shape}, {target} {X3_q.shape}")

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
    ref: str = "gemma",
    target: str = "qwen",
    n_paired: int | None = None,
) -> None:
    """Per-layer CKA heatmap (`ref` layer × `target` layer) plus
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
    ax.set_xlabel(f"{target} fractional depth")
    ax.set_ylabel(f"{ref} fractional depth")
    ax.set_title(f"linear CKA across all ({ref}, {target}) layer pairs")
    fig.colorbar(im, ax=ax, shrink=0.7, label="CKA")
    ax.plot([0, 1], [0, 1], color="white", linewidth=0.6, linestyle="--",
            alpha=0.7)

    ax = axes[1]
    # Diagonal trace: at each fractional depth, take CKA between the
    # closest ref layer and closest target layer.
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

    n_str = f"({n_paired} paired rows, " if n_paired is not None else "("
    fig.suptitle(f"v3 cross-model CKA — {ref} ↔ {target} "
                 f"{n_str}both kaomoji-bearing)")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="gemma",
                        help="Reference model. Default: gemma")
    parser.add_argument("--target", default="qwen",
                        help="Target model (rotated to fit ref). Default: qwen")
    args = parser.parse_args()
    ref, target = args.ref, args.target

    if ref not in MODEL_REGISTRY:
        raise SystemExit(f"unknown --ref {ref!r}; known: {sorted(MODEL_REGISTRY)}")
    if target not in MODEL_REGISTRY:
        raise SystemExit(f"unknown --target {target!r}; known: {sorted(MODEL_REGISTRY)}")
    if ref == target:
        raise SystemExit("--ref and --target must differ")

    if not (
        MODEL_REGISTRY[ref].emotional_data_path.exists()
        and MODEL_REGISTRY[target].emotional_data_path.exists()
    ):
        print(f"need both {ref} and {target} v3 runs on disk")
        sys.exit(1)

    print(f"alignment pair: {ref} (ref) ↔ {target} (target)")
    _result = _load_v3_paired(ref, target)
    X3_g, layers_g = _result[1], _result[2]
    X3_q, layers_q = _result[4], _result[5]

    # CCA computes on each model's deepest layer — canonical,
    # silhouette-independent choice. The all-layers CKA grid below
    # surfaces any cross-depth alignment without committing to a
    # single layer.
    g_idx = len(layers_g) - 1
    q_idx = len(layers_q) - 1
    g_target = layers_g[g_idx]
    q_target = layers_q[q_idx]
    Xg_deep = X3_g[:, g_idx, :]
    Xq_deep = X3_q[:, q_idx, :]

    print(f"\n{ref} layer L{g_target} (deepest) -> {Xg_deep.shape}")
    print(f"{target} layer L{q_target} (deepest) -> {Xq_deep.shape}")

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
    print(f"  deepest-deepest pair ({ref} L{layers_g[-1]} ↔ {target} L{layers_q[-1]}): "
          f"CKA={cka_grid[-1, -1]:.3f}")
    # Best-aligned cross-layer pair.
    bi, bj = np.unravel_index(np.argmax(cka_grid), cka_grid.shape)
    print(f"  best alignment: {ref} L{layers_g[int(bi)]} ↔ {target} L{layers_q[int(bj)]}  "
          f"CKA={cka_grid[bi, bj]:.3f}")

    out_dir = FIGURES_DIR / "local"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default ref=gemma, target=qwen → keep canonical filenames for
    # back-compat with prior artifacts; non-default pairs get a suffix.
    is_default = (ref == "gemma" and target == "qwen")
    pair_suffix = "" if is_default else f"__{ref}_vs_{target}"

    cka_path = out_dir / f"fig_v3_cka_per_layer{pair_suffix}.png"
    _plot_cka_per_layer(cka_grid, layers_g, layers_q, cka_path,
                        ref=ref, target=target,
                        n_paired=int(Xg_deep.shape[0]))
    print(f"wrote {cka_path}")

    # Save CKA grid as TSV for downstream poking.
    cka_df = pd.DataFrame(
        cka_grid, index=[f"L{l}" for l in layers_g],
        columns=[f"L{l}" for l in layers_q],
    )
    cka_df.to_csv(out_dir / f"v3_cka_per_layer{pair_suffix}.tsv", sep="\t")

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
          f"({ref} PCA var kept: {pg.explained_variance_ratio_.sum()*100:.1f}%, "
          f"{target}: {pq.explained_variance_ratio_.sum()*100:.1f}%)...")
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

    cca_path = out_dir / f"fig_v3_cca_canonical_correlations{pair_suffix}.png"
    _plot_cca_canonical(correlations, cca_path)
    print(f"wrote {cca_path}")

    # Quadrant-geometry / centroid Procrustes lives in script 31 now
    # (3D, all-rows, N-model). See module docstring.

    # Save scalar summary.
    summary = {
        "ref_model": ref,
        "target_model": target,
        "n_paired_rows": int(Xg_deep.shape[0]),
        f"{ref}_deepest_layer": int(layers_g[-1]),
        f"{target}_deepest_layer": int(layers_q[-1]),
        "cka_deepest_pair": float(cka_grid[-1, -1]),
        "cka_max": float(cka_grid.max()),
        "cka_max_layers": (int(layers_g[bi]), int(layers_q[bj])),
        "cca_top10_correlations": correlations.tolist(),
    }
    import json
    summary_path = out_dir / f"v3_cross_model_summary{pair_suffix}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
