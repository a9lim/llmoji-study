# pyright: reportAttributeAccessIssue=false
"""v3 hidden-state variance decomposition by grouping source.

Idea 8 from the 2026-05-03 face-stability brainstorm. Existing infra
(Fig B / script 22) tests per-face cluster compactness against a random
null. This is necessary but not the interesting question. The
interesting one is: of the total dispersion in hidden-state space at
h_first layer-stack, how much is explained by which face was
emitted vs which prompt produced the row vs which Russell quadrant the
prompt belongs to vs which seed was drawn? And — critically — how much
does face explain *after* prompt-mean is subtracted out?

Surprise from the first run: η²(prompt_id) = 1.0 and η²(seed) = 0.0
exactly across all 3 models. h_first is the hidden state at the
position the first generated token will be drawn from — that state is
fully determined by the prompt; only the *sampled token* (and so the
*emitted face*) varies across seeds. So the | prompt_id conditional is
degenerate at h_first (residual is identically zero), and the
informative conditional is η²(face | quadrant_split) — does the face
add structure beyond Russell-quadrant content?

For h_mean / h_last (states *during* generation, post-sampling), the
| prompt_id conditional becomes non-degenerate and asks the original
question. Run with $LLMOJI_WHICH=h_mean to switch.

Decomposition (centered hidden-state matrix X ∈ ℝ^{n×d}):

    TSS       = Σᵢ ‖xᵢ − x̄‖²
    BSS(G)    = Σ_g n_g ‖μ_g − x̄‖²              (between-group SS)
    η²(G)     = BSS(G) / TSS

    η²(G | H): subtract the H-group mean from each row, recompute
               η²(G) on the residual matrix.

Groupings: face (first_word, canonicalized), prompt_id (120 prompts),
quadrant_split (HP / LP / HN-D / HN-S / LN / NB), seed (8 seeds).

Outputs:
  figures/local/cross_model/v3_face_stability_eta2.tsv
  figures/local/cross_model/fig_v3_face_stability_eta2.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmoji_study.config import FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    _use_cjk_font,
    load_emotional_features_stack,
)


GROUPINGS = ["face", "prompt_id", "quadrant_split", "seed"]
CONDITIONALS = [
    ("face", "prompt_id"),
    ("quadrant_split", "prompt_id"),
    ("face", "quadrant_split"),
]


def _eta_squared(X: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """η²(G) = BSS(G) / TSS for grouping G defined by ``labels``.

    Returns (eta2, n_groups). Total SS uses the full matrix mean as the
    centroid; between-group SS sums n_g ‖μ_g − x̄‖² over groups."""
    if len(X) == 0:
        return float("nan"), 0
    grand = X.mean(axis=0, keepdims=True)
    centered = X - grand
    tss = float((centered ** 2).sum())
    if tss <= 0:
        return float("nan"), 0
    bss = 0.0
    n_groups = 0
    for g in pd.unique(labels):
        mask = labels == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        mu_g = X[mask].mean(axis=0, keepdims=True)
        bss += n_g * float(((mu_g - grand) ** 2).sum())
        n_groups += 1
    return bss / tss, n_groups


def _conditional_eta_squared(
    X: np.ndarray, g_labels: np.ndarray, h_labels: np.ndarray,
) -> float:
    """η²(G | H): subtract per-H group means from each row, recompute
    η²(G) on the residual matrix. Quantifies how much G explains beyond
    what H already accounts for.

    Returns NaN when the residual is identically zero — happens at
    h_first when H is prompt_id, since h_first is fully determined by
    the prompt (same hidden state across all seeds; the only thing
    seeds vary is which token is sampled out of that state)."""
    if len(X) == 0:
        return float("nan")
    residual = np.empty_like(X)
    for h in pd.unique(h_labels):
        mask = h_labels == h
        if not mask.any():
            continue
        residual[mask] = X[mask] - X[mask].mean(axis=0, keepdims=True)
    if float((residual ** 2).sum()) <= 0:
        return float("nan")
    eta2, _ = _eta_squared(residual, g_labels)
    return eta2


def _build_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Materialize one labels array per grouping. ``face`` is the
    canonicalized first_word; ``quadrant_split`` is the 6-category
    HN-bisected column already attached by load_emotional_features
    when split_hn=True; ``seed`` is the per-row decoder seed."""
    return {
        "face": df["first_word"].astype(str).to_numpy(),
        "prompt_id": df["prompt_id"].astype(str).to_numpy(),
        "quadrant_split": df["quadrant"].astype(str).to_numpy(),
        "seed": df["seed"].astype(int).to_numpy(),
    }


def _decompose_one_model(short: str) -> pd.DataFrame:
    M = MODEL_REGISTRY[short]
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no v3 data at {M.emotional_data_path}; skipping")
        return pd.DataFrame()

    print(f"\n{short}  (h_first, layer-stack)")
    df, X = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
    )
    if len(df) == 0:
        print(f"  [{short}] zero rows after kaomoji filter; skipping")
        return pd.DataFrame()

    print(f"  {len(df)} kaomoji-bearing rows, X {X.shape}")
    labels = _build_labels(df)

    rows: list[dict] = []
    for g in GROUPINGS:
        eta2, n_groups = _eta_squared(X, labels[g])
        rows.append({
            "model": short,
            "kind": "marginal",
            "factor": g,
            "given": "",
            "eta2": eta2,
            "n_groups": n_groups,
            "n_rows": int(len(df)),
        })
        print(f"    η²({g:14s})           = {eta2:.4f}   "
              f"({n_groups} groups)")

    for g, h in CONDITIONALS:
        eta2 = _conditional_eta_squared(X, labels[g], labels[h])
        rows.append({
            "model": short,
            "kind": "conditional",
            "factor": g,
            "given": h,
            "eta2": eta2,
            "n_groups": int(len(np.unique(labels[g]))),
            "n_rows": int(len(df)),
        })
        rendered = "(degenerate)" if np.isnan(eta2) else f"{eta2:.4f}"
        print(f"    η²({g:14s} | {h:14s}) = {rendered}")

    return pd.DataFrame(rows)


def _plot_eta2_grid(summary: pd.DataFrame, out_path: Path) -> None:
    """Two-panel bar grid: marginal η² and conditional η² across models.

    Marginal panel: one bar per (model, factor); conditional panel: one
    bar per (model, factor | given). Shared y-axis at [0, 1] for direct
    visual comparison; conditional bars often tiny and that's the
    point."""
    _use_cjk_font()
    if len(summary) == 0:
        return

    marginal = summary[summary["kind"] == "marginal"]
    conditional = summary[summary["kind"] == "conditional"]

    models = sorted(summary["model"].unique())
    model_color = {
        "gemma":     "#2a7fbf",
        "qwen":      "#d49b3a",
        "ministral": "#4aa66a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [1.05, 1.0]})

    factors = GROUPINGS
    width = 0.8 / max(len(models), 1)
    x_base = np.arange(len(factors))
    ax = axes[0]
    for i, m in enumerate(models):
        row_lookup = {
            r["factor"]: r["eta2"]
            for _, r in marginal[marginal["model"] == m].iterrows()
        }
        ys = [row_lookup.get(f, np.nan) for f in factors]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x_base + offset, ys, width=width * 0.95,
               color=model_color.get(m, "#666"),
               label=m, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x_base)
    ax.set_xticklabels(factors, rotation=20, ha="right")
    ax.set_ylabel("η² = BSS(G) / TSS")
    ax.set_ylim(0, 1.0)
    ax.set_title("marginal η² by grouping factor\n"
                 "(fraction of total hidden-state dispersion explained)")
    ax.axhline(0, color="#888", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    cond_keys = [f"{g}\n| {h}" for g, h in CONDITIONALS]
    x_base = np.arange(len(CONDITIONALS))
    ax = axes[1]
    for i, m in enumerate(models):
        sub = conditional[conditional["model"] == m]
        row_lookup = {
            (r["factor"], r["given"]): r["eta2"]
            for _, r in sub.iterrows()
        }
        ys = [row_lookup.get((g, h), np.nan) for g, h in CONDITIONALS]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x_base + offset, ys, width=width * 0.95,
               color=model_color.get(m, "#666"),
               label=m, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x_base)
    ax.set_xticklabels(cond_keys, fontsize=9)
    ax.set_ylabel("conditional η² (residual after fixing the | factor)")
    ax.set_ylim(0, 1.0)
    ax.set_title("conditional η²\n"
                 "(how much each factor adds beyond the conditioning factor)")
    ax.axhline(0, color="#888", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    import os
    which = os.environ.get("LLMOJI_WHICH", "h_first")
    note = (
        "  (h_first is prompt-deterministic — | prompt_id conditionals "
        "vanish; informative conditional is | quadrant_split)"
        if which == "h_first" else ""
    )
    fig.suptitle(
        f"v3 hidden-state variance decomposition at {which} (layer-stack)"
        + note,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for short in MODEL_REGISTRY:
        parts.append(_decompose_one_model(short))
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(summary) == 0:
        print("no models had loadable data; nothing to write")
        return

    import os
    which = os.environ.get("LLMOJI_WHICH", "h_first")
    tsv_path = out_dir / f"v3_face_stability_eta2_{which}.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {tsv_path}")

    fig_path = out_dir / f"fig_v3_face_stability_eta2_{which}.png"
    _plot_eta2_grid(summary, fig_path)
    print(f"wrote {fig_path}")

    print("\n--- key numbers ---")
    for m in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == m]
        prompt = float(sub[(sub["kind"] == "marginal")
                           & (sub["factor"] == "prompt_id")]["eta2"].iloc[0])
        face = float(sub[(sub["kind"] == "marginal")
                         & (sub["factor"] == "face")]["eta2"].iloc[0])
        face_given_q = float(sub[(sub["kind"] == "conditional")
                                  & (sub["factor"] == "face")
                                  & (sub["given"] == "quadrant_split")]["eta2"].iloc[0])
        print(
            f"  {m:9s}  η²(prompt)={prompt:.3f}  "
            f"η²(face)={face:.3f}  "
            f"η²(face | quadrant)={face_given_q:.3f}"
        )


if __name__ == "__main__":
    main()
