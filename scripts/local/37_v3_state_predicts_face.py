# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""v3 forward-direction test: does h_first predict the face distribution?

Companion to script 36's reverse-direction test (η²(face | prompt) at
h_mean). Script 36 found that face is a real hidden-state commitment
not just a prompt-content readout, but its forward number η²(face) at
h_first conflated two things: (a) prompt clusters in h_first space
(trivial — different prompts have different states), and (b) face-
coherence within those clusters (the actual mechanism).

This script isolates (b) by going pair-level. For every pair of
prompts (p₁, p₂), compute:

  cosine_sim(h_first_p₁, h_first_p₂)
  JSD(face_dist_p₁, face_dist_p₂)

If h_first geometry forward-predicts face emission, prompts with
similar h_first should emit similar face distributions — Spearman
correlation between cosine_sim and -JSD should be positive and large.

h_first is deterministic per prompt (script 36 finding), so we collapse
to one h_first vector per prompt and the empirical face distribution is
the histogram of emitted faces across the 8 seeds for that prompt.

Outputs (per model + cross-model):
  figures/local/cross_model/v3_state_predicts_face.tsv
    one row per model with spearman_r, n_pairs, n_prompts
  figures/local/<short>/fig_v3_state_predicts_face.png
    hexbin scatter: x=cosine_sim(h_first), y=face-dist similarity (1-JSD)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from llmoji_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    _use_cjk_font,
    load_emotional_features,
)


def _per_prompt_state_and_dist(
    df: pd.DataFrame, X: np.ndarray,
) -> tuple[list[str], np.ndarray, list[dict[str, int]]]:
    """Collapse per-row data to per-prompt: one h_first vector + face-
    emission counter per prompt. h_first is deterministic per prompt
    (script 36 finding), so any seed's row gives the same vector — we
    take the mean across seeds for safety against numerical noise."""
    states: list[np.ndarray] = []
    dists: list[dict[str, int]] = []
    pids: list[str] = []
    for pid, sub in df.groupby("prompt_id"):
        idxs = sub.index.to_numpy()
        states.append(X[idxs].mean(axis=0))
        counts: dict[str, int] = {}
        for face in sub["first_word"]:
            face_s = str(face)
            counts[face_s] = counts.get(face_s, 0) + 1
        dists.append(counts)
        pids.append(str(pid))
    return pids, np.asarray(states, dtype=np.float32), dists


def _jsd(a: dict[str, int], b: dict[str, int]) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence) over the union
    of face vocabularies. scipy returns the metric form, ∈ [0, 1]
    when log base = e (1 / sqrt(ln 2) when base = 2; we let scipy use
    its default base-e here and rescale to [0, 1] via the metric's
    natural cap of sqrt(ln 2) ≈ 0.832 — divide by that to get a
    similarity-friendly [0, 1] range; "1 - dist_norm" is the
    similarity).

    Faces present in only one of (a, b) get probability 0 in the
    other, which is the standard handling and what JSD wants."""
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0) for k in keys], dtype=float)
    sa = pa.sum()
    sb = pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa /= sa
    pb /= sb
    d = float(jensenshannon(pa, pb))
    return d / float(np.sqrt(np.log(2)))   # → [0, 1]


def _all_pairs(
    states: np.ndarray, dists: list[dict[str, int]],
    *, center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cos_arr, jsd_arr) of length n*(n-1)/2 — one entry per
    distinct pair (i < j). Centered cosine matches the rest of the
    repo's hidden-state similarity convention (shared-baseline
    direction subtracted)."""
    n = len(states)
    if center:
        states = states - states.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Xn = states / norms
    cos_full = Xn @ Xn.T

    iu = np.triu_indices(n, k=1)
    cos_arr = cos_full[iu]
    jsd_arr = np.empty(len(cos_arr), dtype=np.float32)
    pairs = list(zip(*iu))
    for k, (i, j) in enumerate(pairs):
        jsd_arr[k] = _jsd(dists[i], dists[j])
    return cos_arr, jsd_arr


def _plot_pair_scatter(
    cos_arr: np.ndarray, jsd_arr: np.ndarray,
    *, model: str, rho: float, out_path: Path,
) -> None:
    _use_cjk_font()
    sim = 1.0 - jsd_arr   # face-dist similarity
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(cos_arr, sim, gridsize=40, cmap="viridis", mincnt=1)
    ax.set_xlabel("cosine_sim(h_first centered) — pairwise")
    ax.set_ylabel("face-dist similarity = 1 − JSD")
    ax.set_title(
        f"{model}: pair-level forward direction\n"
        f"do similar h_first emit similar face distributions?\n"
        f"Spearman ρ = {rho:+.3f}  (n_pairs = {len(cos_arr)})"
    )
    cb = fig.colorbar(hb, ax=ax, label="pair count")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _per_model(short: str) -> dict | None:
    M = MODEL_REGISTRY[short]
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no v3 data; skipping")
        return None

    layer_label = "max" if M.preferred_layer is None else f"L{M.preferred_layer}"
    print(f"\n{short}  (h_first, layer={layer_label})")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_first",
        layer=M.preferred_layer,
        split_hn=True,
    )
    if len(df) == 0:
        print(f"  [{short}] no kaomoji-bearing rows")
        return None

    pids, states, dists = _per_prompt_state_and_dist(df, X)
    print(f"  {len(pids)} prompts, {len(df)} rows total")

    cos_arr, jsd_arr = _all_pairs(states, dists)
    sim = 1.0 - jsd_arr
    rho, p = spearmanr(cos_arr, sim)
    rho = float(rho)
    p = float(p)
    print(f"  pairs: {len(cos_arr)}, Spearman ρ(cosine, 1-JSD) = {rho:+.4f}  "
          f"(p = {p:.2e})")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_png = M.figures_dir / "fig_v3_state_predicts_face.png"
    _plot_pair_scatter(cos_arr, jsd_arr, model=short, rho=rho, out_path=out_png)
    print(f"  wrote {out_png}")

    return {
        "model": short,
        "n_prompts": int(len(pids)),
        "n_pairs": int(len(cos_arr)),
        "n_rows": int(len(df)),
        "spearman_rho": rho,
        "spearman_p": p,
    }


def main() -> None:
    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for short in MODEL_REGISTRY:
        r = _per_model(short)
        if r is not None:
            rows.append(r)

    if not rows:
        print("no models produced output")
        return

    summary = pd.DataFrame(rows)
    tsv_path = out_dir / "v3_state_predicts_face.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {tsv_path}")

    print("\n--- forward direction (state → face) summary ---")
    for _, r in summary.iterrows():
        print(f"  {r['model']:9s}  ρ={r['spearman_rho']:+.3f}  "
              f"n_pairs={int(r['n_pairs'])}  "
              f"p={r['spearman_p']:.1e}")


if __name__ == "__main__":
    main()
