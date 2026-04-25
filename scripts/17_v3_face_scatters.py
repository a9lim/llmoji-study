"""v3 face-level scatters (descriptive, post-run).

Two figures:
  - figures/fig_v3_face_pca_by_valence.png:
    Per-kaomoji mean hidden-state vector → PCA → 2D scatter
    colored by the face's dominant prompt valence (positive
    HP+LP / negative HN+LN / neutral NB / mixed across).
  - figures/fig_v3_face_probe_scatter.png:
    Per-kaomoji mean (happy.sad, angry.calm) probe-score scatter
    using probe_scores_t0; same valence coloring.

Operates on v3 data only (no v1/v2 baseline overlay). Both
figures annotate the top-30 most-frequent kaomoji.
"""

from __future__ import annotations

import sys
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
    PROBES,
)
from llmoji.emotional_analysis import _use_cjk_font, load_emotional_features


VALENCE_BY_QUADRANT = {
    "HP": "positive",
    "LP": "positive",
    "HN": "negative",
    "LN": "negative",
    "NB": "neutral",
}

VALENCE_COLORS = {
    "positive": "#2ca02c",  # green
    "negative": "#d62728",  # red
    "neutral":  "#7f7f7f",  # gray
    "mixed":    "#9467bd",  # purple
}


def _per_face_valence(df: pd.DataFrame) -> dict[str, str]:
    """For each first_word, return its valence label across emissions."""
    out: dict[str, str] = {}
    for fw, sub in df.groupby("first_word"):
        valences = {VALENCE_BY_QUADRANT[q] for q in sub["quadrant"]
                    if q in VALENCE_BY_QUADRANT}
        if len(valences) == 1:
            out[fw] = next(iter(valences))
        else:
            out[fw] = "mixed"
    return out


def _add_valence_legend(ax) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   color=VALENCE_COLORS[v], label=v)
        for v in ("positive", "negative", "neutral", "mixed")
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9)


def plot_face_pca_by_valence(
    df: pd.DataFrame,
    X: np.ndarray,
    out_path: Path,
    *,
    annotate_top: int = 30,
) -> dict:
    """Per-face mean hidden state → PCA → scatter colored by valence."""
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

    valence = _per_face_valence(df)
    colors = [VALENCE_COLORS[valence.get(fw, "mixed")] for fw in fdf["first_word"]]
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
        f"v3 per-kaomoji mean hidden-state PCA  ({len(fdf)} kaomoji)\n"
        "colored by face's valence across emissions"
    )
    _add_valence_legend(ax)
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
    """Per-face mean (happy.sad, angry.calm) probe scatter."""
    happy_sad_idx = PROBES.index("happy.sad")
    angry_calm_idx = PROBES.index("angry.calm")

    rows = []
    for fw, sub in df.groupby("first_word"):
        t0 = np.asarray(sub["probe_scores_t0"].tolist(), dtype=float)
        rows.append({
            "first_word": fw,
            "n": len(sub),
            "happy_sad": float(t0[:, happy_sad_idx].mean()),
            "angry_calm": float(t0[:, angry_calm_idx].mean()),
        })
    fdf = pd.DataFrame(rows)

    valence = _per_face_valence(df)
    colors = [VALENCE_COLORS[valence.get(fw, "mixed")] for fw in fdf["first_word"]]
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

    ax.set_xlabel("mean happy.sad probe (token 0)  →  positive = happy")
    ax.set_ylabel("mean angry.calm probe (token 0)  →  positive = angry")
    ax.set_title(
        f"v3 per-kaomoji probe scatter  ({len(fdf)} kaomoji)\n"
        "saklas bipolar probes, colored by face's valence"
    )
    _add_valence_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also report Pearson correlation of the two probe means across faces
    from scipy.stats import pearsonr
    r, p = pearsonr(fdf["happy_sad"].to_numpy(), fdf["angry_calm"].to_numpy())
    return {
        "n_faces": len(fdf),
        "n_emissions": int(fdf["n"].sum()),
        "probe_pair_pearson_r": float(r),
        "probe_pair_p": float(p),
    }


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}")
        sys.exit(1)
    _use_cjk_font()

    print("loading v3 hidden-state features (which=h_last)...")
    df, X = load_emotional_features(
        str(EMOTIONAL_DATA_PATH), DATA_DIR,
        experiment=EMOTIONAL_EXPERIMENT, which="h_last",
    )
    df = df[df["first_word"].notna() & (df["first_word"] != "")].reset_index(drop=True)
    print(f"  {len(df)} kaomoji-bearing rows; {df['first_word'].nunique()} unique faces; X {X.shape}")

    # quadrant breakdown of unique faces
    valence = _per_face_valence(df)
    from collections import Counter
    counts = Counter(valence.values())
    print("  faces by valence:", dict(counts))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out1 = FIGURES_DIR / "fig_v3_face_pca_by_valence.png"
    print(f"\nplotting per-face PCA...")
    s1 = plot_face_pca_by_valence(df, X, out1)
    print(f"  wrote {out1}")
    print(f"  PC1 {s1['explained_variance_ratio'][0]*100:.1f}%, "
          f"PC2 {s1['explained_variance_ratio'][1]*100:.1f}%")

    out2 = FIGURES_DIR / "fig_v3_face_probe_scatter.png"
    print(f"\nplotting per-face probe scatter...")
    s2 = plot_face_probe_scatter(df, out2)
    print(f"  wrote {out2}")
    print(f"  {s2['n_faces']} faces, {s2['n_emissions']} total emissions")
    print(f"  Pearson(mean happy.sad, mean angry.calm) across faces: "
          f"r={s2['probe_pair_pearson_r']:+.3f}, p={s2['probe_pair_p']:.3g}")


if __name__ == "__main__":
    main()
