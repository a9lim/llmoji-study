"""Regenerate the introspection-via-kaomoji blog post's PNG figures.

Writes light + dark variants of the four static figures embedded in
[a9l.im/blog/introspection-via-kaomoji](https://a9l.im/blog/introspection-via-kaomoji),
into a fixed output directory under the a9l.im repo:

    a9l.im/blog-assets/introspection-via-kaomoji/
        fig_v3_layerwise_emergence_compare_{light,dark}.png
        fig_emo_a_kaomoji_sim_{gemma,qwen,ministral}_{light,dark}.png

Background, text, axis, grid, and per-model line colors are taken from
the site's canonical palette in `shared-tokens.js` (light/dark elevated
panel + text + text-muted + grid; per-model accent / secondary /
extended.green for the layerwise plot). The cosine heatmaps keep
matplotlib's `RdBu_r` divergent colormap.

Run after re-running `scripts/harness/10_emit_analysis.py` and
`scripts/20_v3_layerwise_emergence.py` for any model whose data has
changed; the heatmap path reads sidecars + cached h_first via
`load_emotional_features`, the layerwise path reads the per-model
`v3_layerwise_emergence.tsv` written by script 21.

Configure `BLOG_ASSETS_DIR` if the a9l.im checkout lives elsewhere.

Usage:
    .venv/bin/python scripts/99_regen_blog_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(STUDY_ROOT))

from llmoji_study.config import MODEL_REGISTRY  # noqa: E402
from llmoji_study.emotional_analysis import (  # noqa: E402
    load_emotional_features_stack,
    plot_kaomoji_cosine_heatmap,
)

# Output target — adjust if the a9l.im checkout lives elsewhere.
BLOG_ASSETS_DIR = (
    STUDY_ROOT.parent / "a9lim.github.io" / "blog-assets" / "introspection-via-kaomoji"
)
DATA_DIR = STUDY_ROOT / "data"
# Heatmap is per-model; regenerate for the full v3 main lineup.
HEATMAP_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
# Layerwise comparison only includes models with v3_layerwise_emergence.tsv;
# script 21 wasn't re-run on the gpt_oss / granite additions because the
# layer-stack rep supersedes single-layer silhouette as the canonical
# read. Keep the 3-model trajectory as a depth-emergence illustration.
LAYERWISE_MODELS = ("gemma", "qwen", "ministral")

# Theme palettes mirror `shared-tokens.js` light/dark surface + text vars.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "figure.facecolor":  "#F7F8FB",
        "axes.facecolor":    "#F7F8FB",
        "savefig.facecolor": "#F7F8FB",
        "text.color":        "#090B0F",
        "axes.labelcolor":   "#090B0F",
        "axes.titlecolor":   "#090B0F",
        "xtick.color":       "#75777C",
        "ytick.color":       "#75777C",
        "axes.edgecolor":    "#75777C",
        "grid.color":        "#E6E8ED",
        "figure.edgecolor":  "#F7F8FB",
        "savefig.edgecolor": "#F7F8FB",
    },
    "dark": {
        "figure.facecolor":  "#191A1F",
        "axes.facecolor":    "#191A1F",
        "savefig.facecolor": "#191A1F",
        "text.color":        "#E0E1E5",
        "axes.labelcolor":   "#E0E1E5",
        "axes.titlecolor":   "#E0E1E5",
        "xtick.color":       "#84868B",
        "ytick.color":       "#84868B",
        "axes.edgecolor":    "#84868B",
        "grid.color":        "#252630",
        "figure.edgecolor":  "#191A1F",
        "savefig.edgecolor": "#191A1F",
    },
}

# Per-model line colors on the layerwise comparison.
# accent / secondary / extended.green from shared-tokens.js _PALETTE,
# extended with two more accents for the post-2026-05-04 5-model lineup.
LINE_PALETTE = {
    "gemma":       "#E11107",
    "qwen":        "#488ACB",
    "ministral":   "#009F68",
    "gpt_oss_20b": "#D49A1A",
    "granite":     "#8B5CF6",
}


def with_theme(theme: str) -> None:
    plt.rcdefaults()
    for k, v in THEMES[theme].items():
        plt.rcParams[k] = v


def plot_layerwise(theme: str) -> None:
    with_theme(theme)
    per_model: dict[str, pd.DataFrame] = {}
    for short in LAYERWISE_MODELS:
        M = MODEL_REGISTRY[short]
        tsv = M.figures_dir / "v3_layerwise_emergence.tsv"
        if not tsv.exists():
            print(f"  [layerwise] no TSV at {tsv} — skipping {short}")
            continue
        per_model[short] = pd.read_csv(tsv, sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (axes[0], "silhouette",
         "silhouette score (quadrants in PC1-PC2)", "quadrant separation"),
        (axes[1], "between_centroid_std_pc1",
         "between-centroid std on PC1", "PC1 spread"),
        (axes[2], "between_centroid_std_pc2",
         "between-centroid std on PC2", "PC2 spread"),
    ]
    for ax, ycol, ylabel, title in panels:
        for short, metrics in per_model.items():
            layers = metrics["layer"].to_numpy()
            denom = max(1.0, float(layers.max() - layers.min()))
            depth = (layers - layers.min()) / denom
            ax.plot(depth, metrics[ycol],
                    color=LINE_PALETTE[short], linewidth=1.6, label=short)
            ax.scatter(depth, metrics[ycol],
                       s=14, color=LINE_PALETTE[short])
        if ycol == "silhouette":
            ax.axhline(0, color=plt.rcParams["axes.edgecolor"],
                       linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_xlabel("fractional depth  (layer normalized to [0, 1])")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9, frameon=False,
                  labelcolor=plt.rcParams["text.color"])

    fig.suptitle("v3 layer-wise emergence — cross-model comparison",
                 color=plt.rcParams["text.color"])
    fig.tight_layout()
    out = BLOG_ASSETS_DIR / f"fig_v3_layerwise_emergence_compare_{theme}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close(fig)
    print(f"  wrote {out}")


def plot_heatmap(short: str, theme: str) -> None:
    with_theme(theme)
    df, X = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
    )
    out = BLOG_ASSETS_DIR / f"fig_emo_a_kaomoji_sim_{short}_{theme}.png"
    plot_kaomoji_cosine_heatmap(df, X, str(out), min_count=3)
    print(f"  wrote {out}")


def main() -> None:
    if not BLOG_ASSETS_DIR.parent.exists():
        print(f"blog-assets parent does not exist: {BLOG_ASSETS_DIR.parent}")
        print("  edit BLOG_ASSETS_DIR at the top of this script if your "
              "a9l.im checkout lives somewhere else.")
        sys.exit(1)
    BLOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        print(f"\n=== theme={theme} ===")
        plot_layerwise(theme)
        for short in HEATMAP_MODELS:
            plot_heatmap(short, theme)


if __name__ == "__main__":
    main()
