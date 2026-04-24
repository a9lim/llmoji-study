"""Eriskii-replication step 3: analysis + figures.

Sections in build order:
  - axis projection: data/eriskii_axes.tsv +
    figures/eriskii_axis_<name>.png × 21
  - clusters: data/eriskii_clusters.tsv +
    figures/eriskii_clusters_tsne.png
  - per-model: data/eriskii_per_model.tsv +
    figures/eriskii_per_model_axes_{mean,std}.png
  - per-project: data/eriskii_per_project.tsv +
    figures/eriskii_per_project_axes_{mean,std}.png
  - mechanistic bridge: data/eriskii_user_kaomoji_axis_corr.tsv +
    figures/eriskii_user_kaomoji_axis_corr.png
  - narrative writeup: data/eriskii_comparison.md

Usage:
  python scripts/16_eriskii_replication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji.claude_faces import EMBED_MODEL, load_embeddings
from llmoji.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    CLAUDE_KAOMOJI_PATH,
    DATA_DIR,
    ERISKII_AXES,
    ERISKII_AXES_TSV,
    FIGURES_DIR,
)
from llmoji.eriskii import compute_axis_vectors, project_onto_axes
from llmoji.eriskii_prompts import AXIS_ANCHORS


def _use_cjk_font() -> None:
    """Same fallback chain used in analysis.py / emotional_analysis.py /
    09_claude_faces_plot.py — copy here for consistency."""
    import matplotlib
    import matplotlib.font_manager as fm
    chain = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif",
        "Tahoma", "Noto Sans Canadian Aboriginal", "Heiti TC",
        "Hiragino Sans", "Apple Symbols",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


def section_axes(
    fw: list[str],
    n: np.ndarray,
    P: np.ndarray,
) -> pd.DataFrame:
    """Write eriskii_axes.tsv + one ranked-bar figure per axis."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"first_word": fw, "n": n})
    for j, name in enumerate(ERISKII_AXES):
        df[name] = P[:, j]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ERISKII_AXES_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_AXES_TSV}  ({len(df)} kaomoji × {len(ERISKII_AXES)} axes)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(ERISKII_AXES):
        scores = P[:, j]
        order = np.argsort(-scores)
        top = order[:15]
        bot = order[-15:][::-1]
        idxs = list(top) + list(bot)
        labels = [fw[i] for i in idxs]
        vals = [scores[i] for i in idxs]
        counts = [n[i] for i in idxs]

        fig, ax = plt.subplots(figsize=(6, 8))
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        y = np.arange(len(idxs))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.axhline(14.5, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(f"{name} projection (cosine)")
        ax.set_title(f"top-15 / bottom-15 on {name}\n(bar color = emission count)")
        fig.tight_layout()
        out = FIGURES_DIR / f"eriskii_axis_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
    return df


def section_clusters(
    fw: list[str],
    n: np.ndarray,
    E: np.ndarray,
    descriptions_by_fw: dict[str, str],
) -> pd.DataFrame:
    """t-SNE + KMeans(k=15) + Haiku per-cluster labels.

    `descriptions_by_fw` maps first_word → synthesized one-sentence
    description (the Stage-B output of script 14)."""
    import os
    import anthropic
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    from llmoji.config import (
        ERISKII_CLUSTERS_TSV, HAIKU_MODEL_ID,
    )
    from llmoji.eriskii import label_cluster_via_haiku
    from llmoji.eriskii_prompts import CLUSTER_LABEL_PROMPT

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    print("computing t-SNE...")
    perp = max(5, min(30, (len(fw) - 1) // 4))
    xy = TSNE(
        n_components=2, metric="cosine", perplexity=perp,
        init="pca", learning_rate="auto", random_state=0,
    ).fit_transform(E)

    print("computing KMeans(k=15)...")
    k = min(15, len(fw))
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    clusters = km.fit_predict(E)

    print("requesting cluster labels from Haiku...")
    client = anthropic.Anthropic()
    cluster_labels: dict[int, str] = {}
    cluster_rows = []
    for c in sorted(set(int(x) for x in clusters)):
        member_idx = [i for i, ci in enumerate(clusters) if int(ci) == c]
        members: list[tuple[str, str]] = [
            (fw[i], descriptions_by_fw.get(fw[i], "")) for i in member_idx
        ]
        try:
            label = label_cluster_via_haiku(
                client, members,
                model_id=HAIKU_MODEL_ID,
                prompt_template=CLUSTER_LABEL_PROMPT,
            )
        except Exception as e:
            print(f"  cluster {c}: Haiku error {e}; using placeholder")
            label = f"cluster-{c}"
        cluster_labels[c] = label
        members_str = ", ".join(fw[i] for i in member_idx)
        cluster_rows.append({
            "cluster_id": c,
            "label": label,
            "n": len(member_idx),
            "members": members_str,
        })
        print(f"  cluster {c} (n={len(member_idx)}): {label}")

    df_clusters = pd.DataFrame(cluster_rows)
    df_clusters.to_csv(ERISKII_CLUSTERS_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_CLUSTERS_TSV}")

    # labeled t-SNE figure
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors
    fig, ax = plt.subplots(figsize=(14, 10))
    sizes = np.clip(15 + 60 * np.log1p(n), 15, 250)
    colors = [palette[int(c) % len(palette)] for c in clusters]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    # annotate top-30 most frequent kaomoji
    top_idx = np.argsort(-n)[:30]
    for i in top_idx:
        ax.annotate(fw[i], xy=(xy[i, 0], xy[i, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, color="#222")

    # cluster name at each cluster centroid
    for c in sorted(cluster_labels):
        mask = clusters == c
        cx = float(xy[mask, 0].mean())
        cy = float(xy[mask, 1].mean())
        ax.text(cx, cy, cluster_labels[c],
                fontsize=10, fontweight="bold", color="#111",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=palette[c % len(palette)], alpha=0.9))

    ax.set_title(f"Eriskii-replication t-SNE + KMeans(k={k}), Haiku-labeled clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out = FIGURES_DIR / "eriskii_clusters_tsne.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df_clusters


def _heatmap(
    df_long: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    out_path: Path,
    title: str,
):
    """Pivot long-form to (group × axis), draw heatmap, save."""
    import matplotlib.pyplot as plt
    pivot = df_long.pivot(index=group_col, columns="axis", values=value_col)
    # preserve canonical axis order
    pivot = pivot[ERISKII_AXES]
    pivot = pivot.sort_index()  # alphabetical groups; tweak if needed

    fig, ax = plt.subplots(figsize=(13, max(2, 0.5 * len(pivot) + 2)))
    vmin, vmax = float(np.nanmin(pivot.values)), float(np.nanmax(pivot.values))
    if value_col == "mean":
        # diverging: center at 0
        vabs = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        im = ax.imshow(pivot.values, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
    else:
        cmap = "viridis"
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ERISKII_AXES)))
    ax.set_xticklabels(ERISKII_AXES, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def section_per_model(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    from llmoji.config import ERISKII_PER_MODEL_TSV
    from llmoji.eriskii import weighted_group_axis_stats
    cc = rows[rows["source"] == "claude-code"].copy()
    cc["model"] = cc["model"].fillna("(unknown)")
    df = weighted_group_axis_stats(
        cc, axes_df,
        group_col="model", axis_names=ERISKII_AXES, min_emissions=10,
    )
    df.to_csv(ERISKII_PER_MODEL_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_PER_MODEL_TSV}")
    if not df.empty:
        _heatmap(df, group_col="model", value_col="mean",
                 out_path=FIGURES_DIR / "eriskii_per_model_axes_mean.png",
                 title="per-model axis mean (claude-code only, n≥10 emissions)")
        _heatmap(df, group_col="model", value_col="std",
                 out_path=FIGURES_DIR / "eriskii_per_model_axes_std.png",
                 title="per-model axis std (range)")
    return df


def section_per_project(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    from llmoji.config import ERISKII_PER_PROJECT_TSV
    from llmoji.eriskii import weighted_group_axis_stats
    cc = rows[rows["source"] == "claude-code"].copy()
    df = weighted_group_axis_stats(
        cc, axes_df,
        group_col="project_slug", axis_names=ERISKII_AXES,
        min_emissions=10,
    )
    df.to_csv(ERISKII_PER_PROJECT_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_PER_PROJECT_TSV}")
    if not df.empty:
        _heatmap(df, group_col="project_slug", value_col="mean",
                 out_path=FIGURES_DIR / "eriskii_per_project_axes_mean.png",
                 title="per-project axis mean (n≥10 emissions)")
        _heatmap(df, group_col="project_slug", value_col="std",
                 out_path=FIGURES_DIR / "eriskii_per_project_axes_std.png",
                 title="per-project axis std (range)")
    return df


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
              "run scripts/15 first")
        sys.exit(1)
    _use_cjk_font()

    print("loading description embeddings...")
    fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} kaomoji, {E.shape[1]}-dim")

    print("computing axis vectors...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    axes = compute_axis_vectors(embedder, AXIS_ANCHORS)

    print("projecting kaomoji onto axes...")
    P = project_onto_axes(E, axes, ERISKII_AXES)

    print("\n=== Section: axes ===")
    df_axes = section_axes(fw, n, P)

    print("\n=== Section: clusters ===")
    # synthesized descriptions: one canonical string per kaomoji
    import json
    from llmoji.config import CLAUDE_HAIKU_SYNTHESIZED_PATH
    descriptions_by_fw: dict[str, str] = {}
    with open(CLAUDE_HAIKU_SYNTHESIZED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            descriptions_by_fw[r["first_word"]] = r["synthesized"]
    section_clusters(fw, n, E, descriptions_by_fw)

    print("\n=== Section: per-model ===")
    rows = pd.read_json(CLAUDE_KAOMOJI_PATH, lines=True)
    section_per_model(rows, df_axes)
    print("\n=== Section: per-project ===")
    section_per_project(rows, df_axes)


if __name__ == "__main__":
    main()
