# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Eriskii-style Claude-faces plot.

Panel A: t-SNE + HDBSCAN auto-clustering (noise in gray).
Panel B: t-SNE + KMeans(k=15) for eriskii parity.
Both panels: top-30 most-frequent kaomoji annotated; point size ~
log(count); cluster-centroid id labels.

Also writes figures/claude_faces_interactive.html (plotly) with
hover-tooltips showing kaomoji + count + cluster id, matching eriskii's
interactive explorer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_faces import load_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_PATH, FIGURES_DIR


def _use_cjk_font() -> None:
    import matplotlib
    import matplotlib.font_manager as fm
    preferred = [
        "Noto Sans CJK JP",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Maru Gothic ProN",
        "Apple Color Emoji", "Yu Gothic", "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return


def _tsne_2d(E: np.ndarray, *, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = len(E)
    perplexity = max(5, min(30, (n - 1) // 4))
    model = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity,
        init="pca", learning_rate="auto", random_state=seed,
    )
    return model.fit_transform(E)


def _hdbscan(E: np.ndarray) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    model = HDBSCAN(metric="cosine", min_cluster_size=3, min_samples=2)
    return model.fit_predict(E)


def _kmeans(E: np.ndarray, *, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    if len(E) <= k:
        # degenerate: one cluster per point
        return np.arange(len(E))
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return model.fit_predict(E)


def _plot_panel(
    ax,
    xy: np.ndarray,
    labels: list[str],
    counts: np.ndarray,
    clusters: np.ndarray,
    *,
    title: str,
    annotate_top: int = 30,
):
    import matplotlib.pyplot as plt

    uniq = sorted(set(int(c) for c in clusters))
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors  # 40 colors
    cluster_color = {c: ("#bbbbbb" if c == -1 else palette[i % len(palette)])
                     for i, c in enumerate(uniq)}
    colors = [cluster_color[int(c)] for c in clusters]

    # size by log(count), floor at 15, cap at 250
    sizes = 15 + 60 * np.log1p(counts)
    sizes = np.clip(sizes, 15, 250)

    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    # annotate top-N most frequent
    top_idx = np.argsort(-counts)[:annotate_top]
    for i in top_idx:
        ax.annotate(
            labels[i], xy=(xy[i, 0], xy[i, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#222",
        )

    # cluster id at each cluster centroid (skip -1 noise)
    for c in uniq:
        if c == -1:
            continue
        mask = clusters == c
        cx = float(xy[mask, 0].mean())
        cy = float(xy[mask, 1].mean())
        ax.text(cx, cy, str(c), fontsize=11, fontweight="bold",
                color="#111", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=cluster_color[c], alpha=0.9))

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])


def _write_interactive(
    xy: np.ndarray, labels: list[str], counts: np.ndarray, clusters: np.ndarray,
    out_path: Path,
) -> None:
    """Optional plotly HTML. Falls back silently if plotly isn't installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  (plotly not installed; skipping {out_path.name})")
        return
    hover_text = [
        f"{lab}  n={n}  cluster={c}"
        for lab, n, c in zip(labels, counts, clusters)
    ]
    fig = go.Figure(data=[
        go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode="markers",
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                size=np.clip(6 + 4 * np.log1p(counts), 6, 30).tolist(),
                color=clusters.tolist(),
                colorscale="Turbo",
                showscale=True,
                line=dict(color="white", width=0.4),
            ),
        )
    ])
    fig.update_layout(
        title="Claude faces — t-SNE (hover for kaomoji + cluster)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000, height=800,
    )
    fig.write_html(str(out_path))
    print(f"  wrote {out_path}")


def main() -> None:
    if not CLAUDE_FACES_EMBED_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_PATH}; run scripts/08_claude_faces_embed.py first")
        return
    _use_cjk_font()
    labels, counts, E = load_embeddings(CLAUDE_FACES_EMBED_PATH)
    print(f"loaded {len(labels)} kaomoji embeddings, dim={E.shape[1]}")

    if len(labels) < 3:
        print("need at least 3 kaomoji to plot; exiting")
        return

    import matplotlib.pyplot as plt

    print("computing t-SNE...")
    xy = _tsne_2d(E)
    print("computing HDBSCAN...")
    clusters_hdb = _hdbscan(E)
    print("computing KMeans(k=15)...")
    clusters_km = _kmeans(E, k=15)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _plot_panel(
        axes[0], xy, labels, counts, clusters_hdb,
        title=f"HDBSCAN (auto-k; {len(set(int(c) for c in clusters_hdb) - {-1})} clusters + noise)",
    )
    _plot_panel(
        axes[1], xy, labels, counts, clusters_km,
        title="KMeans (k=15, eriskii parity)",
    )
    fig.suptitle(
        f"Claude-faces t-SNE ({len(labels)} kaomoji, response-based embedding)",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = FIGURES_DIR / "claude_faces_tsne.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    out_html = FIGURES_DIR / "claude_faces_interactive.html"
    _write_interactive(xy, labels, counts, clusters_km, out_html)


if __name__ == "__main__":
    main()
