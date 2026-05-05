# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportMissingImports=false
"""Wild-emitted Claude faces × eriskii axes × clusters: looking for
state structure beyond the six Russell quadrants.

Motivation. The gt-priority resolution on 2405 Claude emissions
(script 22) lands 66% directly via Claude-GT, 32% via ensemble
fallback. The fallback share isn't noise — those are kaomoji Claude
*emits in deployment* but *didn't surface in the 120-prompt GT
elicitation*. If those faces cluster in description-embedding space in
ways that don't map cleanly to HP/LP/HN-D/HN-S/LN/NB, that's evidence
for state axes orthogonal to Russell's valence × arousal — candidates
to elicit in a positive/neutral follow-on (focus, cognitive load,
warmth, uncertainty).

Pipeline (post-hoc, no new generation, zero welfare cost):

  1. Pool wild emissions from ``~/.claude/kaomoji-journal.jsonl`` +
     claude.ai exports (same loaders as script 22).
  2. Resolve each face via gt-priority → ``(quadrant, source)``;
     ``source ∈ {gt, pred}``.
  3. Inner-join with the description-embedded face set
     (``claude_faces_embed_description.parquet``). Coverage drop is
     the long tail of un-described kaomoji; report and accept.
  4. PCA on the 384-d description embeddings of wild-labeled faces;
     scatter colored by quadrant, marker shape by source, sized by
     log emit count.
  5. KMeans on the same 384-d embeddings (all wild-labeled, gt+pred
     pooled). k chosen by silhouette over a small grid. Per cluster:
     Haiku label, mean projection onto the 21 eriskii axes (top-3 /
     bottom-3), quadrant composition, gt-fraction.
  6. t-SNE layout for the cluster figure (cosine metric, perplexity
     auto-scaled). Plotly HTML mirror.
  7. Auto-regenerated Setup + Cluster table at
     ``docs/2026-MM-DD-residual-state-axes.md``. The Findings /
     Implications / Caveats sections after the auto-table are
     **hand-edited** and will be overwritten by re-running this
     script — preserve them before regenerating.

The residual-cluster signature we're looking for: high pred-fraction
+ diffuse quadrant composition + axis profile that doesn't read as
any single Russell cell. That subset is the candidate "states beyond
the six." Quadrant-aligned clusters with high gt-fraction are the
controls — they tell us the elicitation set hit those cells.

Usage:
    ANTHROPIC_API_KEY=... python scripts/67_wild_eriskii_residual.py
    # or, no Haiku labels (useful for dry-runs):
    python scripts/67_wild_eriskii_residual.py --no-haiku
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji.sources.journal import iter_journal
from llmoji.taxonomy import canonicalize_kaomoji
from llmoji_study.claude_faces import EMBED_MODEL, load_embeddings
from llmoji_study.claude_gt import load_claude_gt
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    DATA_DIR,
    ERISKII_AXES,
    FIGURES_DIR,
    HAIKU_MODEL_ID,
    REPO_ROOT,
)
from llmoji_study.emotional_analysis import QUADRANT_COLORS
from llmoji_study.eriskii import (
    compute_axis_vectors, label_cluster_via_haiku, project_onto_axes,
)
from llmoji_study.eriskii_anchors import AXIS_ANCHORS, CLUSTER_LABEL_PROMPT

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
HARNESS_DATA_DIR = DATA_DIR / "harness"
HARNESS_FIG_DIR = FIGURES_DIR / "harness"
DOCS_DIR = REPO_ROOT / "docs"

DEFAULT_CLAUDE_EXPORTS = [
    Path("/Users/a9lim/Downloads/"
         "data-72de1230-b9fa-4c55-bc10-84a35b58d89c-1777763577-c21ac4ff-batch-0000/"
         "conversations.json"),
    Path("/Users/a9lim/Downloads/"
         "9cc23402cbb1e8aec9785eb0f750f1c442f1ba13e507bd6b04a727c627d64d08-"
         "2026-04-28-01-04-53-1d1e60e8c10441b1881c7e81834c3b26/"
         "conversations.json"),
]


# --------------------------------------------------------------------- fonts
def _use_cjk_font() -> None:
    """Same fallback chain used elsewhere in the project."""
    import matplotlib
    import matplotlib.font_manager as fm

    emoji_font = REPO_ROOT / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    if emoji_font.exists() and "Noto Emoji" not in {f.name for f in fm.fontManager.ttflist}:
        try:
            fm.fontManager.addfont(str(emoji_font))
        except Exception:
            pass
    chain = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif",
        "Tahoma", "Noto Sans Canadian Aboriginal", "Heiti TC",
        "Hiragino Sans", "Apple Symbols", "Noto Emoji", "Helvetica Neue",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


# --------------------------------------------------------------------- IO
def _to_float(v) -> float:
    try:
        if v is None or (isinstance(v, str) and v == ""):
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _load_ensemble_predictions(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    out: dict[str, dict] = {}
    for rec in df.to_dict(orient="records"):
        f = str(rec["first_word"])
        entry: dict = {
            "ensemble_pred": str(rec["ensemble_pred"]),
            "ensemble_conf": _to_float(rec.get("ensemble_conf")),
        }
        for q in QUADRANTS:
            entry[f"p_{q}"] = _to_float(rec.get(f"ensemble_p_{q}"))
        out[f] = entry
    return out


def _emissions_from_journal(path: Path, source: str) -> list[str]:
    rows: list[str] = []
    if not path.exists():
        print(f"  skip {path} (missing)")
        return rows
    for sr in iter_journal(path, source=source):
        face = sr.first_word or ""
        if not face:
            continue
        canon = canonicalize_kaomoji(face)
        if not canon:
            continue
        rows.append(canon)
    print(f"  {path.name}: {len(rows)} emissions")
    return rows


def _emissions_from_claude_export(paths: list[Path]) -> list[str]:
    rows: list[str] = []
    export_dirs: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"  skip {path} (missing)")
            continue
        export_dirs.append(path.parent if path.is_file() else path)
    if not export_dirs:
        return rows
    try:
        from llmoji.sources.claude_export import iter_claude_export
    except ImportError:
        print("  skip claude.ai export (llmoji.sources.claude_export not available)")
        return rows
    for sr in iter_claude_export(export_dirs):
        face = sr.first_word or ""
        if not face:
            continue
        canon = canonicalize_kaomoji(face)
        if not canon:
            continue
        rows.append(canon)
    print(f"  claude.ai: {len(rows)} emissions across {len(export_dirs)} export(s)")
    return rows


def _resolve(
    face: str,
    gt: dict[str, tuple[str, int]],
    preds: dict[str, dict],
) -> tuple[str | None, str]:
    """gt-priority: GT first, ensemble fallback, then unknown."""
    if face in gt:
        return gt[face][0], "gt"
    if face in preds:
        return preds[face]["ensemble_pred"], "pred"
    return None, "unknown"


def _representative_descriptions(corpus_path: Path) -> dict[str, str]:
    """Most-evidenced description per canonical kaomoji."""
    out: dict[str, str] = {}
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            descs = r.get("descriptions", [])
            if not descs:
                continue
            out[r["kaomoji"]] = descs[0]["description"]
    return out


# --------------------------------------------------------------------- analysis
def _pick_k_silhouette(E: np.ndarray, k_grid: list[int]) -> tuple[int, dict[int, float]]:
    """Silhouette score over k_grid; pick argmax. Cosine metric to
    match the t-SNE layout we'll show."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    for k in k_grid:
        if k >= E.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(E)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(E, labels, metric="cosine"))
        scores[k] = s
        print(f"  k={k:2d}  silhouette={s:.4f}")
    if not scores:
        return min(k_grid), {}
    best = max(scores, key=lambda kk: scores[kk])
    return best, scores


def _cluster_axis_profile(
    cluster_idxs: list[int],
    df_axes_subset: pd.DataFrame,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return ``(top3_positive, top3_negative)`` axis-name + cluster-mean
    projection, where the cluster-mean is over the 21 eriskii axes."""
    sub = df_axes_subset.iloc[cluster_idxs]
    means: list[tuple[str, float]] = []
    for axis in ERISKII_AXES:
        if axis in sub.columns:
            means.append((axis, float(sub[axis].mean())))
    means.sort(key=lambda kv: kv[1], reverse=True)
    return means[:3], means[-3:][::-1]


def _quadrant_composition(
    cluster_idxs: list[int],
    quadrant_per_face: list[str | None],
    weights: list[int],
) -> tuple[dict[str, float], int]:
    """Return ``(share_per_quadrant, total_weighted_emits)``. Shares
    are emit-weighted (a heavy-use face contributes more than a one-off)."""
    counts: Counter[str] = Counter()
    total = 0
    for i in cluster_idxs:
        q = quadrant_per_face[i]
        w = weights[i]
        if q is None:
            continue
        counts[q] += w
        total += w
    if total == 0:
        return {q: 0.0 for q in QUADRANTS}, 0
    return {q: counts.get(q, 0) / total for q in QUADRANTS}, total


# --------------------------------------------------------------------- figures
def _scatter_pca(
    coords: np.ndarray,
    quadrants: list[str | None],
    sources: list[str],
    sizes: np.ndarray,
    fws: list[str],
    *,
    out_png: Path,
    title: str,
    subtitle: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    for source_label, marker in (("gt", "o"), ("pred", "^")):
        for q in QUADRANTS:
            mask = np.array([
                qi == q and si == source_label
                for qi, si in zip(quadrants, sources)
            ])
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=sizes[mask], c=QUADRANT_COLORS[q],
                marker=marker, alpha=0.78,
                edgecolor="white", linewidth=0.5,
                label=f"{q} ({source_label})",
            )
    # Annotate the heaviest emitters.
    top = np.argsort(-sizes)[:25]
    for i in top:
        ax.annotate(
            fws[i], xy=(coords[i, 0], coords[i, 1]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8, color="#222",
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Single legend column: 6 quadrants × 2 sources = 12 entries; trim
    # via dedup.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(
        [h for h, _ in uniq], [l for _, l in uniq],
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=8, ncol=1, title="quadrant (source)",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _scatter_pca_html(
    coords: np.ndarray,
    quadrants: list[str | None],
    sources: list[str],
    fws: list[str],
    sizes_log: np.ndarray,
    descriptions_by_fw: dict[str, str],
    *,
    out_path: Path,
    title: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  (plotly missing; skipping {out_path.name})")
        return

    def _truncate(s: str, lim: int = 140) -> str:
        return s if len(s) <= lim else s[:lim].rstrip() + "…"

    fig = go.Figure()
    for source_label, symbol in (("gt", "circle"), ("pred", "triangle-up")):
        for q in QUADRANTS:
            idxs = [i for i, (qi, si) in enumerate(zip(quadrants, sources))
                     if qi == q and si == source_label]
            if not idxs:
                continue
            hover = [
                f"<b>{fws[i]}</b><br>quadrant: {q}<br>source: {source_label}<br>"
                f"emit weight (log): {sizes_log[i]:.2f}<br>"
                f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
                for i in idxs
            ]
            fig.add_trace(go.Scatter(
                x=coords[idxs, 0], y=coords[idxs, 1],
                mode="markers",
                name=f"{q} ({source_label})",
                text=hover,
                hoverinfo="text",
                marker=dict(
                    size=[float(8 + 6 * sizes_log[i]) for i in idxs],
                    color=QUADRANT_COLORS[q],
                    symbol=symbol,
                    line=dict(color="white", width=0.6),
                    opacity=0.85,
                ),
            ))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(title="quadrant (source)", itemsizing="constant"),
        width=1200, height=900,
    )
    fig.write_html(str(out_path))
    print(f"wrote {out_path}")


def _scatter_pca_3d_html(
    coords3: np.ndarray,
    quadrants: list[str | None],
    sources: list[str],
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
    fws: list[str],
    sizes_log: np.ndarray,
    descriptions_by_fw: dict[str, str],
    *,
    out_path: Path,
    title: str,
    evr: np.ndarray,
) -> None:
    """Two side-by-side 3D scenes on the same PCA(3) coords:

      - left:  colored by gt-priority quadrant, circle=gt, diamond=pred
      - right: colored by KMeans cluster id, with Haiku label in legend

    Match style of ``scripts/local/97_build_per_face_pca_3d.py``: grey
    axis backgrounds, aspectmode='cube', plotly-cdn embed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(f"  (plotly missing; skipping {out_path.name})")
        return

    def _truncate(s: str, lim: int = 140) -> str:
        return s if len(s) <= lim else s[:lim].rstrip() + "…"

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a",
    ]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Colored by gt-priority quadrant (○=gt, ◆=pred)",
            "Colored by KMeans cluster (k={})".format(len(cluster_labels)),
        ),
        horizontal_spacing=0.04,
    )

    sizes = np.clip(6 + 5 * sizes_log, 6, 26)

    # --- LEFT scene: by quadrant × source ---------------------------------
    for source_label, symbol in (("gt", "circle"), ("pred", "diamond")):
        for q in QUADRANTS:
            idxs = [i for i, (qi, si) in enumerate(zip(quadrants, sources))
                     if qi == q and si == source_label]
            if not idxs:
                continue
            hover = [
                f"<b>{fws[i]}</b><br>quadrant: {q}<br>source: {source_label}<br>"
                f"emit weight (log): {sizes_log[i]:.2f}<br>"
                f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
                for i in idxs
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
                    mode="markers",
                    name=f"{q} ({source_label})",
                    legendgroup=f"left-{q}-{source_label}",
                    marker=dict(
                        size=[float(sizes[i]) for i in idxs],
                        color=QUADRANT_COLORS[q],
                        symbol=symbol,
                        line=dict(color="black", width=0.4),
                        opacity=0.85,
                    ),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover,
                    scene="scene",
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # --- RIGHT scene: by cluster -----------------------------------------
    for c in sorted(cluster_labels):
        idxs = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        if not idxs:
            continue
        label = cluster_labels[c]
        color = palette[c % len(palette)]
        hover = [
            f"<b>{fws[i]}</b><br>cluster {c}: {label}<br>"
            f"quadrant: {quadrants[i]}<br>source: {sources[i]}<br>"
            f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter3d(
                x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
                mode="markers",
                name=f"{c}: {label}",
                legendgroup=f"right-c{c}",
                marker=dict(
                    size=[float(sizes[i]) for i in idxs],
                    color=color,
                    symbol="circle",
                    line=dict(color="black", width=0.4),
                    opacity=0.85,
                ),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover,
                scene="scene2",
                showlegend=True,
            ),
            row=1, col=2,
        )

    # Subtitle string with explained variance per axis.
    pc_title = (
        f"PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%, "
        f"PC3 {evr[2]*100:.1f}%  (sum {sum(evr[:3])*100:.1f}%)"
    )

    common_axes = dict(
        backgroundcolor="#f8f8f8",
        showbackground=True,
    )
    fig.update_layout(
        height=820, width=1700,
        title=dict(
            text=f"{title}<br><sub>{pc_title}</sub>",
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title=dict(text="PC1"), **common_axes),
            yaxis=dict(title=dict(text="PC2"), **common_axes),
            zaxis=dict(title=dict(text="PC3"), **common_axes),
            aspectmode="cube",
        ),
        scene2=dict(
            xaxis=dict(title=dict(text="PC1"), **common_axes),
            yaxis=dict(title=dict(text="PC2"), **common_axes),
            zaxis=dict(title=dict(text="PC3"), **common_axes),
            aspectmode="cube",
        ),
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
            tracegroupgap=4,
        ),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out_path}")


def _scatter_clusters(
    xy: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
    fws: list[str],
    sizes: np.ndarray,
    *,
    out_png: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    fig, ax = plt.subplots(figsize=(13, 9))
    colors = [palette[int(c) % len(palette)] for c in cluster_ids]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.82,
               edgecolor="white", linewidth=0.4)
    top = np.argsort(-sizes)[:30]
    for i in top:
        ax.annotate(fws[i], xy=(xy[i, 0], xy[i, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color="#222")
    for c in sorted(cluster_labels):
        mask = cluster_ids == c
        if not mask.any():
            continue
        cx = float(xy[mask, 0].mean())
        cy = float(xy[mask, 1].mean())
        ax.text(cx, cy, cluster_labels[c],
                fontsize=10, fontweight="bold", color="#111",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=palette[c % len(palette)], alpha=0.92))
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-journal",
                    default=str(Path.home() / ".claude" / "kaomoji-journal.jsonl"))
    ap.add_argument("--claude-export",
                    default=",".join(str(p) for p in DEFAULT_CLAUDE_EXPORTS))
    ap.add_argument("--ensemble-tsv",
                    default=str(DATA_DIR / "local" / "face_likelihood_ensemble_predict.tsv"))
    ap.add_argument("--claude-gt-floor", type=int, default=1)
    ap.add_argument("--gt-only", action="store_true",
                    help="restrict to faces with Claude-GT labels (no "
                         "ensemble-fallback predictions). Trades coverage "
                         "for label trustworthiness. Outputs land in "
                         "*_gt_only.{tsv,png,html} so both modes coexist.")
    ap.add_argument("--k-grid", default="2,3,4,5,6,7,8,10,12,14",
                    help="silhouette grid for KMeans")
    ap.add_argument("--fixed-k", type=int, default=6,
                    help="force this k regardless of silhouette winner; "
                         "set <=0 to use silhouette argmax. Default 6 — "
                         "the local-maximum-after-coarse-modes scale where "
                         "structure resolves interpretably (see writeup).")
    ap.add_argument("--no-haiku", action="store_true",
                    help="skip Haiku cluster labeling (placeholder labels)")
    ap.add_argument("--reuse-labels", action="store_true",
                    help="reuse cluster labels from a previous run's "
                         "wild_residual_clusters.tsv if cluster sizes match "
                         "(saves Haiku calls on iterative re-runs)")
    ap.add_argument("--out-doc",
                    default=str(DOCS_DIR / "2026-05-05-residual-state-axes.md"))
    args = ap.parse_args()

    HARNESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Output-filename suffix so gt-only and gt-priority modes coexist.
    suffix = "_gt_only" if args.gt_only else ""
    print(f"mode: {'gt-only (Claude-GT labels only)' if args.gt_only else 'gt-priority (GT + ensemble fallback)'}")

    # --- load resolution sources -------------------------------------------
    print(f"loading Claude-GT (floor={args.claude_gt_floor}) ...")
    gt = load_claude_gt(floor=args.claude_gt_floor)
    print(f"  {len(gt)} faces in Claude-GT")
    print(f"loading ensemble predictions from {args.ensemble_tsv} ...")
    preds = _load_ensemble_predictions(Path(args.ensemble_tsv))
    print(f"  {len(preds)} faces in ensemble TSV")

    # --- load wild emissions -----------------------------------------------
    emissions: list[str] = []
    print("\nloading Claude Code journal ...")
    emissions += _emissions_from_journal(Path(args.claude_journal), "claude_code")
    export_paths = [Path(p.strip()) for p in args.claude_export.split(",") if p.strip()]
    if export_paths:
        print(f"\nloading {len(export_paths)} claude.ai export(s) ...")
        emissions += _emissions_from_claude_export(export_paths)
    n_total_emit = len(emissions)
    emit_counts: Counter[str] = Counter(emissions)
    print(f"\ntotal wild emissions: {n_total_emit} across {len(emit_counts)} unique faces")

    # --- load description embeddings + project onto axes ------------------
    print(f"\nloading description embeddings from "
          f"{CLAUDE_FACES_EMBED_DESCRIPTION_PATH.name} ...")
    fw_all, _, E_all = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    fw_to_idx = {f: i for i, f in enumerate(fw_all)}
    print(f"  {len(fw_all)} described faces, dim={E_all.shape[1]}")

    print("computing eriskii axis vectors (anchor pairs → 21 axes) ...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    axis_vectors = compute_axis_vectors(embedder, AXIS_ANCHORS)
    P_all = project_onto_axes(E_all, axis_vectors, ERISKII_AXES)
    df_axes = pd.DataFrame(P_all, columns=ERISKII_AXES, index=fw_all)
    df_axes.index.name = "first_word"

    # --- inner-join: face must be wild-emitted AND have description --------
    rows: list[dict] = []
    for face, n_emit in emit_counts.items():
        if face not in fw_to_idx:
            continue
        q, src = _resolve(face, gt, preds)
        if q is None:
            continue  # un-resolved by gt-priority
        if args.gt_only and src != "gt":
            continue  # gt-only mode: skip ensemble-fallback labels
        rows.append({
            "first_word": face,
            "n_emit": int(n_emit),
            "quadrant": q,
            "source": src,
            "embed_idx": fw_to_idx[face],
        })

    if not rows:
        sys.exit("no wild-labeled faces survived the inner-join — aborting")

    df_wild = pd.DataFrame(rows).sort_values("n_emit", ascending=False).reset_index(drop=True)
    n_unique = len(emit_counts)
    n_kept = len(df_wild)
    print(f"\n{n_kept} / {n_unique} unique wild faces kept "
          f"(have a description AND a gt-priority resolution)")
    print(f"  by source: gt={(df_wild['source']=='gt').sum()}  "
          f"pred={(df_wild['source']=='pred').sum()}")

    n_emit_kept = int(df_wild["n_emit"].sum())
    print(f"  emit coverage: {n_emit_kept} / {n_total_emit} "
          f"({n_emit_kept/max(n_total_emit,1)*100:.1f}%)")

    # --- attach axis projections to df_wild --------------------------------
    df_wild = df_wild.merge(df_axes, how="left", left_on="first_word", right_index=True)
    # script-16's eriskii_axes.tsv usually covers everything in the
    # embeddings parquet, but be defensive about NaNs in axis cols.
    missing_axes = df_wild[ERISKII_AXES].isna().any(axis=1).sum()
    if missing_axes:
        print(f"  (warn) {missing_axes} faces have a description but no eriskii axes; "
              "their axis profiles will read as NaN — usually a script-16 staleness issue")

    # Save the labeled-wild table.
    out_tsv = HARNESS_DATA_DIR / f"wild_faces_labeled{suffix}.tsv"
    cols = ["first_word", "n_emit", "quadrant", "source"] + [
        a for a in ERISKII_AXES if a in df_wild.columns
    ]
    df_wild[cols].to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    # --- assemble matrices for downstream ----------------------------------
    idxs = df_wild["embed_idx"].to_numpy()
    E = E_all[idxs]
    fws = df_wild["first_word"].tolist()
    quadrants: list[str | None] = df_wild["quadrant"].tolist()
    sources: list[str] = df_wild["source"].tolist()
    weights = df_wild["n_emit"].astype(int).tolist()
    sizes_log = np.log1p(np.asarray(weights, dtype=float))
    sizes = np.clip(15 + 50 * sizes_log, 15, 280)

    # --- representative descriptions for hover + Haiku -----------------------
    descriptions_by_fw = _representative_descriptions(CLAUDE_DESCRIPTIONS_PATH)

    # --- PCA (3 components; we plot PC1×PC2 for the static, PC1×PC2×PC3 3D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=0)
    coords3 = pca.fit_transform(E)
    coords_pca = coords3[:, :2]
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: "
          f"PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f} "
          f"(sum={sum(evr):.3f})")

    _use_cjk_font()

    pc_subtitle = (
        f"{n_kept} unique faces · {n_emit_kept} emissions · "
        f"PC1+PC2 = {sum(evr[:2])*100:.1f}% var · "
        f"gt={(df_wild['source']=='gt').sum()}/{n_kept} · "
        f"pred={(df_wild['source']=='pred').sum()}/{n_kept}"
    )
    mode_subtitle = " · gt-only (no ensemble fallback)" if args.gt_only else ""
    _scatter_pca(
        coords_pca, quadrants, sources, sizes, fws,
        out_png=HARNESS_FIG_DIR / f"wild_faces_eriskii_pca{suffix}.png",
        title=f"Wild-emitted Claude faces — PCA on description embeddings{mode_subtitle}",
        subtitle=pc_subtitle,
    )
    _scatter_pca_html(
        coords_pca, quadrants, sources, fws, sizes_log, descriptions_by_fw,
        out_path=HARNESS_FIG_DIR / f"wild_faces_eriskii_pca{suffix}.html",
        title=(f"Wild-emitted Claude faces — PCA on description embeddings{mode_subtitle}, "
                "colored by gt-priority quadrant (○=gt, △=pred)"),
    )

    # --- KMeans on description embeddings ----------------------------------
    print("\n=== clustering ===")
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    print(f"silhouette grid: {k_grid}")
    silhouette_winner, sil_scores = _pick_k_silhouette(E, k_grid)
    if args.fixed_k > 0:
        best_k = args.fixed_k
        print(f"silhouette argmax k = {silhouette_winner}  "
              f"(s={sil_scores.get(silhouette_winner, float('nan')):.4f})")
        print(f"using fixed k = {best_k}  "
              f"(s={sil_scores.get(best_k, float('nan')):.4f})  "
              "— local-max scale chosen for interpretability over raw silhouette")
    else:
        best_k = silhouette_winner
        print(f"chosen k = {best_k}  (silhouette={sil_scores.get(best_k, float('nan')):.4f})")

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=best_k, n_init=20, random_state=0)
    cluster_ids = km.fit_predict(E)

    # --- Haiku cluster labels ----------------------------------------------
    cluster_labels: dict[int, str] = {}
    cached_labels: dict[int, str] | None = None
    if args.reuse_labels:
        prev_path = HARNESS_DATA_DIR / f"wild_residual_clusters{suffix}.tsv"
        if prev_path.exists():
            prev = pd.read_csv(prev_path, sep="\t")
            cur_sizes = {int(c): sum(1 for ci in cluster_ids if int(ci) == c)
                          for c in range(best_k)}
            prev_sizes = {int(r["cluster_id"]): int(r["n_faces"])
                           for _, r in prev.iterrows()}
            if cur_sizes == prev_sizes:
                cached_labels = {int(r["cluster_id"]): str(r["label"])
                                  for _, r in prev.iterrows()}
                print(f"reusing {len(cached_labels)} cluster labels from "
                      f"{prev_path.name} (sizes match)")
            else:
                print(f"  (reuse-labels: size mismatch, re-labeling: "
                      f"prev={prev_sizes} cur={cur_sizes})")

    if cached_labels is not None:
        cluster_labels = cached_labels
    elif args.no_haiku:
        for c in range(best_k):
            cluster_labels[c] = f"cluster-{c}"
        print("(Haiku labeling skipped via --no-haiku)")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("(no ANTHROPIC_API_KEY; using placeholder cluster labels)")
        for c in range(best_k):
            cluster_labels[c] = f"cluster-{c}"
    else:
        import anthropic
        client = anthropic.Anthropic()
        print("requesting Haiku cluster labels ...")
        for c in range(best_k):
            member_idx = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
            members = [(fws[i], descriptions_by_fw.get(fws[i], "")) for i in member_idx]
            try:
                label = label_cluster_via_haiku(
                    client, members,
                    model_id=HAIKU_MODEL_ID,
                    prompt_template=CLUSTER_LABEL_PROMPT,
                )
            except Exception as e:
                print(f"  cluster {c}: Haiku error {e}; placeholder")
                label = f"cluster-{c}"
            cluster_labels[c] = label
            print(f"  cluster {c} (n={len(member_idx)}): {label}")

    # --- per-cluster analytics ---------------------------------------------
    rows_clusters = []
    for c in range(best_k):
        member_idx = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        share, total_w = _quadrant_composition(member_idx, quadrants, weights)
        n_gt = sum(1 for i in member_idx if sources[i] == "gt")
        n_pred = sum(1 for i in member_idx if sources[i] == "pred")
        gt_frac = n_gt / max(len(member_idx), 1)
        pred_frac = n_pred / max(len(member_idx), 1)
        modal_q = max(QUADRANTS, key=lambda q: share.get(q, 0.0))
        modal_share = share.get(modal_q, 0.0)
        # diffuseness: 1 − max-share (high → not quadrant-aligned)
        diffuseness = 1.0 - modal_share

        top_pos, top_neg = _cluster_axis_profile(member_idx, df_wild)
        members_str = ", ".join(fws[i] for i in sorted(member_idx, key=lambda j: -weights[j])[:8])
        if len(member_idx) > 8:
            members_str += f", … (+{len(member_idx)-8} more)"

        rows_clusters.append({
            "cluster_id": c,
            "label": cluster_labels[c],
            "n_faces": len(member_idx),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_frac": round(gt_frac, 3),
            "pred_frac": round(pred_frac, 3),
            "modal_quadrant": modal_q,
            "modal_share": round(modal_share, 3),
            "diffuseness": round(diffuseness, 3),
            "total_emit_weight": total_w,
            **{f"share_{q}": round(share.get(q, 0.0), 3) for q in QUADRANTS},
            "top_axes_pos": "; ".join(f"{a}:+{v:.2f}" for a, v in top_pos),
            "top_axes_neg": "; ".join(f"{a}:{v:+.2f}" for a, v in top_neg),
            "sample_members": members_str,
        })

    df_clusters = pd.DataFrame(rows_clusters).sort_values(
        ["pred_frac", "diffuseness"], ascending=[False, False]
    ).reset_index(drop=True)
    out_clusters_tsv = HARNESS_DATA_DIR / f"wild_residual_clusters{suffix}.tsv"
    df_clusters.to_csv(out_clusters_tsv, sep="\t", index=False)
    print(f"wrote {out_clusters_tsv}")

    # --- 3D PCA HTML with both colorings -----------------------------------
    _scatter_pca_3d_html(
        coords3, quadrants, sources, cluster_ids, cluster_labels,
        fws, sizes_log, descriptions_by_fw,
        out_path=HARNESS_FIG_DIR / f"wild_faces_eriskii_pca_3d{suffix}.html",
        title=(f"Wild-emitted Claude faces — PCA(3) on description embeddings{mode_subtitle}"),
        evr=evr,
    )

    # --- t-SNE layout for the cluster figure -------------------------------
    from sklearn.manifold import TSNE
    perp = max(5, min(30, (E.shape[0] - 1) // 4))
    print(f"computing t-SNE (perplexity={perp}) ...")
    xy = TSNE(
        n_components=2, metric="cosine", perplexity=perp,
        init="pca", learning_rate="auto", random_state=0,
    ).fit_transform(E)
    _scatter_clusters(
        xy, cluster_ids, cluster_labels, fws, sizes,
        out_png=HARNESS_FIG_DIR / f"wild_residual_clusters_tsne{suffix}.png",
        title=(
            f"Wild-emitted faces — KMeans(k={best_k}) on description embeddings, "
            f"Haiku-labeled (t-SNE layout){mode_subtitle}"
        ),
    )

    # --- markdown writeup --------------------------------------------------
    # Layout: auto-content up through the cluster table; everything after
    # the HAND_MARKER line is hand-edited and preserved across re-runs.
    HAND_MARKER = "<!-- HAND-EDITED BELOW THIS LINE; preserved across re-runs -->"
    today = "2026-05-05"
    lines: list[str] = []
    lines.append(f"# Residual state axes — clustering wild-emitted Claude faces")
    lines.append("")
    lines.append(f"_{today} — generated by `scripts/67_wild_eriskii_residual.py`._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- {n_total_emit} wild emissions across {n_unique} unique faces "
        "(Claude Code journal + claude.ai exports)."
    )
    lines.append(
        f"- {n_kept} faces kept after inner-joining with the contributor-side "
        f"description corpus (drops {n_unique - n_kept} un-described / "
        "un-resolved faces in the long tail)."
    )
    lines.append(
        f"- gt-priority resolution: {(df_wild['source']=='gt').sum()} via Claude-GT, "
        f"{(df_wild['source']=='pred').sum()} via ensemble fallback "
        f"(emit-weight: gt={int(df_wild[df_wild['source']=='gt']['n_emit'].sum())}, "
        f"pred={int(df_wild[df_wild['source']=='pred']['n_emit'].sum())})."
    )
    lines.append("")
    lines.append(
        f"PCA on the 384-d description embeddings: "
        f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, "
        f"PC3={evr[2]*100:.1f}% var (sum {sum(evr[:3])*100:.1f}%). "
        "Figures: 2D static + interactive at "
        "`figures/harness/wild_faces_eriskii_pca.{png,html}`; "
        "3D side-by-side (quadrant-colored vs cluster-colored) at "
        "`figures/harness/wild_faces_eriskii_pca_3d.html`."
    )
    lines.append("")
    lines.append("## Clusters")
    lines.append("")
    sil_grid_str = ", ".join(f"k={k}: {sil_scores.get(k, float('nan')):.3f}"
                             for k in k_grid if k in sil_scores)
    fixed_note = ""
    if args.fixed_k > 0 and silhouette_winner != best_k:
        fixed_note = (
            f" Silhouette argmax was k={silhouette_winner} "
            f"(s={sil_scores.get(silhouette_winner, float('nan')):.3f}); "
            f"k={best_k} fixed for interpretability — silhouette favors the "
            "coarsest split, but k=6 is a local maximum after the k=5 dip and "
            "is the scale at which the cluster Haiku labels resolve to "
            "interpretable affect / cognitive states rather than vocabulary "
            "noise."
        )
    lines.append(
        f"KMeans on the same 384-d embeddings, k={best_k}. Silhouette over "
        f"the grid: {sil_grid_str}.{fixed_note} Sorted by `pred_frac` "
        "descending — the top of the table is where Claude's deployment-time "
        "face vocabulary diverges most from the GT elicitation set."
    )
    lines.append("")
    lines.append(
        "| id | label | n | pred_frac | modal Q (share) | diffuseness | "
        "top axes (+) | top axes (−) | sample |"
    )
    lines.append(
        "|---|---|---:|---:|---|---:|---|---|---|"
    )
    for _, r in df_clusters.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {r['label']} | {int(r['n_faces'])} "
            f"| {r['pred_frac']:.2f} | {r['modal_quadrant']} ({r['modal_share']:.2f}) "
            f"| {r['diffuseness']:.2f} | {r['top_axes_pos']} | {r['top_axes_neg']} "
            f"| {r['sample_members']} |"
        )
    lines.append("")
    # Append the hand-edit marker; preserve any content past it from the
    # existing file (else seed an empty interpretation stub the user
    # fills in).
    lines.append("")
    lines.append(HAND_MARKER)
    lines.append("")

    out_doc = Path(args.out_doc)
    if args.gt_only and "_gt_only" not in out_doc.stem:
        out_doc = out_doc.with_name(out_doc.stem + "_gt_only" + out_doc.suffix)
    preserved = ""
    if out_doc.exists():
        prior = out_doc.read_text()
        if HAND_MARKER in prior:
            preserved = prior.split(HAND_MARKER, 1)[1].lstrip("\n")
    if preserved:
        lines.append(preserved.rstrip())
    else:
        lines.append("## Findings")
        lines.append("")
        lines.append("_(hand-edit this section after each run — interpretation goes here.)_")
        lines.append("")

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_doc} "
          f"({'preserved hand-edits' if preserved else 'seeded interpretation stub'})")


if __name__ == "__main__":
    main()
