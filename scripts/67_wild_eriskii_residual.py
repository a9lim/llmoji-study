# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportMissingImports=false
"""Wild-emitted Claude faces × eriskii axes × clusters: looking for
state structure beyond the six Russell quadrants.

Refactored 2026-05-05 to consume the **full HF corpus**
(``claude_descriptions.jsonl`` — 306 canonical kaomoji pooled from
``a9lim/llmoji``) rather than the local-machine wild journal alone.
Wild-only faces (not in Claude GT) are no longer displayed by their
ensemble-*predicted* quadrant; that label was a confidence-shaped
fiction. Each kept face is partitioned into one of three categories
whose **status w.r.t. the GT elicitation set is the actual signal**:

  - ``shared`` — in Claude GT ∩ HF corpus.
    Marker: circle. Color: the GT modal quadrant.
  - ``wild_claude`` — in HF corpus, claude-opus emitted at least once,
    not in Claude GT. Marker: black circle.
  - ``wild_other`` — in HF corpus, no claude-opus emission, not in
    Claude GT. Marker: black square.

The cluster table still reports quadrant composition for the shared
subset within each cluster, plus per-category face counts and a
``wild_frac``. High-``wild_frac`` / high-``wild_claude_frac`` clusters
remain the candidate "states beyond the six" — emissions from prompt
contexts the GT elicitation didn't probe.

Pipeline (post-hoc, no new generation, zero welfare cost):

  1. Load HF corpus (``claude_descriptions.jsonl``) →
     ``{face: count_total, claude_emit, top_description, ...}``.
  2. Load Claude-GT modal labels (claude-runs naturalistic +
     introspection arm).
  3. Inner-join with the description-embedded face set
     (``claude_faces_embed_description.parquet``); categorize each kept
     face.
  4. PCA(3) on the 384-d description embeddings; produce a single 3D
     side-by-side plotly HTML — left scene colored by category
     (shared = GT modal quadrant, wild_claude = black ●, wild_other =
     black ■); right scene colored by KMeans cluster id.
  5. KMeans on the same 384-d embeddings; k via silhouette grid but
     defaulted to k=6 for interpretability (see writeup).
  6. Per cluster: Haiku label, mean projection onto the 21 eriskii
     axes (top/bottom 3), category composition (shared / wild_claude /
     wild_other) plus quadrant composition over the shared subset.
  7. Auto-regenerated Setup + Cluster table at
     ``docs/2026-MM-DD-residual-state-axes.md``. Content past the
     HAND_MARKER line is preserved across re-runs.

The 2D static PNG, 2D plotly HTML, and t-SNE cluster PNG were all
deleted in the 2026-05-05 cleanup — the side-by-side 3D scene already
carries both the category-coloring and cluster-coloring views, and the
2D outputs were redundant.

Usage:
    ANTHROPIC_API_KEY=... python scripts/67_wild_eriskii_residual.py
    python scripts/67_wild_eriskii_residual.py --no-haiku
    python scripts/67_wild_eriskii_residual.py --gt-only  # shared subset only
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
CATEGORIES = ["shared", "wild_claude", "wild_other"]
WILD_COLOR = "#111111"  # uniform black for wild-only faces
HARNESS_DATA_DIR = DATA_DIR / "harness"
HARNESS_FIG_DIR = FIGURES_DIR / "harness"
DOCS_DIR = REPO_ROOT / "docs"


# --------------------------------------------------------------------- IO
def _is_claude_source_model(source_model: str) -> bool:
    """True iff a description's source_model attests a real Claude
    emission. Treats anything containing 'claude' as Claude-emitted;
    the literal '<synthetic>' source_model (haiku-generated synthesis
    prompts) is *not* an emission and naturally fails the substring
    check."""
    return "claude" in source_model.lower()


def _load_hf_corpus(path: Path) -> tuple[dict[str, dict], int]:
    """Read ``claude_descriptions.jsonl`` into per-face metadata.

    Returns ``(per_face_meta, total_emissions)`` where ``per_face_meta``
    maps canonical face → {count_total, claude_emit, top_description,
    descriptions, source_models}.
    """
    out: dict[str, dict] = {}
    total = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            face = canonicalize_kaomoji(r.get("kaomoji", ""))
            if not face:
                continue
            descs = r.get("descriptions", []) or []
            descs_sorted = sorted(descs, key=lambda d: -int(d.get("count", 0)))
            top_desc = descs_sorted[0].get("description", "") if descs_sorted else ""
            claude_emit = sum(
                int(d.get("count", 0)) for d in descs
                if _is_claude_source_model(d.get("source_model", ""))
            )
            count_total = int(r.get("count_total", 0))
            sms = sorted({d.get("source_model", "?") for d in descs})
            out[face] = {
                "count_total": count_total,
                "claude_emit": int(claude_emit),
                "top_description": top_desc,
                "descriptions": descs_sorted,
                "source_models": sms,
            }
            total += count_total
    return out, total


def _categorize(
    face: str,
    gt: dict[str, tuple[str, int]],
    claude_emit: int,
) -> tuple[str, str | None]:
    """Return ``(category, gt_quadrant_or_None)``.

    ``category`` ∈ {'shared', 'wild_claude', 'wild_other'}. Only
    ``shared`` faces carry a quadrant label — wild-only quadrants would
    be ensemble guesses, which this refactor explicitly drops.
    """
    if face in gt:
        return "shared", gt[face][0]
    if claude_emit > 0:
        return "wild_claude", None
    return "wild_other", None


# Color/marker mapping for the 3D scene's left pane:
#   shared      → QUADRANT_COLORS[gt_quadrant], circle
#   wild_claude → WILD_COLOR (black), circle
#   wild_other  → WILD_COLOR (black), square
# Inlined in `_scatter_pca_3d_html` rather than factored out — the
# plotly trace API wants the per-category branches anyway.


# --------------------------------------------------------------------- analysis
def _pick_k_silhouette(E: np.ndarray, k_grid: list[int]) -> tuple[int, dict[int, float]]:
    """Silhouette score over k_grid; pick argmax. Cosine metric — the
    description embeddings are normalized in cosine space."""
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
# Only the 3D PCA HTML survives the 2026-05-05 cleanup. The 2D static
# PNG, 2D plotly HTML, and t-SNE cluster PNG were all redundant with the
# side-by-side 3D scene (which carries both the category-coloring and
# the cluster-coloring views) and have been removed.
def _scatter_pca_3d_html(
    coords3: np.ndarray,
    categories: list[str],
    quadrants: list[str | None],
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

      - left:  shared faces colored by GT modal quadrant (circles);
               wild-only Claude faces black circles; wild-only non-Claude
               faces black squares.
      - right: colored by KMeans cluster id, with Haiku label in legend.

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
            "Shared (○ colored by Q) · wild-Claude (● black) · wild-other (■ black)",
            "Colored by KMeans cluster (k={})".format(len(cluster_labels)),
        ),
        horizontal_spacing=0.04,
    )

    sizes = np.clip(6 + 5 * sizes_log, 6, 26)

    # --- LEFT scene: by category (+ quadrant for shared) ------------------
    for q in QUADRANTS:
        idxs = [
            i for i, (cat, qi) in enumerate(zip(categories, quadrants))
            if cat == "shared" and qi == q
        ]
        if not idxs:
            continue
        hover = [
            f"<b>{fws[i]}</b><br>category: shared<br>quadrant: {q}<br>"
            f"emit weight (log): {sizes_log[i]:.2f}<br>"
            f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter3d(
                x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
                mode="markers",
                name=f"shared · {q}",
                legendgroup=f"left-shared-{q}",
                marker=dict(
                    size=[float(sizes[i]) for i in idxs],
                    color=QUADRANT_COLORS[q],
                    symbol="circle",
                    line=dict(color="black", width=0.4),
                    opacity=0.88,
                ),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover,
                scene="scene",
                showlegend=True,
            ),
            row=1, col=1,
        )

    for cat, label, symbol in (
        ("wild_claude", "wild-only · Claude", "circle"),
        ("wild_other", "wild-only · non-Claude", "square"),
    ):
        idxs = [i for i, c in enumerate(categories) if c == cat]
        if not idxs:
            continue
        hover = [
            f"<b>{fws[i]}</b><br>category: {cat}<br>"
            f"emit weight (log): {sizes_log[i]:.2f}<br>"
            f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter3d(
                x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
                mode="markers",
                name=label,
                legendgroup=f"left-{cat}",
                marker=dict(
                    size=[float(sizes[i]) for i in idxs],
                    color=WILD_COLOR,
                    symbol=symbol,
                    line=dict(color="white", width=0.4),
                    opacity=0.82,
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
            f"category: {categories[i]}"
            + (f" · quadrant: {quadrants[i]}" if quadrants[i] else "")
            + f"<br><i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
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


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",
                    default=str(CLAUDE_DESCRIPTIONS_PATH),
                    help="HF-corpus JSONL of {kaomoji, count_total, "
                         "descriptions, ...}. Default is the snapshot at "
                         "data/harness/claude_descriptions.jsonl.")
    ap.add_argument("--claude-gt-floor", type=int, default=1)
    ap.add_argument("--gt-only", action="store_true",
                    help="restrict to shared faces (in both Claude GT and "
                         "the HF corpus) — drops both wild-only categories. "
                         "Outputs land in *_gt_only.{tsv,png,html} so both "
                         "modes coexist.")
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

    # Output-filename suffix so gt-only and full-corpus modes coexist.
    suffix = "_gt_only" if args.gt_only else ""
    print(f"mode: {'gt-only (shared subset)' if args.gt_only else 'full HF corpus'}")

    # --- load Claude-GT modal labels ---------------------------------------
    print(f"loading Claude-GT (floor={args.claude_gt_floor}) ...")
    gt = load_claude_gt(floor=args.claude_gt_floor)
    print(f"  {len(gt)} faces in Claude-GT")

    # --- load HF corpus -----------------------------------------------------
    corpus_path = Path(args.corpus)
    print(f"\nloading HF corpus from {corpus_path} ...")
    hf_meta, n_total_emit = _load_hf_corpus(corpus_path)
    n_unique = len(hf_meta)
    n_claude_emit_total = sum(m["claude_emit"] for m in hf_meta.values())
    print(f"  {n_unique} canonical faces · {n_total_emit} total emissions "
          f"({n_claude_emit_total} from claude-opus)")

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

    # --- inner-join HF corpus × embeddings, then categorize ----------------
    rows: list[dict] = []
    for face, meta in hf_meta.items():
        if face not in fw_to_idx:
            continue  # un-described long tail
        cat, q = _categorize(face, gt, meta["claude_emit"])
        if args.gt_only and cat != "shared":
            continue  # gt-only mode: drop wild-only categories
        rows.append({
            "first_word": face,
            "n_emit": int(meta["count_total"]),
            "claude_emit": int(meta["claude_emit"]),
            "category": cat,
            "quadrant": q if q is not None else "",
            "embed_idx": fw_to_idx[face],
        })

    if not rows:
        sys.exit("no faces survived the inner-join — aborting")

    df_wild = pd.DataFrame(rows).sort_values("n_emit", ascending=False).reset_index(drop=True)
    n_kept = len(df_wild)
    cat_counts = df_wild["category"].value_counts().to_dict()
    n_shared = int(cat_counts.get("shared", 0))
    n_wild_claude = int(cat_counts.get("wild_claude", 0))
    n_wild_other = int(cat_counts.get("wild_other", 0))
    print(f"\n{n_kept} / {n_unique} HF faces kept (have a description embedding)")
    print(f"  shared (in GT)        : {n_shared}")
    print(f"  wild-only · Claude    : {n_wild_claude}")
    print(f"  wild-only · non-Claude: {n_wild_other}")

    n_emit_kept = int(df_wild["n_emit"].sum())
    print(f"  emit coverage: {n_emit_kept} / {n_total_emit} "
          f"({n_emit_kept/max(n_total_emit,1)*100:.1f}%)")

    # --- attach axis projections to df_wild --------------------------------
    df_wild = df_wild.merge(df_axes, how="left", left_on="first_word", right_index=True)
    missing_axes = df_wild[ERISKII_AXES].isna().any(axis=1).sum()
    if missing_axes:
        print(f"  (warn) {missing_axes} faces have a description but no eriskii axes; "
              "their axis profiles will read as NaN — usually a script-16 staleness issue")

    # Save the labeled table. `quadrant` is empty for wild-only faces by
    # design — those rows are intentionally without a Russell label.
    out_tsv = HARNESS_DATA_DIR / f"wild_faces_labeled{suffix}.tsv"
    cols = ["first_word", "n_emit", "claude_emit", "category", "quadrant"] + [
        a for a in ERISKII_AXES if a in df_wild.columns
    ]
    df_wild[cols].to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    # --- assemble matrices for downstream ----------------------------------
    idxs = df_wild["embed_idx"].to_numpy()
    E = E_all[idxs]
    fws = df_wild["first_word"].tolist()
    categories: list[str] = df_wild["category"].tolist()
    quadrants: list[str | None] = [
        (q if q else None) for q in df_wild["quadrant"].tolist()
    ]
    weights = df_wild["n_emit"].astype(int).tolist()
    sizes_log = np.log1p(np.asarray(weights, dtype=float))

    # --- representative descriptions for hover + Haiku ---------------------
    descriptions_by_fw = {f: m["top_description"] for f, m in hf_meta.items()}

    # --- PCA(3) for the 3D scene ------------------------------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=0)
    coords3 = pca.fit_transform(E)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: "
          f"PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f} "
          f"(sum={sum(evr):.3f})")

    mode_subtitle = " · gt-only (shared subset)" if args.gt_only else ""

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
    # Quadrant composition is computed over the *shared* subset of each
    # cluster only — wild-only faces deliberately have no Russell label,
    # so excluding them keeps `share_*` honest. `wild_*_frac` reports
    # the per-cluster category split separately.
    rows_clusters = []
    for c in range(best_k):
        member_idx = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        share, total_w_shared = _quadrant_composition(member_idx, quadrants, weights)
        n_shared_c = sum(1 for i in member_idx if categories[i] == "shared")
        n_wild_claude_c = sum(1 for i in member_idx if categories[i] == "wild_claude")
        n_wild_other_c = sum(1 for i in member_idx if categories[i] == "wild_other")
        n_total_c = max(len(member_idx), 1)
        shared_frac = n_shared_c / n_total_c
        wild_claude_frac = n_wild_claude_c / n_total_c
        wild_other_frac = n_wild_other_c / n_total_c
        wild_frac = wild_claude_frac + wild_other_frac

        if n_shared_c > 0:
            modal_q = max(QUADRANTS, key=lambda q: share.get(q, 0.0))
            modal_share = share.get(modal_q, 0.0)
        else:
            modal_q = ""
            modal_share = 0.0
        # diffuseness: 1 − max-share (high → not quadrant-aligned).
        # Defined over the shared subset; meaningless for wild-pure clusters.
        diffuseness = 1.0 - modal_share if n_shared_c > 0 else float("nan")

        top_pos, top_neg = _cluster_axis_profile(member_idx, df_wild)
        members_str = ", ".join(fws[i] for i in sorted(member_idx, key=lambda j: -weights[j])[:8])
        if len(member_idx) > 8:
            members_str += f", … (+{len(member_idx)-8} more)"

        rows_clusters.append({
            "cluster_id": c,
            "label": cluster_labels[c],
            "n_faces": len(member_idx),
            "n_shared": n_shared_c,
            "n_wild_claude": n_wild_claude_c,
            "n_wild_other": n_wild_other_c,
            "shared_frac": round(shared_frac, 3),
            "wild_claude_frac": round(wild_claude_frac, 3),
            "wild_other_frac": round(wild_other_frac, 3),
            "wild_frac": round(wild_frac, 3),
            "modal_quadrant": modal_q,
            "modal_share": round(modal_share, 3),
            "diffuseness": round(diffuseness, 3) if n_shared_c > 0 else "",
            "total_emit_weight_shared": total_w_shared,
            **{f"share_{q}": round(share.get(q, 0.0), 3) for q in QUADRANTS},
            "top_axes_pos": "; ".join(f"{a}:+{v:.2f}" for a, v in top_pos),
            "top_axes_neg": "; ".join(f"{a}:{v:+.2f}" for a, v in top_neg),
            "sample_members": members_str,
        })

    # Sort by wild_frac desc — top of the table is where Claude's
    # deployment-shaped vocabulary diverges most from the GT elicitation.
    df_clusters = pd.DataFrame(rows_clusters).sort_values(
        ["wild_frac", "n_faces"], ascending=[False, False]
    ).reset_index(drop=True)
    out_clusters_tsv = HARNESS_DATA_DIR / f"wild_residual_clusters{suffix}.tsv"
    df_clusters.to_csv(out_clusters_tsv, sep="\t", index=False)
    print(f"wrote {out_clusters_tsv}")

    # --- 3D PCA HTML with both colorings -----------------------------------
    _scatter_pca_3d_html(
        coords3, categories, quadrants, cluster_ids, cluster_labels,
        fws, sizes_log, descriptions_by_fw,
        out_path=HARNESS_FIG_DIR / f"wild_faces_eriskii_pca_3d{suffix}.html",
        title=(f"HF-corpus Claude faces — PCA(3) on description embeddings{mode_subtitle}"),
        evr=evr,
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
        f"- HF corpus: {n_total_emit} total emissions across {n_unique} canonical "
        f"kaomoji ({n_claude_emit_total} from claude-opus source models). "
        f"Source: `{corpus_path.name}` — pooled contributor data from "
        "`a9lim/llmoji`."
    )
    lines.append(
        f"- {n_kept} faces kept after inner-joining with the description "
        f"embedding parquet (drops {n_unique - n_kept} un-described faces "
        "in the long tail)."
    )
    lines.append(
        f"- Categories: **shared** (in Claude GT ∩ HF corpus) "
        f"= **{n_shared}**; "
        f"**wild-only · Claude** (HF + claude-opus emit, not in GT) "
        f"= **{n_wild_claude}**; "
        f"**wild-only · non-Claude** (HF + no claude-opus emit, not in GT) "
        f"= **{n_wild_other}**."
    )
    lines.append(
        f"- Display: shared faces = colored circles (GT modal quadrant); "
        f"wild-only Claude = black circles; wild-only non-Claude = black "
        f"squares. Wild-only faces deliberately carry no Russell label — the "
        f"prior ensemble-fallback predictions were a confidence-shaped "
        f"fiction and have been removed."
    )
    lines.append("")
    lines.append(
        f"PCA on the 384-d description embeddings: "
        f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, "
        f"PC3={evr[2]*100:.1f}% var (sum {sum(evr[:3])*100:.1f}%). "
        "Figure: 3D side-by-side (category-colored vs cluster-colored) at "
        f"`figures/harness/wild_faces_eriskii_pca_3d{suffix}.html`. "
        "(2D PNG / 2D HTML / t-SNE PNG were dropped 2026-05-05 — "
        "the 3D scene is the only chart output.)"
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
        f"the grid: {sil_grid_str}.{fixed_note} Sorted by `wild_frac` "
        "descending — the top of the table is where the HF-corpus face "
        "vocabulary diverges most from the GT elicitation set. `share_*` "
        "and `modal_quadrant` are computed over the shared subset of each "
        "cluster only."
    )
    lines.append("")
    lines.append(
        "| id | label | n | shared / wC / wO | wild_frac | modal Q (share) | "
        "diffuseness | top axes (+) | top axes (−) | sample |"
    )
    lines.append(
        "|---|---|---:|---|---:|---|---:|---|---|---|"
    )
    for _, r in df_clusters.iterrows():
        modal_q = str(r["modal_quadrant"])
        diff_raw = r["diffuseness"]
        diff_str = f"{float(diff_raw):.2f}" if str(diff_raw) != "" else "—"
        modal_str = (
            f"{modal_q} ({float(r['modal_share']):.2f})" if modal_q else "—"
        )
        lines.append(
            f"| {int(r['cluster_id'])} | {r['label']} | {int(r['n_faces'])} "
            f"| {int(r['n_shared'])} / {int(r['n_wild_claude'])} / "
            f"{int(r['n_wild_other'])} "
            f"| {float(r['wild_frac']):.2f} | {modal_str} | {diff_str} "
            f"| {r['top_axes_pos']} | {r['top_axes_neg']} "
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
