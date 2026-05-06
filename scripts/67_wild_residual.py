# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportMissingImports=false
"""Wild-emitted Claude faces × bag-of-lexicon (BoL) clusters: looking
for state structure beyond the six Russell quadrants.

Refactored 2026-05-06 to consume the **BoL parquet** instead of the
prior 384-d MiniLM-on-prose embeddings (and to drop the eriskii
21-axis projection that was the load-bearing analytical layer
pre-refactor). The new representation is the synthesizer's structured
commit over the locked llmoji v2 LEXICON: 48-d soft distribution per
canonical face, with 19 of those words tagged with explicit Russell
quadrants. See ``llmoji_study.lexicon``.

Two parallel categorizations live in the data:

  - **GT-overlap category** (``shared`` / ``wild_claude`` /
    ``wild_other``): drives the cluster-table breakdown
    (``wild_frac``, modal-quadrant-shared etc).
      * shared — in Claude GT ∩ HF corpus
      * wild_claude — HF + claude-opus emit, not in GT
      * wild_other — HF + no claude-opus emit, not in GT
  - **Deployment surface** (``claude_code`` / ``claude_ai`` /
    ``other``): drives the chart marker shape (post-2026-05-06).
    Sourced from a9's local ``~/.claude/kaomoji-journal.jsonl`` +
    claude.ai exports via :mod:`llmoji_study.local_emissions`.
      * claude_code — face appears in the Claude Code journal AND
        NOT in any claude.ai export
      * claude_ai — face appears in any claude.ai export (whether
        or not also in Code)
      * other — face is in the HF corpus but not in either of a9's
        local sources (other contributors / non-Claude source models
        in a9's bundle that didn't surface locally)

Color is **BoL modal quadrant uniformly** for every face — even where
GT is available. Pre-2026-05-06 the script used GT modal for shared
faces and BoL only for wild-* ones; that collapsed the synthesis-vs-
emission gap exactly where it's most visible. The new convention
makes the chart show what the synthesizer says about every face,
independent of whether elicitation also caught it.

The complementary view is available via ``--color-by gt``: each point
is colored by its **Claude-GT modal quadrant**, with faces *not* in
the GT set rendered black. That's the elicitation-honest view —
visually answers "what part of PCA-space did the GT pilot actually
cover, and which wild faces fell outside it?" It composes with
``--gt-only`` (in which case nothing is black, since every face is in
GT) and writes to ``*_gtcolor.{tsv,html}`` so the two color modes
coexist on disk.

Pipeline (post-hoc, no new generation, zero welfare cost):

  1. Load BoL parquet (``claude_faces_lexicon_bag.parquet``) →
     48-d per-face soft distribution.
  2. Load Claude-GT modal labels (claude-runs naturalistic +
     introspection arm).
  3. Cross-load HF corpus metadata (claude_descriptions.jsonl) for
     emit counts + claude-opus emit attribution.
  4. PCA(3) on the 48-d BoL; produce a side-by-side 3D plotly HTML —
     left scene colored by quadrant (shared = GT modal, wild = BoL
     modal); right scene colored by KMeans cluster id.
  5. KMeans on BoL; k via silhouette grid but defaulted to k=6 for
     interpretability (see writeup).
  6. Per cluster: deterministic modal-lexicon-word label (top-2
     words by cluster-mean BoL), category composition (shared /
     wild_claude / wild_other), quadrant composition over the shared
     subset AND over the full BoL-inferred set, top lexicon words by
     cluster mean.
  7. Auto-regenerated Setup + Cluster table at
     ``docs/2026-MM-DD-residual-state-axes.md``. Content past the
     HAND_MARKER line is preserved across re-runs.

Usage:
    python scripts/67_wild_residual.py
    python scripts/67_wild_residual.py --gt-only  # shared subset only
    python scripts/67_wild_residual.py --fixed-k 6
    python scripts/67_wild_residual.py --color-by gt  # GT quadrant; non-GT = black
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji.taxonomy import canonicalize_kaomoji
from llmoji_study.claude_faces import load_bol_parquet
from llmoji_study.claude_gt import load_claude_gt
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_LEXICON_BAG_PATH,
    DATA_DIR,
    FIGURES_DIR,
    REPO_ROOT,
)
from llmoji_study.emotional_analysis import QUADRANT_COLORS
from llmoji_study.lexicon import (
    QUADRANTS,
    bol_modal_quadrant,
    top_lexicon_words,
)
from llmoji_study.local_emissions import (
    SOURCE_CLAUDE_AI,
    SOURCE_CLAUDE_CODE,
    load_face_source_counts,
)

CATEGORIES = ["shared", "wild_claude", "wild_other"]
HARNESS_DATA_DIR = DATA_DIR / "harness"
HARNESS_FIG_DIR = FIGURES_DIR / "harness"
DOCS_DIR = REPO_ROOT / "docs"

# Per-face deployment surface (from a9's local journal + claude.ai
# exports via llmoji_study.local_emissions). Drives the marker shape
# in the 3D PCA scene as of 2026-05-06: the analytical question is
# "where in deployment does this face live?" rather than the prior
# "where does this face sit relative to the GT elicitation set?"
# Color (= quadrant) and the cluster-table categorization
# (shared/wild_claude/wild_other) are unchanged.
SURFACES = ["claude_code", "claude_ai", "other"]
SURFACE_LABELS = {
    "claude_code": "Claude Code (only)",
    "claude_ai": "claude.ai (any — incl. faces also in Code)",
    "other": "neither (HF corpus only)",
}
# Marker shape per surface. Diamond is intentionally claude.ai
# (the new analytical focus); circles cover the Code-only deployment;
# squares mark "in HF corpus, not in either of a9's local sources"
# (faces from other contributors / providers).
SURFACE_MARKERS = {
    "claude_code": "circle",
    "claude_ai": "diamond",
    "other": "square",
}


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
) -> str:
    """Return the source category. Shared ↔ in Claude GT; the two
    wild-* categories split by whether claude-opus emitted the face."""
    if face in gt:
        return "shared"
    if claude_emit > 0:
        return "wild_claude"
    return "wild_other"


def _deployment_surface(
    face: str,
    source_counts: dict[str, dict[str, int]],
) -> str:
    """Return one of :data:`SURFACES` for the chart marker.

    Rule: ``claude_ai`` if the face appears in any claude.ai export
    (even if also in Claude Code); ``claude_code`` if it's
    exclusively in the journal; ``other`` if it's in the HF corpus
    but not in either of a9's local sources (i.e. an emit from
    another contributor or a non-Claude source model in a9's own
    bundle that didn't surface locally).
    """
    src = source_counts.get(face, {})
    if src.get(SOURCE_CLAUDE_AI, 0) > 0:
        return "claude_ai"
    if src.get(SOURCE_CLAUDE_CODE, 0) > 0:
        return "claude_code"
    return "other"


# --------------------------------------------------------------------- analysis
def _pick_k_silhouette(B: np.ndarray, k_grid: list[int]) -> tuple[int, dict[int, float]]:
    """Silhouette score over k_grid; pick argmax. Euclidean metric —
    BoL rows are L1-normalized soft distributions over the lexicon,
    so euclidean distance is principled."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    for k in k_grid:
        if k >= B.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(B)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(B, labels))
        scores[k] = s
        print(f"  k={k:2d}  silhouette={s:.4f}")
    if not scores:
        return min(k_grid), {}
    best = max(scores, key=lambda kk: scores[kk])
    return best, scores


def _quadrant_composition(
    cluster_idxs: list[int],
    quadrant_per_face: list[str | None],
    weights: list[int],
    *,
    restrict_to_categories: set[str] | None = None,
    categories_per_face: list[str] | None = None,
) -> tuple[dict[str, float], int]:
    """Return ``(share_per_quadrant, total_weighted_emits)`` for the
    cluster. Shares are emit-weighted (a heavy-use face contributes
    more than a one-off). When ``restrict_to_categories`` is set, only
    faces whose category is in that set count toward the composition
    (used for the shared-only view that the cluster table reports
    alongside the BoL-inferred view)."""
    counts: Counter[str] = Counter()
    total = 0
    for i in cluster_idxs:
        if restrict_to_categories is not None:
            if categories_per_face is None or categories_per_face[i] not in restrict_to_categories:
                continue
        q = quadrant_per_face[i]
        w = weights[i]
        if q is None:
            continue
        counts[q] += w
        total += w
    if total == 0:
        return {q: 0.0 for q in QUADRANTS}, 0
    return {q: counts.get(q, 0) / total for q in QUADRANTS}, total


def _cluster_top_words(
    cluster_idxs: list[int],
    B: np.ndarray,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Top-k modal lexicon words for the cluster (cluster-mean BoL)."""
    cluster_bol = B[cluster_idxs].mean(axis=0)
    return top_lexicon_words(cluster_bol, k=k)


# --------------------------------------------------------------------- figures
def _scatter_pca_3d_html(
    coords3: np.ndarray,
    categories: list[str],
    surfaces: list[str],
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
    color_mode: str = "bol",
) -> None:
    """Two side-by-side 3D scenes on the same PCA(3) coords:

      - left:  faces colored by **quadrant**; ``color_mode`` selects
               the source. ``"bol"`` (default) reads BoL modal
               quadrant for every face; ``"gt"`` reads Claude-GT
               modal where available and renders the rest black
               (non-GT bucket). **Marker shape = deployment surface**
               (circle = Claude Code only, diamond = any claude.ai,
               square = neither).
      - right: colored by KMeans cluster id; legend carries the
               deterministic top-2 modal-lexicon-word label.
    """
    if color_mode == "gt":
        color_label = "Claude-GT modal quadrant"
        missing_color = "#000000"
        missing_label = "not in GT"
        hover_q_prefix = "GT quadrant"
    else:
        color_label = "BoL modal quadrant"
        missing_color = "#cccccc"
        missing_label = "no Q"
        hover_q_prefix = "BoL quadrant"
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
            f"Color = {color_label}; marker shape = deployment surface "
            "(○ = Claude Code only, ◇ = any claude.ai, □ = neither)",
            "Colored by KMeans cluster (k={}); labels = top-2 modal lexicon words".format(
                len(cluster_labels),
            ),
        ),
        horizontal_spacing=0.04,
    )

    sizes = np.clip(6 + 5 * sizes_log, 6, 26)

    # --- LEFT scene: by quadrant × deployment surface ---------------------
    # Two channels carry independent information: marker shape =
    # deployment surface (where the face lives in a9's local data),
    # marker color = BoL modal quadrant (what the synthesizer commits
    # to). GT modal is no longer used for color — surfacing the
    # synthesis-vs-emission gap requires showing BoL's read everywhere
    # rather than collapsing shared faces to GT.
    seen_q: list[str] = list(QUADRANTS) + ["none"]
    for surface in SURFACES:
        marker_symbol = SURFACE_MARKERS[surface]
        surface_label = SURFACE_LABELS[surface]
        for q in seen_q:
            if q == "none":
                idxs = [
                    i for i, (si, qi) in enumerate(zip(surfaces, quadrants))
                    if si == surface and qi is None
                ]
                color = missing_color
                qlabel = missing_label
            else:
                idxs = [
                    i for i, (si, qi) in enumerate(zip(surfaces, quadrants))
                    if si == surface and qi == q
                ]
                color = QUADRANT_COLORS[q]
                qlabel = q
            if not idxs:
                continue
            hover = [
                f"<b>{fws[i]}</b><br>"
                f"surface: {surface_label}<br>"
                f"{hover_q_prefix}: {qlabel}<br>"
                f"GT-overlap category: {categories[i]}<br>"
                f"emit weight (log): {sizes_log[i]:.2f}<br>"
                f"<i>{_truncate(descriptions_by_fw.get(fws[i], ''))}</i>"
                for i in idxs
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
                    mode="markers",
                    name=f"{surface} · {qlabel}",
                    legendgroup=f"left-{surface}-{q}",
                    marker=dict(
                        size=[float(sizes[i]) for i in idxs],
                        color=color,
                        symbol=marker_symbol,
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

    # --- RIGHT scene: by cluster -----------------------------------------
    for c in sorted(cluster_labels):
        idxs = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        if not idxs:
            continue
        label = cluster_labels[c]
        color = palette[c % len(palette)]
        hover = [
            f"<b>{fws[i]}</b><br>cluster {c}: {label}<br>"
            f"surface: {SURFACE_LABELS[surfaces[i]]}<br>"
            f"GT-overlap category: {categories[i]}"
            + (f" · {hover_q_prefix}: {quadrants[i]}" if quadrants[i] else "")
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
    ap.add_argument("--bol-parquet",
                    default=str(CLAUDE_FACES_LEXICON_BAG_PATH),
                    help="BoL parquet from scripts/harness/62_corpus_lexicon.py")
    ap.add_argument("--claude-gt-floor", type=int, default=1)
    ap.add_argument("--gt-only", action="store_true",
                    help="restrict to shared faces (in both Claude GT and "
                         "the HF corpus) — drops both wild-only categories. "
                         "Outputs land in *_gt_only.{tsv,html} so both "
                         "modes coexist.")
    ap.add_argument("--color-by", choices=["bol", "gt"], default="bol",
                    help="left-scene color source. 'bol' (default) uses BoL "
                         "modal quadrant for every face; 'gt' uses Claude-GT "
                         "modal where available and renders non-GT faces "
                         "black. The 'gt' mode tags outputs with _gtcolor "
                         "so it coexists with the default on disk. "
                         "Composes with --gt-only (in which case there are "
                         "no black points since every face is in GT).")
    ap.add_argument("--k-grid", default="2,3,4,5,6,7,8,10,12,14",
                    help="silhouette grid for KMeans")
    ap.add_argument("--fixed-k", type=int, default=6,
                    help="force this k regardless of silhouette winner; "
                         "set <=0 to use silhouette argmax. Default 6 — "
                         "the local-maximum-after-coarse-modes scale where "
                         "structure resolves interpretably (see writeup).")
    ap.add_argument("--out-doc",
                    default=str(DOCS_DIR / "2026-05-05-residual-state-axes.md"))
    args = ap.parse_args()

    HARNESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # TSV outputs are color-independent (they expose both BoL + GT
    # quadrants as columns and cluster-composition math always uses
    # BoL); only the HTML figure + auto-generated markdown setup
    # paragraph differ by color choice. Keep a separate fig_suffix so
    # we don't thrash TSV filenames between runs.
    suffix = "_gt_only" if args.gt_only else ""
    fig_suffix = suffix + ("_gtcolor" if args.color_by == "gt" else "")
    print(f"mode: {'gt-only (shared subset)' if args.gt_only else 'full HF corpus'}")
    print(f"left-scene color: {'Claude-GT modal (non-GT = black)' if args.color_by == 'gt' else 'BoL modal'}")

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

    # --- load BoL vectors ---------------------------------------------------
    bol_path = Path(args.bol_parquet)
    print(f"\nloading BoL vectors from {bol_path.name} ...")
    fw_all, _, _, B_all = load_bol_parquet(bol_path)
    fw_to_idx = {f: i for i, f in enumerate(fw_all)}
    print(f"  {len(fw_all)} bagged faces, dim={B_all.shape[1]} (lexicon)")

    # --- load a9's local emission sources (drives chart marker shape) -----
    print("\nloading local emission sources (Claude Code journal + claude.ai exports) ...")
    source_counts = load_face_source_counts()
    print(f"  {len(source_counts)} unique faces in local sources")

    # --- inner-join HF corpus × BoL, then categorize -----------------------
    # Two parallel categorizations per face:
    #   - GT-overlap category (shared / wild_claude / wild_other) →
    #     drives the cluster-table breakdown (wild_frac etc).
    #   - Deployment surface (claude_code / claude_ai / other) →
    #     drives the chart marker shape (post-2026-05-06 rework).
    # Color is BoL modal quadrant uniformly, even for shared faces —
    # surfacing the synthesis-vs-emission gap requires showing BoL's
    # read everywhere rather than collapsing shared faces to GT.
    rows: list[dict] = []
    for face, meta in hf_meta.items():
        if face not in fw_to_idx:
            continue
        cat = _categorize(face, gt, meta["claude_emit"])
        if args.gt_only and cat != "shared":
            continue
        b_idx = fw_to_idx[face]
        bol_quadrant = bol_modal_quadrant(B_all[b_idx])
        gt_quadrant = gt[face][0] if face in gt else ""
        surface = _deployment_surface(face, source_counts)
        rows.append({
            "first_word": face,
            "n_emit": int(meta["count_total"]),
            "claude_emit": int(meta["claude_emit"]),
            "category": cat,
            "surface": surface,
            "bol_quadrant": bol_quadrant if bol_quadrant is not None else "",
            "gt_quadrant": gt_quadrant,
            "bol_idx": b_idx,
        })

    if not rows:
        sys.exit("no faces survived the inner-join — aborting")

    df_wild = pd.DataFrame(rows).sort_values("n_emit", ascending=False).reset_index(drop=True)
    n_kept = len(df_wild)
    cat_counts = df_wild["category"].value_counts().to_dict()
    n_shared = int(cat_counts.get("shared", 0))
    n_wild_claude = int(cat_counts.get("wild_claude", 0))
    n_wild_other = int(cat_counts.get("wild_other", 0))
    surface_counts = df_wild["surface"].value_counts().to_dict()
    n_code = int(surface_counts.get("claude_code", 0))
    n_ai = int(surface_counts.get("claude_ai", 0))
    n_other = int(surface_counts.get("other", 0))
    print(f"\n{n_kept} / {n_unique} HF faces kept (have a BoL vector)")
    print(f"  GT-overlap categories: shared={n_shared}, "
          f"wild_claude={n_wild_claude}, wild_other={n_wild_other}")
    print(f"  Deployment surfaces:   claude_code={n_code}, "
          f"claude_ai={n_ai}, other={n_other}")

    n_emit_kept = int(df_wild["n_emit"].sum())
    print(f"  emit coverage: {n_emit_kept} / {n_total_emit} "
          f"({n_emit_kept/max(n_total_emit,1)*100:.1f}%)")

    n_no_bol_q = int((df_wild["bol_quadrant"] == "").sum())
    if n_no_bol_q:
        print(
            f"  {n_no_bol_q} face(s) have no BoL circumplex commitment "
            "(extension-only picks)"
        )

    out_tsv = HARNESS_DATA_DIR / f"wild_faces_labeled{suffix}.tsv"
    cols = [
        "first_word", "n_emit", "claude_emit", "category", "surface",
        "bol_quadrant", "gt_quadrant",
    ]
    df_wild[cols].to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    # --- assemble matrices for downstream ----------------------------------
    bol_idxs = df_wild["bol_idx"].to_numpy()
    B = B_all[bol_idxs]
    fws = df_wild["first_word"].tolist()
    categories: list[str] = df_wild["category"].tolist()
    surfaces: list[str] = df_wild["surface"].tolist()
    # Per-face cluster-table analytics always use BoL (the cluster
    # composition columns are computed against BoL modal); only the
    # left-scene color picks between BoL and GT per --color-by.
    bol_quadrants: list[str | None] = [
        (q if q else None) for q in df_wild["bol_quadrant"].tolist()
    ]
    if args.color_by == "gt":
        # GT modal where available; None (= black bucket) for wild-* faces.
        color_quadrants: list[str | None] = [
            (q if q else None) for q in df_wild["gt_quadrant"].tolist()
        ]
    else:
        color_quadrants = bol_quadrants
    # Cluster-composition analytics consume BoL modal — that's the
    # invariant the rest of the pipeline (and the markdown table) is
    # written against. Color choice only flips the chart, not the math.
    quadrants = bol_quadrants
    weights = df_wild["n_emit"].astype(int).tolist()
    sizes_log = np.log1p(np.asarray(weights, dtype=float))

    descriptions_by_fw = {f: m["top_description"] for f, m in hf_meta.items()}

    # --- PCA(3) for the 3D scene ------------------------------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=0)
    coords3 = pca.fit_transform(B)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: "
          f"PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f} "
          f"(sum={sum(evr):.3f})")

    mode_subtitle = " · gt-only (shared subset)" if args.gt_only else ""
    if args.color_by == "gt":
        mode_subtitle += " · color = Claude-GT (non-GT = black)"

    # --- KMeans on BoL ----------------------------------------------------
    print("\n=== clustering ===")
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    print(f"silhouette grid: {k_grid}")
    silhouette_winner, sil_scores = _pick_k_silhouette(B, k_grid)
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
    cluster_ids = km.fit_predict(B)

    # --- deterministic cluster labels (no Haiku call) ----------------------
    cluster_labels: dict[int, str] = {}
    cluster_top_words: dict[int, list[tuple[str, float]]] = {}
    for c in range(best_k):
        member_idx = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        top = _cluster_top_words(member_idx, B, k=5)
        cluster_top_words[c] = top
        cluster_labels[c] = "/".join(w for w, _ in top[:2]) if top else f"cluster-{c}"
        print(
            f"  cluster {c} (n={len(member_idx)}): "
            + ", ".join(f"{w}:{v:.2f}" for w, v in top[:5])
        )

    # --- per-cluster analytics ---------------------------------------------
    rows_clusters = []
    for c in range(best_k):
        member_idx = [i for i, ci in enumerate(cluster_ids) if int(ci) == c]
        share_shared, total_w_shared = _quadrant_composition(
            member_idx, quadrants, weights,
            restrict_to_categories={"shared"},
            categories_per_face=categories,
        )
        share_full, total_w_full = _quadrant_composition(
            member_idx, quadrants, weights,
        )
        n_shared_c = sum(1 for i in member_idx if categories[i] == "shared")
        n_wild_claude_c = sum(1 for i in member_idx if categories[i] == "wild_claude")
        n_wild_other_c = sum(1 for i in member_idx if categories[i] == "wild_other")
        n_total_c = max(len(member_idx), 1)
        shared_frac = n_shared_c / n_total_c
        wild_claude_frac = n_wild_claude_c / n_total_c
        wild_other_frac = n_wild_other_c / n_total_c
        wild_frac = wild_claude_frac + wild_other_frac

        if n_shared_c > 0:
            modal_q = max(QUADRANTS, key=lambda q: share_shared.get(q, 0.0))
            modal_share = share_shared.get(modal_q, 0.0)
        else:
            modal_q = ""
            modal_share = 0.0
        modal_q_full = max(QUADRANTS, key=lambda q: share_full.get(q, 0.0)) if total_w_full > 0 else ""
        modal_share_full = share_full.get(modal_q_full, 0.0) if modal_q_full else 0.0

        diffuseness = 1.0 - modal_share if n_shared_c > 0 else float("nan")

        top = cluster_top_words[c]
        top_words_str = "; ".join(f"{w}:{v:.2f}" for w, v in top)

        members_str = ", ".join(
            fws[i] for i in sorted(member_idx, key=lambda j: -weights[j])[:8]
        )
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
            "modal_quadrant_shared": modal_q,
            "modal_share_shared": round(modal_share, 3),
            "modal_quadrant_full": modal_q_full,
            "modal_share_full": round(modal_share_full, 3),
            "diffuseness_shared": round(diffuseness, 3) if n_shared_c > 0 else "",
            "total_emit_weight_shared": total_w_shared,
            "total_emit_weight_full": total_w_full,
            **{f"share_shared_{q}": round(share_shared.get(q, 0.0), 3) for q in QUADRANTS},
            **{f"share_full_{q}": round(share_full.get(q, 0.0), 3) for q in QUADRANTS},
            "top_lexicon_words": top_words_str,
            "sample_members": members_str,
        })

    df_clusters = pd.DataFrame(rows_clusters).sort_values(
        ["wild_frac", "n_faces"], ascending=[False, False]
    ).reset_index(drop=True)
    out_clusters_tsv = HARNESS_DATA_DIR / f"wild_residual_clusters{suffix}.tsv"
    df_clusters.to_csv(out_clusters_tsv, sep="\t", index=False)
    print(f"wrote {out_clusters_tsv}")

    # --- 3D PCA HTML with both colorings -----------------------------------
    _scatter_pca_3d_html(
        coords3, categories, surfaces, color_quadrants, cluster_ids, cluster_labels,
        fws, sizes_log, descriptions_by_fw,
        out_path=HARNESS_FIG_DIR / f"wild_faces_pca_3d{fig_suffix}.html",
        title=f"HF-corpus Claude faces - PCA(3) on BoL{mode_subtitle}",
        evr=evr,
        color_mode=args.color_by,
    )

    # --- markdown writeup --------------------------------------------------
    HAND_MARKER = "<!-- HAND-EDITED BELOW THIS LINE; preserved across re-runs -->"
    today = "2026-05-06"
    lines: list[str] = []
    lines.append("# Residual state structure - clustering wild-emitted Claude faces (BoL)")
    lines.append("")
    lines.append(f"_{today} - generated by `scripts/67_wild_residual.py`._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"- HF corpus: {n_total_emit} total emissions across {n_unique} canonical "
        f"kaomoji ({n_claude_emit_total} from claude-opus source models). "
        f"Source: `{corpus_path.name}` - pooled contributor data from "
        "`a9lim/llmoji`."
    )
    lines.append(
        f"- {n_kept} faces kept after inner-joining with the BoL parquet "
        f"(drops {n_unique - n_kept} faces without v2 synthesis)."
    )
    lines.append(
        f"- GT-overlap categories (drive the cluster table below): "
        f"**shared** (in Claude GT ∩ HF corpus) = **{n_shared}**; "
        f"**wild-only · Claude** (HF + claude-opus emit, not in GT) "
        f"= **{n_wild_claude}**; "
        f"**wild-only · non-Claude** (HF + no claude-opus emit, not in GT) "
        f"= **{n_wild_other}**."
    )
    lines.append(
        f"- Deployment surfaces (drive the chart marker shape, post-2026-05-06): "
        f"**Claude Code (only)** = **{n_code}**; "
        f"**any claude.ai** (incl. faces also in Code) = **{n_ai}**; "
        f"**neither / HF corpus only** = **{n_other}**. "
        f"Source: a9's local `~/.claude/kaomoji-journal.jsonl` + "
        f"the claude.ai exports listed in "
        f"`llmoji_study.local_emissions.DEFAULT_CLAUDE_EXPORTS`."
    )
    if args.color_by == "gt":
        lines.append(
            "- Display: **color = Claude-GT modal quadrant** where "
            "available; faces *not* in the GT set render **black** "
            "(`--color-by gt`). The elicitation-honest view: which "
            "wild-corpus faces fell outside the GT pilot's coverage. "
            "**Marker shape = deployment surface**: "
            "○ = Claude Code only, ◇ = any claude.ai, □ = neither. "
            "Hover surfaces the GT-overlap category, GT quadrant, and "
            "emit weight."
        )
    else:
        lines.append(
            "- Display: **color = BoL modal quadrant** for every face "
            "(uniformly, even where GT is available — surfaces the "
            "synthesis-vs-emission gap rather than collapsing shared "
            "faces to GT). **Marker shape = deployment surface**: "
            "○ = Claude Code only, ◇ = any claude.ai, □ = neither. "
            "Hover surfaces the GT-overlap category, BoL quadrant, and "
            "emit weight. (Companion view: re-run with "
            "`--color-by gt` to color by Claude-GT instead, with "
            "non-GT faces black.)"
        )
    lines.append("")
    lines.append(
        f"PCA on the 48-d BoL: "
        f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, "
        f"PC3={evr[2]*100:.1f}% var (sum {sum(evr[:3])*100:.1f}%). "
        "Figure: 3D side-by-side (quadrant × surface vs cluster) at "
        f"`figures/harness/wild_faces_pca_3d{fig_suffix}.html`."
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
            f"k={best_k} fixed for interpretability - silhouette favors the "
            "coarsest split, but k=6 is a local maximum after the k=5 dip."
        )
    lines.append(
        f"KMeans on the 48-d BoL, k={best_k}. Silhouette over the grid: "
        f"{sil_grid_str}.{fixed_note} Sorted by `wild_frac` descending - "
        "the top of the table is where the HF-corpus face vocabulary "
        "diverges most from the GT elicitation set. `share_shared_*` and "
        "`modal_quadrant_shared` are computed over the shared subset only "
        "(GT-honest); `share_full_*` and `modal_quadrant_full` use BoL-"
        "inferred quadrants for wild-* faces too. Cluster labels are the "
        "top-2 modal lexicon words (deterministic, no model call)."
    )
    lines.append("")
    lines.append(
        "| id | label | n | shared / wC / wO | wild_frac | "
        "modal Q (shared) | modal Q (full) | top lexicon words | sample |"
    )
    lines.append(
        "|---|---|---:|---|---:|---|---|---|---|"
    )
    for _, r in df_clusters.iterrows():
        modal_q_shared = str(r["modal_quadrant_shared"])
        modal_str_shared = (
            f"{modal_q_shared} ({float(r['modal_share_shared']):.2f})"
            if modal_q_shared else "-"
        )
        modal_q_full = str(r["modal_quadrant_full"])
        modal_str_full = (
            f"{modal_q_full} ({float(r['modal_share_full']):.2f})"
            if modal_q_full else "-"
        )
        lines.append(
            f"| {int(r['cluster_id'])} | {r['label']} | {int(r['n_faces'])} "
            f"| {int(r['n_shared'])} / {int(r['n_wild_claude'])} / "
            f"{int(r['n_wild_other'])} "
            f"| {float(r['wild_frac']):.2f} | {modal_str_shared} "
            f"| {modal_str_full} | {r['top_lexicon_words']} "
            f"| {r['sample_members']} |"
        )
    lines.append("")
    lines.append("")
    lines.append(HAND_MARKER)
    lines.append("")

    out_doc = Path(args.out_doc)
    if args.gt_only and "_gt_only" not in out_doc.stem:
        out_doc = out_doc.with_name(out_doc.stem + "_gt_only" + out_doc.suffix)
    if args.color_by == "gt" and "_gtcolor" not in out_doc.stem:
        out_doc = out_doc.with_name(out_doc.stem + "_gtcolor" + out_doc.suffix)
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
        lines.append("_(hand-edit this section after each run - interpretation goes here.)_")
        lines.append("")

    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_doc} "
          f"({'preserved hand-edits' if preserved else 'seeded interpretation stub'})")


if __name__ == "__main__":
    main()
