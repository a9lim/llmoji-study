# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Eriskii-replication: axis projection + Haiku-labeled clusters + writeup.

Sections in build order:
  - axis projection: ``data/harness/eriskii_axes.tsv`` (per-kaomoji × 21 axes)
  - clusters: ``data/harness/eriskii_clusters.tsv`` +
    ``figures/harness/eriskii_clusters_tsne.png`` + interactive plotly
    HTML at ``figures/harness/claude_faces_interactive.html``
  - narrative writeup: ``data/harness/eriskii_comparison.md``

Pre-2026-04-27, this script also produced
``eriskii_per_model.tsv`` / ``eriskii_per_project.tsv`` and the
``surrounding_user → kaomoji`` mechanistic-bridge correlation. Those
all needed per-row ``model`` / ``project_slug`` / ``surrounding_user``
fields, which only existed in the local-scrape pipeline. Post-refactor
the corpus is the HF dataset ``a9lim/llmoji``. As of the dataset's 1.1
layout (one ``<sanitized-source-model>.jsonl`` per source model per
bundle), per-source-model breakdowns are back on the table — but
project_slug / surrounding_user remain pooled-out and aren't coming
back. The eriskii axis projection + clustering pieces still pool
across source models for the figures here; per-source-model views
should land in a follow-up script.

Inputs:
  - ``data/harness/claude_faces_embed_description.parquet`` (from script 15)
  - ``data/harness/claude_descriptions.jsonl`` (from script 06; for cluster-
    label prompts, which need the description string per kaomoji)

Usage:
    ANTHROPIC_API_KEY=... python scripts/64_eriskii_replication.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.claude_faces import EMBED_MODEL, load_embeddings
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    DATA_DIR,
    ERISKII_AXES,
    ERISKII_AXES_TSV,
    ERISKII_CLUSTERS_TSV,
    ERISKII_COMPARISON_MD,
    FIGURES_DIR,
    HAIKU_MODEL_ID,
)
from llmoji_study.eriskii import (
    compute_axis_vectors, label_cluster_via_haiku, project_onto_axes,
)
from llmoji_study.eriskii_anchors import AXIS_ANCHORS, CLUSTER_LABEL_PROMPT


def _use_cjk_font() -> None:
    """Same fallback chain used in analysis.py / emotional_analysis.py /
    63_corpus_pca.py — keep these in sync."""
    import matplotlib
    import matplotlib.font_manager as fm
    repo_root = Path(__file__).resolve().parent.parent
    emoji_font = repo_root / "data" / "fonts" / "NotoEmoji-Regular.ttf"
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


def _representative_descriptions(corpus_rows: list[dict]) -> dict[str, str]:
    """Map canonical kaomoji → a single representative description
    (the highest-count per-bundle synthesis). Used as the description
    string passed to Haiku in the cluster-labeling prompt."""
    out: dict[str, str] = {}
    for r in corpus_rows:
        descs = r.get("descriptions", [])
        if not descs:
            continue
        # corpus rows are already sorted (-count, contributor) by the
        # pull script, so the first entry is the most-evidenced one.
        out[r["kaomoji"]] = descs[0]["description"]
    return out


def section_axes(
    fw: list[str], n: np.ndarray, P: np.ndarray,
) -> pd.DataFrame:
    """Write eriskii_axes.tsv (one row per kaomoji × 21 axis cols)."""
    df = pd.DataFrame({"first_word": fw, "n": n})
    for j, name in enumerate(ERISKII_AXES):
        df[name] = P[:, j]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ERISKII_AXES_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_AXES_TSV}  ({len(df)} kaomoji × {len(ERISKII_AXES)} axes)")
    return df


def section_clusters(
    fw: list[str],
    n: np.ndarray,
    E: np.ndarray,
    descriptions_by_fw: dict[str, str],
) -> pd.DataFrame:
    """t-SNE + KMeans(k=15) + Haiku per-cluster labels."""
    import anthropic
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

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
        members = [(fw[i], descriptions_by_fw.get(fw[i], "")) for i in member_idx]
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

    # Static labeled t-SNE figure.
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors
    fig, ax = plt.subplots(figsize=(14, 10))
    sizes = np.clip(15 + 60 * np.log1p(n), 15, 250)
    colors = [palette[int(c) % len(palette)] for c in clusters]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    top_idx = np.argsort(-n)[:30]
    for i in top_idx:
        ax.annotate(fw[i], xy=(xy[i, 0], xy[i, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, color="#222")

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
    harness_dir = FIGURES_DIR / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    out = harness_dir / "eriskii_clusters_tsne.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    _write_interactive_clusters_html(
        xy, fw, n, clusters, cluster_labels, descriptions_by_fw,
        out_path=harness_dir / "claude_faces_interactive.html",
    )
    return df_clusters


def _write_interactive_clusters_html(
    xy: np.ndarray,
    fw: list[str],
    n: np.ndarray,
    clusters: np.ndarray,
    cluster_labels: dict[int, str],
    descriptions_by_fw: dict[str, str],
    *,
    out_path: Path,
) -> None:
    """Plotly HTML mirroring the static eriskii_clusters_tsne.png
    layout. One trace per cluster so the legend toggles clusters
    on/off. Hover surfaces kaomoji + cluster + count + description."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  (plotly not installed; skipping {out_path.name})")
        return

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a",
    ]
    sizes = np.clip(8 + 6 * np.log1p(n), 8, 36)

    def _truncate(s: str, lim: int = 140) -> str:
        if len(s) <= lim:
            return s
        return s[:lim].rstrip() + "…"

    fig = go.Figure()
    for c in sorted(cluster_labels):
        idxs = [i for i, ci in enumerate(clusters) if int(ci) == c]
        if not idxs:
            continue
        label = cluster_labels[c]
        hover = [
            (
                f"<b>{fw[i]}</b><br>"
                f"cluster {c}: {label}<br>"
                f"n = {int(n[i])}<br>"
                f"<i>{_truncate(descriptions_by_fw.get(fw[i], ''))}</i>"
            )
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter(
                x=xy[idxs, 0], y=xy[idxs, 1],
                mode="markers",
                name=f"{c}: {label}",
                text=hover,
                hoverinfo="text",
                marker=dict(
                    size=[float(sizes[i]) for i in idxs],
                    color=palette[c % len(palette)],
                    line=dict(color="white", width=0.6),
                    opacity=0.85,
                ),
            )
        )
    fig.update_layout(
        title=(
            f"Claude faces — t-SNE + KMeans(k={len(cluster_labels)}), "
            "Haiku-labeled clusters (canonicalized; description-based embeddings)"
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(title="cluster", itemsizing="constant"),
        width=1200, height=900,
    )
    fig.write_html(str(out_path))
    print(f"wrote {out_path}")


def section_writeup(
    df_axes: pd.DataFrame,
    df_clusters: pd.DataFrame,
    corpus_rows: list[dict],
) -> None:
    """Narrative comparison vs eriskii's published page."""
    top_us = df_axes.nlargest(20, "n")[["first_word", "n"]].copy()
    top_us["pct"] = (top_us["n"] / df_axes["n"].sum() * 100).round(1)

    eriskii_top20 = [
        ("(´・ω・`)", 248, "7.4"), ("(・ω・)", 213, "6.3"),
        ("(・∀・)",   194, "5.8"), ("(◕‿◕)",  145, "4.3"),
        ("(´-ω-`)",  120, "3.6"), ("(￣▽￣)",  84,  "2.5"),
        ("(｀・ω・´)", 74, "2.2"), ("(⊙_⊙)",   63,  "1.9"),
        ("(・ω・)ノ", 60, "1.8"), ("(°△°)",    57,  "1.7"),
        ("(╯°□°)╯︵ ┻━┻", 56, "1.7"), ("(・_・)", 52, "1.5"),
        ("(・_・;)",  51, "1.5"), ("(￣ω￣)",   51,  "1.5"),
        ("(☆▽☆)",    50, "1.5"), ("(；・∀・)", 39,  "1.2"),
        ("(・_・ )",  38, "1.1"), ("(・_・?)",  36,  "1.1"),
        ("(◕‿◕✿)",   36, "1.1"), ("(・・?)",   31,  "0.9"),
    ]
    eriskii_clusters = [
        "Warm reassuring support (50 faces)",
        "Warm supportive affirmation (37 faces)",
        "Warm affirmation and agreement",
        "Warm technical enthusiasm",
        "Wry Resignation",
        "Wry sympathy",
        "Clever Wry Delight",
        "Empathetic honesty",
        "Compassionate acknowledgment",
        "Sympathetic acknowledgment of difficulties",
        "Sheepish acknowledgment",
        "Eager to help",
        "Clever Admiration",
        "Thoughtful skepticism",
        "Thoughtful intellectual appreciation",
    ]

    our_kaomoji = set(df_axes["first_word"].tolist())
    overlap = []
    for fw, en, ept in eriskii_top20:
        in_ours = fw in our_kaomoji
        if in_ours:
            r = df_axes[df_axes["first_word"] == fw].iloc[0]
            our_n = int(r["n"])
            warmth = float(r["warmth"])
            wetness = float(r["wetness"])
            overlap.append((fw, en, ept, our_n, warmth, wetness, True))
        else:
            overlap.append((fw, en, ept, 0, 0.0, 0.0, False))
    overlap_count = sum(1 for *_, present in overlap if present)

    n_contributors = len({d["contributor"]
                          for r in corpus_rows for d in r["descriptions"]})
    backends_seen: set[str] = set()
    source_models_seen: set[str] = set()
    for r in corpus_rows:
        for sm in r.get("source_models", []) or []:
            source_models_seen.add(sm)
        for b in r.get("synthesis_backends", []) or []:
            backends_seen.add(b)
    backends_str = ", ".join(sorted(b for b in backends_seen if b)) or "anthropic (legacy)"
    n_source_models = len(source_models_seen)

    lines: list[str] = []
    lines.append("# Eriskii-replication: narrative comparison")
    lines.append("")
    lines.append(
        "Generated by `scripts/64_eriskii_replication.py` from the "
        "`a9lim/llmoji` HuggingFace dataset (contributor-submitted "
        "kaomoji corpus, contributor-side-synthesized per-face meanings)."
    )
    lines.append("")
    lines.append(
        f"Corpus snapshot: {len(corpus_rows)} canonical kaomoji from "
        f"{n_contributors} contributor(s), spanning {n_source_models} "
        f"source model(s); synthesis backends in this snapshot: {backends_str}."
    )
    lines.append("")
    lines.append("## Methodology recap")
    lines.append("")
    lines.append("Replicates eriskii.net/projects/claude-faces' two-stage pipeline:")
    lines.append("1. Stage A (contributor-side, in the `llmoji` package): per `(source_model, kaomoji)` cell, sample instances of the kaomoji written by that source model; mask the leading kaomoji with `[FACE]`; ask the synthesizer (one of Haiku, GPT-5.4 mini via the Responses API, or any local OpenAI-compatible model) to describe what the masked face conveys.")
    lines.append("2. Stage B (contributor-side): per `(source_model, kaomoji)`, synthesize the per-instance descriptions into one canonical one-or-two-sentence meaning. Each cell ships as one row in `<sanitized-source-model>.jsonl` in the bundle.")
    lines.append("3. (Here) embed each per-bundle / per-source-model synthesis with sentence-transformers/all-MiniLM-L6-v2; weighted-mean by per-bundle count across contributors and source models; L2-normalize.")
    lines.append("4. Project onto eriskii's 21 semantic axes (Warmth, Energy, …, Exhaustion).")
    lines.append("5. t-SNE + KMeans(k=15); Haiku per-cluster labels.")
    lines.append("")
    lines.append("Pre-2026-04-27 we also ran per-model and per-project axis breakdowns plus a `surrounding_user → kaomoji` axis-correlation bridge from a local Claude.ai-export + journal scrape. The HF dataset pools per-machine before upload, so project / user-text isn't available and those breakdowns are gone. Per-(source_model) breakdowns are recoverable now that the dataset's 1.1 layout splits each bundle by source model — landing in a follow-up script, not in this one.")
    lines.append("")
    lines.append(f"## Top-20 frequency overlap with eriskii: {overlap_count}/20")
    lines.append("")
    lines.append("| eriskii rank | kaomoji | eriskii n (%) | in our data | our n | our warmth | our wetness |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, (fw, en, ept, our_n, warmth, wetness, present) in enumerate(overlap, start=1):
        if present:
            lines.append(f"| {i} | {fw} | {en} ({ept}%) | yes | {our_n} | {warmth:+.2f} | {wetness:+.2f} |")
        else:
            lines.append(f"| {i} | {fw} | {en} ({ept}%) | no | — | — | — |")
    lines.append("")
    lines.append("## Our top-20 most-emitted kaomoji")
    lines.append("")
    lines.append("| rank | kaomoji | n | % of emissions |")
    lines.append("|---|---|---|---|")
    for i, (_, r) in enumerate(top_us.iterrows(), start=1):
        lines.append(f"| {i} | {r['first_word']} | {int(r['n'])} | {r['pct']}% |")
    lines.append("")

    lines.append("## Our 15 cluster labels vs eriskii's 15")
    lines.append("")
    lines.append("Eriskii's full 15 cluster names are visible on their public page. Direct numeric per-kaomoji cluster-membership comparison isn't possible (eriskii's per-kaomoji assignments aren't published) but theme-level comparison is.")
    lines.append("")
    lines.append("**Eriskii's 15** (in roughly decreasing size order; sizes shown where stated):")
    lines.append("")
    for label in eriskii_clusters:
        lines.append(f"- {label}")
    lines.append("")
    lines.append("**Ours** (15, sorted by size):")
    lines.append("")
    lines.append("| id | n | label | members (truncated) |")
    lines.append("|---|---|---|---|")
    for _, r in df_clusters.sort_values("n", ascending=False).iterrows():
        members = r["members"]
        if len(members) > 60:
            members = members[:60] + "…"
        lines.append(f"| {int(r['cluster_id'])} | {int(r['n'])} | "
                     f"{r['label']} | {members} |")
    lines.append("")

    lines.append("## Caveats and known limitations")
    lines.append("")
    lines.append("- **Wetness anchor is a9lim's enrichment, not eriskii's**. Eriskii explicitly used the bare strings `wetness ↔ dryness` ('three seashells' joke; intentionally undefined). Our anchor reads 'waxing poetic about emotions, lyrical and self-expressive ↔ helpful assistant tone, task-focused, businesslike'. Wetness rankings are accordingly more meaningful than eriskii's but not directly comparable.")
    lines.append("- **Eriskii's per-kaomoji cluster assignments are not public** — comparison is theme-level only.")
    lines.append("- **The mask token `[FACE]` is sometimes referenced literally** in Haiku descriptions. Stage-B synthesis usually corrects for this but a few descriptions retain artifacts.")
    lines.append("- **Per-machine, per-source-model pooling already happened on the contributor side.** Counts here are sums of per-bundle / per-source-model counts; a single heavy contributor can swing a face's rank. The corpus is small (n<1K canonical kaomoji as of writing) so this is worth watching.")
    lines.append("- **Per-project axis breakdowns are gone.** They needed per-row metadata that the HF dataset doesn't carry. If we want them back, the right move is a separate research-side scrape of a single contributor's local journal — not adding fields to the public dataset. Per-(source_model) breakdowns are coming back via the 1.1 layout but aren't in this script yet.")
    lines.append("- **Mixed synthesizers in one corpus.** The 1.1 layout records `synthesis_backend` per bundle (anthropic / openai / local). The figures here pool prose written by all three; if cross-backend prose-style drift contaminates the axis projections, a per-backend split is the move.")
    lines.append("")

    ERISKII_COMPARISON_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {ERISKII_COMPARISON_MD}")


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(
            f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
            "run scripts/15 first"
        )
        sys.exit(1)
    if not CLAUDE_DESCRIPTIONS_PATH.exists():
        print(
            f"no corpus at {CLAUDE_DESCRIPTIONS_PATH}; "
            "run scripts/60_corpus_pull.py first"
        )
        sys.exit(1)

    _use_cjk_font()

    print("loading description embeddings...")
    fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} canonical kaomoji, {E.shape[1]}-dim")

    print("loading raw corpus (for cluster-label descriptions)...")
    corpus_rows: list[dict] = []
    with CLAUDE_DESCRIPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            corpus_rows.append(json.loads(line))
    descriptions_by_fw = _representative_descriptions(corpus_rows)
    print(f"  {len(descriptions_by_fw)} representative descriptions")

    print("computing axis vectors...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    axes = compute_axis_vectors(embedder, AXIS_ANCHORS)

    print("projecting kaomoji onto axes...")
    P = project_onto_axes(E, axes, ERISKII_AXES)

    print("\n=== Section: axes ===")
    df_axes = section_axes(fw, n, P)

    print("\n=== Section: clusters ===")
    df_clusters = section_clusters(fw, n, E, descriptions_by_fw)

    print("\n=== Section: writeup ===")
    section_writeup(df_axes, df_clusters, corpus_rows)


if __name__ == "__main__":
    main()
