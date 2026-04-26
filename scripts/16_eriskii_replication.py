"""Eriskii-replication step 3: analysis + figures.

Sections in build order:
  - axis projection: data/eriskii_axes.tsv (TSV only — per-axis
    ranked-bar figures dropped, the per-model / per-project
    heatmaps already summarize axis structure)
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
    from pathlib import Path
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


def section_axes(
    fw: list[str],
    n: np.ndarray,
    P: np.ndarray,
) -> pd.DataFrame:
    """Write eriskii_axes.tsv. Per-axis ranked-bar PNGs intentionally
    dropped — the per-model / per-project axis heatmaps already
    summarize the same axis structure, and 21 single-axis bar charts
    cluttered figures/ without adding insight."""
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

    # Interactive plotly HTML — same xy, clusters, and Haiku labels as
    # the static figure so hovering matches what's in eriskii_clusters_tsne.
    _write_interactive_clusters_html(
        xy, fw, n, clusters, cluster_labels, descriptions_by_fw,
        out_path=FIGURES_DIR / "claude_faces_interactive.html",
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
    """Plotly HTML with the eriskii-clusters_tsne layout and clustering.

    One trace per cluster so the legend toggles clusters on/off and the
    Haiku-derived label appears in the legend. Hover shows the kaomoji,
    cluster label, emission count, and (truncated) synthesized
    description."""
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


def section_bridge(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    from llmoji.config import ERISKII_USER_KAOMOJI_CORR_TSV
    from llmoji.eriskii import user_kaomoji_axis_correlation

    embedder = SentenceTransformer(EMBED_MODEL)
    df = user_kaomoji_axis_correlation(
        rows, axes_df,
        embedder=embedder, axis_anchors=AXIS_ANCHORS, axis_order=ERISKII_AXES,
    )
    df.to_csv(ERISKII_USER_KAOMOJI_CORR_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_USER_KAOMOJI_CORR_TSV}")

    if df.empty:
        print("  no rows with surrounding_user; skipping figure")
        return df

    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("r", ascending=True)
    ax.barh(df_sorted["axis"], df_sorted["r"],
            color=["#444" if pb < 0.05 else "#bbb"
                   for pb in df_sorted["p_bonf"]])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson r (user-text axis projection × kaomoji axis projection)")
    n_used = int(df["n"].iloc[0]) if len(df) else 0
    ax.set_title(f"surrounding_user → kaomoji axis correlation\n"
                 f"n={n_used}; dark bars: p_bonf < 0.05 (Bonferroni across 21 axes)")
    fig.tight_layout()
    out = FIGURES_DIR / "eriskii_user_kaomoji_axis_corr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df


def section_writeup(
    df_axes: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_per_model: pd.DataFrame,
    df_per_project: pd.DataFrame,
    df_bridge: pd.DataFrame,
) -> None:
    from llmoji.config import ERISKII_COMPARISON_MD

    # Top-N kaomoji from our data (by emission count).
    top_us = df_axes.nlargest(20, "n")[["first_word", "n"]].copy()
    top_us["pct"] = (top_us["n"] / df_axes["n"].sum() * 100).round(1)

    # Eriskii's published top-20 kaomoji + frequencies (from the page text).
    eriskii_top20 = [
        ("(´・ω・`)", 248, "7.4"),
        ("(・ω・)",   213, "6.3"),
        ("(・∀・)",   194, "5.8"),
        ("(◕‿◕)",    145, "4.3"),
        ("(´-ω-`)",  120, "3.6"),
        ("(￣▽￣)",   84,  "2.5"),
        ("(｀・ω・´)", 74,  "2.2"),
        ("(⊙_⊙)",    63,  "1.9"),
        ("(・ω・)ノ", 60,  "1.8"),
        ("(°△°)",    57,  "1.7"),
        ("(╯°□°)╯︵ ┻━┻", 56, "1.7"),
        ("(・_・)",   52,  "1.5"),
        ("(・_・;)",  51,  "1.5"),
        ("(￣ω￣)",   51,  "1.5"),
        ("(☆▽☆)",    50,  "1.5"),
        ("(；・∀・)", 39,  "1.2"),
        ("(・_・ )",  38,  "1.1"),
        ("(・_・?)",  36,  "1.1"),
        ("(◕‿◕✿)",   36,  "1.1"),
        ("(・・?)",   31,  "0.9"),
    ]

    # Eriskii's full 15 cluster labels (visible on the page).
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

    lines: list[str] = []
    lines.append("# Eriskii-replication: narrative comparison")
    lines.append("")
    lines.append("Generated by `scripts/16_eriskii_replication.py`. Pre-registration record: `docs/superpowers/specs/2026-04-24-eriskii-replication-design.md`.")
    lines.append("")
    lines.append("## Methodology recap")
    lines.append("")
    lines.append("Replicates eriskii.net/projects/claude-faces' two-stage pipeline:")
    lines.append("1. Stage A: per kaomoji, sample up to 4 instances; mask the leading kaomoji with `[FACE]`; ask Haiku to describe what the masked face conveys.")
    lines.append("2. Stage B: per kaomoji, synthesize the per-instance descriptions into one canonical one-sentence meaning.")
    lines.append("3. Embed each synthesis with sentence-transformers/all-MiniLM-L6-v2.")
    lines.append("4. Project onto eriskii's 21 semantic axes (Warmth, Energy, …, Exhaustion).")
    lines.append("5. t-SNE + KMeans(k=15); Haiku per-cluster labels.")
    lines.append("")
    lines.append("Beyond eriskii: per-model axis breakdowns (eriskii's data lacked model info) + per-project axis breakdowns + mechanistic surrounding-user → kaomoji axis-correlation bridge.")
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

    lines.append("## Per-model axis means (claude-code only, n≥10 emissions)")
    lines.append("")
    if not df_per_model.empty:
        pivot = df_per_model.pivot(index="model", columns="axis", values="mean")
        pivot = pivot[ERISKII_AXES]
        lines.append("| model | " + " | ".join(ERISKII_AXES) + " |")
        lines.append("|" + "---|" * (len(ERISKII_AXES) + 1))
        for m in pivot.index:
            cells = [f"{pivot.loc[m, a]:+.2f}" for a in ERISKII_AXES]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        # per-model std summary
        std_pivot = df_per_model.pivot(index="model", columns="axis", values="std")
        avg_std = std_pivot.mean(axis=1).sort_values(ascending=False)
        lines.append("**Per-model average std across all 21 axes** (operationalizes eriskii's qualitative \"opus-4-6 had wider range\" claim):")
        lines.append("")
        lines.append("| model | mean axis std |")
        lines.append("|---|---|")
        for m, s in avg_std.items():
            lines.append(f"| {m} | {s:.4f} |")
        lines.append("")

    lines.append("## Mechanistic bridge: surrounding_user → kaomoji axis correlation")
    lines.append("")
    if not df_bridge.empty:
        lines.append("| axis | Pearson r | p | p_bonf | n |")
        lines.append("|---|---|---|---|---|")
        for _, r in df_bridge.sort_values("r", ascending=False).iterrows():
            sig = "**" if r["p_bonf"] < 0.05 else ""
            lines.append(f"| {r['axis']} | {sig}{r['r']:+.3f}{sig} | "
                         f"{r['p']:.3g} | {r['p_bonf']:.3g} | {int(r['n'])} |")
        lines.append("")
        lines.append("Bold = passes Bonferroni correction across 21 axes (p_bonf < 0.05). Reading: significant positive r on (e.g.) Warmth would mean warmer user messages elicit warmer kaomoji. Caveat: user-text and kaomoji-description embeddings live in the same MiniLM space, so correlation is at-best evidence of register-tracking, not direct evidence of internal state.")
        lines.append("")

    lines.append("## Caveats and known limitations")
    lines.append("")
    lines.append("- **Wetness anchor is a9lim's enrichment, not eriskii's**. Eriskii explicitly used the bare strings `wetness ↔ dryness` ('three seashells' joke; intentionally undefined). Our anchor reads 'waxing poetic about emotions, lyrical and self-expressive ↔ helpful assistant tone, task-focused, businesslike'. Wetness rankings are accordingly more meaningful than eriskii's but not directly comparable.")
    lines.append("- **Eriskii's per-kaomoji cluster assignments are not public** — comparison is theme-level only.")
    lines.append("- **The mask token `[FACE]` is sometimes referenced literally** in Haiku descriptions. Stage-B synthesis usually corrects for this but a few descriptions retain artifacts.")
    lines.append("- **Cross-pipeline embedding methodology**: eriskii embeds Haiku-synthesized meaning; we follow the same pipeline. The response-based embedding (`data/claude_faces_embed.parquet`) is preserved for future ad-hoc comparison.")
    lines.append("")

    ERISKII_COMPARISON_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {ERISKII_COMPARISON_MD}")


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
              "run scripts/15 first")
        sys.exit(1)
    _use_cjk_font()

    print("loading description embeddings (canonicalized)...")
    from llmoji.claude_faces import load_embeddings_canonical
    fw, n, E = load_embeddings_canonical(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} canonical kaomoji, {E.shape[1]}-dim")

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
    # Canonicalize the synthesized-description keys; on collision keep
    # the description from the variant with the most contributing
    # instances (n_descriptions). Mirrors the n-weighted merge that
    # load_embeddings_canonical does for the embedding side.
    from llmoji.taxonomy import canonicalize_kaomoji
    desc_candidates: dict[str, list[tuple[int, str]]] = {}
    with open(CLAUDE_HAIKU_SYNTHESIZED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            canon = canonicalize_kaomoji(r["first_word"])
            desc_candidates.setdefault(canon, []).append(
                (int(r.get("n_descriptions", 1)), str(r["synthesized"])),
            )
    descriptions_by_fw: dict[str, str] = {
        canon: max(cands, key=lambda c: c[0])[1]
        for canon, cands in desc_candidates.items()
    }
    df_clusters = section_clusters(fw, n, E, descriptions_by_fw)

    print("\n=== Section: per-model ===")
    rows = pd.read_json(CLAUDE_KAOMOJI_PATH, lines=True)
    # Canonicalize first_word so per-model / per-project / bridge joins
    # against df_axes (which is canonical) work correctly. Original
    # first_word preserved as first_word_raw for audit.
    rows = rows.assign(
        first_word_raw=rows["first_word"],
        first_word=rows["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    df_per_model = section_per_model(rows, df_axes)
    print("\n=== Section: per-project ===")
    df_per_project = section_per_project(rows, df_axes)

    print("\n=== Section: mechanistic bridge ===")
    df_bridge = section_bridge(rows, df_axes)

    print("\n=== Section: writeup ===")
    section_writeup(df_axes, df_clusters, df_per_model, df_per_project, df_bridge)


if __name__ == "__main__":
    main()
