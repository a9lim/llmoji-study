# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Three-way (use / read / act) per-face analysis of Claude's
kaomoji-quadrant association.

Three structurally different channels for "what quadrant goes with
this face?":

  - **use (Claude-GT)**: ``data/harness/claude-runs*/run-*.jsonl``,
    Opus 4.7 emitting kaomoji under known Russell-prompted conditions.
    Per-face = the empirical distribution over the prompt-quadrants
    that elicited the face. Source: forward emit, controlled stimulus.
  - **read (Opus / Haiku face_likelihood)**:
    ``data/harness/face_likelihood_{opus,haiku}_summary.tsv``. Model
    shown the face cold, asked what affective state it represents.
    Source: symbolic interpretation, no context.
  - **act (BoL)**: ``data/harness/claude_faces_lexicon_bag.parquet``
    → 6-d quadrant distribution via
    :func:`llmoji_study.lexicon.bol_to_quadrant_distribution`. Built by
    pooling Haiku synthesis over many *real in-context emits* of the
    face. Source: in-deployment behavior summarized.

The three channels are not redundant — they're independent windows
on the same face:

  - use ≠ read: deployment context shapes face choice differently than
    the face's literal denoted meaning.
  - act ≠ read: the synthesizer pooling across emits picks up
    contextual signal that introspection-of-the-symbol misses.
  - use ≠ act: GT is elicited under prompt structure (Russell-balanced),
    while BoL is pooled over wild emit contexts (deployment-shaped).
    Disagreement = elicitation-bias measurement.

Outputs (data/harness/ + figures/harness/, all `_three_way`):
  - three_way_per_face.tsv  — per-face 4 distributions + 6 pairwise
    JSDs + modal labels + agreement-pattern code
  - three_way_summary.md    — narrative writeup with the headline
    pairwise-similarity table (face-uniform + emit-weighted),
    per-quadrant agreement table, top-N divergent faces, and the
    8-pattern agreement breakdown
  - three_way_pairwise_heatmap.png — 4×4 similarity heatmap
  - three_way_top_divergent.png — grouped-bar chart of top-N
    most-divergent faces (4 channels × 6 quadrants per face)

Usage:
    python scripts/harness/68_three_way_analysis.py
    python scripts/harness/68_three_way_analysis.py --gt-floor 1
    python scripts/harness/68_three_way_analysis.py --top-n 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmoji_study.claude_faces import load_bol_parquet
from llmoji_study.claude_gt import load_claude_gt_distribution
from llmoji_study.config import (
    CLAUDE_FACES_LEXICON_BAG_PATH,
    DATA_DIR,
    FIGURES_DIR,
)
from llmoji_study.emotional_analysis import QUADRANT_COLORS
from llmoji_study.jsd import js, normalize, similarity
from llmoji_study.lexicon import (
    QUADRANTS,
    bol_to_quadrant_distribution,
)


HARNESS_DIR = DATA_DIR / "harness"
HARNESS_FIG_DIR = FIGURES_DIR / "harness"

# Channel names — keep in sync with column prefixes downstream.
CHANNELS = ("gt", "opus", "haiku", "bol")
CHANNEL_LABELS = {
    "gt": "GT (use)",
    "opus": "Opus (read)",
    "haiku": "Haiku (read)",
    "bol": "BoL (act)",
}
# Six pairs out of the four channels.
PAIRS: list[tuple[str, str]] = [
    ("gt", "opus"),
    ("gt", "haiku"),
    ("gt", "bol"),
    ("opus", "haiku"),
    ("opus", "bol"),
    ("haiku", "bol"),
]


def _use_cjk_font() -> None:
    """Synced with scripts/harness/63_corpus_pca.py /
    llmoji_study/per_project_charts.py — keep these chains in sync."""
    import matplotlib.font_manager as fm
    repo_root = Path(__file__).resolve().parent.parent.parent
    emoji_font = repo_root / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    if emoji_font.exists() and "Noto Emoji" not in {f.name for f in fm.fontManager.ttflist}:
        try:
            fm.fontManager.addfont(str(emoji_font))
        except Exception:
            pass
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans",
        "DejaVu Serif", "Tahoma", "Noto Sans Canadian Aboriginal",
        "Heiti TC", "Noto Emoji", "Helvetica Neue",
    ]


# ---------------------------------------------------------------- loaders
def _load_gt(floor: int) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Per-face Claude-GT distribution (smoothed-normalized over
    QUADRANTS) plus per-face total emit count for emit-weighting."""
    raw = load_claude_gt_distribution(floor=floor)
    out: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for face, qcounts in raw.items():
        out[face] = normalize(
            {q: float(qcounts.get(q, 0)) for q in QUADRANTS},
            QUADRANTS,
        )
        counts[face] = int(sum(qcounts.values()))
    return out, counts


def _load_summary_softmax(path: Path) -> dict[str, list[float]]:
    """Load a face_likelihood summary TSV → {face: softmax-list}."""
    if not path.exists():
        sys.exit(f"missing {path}")
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    out: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        face = str(row["first_word"])
        # Re-normalize with smoothing so JSD stays finite even if a
        # softmax cell underflowed to 0.
        d = {q: float(row.get(f"softmax_{q}", 0.0) or 0.0) for q in QUADRANTS}
        out[face] = normalize(d, QUADRANTS)
    return out


def _load_bol(path: Path) -> dict[str, list[float]]:
    """Per-face BoL → quadrant distribution."""
    if not path.exists():
        sys.exit(f"missing {path}; run scripts/harness/62_corpus_lexicon.py first")
    fw, _, _, B = load_bol_parquet(path)
    out: dict[str, list[float]] = {}
    for i, face in enumerate(fw):
        dist = bol_to_quadrant_distribution(B[i])
        if dist.sum() <= 0:
            continue  # extension-only commit; no quadrant signal
        # normalize() smooths so JSD stays finite vs hard-zero cells
        out[face] = normalize(
            {q: float(dist[j]) for j, q in enumerate(QUADRANTS)},
            QUADRANTS,
        )
    return out


def _modal(dist: list[float]) -> str:
    return QUADRANTS[int(np.argmax(dist))]


def _agreement_pattern(modals: dict[str, str]) -> str:
    """Compress 4 modal labels to an agreement-pattern code.

    Reports which subset of the three non-GT channels agrees with GT.
    Format: "rrr" where each char is 1/0 for (opus==gt, haiku==gt,
    bol==gt). 8 patterns total. Read as a binary signature of which
    channel(s) the model's introspection / synthesis match the
    elicited use distribution on.
    """
    bits = (
        "1" if modals["opus"] == modals["gt"] else "0",
        "1" if modals["haiku"] == modals["gt"] else "0",
        "1" if modals["bol"] == modals["gt"] else "0",
    )
    return "".join(bits)


PATTERN_INTERPRETATIONS = {
    "111": "all channels agree",
    "110": "opus+haiku read GT; BoL acts differently",
    "101": "opus reads + BoL acts agree with GT; haiku diverges",
    "100": "only opus agrees with GT",
    "011": "haiku reads + BoL acts agree with GT; opus diverges",
    "010": "only haiku agrees with GT",
    "001": "only BoL agrees with GT",
    "000": "all introspection/synthesis channels disagree with GT",
}


# ---------------------------------------------------------------- analysis
def _per_face_table(
    faces: list[str],
    gt: dict[str, list[float]],
    opus: dict[str, list[float]],
    haiku: dict[str, list[float]],
    bol: dict[str, list[float]],
    counts: dict[str, int],
) -> pd.DataFrame:
    """Build the per-face TSV. One row per face × all four
    distributions × six pairwise JSDs × modal labels + agreement code."""
    rows = []
    for face in faces:
        d = {"gt": gt[face], "opus": opus[face], "haiku": haiku[face], "bol": bol[face]}
        modals = {ch: _modal(d[ch]) for ch in CHANNELS}
        row: dict = {
            "first_word": face,
            "emit_count": counts.get(face, 0),
        }
        for ch in CHANNELS:
            for j, q in enumerate(QUADRANTS):
                row[f"{ch}_{q}"] = round(d[ch][j], 4)
            row[f"modal_{ch}"] = modals[ch]
        # Pairwise JSDs in nats; report similarity = 1 - JSD/ln2 too.
        jsds: dict[tuple[str, str], float] = {}
        for a, b in PAIRS:
            j_ab = js(d[a], d[b])
            jsds[(a, b)] = j_ab
            row[f"jsd_{a}_{b}"] = round(j_ab, 4)
            row[f"sim_{a}_{b}"] = round(similarity(j_ab), 4)
        row["max_pair_jsd"] = round(max(jsds.values()), 4)
        row["min_pair_jsd"] = round(min(jsds.values()), 4)
        row["agreement_pattern"] = _agreement_pattern(modals)
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_means(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two 4×4 channel-pair similarity tables: face-uniform and
    emit-weighted. Symmetric matrices with 1.0 on the diagonal."""
    face_uniform = pd.DataFrame(
        index=list(CHANNELS), columns=list(CHANNELS), dtype=float,
    )
    emit_weighted = pd.DataFrame(
        index=list(CHANNELS), columns=list(CHANNELS), dtype=float,
    )
    for ch in CHANNELS:
        face_uniform.loc[ch, ch] = 1.0
        emit_weighted.loc[ch, ch] = 1.0
    weights = np.asarray(df["emit_count"].astype(float), dtype=float)
    w_total = float(weights.sum())
    for a, b in PAIRS:
        col = f"sim_{a}_{b}"
        face_uniform.loc[a, b] = float(df[col].mean())
        face_uniform.loc[b, a] = face_uniform.loc[a, b]
        if w_total > 0:
            sims = np.asarray(df[col].astype(float), dtype=float)
            emit_weighted.loc[a, b] = float((sims * weights).sum() / w_total)
        else:
            emit_weighted.loc[a, b] = float("nan")
        emit_weighted.loc[b, a] = emit_weighted.loc[a, b]
    return face_uniform, emit_weighted


def _per_quadrant_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For each GT modal quadrant, report the per-pair mean similarity
    over the faces in that quadrant. Reveals which quadrants each
    channel-pair handles well or poorly."""
    rows = []
    for q in QUADRANTS:
        mask = df["modal_gt"] == q
        n = int(mask.sum())
        if n == 0:
            continue
        emit = float(df.loc[mask, "emit_count"].sum())
        row: dict = {"gt_modal": q, "n_faces": n, "n_emit": int(emit)}
        for a, b in PAIRS:
            col = f"sim_{a}_{b}"
            row[f"sim_{a}_{b}"] = round(float(df.loc[mask, col].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _heatmap_panel(ax, M: pd.DataFrame, title: str, n_label: str) -> None:
    """Render one 4×4 channel-pair similarity heatmap."""
    arr = M.loc[list(CHANNELS), list(CHANNELS)].astype(float).values
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
    for i in range(len(CHANNELS)):
        for j in range(len(CHANNELS)):
            v = arr[i, j]
            txtc = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=txtc, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS], rotation=20, ha="right")
    ax.set_yticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
    ax.set_title(f"{title}\n({n_label})", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity (1 − JSD/ln2)")


def _heatmap_figure(
    M_face: pd.DataFrame,
    M_emit: pd.DataFrame,
    out_path: Path,
    n_faces: int,
    n_emit: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _heatmap_panel(axes[0], M_face, "Face-uniform pairwise similarity",
                   f"{n_faces} faces")
    _heatmap_panel(axes[1], M_emit, "Emit-weighted pairwise similarity",
                   f"{n_emit} emissions across {n_faces} faces")
    fig.suptitle(
        "Three-way per-face analysis: GT (use) vs Opus/Haiku (read) vs BoL (act)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _top_divergent_figure(
    df: pd.DataFrame,
    top_n: int,
    out_path: Path,
) -> None:
    """Grouped-bar chart of the top-N highest max-pair-JSD faces.

    For each face, four 6-bar groups stacked horizontally — one per
    channel — colored by quadrant. The visual question: where do the
    bars *not* line up?
    """
    top = df.nlargest(top_n, "max_pair_jsd").reset_index(drop=True)
    if len(top) == 0:
        return

    # 4 panels per face row, stacked vertically.
    fig, axes = plt.subplots(
        len(top), 1, figsize=(11, max(2.4, 0.85 * len(top))),
        sharex=False,
    )
    if len(top) == 1:
        axes = [axes]

    bar_w = 0.18
    x = np.arange(len(QUADRANTS))
    for row_i, (_, r) in enumerate(top.iterrows()):
        ax = axes[row_i]
        for ch_i, ch in enumerate(CHANNELS):
            vals = [float(r[f"{ch}_{q}"]) for q in QUADRANTS]
            ax.bar(
                x + (ch_i - 1.5) * bar_w, vals, width=bar_w,
                color=[QUADRANT_COLORS[q] for q in QUADRANTS],
                edgecolor="black", linewidth=0.4, alpha=0.85,
                label=CHANNEL_LABELS[ch] if row_i == 0 else None,
            )
            # Channel name above each bar group as a tiny watermark.
            ax.text(
                -0.5 + (ch_i - 1.5) * 0.12, 1.05,
                ch, fontsize=7, color="#666",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(QUADRANTS, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        title = (
            f"{r['first_word']}  ·  emit={int(r['emit_count'])}  ·  "
            f"max-pair JSD={float(r['max_pair_jsd']):.3f}  ·  "
            f"pattern {r['agreement_pattern']}  ·  "
            f"modals: gt={r['modal_gt']} opus={r['modal_opus']} "
            f"haiku={r['modal_haiku']} bol={r['modal_bol']}"
        )
        ax.set_title(title, fontsize=8.5, loc="left")

    fig.suptitle(
        f"Top-{len(top)} most-divergent faces: per-channel quadrant distributions",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def _summary_md(
    df: pd.DataFrame,
    M_face: pd.DataFrame,
    M_emit: pd.DataFrame,
    per_q: pd.DataFrame,
    pattern_counts: Counter,
    pattern_emit: Counter,
    top_n: int,
    out_path: Path,
    n_faces: int,
    n_emit: int,
    gt_floor: int,
) -> None:
    lines: list[str] = []
    lines.append("# Three-way per-face analysis: GT (use) × Opus/Haiku (read) × BoL (act)")
    lines.append("")
    lines.append(
        "Three structurally different windows on the same per-face "
        "quadrant association:"
    )
    lines.append("")
    lines.append(
        "- **GT (use)** — Opus 4.7 emitting the face under known "
        "Russell-prompted conditions (`data/harness/claude-runs*/`). "
        "*What the face is actually emitted under.*"
    )
    lines.append(
        "- **Opus / Haiku (read)** — model shown the face cold, asked "
        "what affective state it represents "
        "(`face_likelihood_{opus,haiku}_summary.tsv`). "
        "*Symbolic interpretation, no context.*"
    )
    lines.append(
        "- **BoL (act)** — Haiku synthesizer pooling adjective-bag "
        "picks across many *real in-context emits* of the face → "
        "6-d quadrant distribution from the 19 circumplex anchors in "
        "the locked v2 LEXICON (`claude_faces_lexicon_bag.parquet`). "
        "*In-deployment behavior summarized.*"
    )
    lines.append("")
    lines.append(
        f"Inner-join shared by all four channels: **{n_faces} faces "
        f"× {n_emit} GT emissions** (Claude-GT floor ≥ {gt_floor})."
    )
    lines.append("")

    # --- pairwise similarity ---
    lines.append("## Pairwise channel similarity")
    lines.append("")
    lines.append(
        "Mean of `1 − JSD/ln2` over the shared face set. Two flavors: "
        "face-uniform (each face counts equally) and emit-weighted "
        "(each face counts as Claude actually emits it). Same-cell "
        "diagonal is 1.0 by definition."
    )
    lines.append("")
    lines.append("**Face-uniform**:")
    lines.append("")
    header = "| · | " + " | ".join(CHANNEL_LABELS[c] for c in CHANNELS) + " |"
    lines.append(header)
    lines.append("|---" * (len(CHANNELS) + 1) + "|")
    for ch in CHANNELS:
        cells = [CHANNEL_LABELS[ch]] + [
            f"{float(M_face.loc[ch, c]):.3f}" for c in CHANNELS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Emit-weighted**:")
    lines.append("")
    lines.append(header)
    lines.append("|---" * (len(CHANNELS) + 1) + "|")
    for ch in CHANNELS:
        cells = [CHANNEL_LABELS[ch]] + [
            f"{float(M_emit.loc[ch, c]):.3f}" for c in CHANNELS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Read-out: which off-diagonal pair is highest? lowest?
    pair_face = []
    pair_emit = []
    for a, b in PAIRS:
        pair_face.append((a, b, float(M_face.loc[a, b])))
        pair_emit.append((a, b, float(M_emit.loc[a, b])))
    pair_face.sort(key=lambda kv: -kv[2])
    pair_emit.sort(key=lambda kv: -kv[2])
    lines.append(
        "**Reading the matrix** — highest pairwise (face-uniform) is "
        + ", ".join(
            f"`{a}↔{b}` ({s:.3f})" for a, b, s in pair_face[:2]
        )
        + "; lowest is "
        + ", ".join(
            f"`{a}↔{b}` ({s:.3f})" for a, b, s in pair_face[-2:]
        )
        + "."
    )
    lines.append("")

    # --- per-quadrant ---
    lines.append("## Per-GT-quadrant pairwise similarity")
    lines.append("")
    lines.append(
        "Restrict to faces with each modal-GT label, then re-mean the "
        "per-pair similarities. Reveals which channels handle which "
        "quadrants well — e.g. NB tends to be a BoL win (the lexicon "
        "has explicit `neutral`/`detached` anchors), HP often a BoL "
        "weakness (deployment use diverges from denoted meaning)."
    )
    lines.append("")
    lines.append("| GT modal | n faces | n emit | "
                 + " | ".join(f"{a}↔{b}" for a, b in PAIRS) + " |")
    lines.append("|---" * (3 + len(PAIRS)) + "|")
    for _, r in per_q.iterrows():
        cells = [
            str(r["gt_modal"]), str(int(r["n_faces"])), str(int(r["n_emit"])),
        ] + [f"{float(r[f'sim_{a}_{b}']):.2f}" for a, b in PAIRS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- agreement patterns ---
    lines.append("## Modal-agreement patterns")
    lines.append("")
    lines.append(
        "Each pattern is a 3-bit code `(opus==gt)(haiku==gt)(bol==gt)` "
        "— which subset of the three non-GT channels agrees with the "
        "GT modal quadrant. 8 patterns total."
    )
    lines.append("")
    lines.append("| pattern | meaning | n faces | % faces | n emit | % emit |")
    lines.append("|---|---|---:|---:|---:|---:|")
    n_total_emit = sum(pattern_emit.values())
    n_total_faces = sum(pattern_counts.values())
    for pat in sorted(PATTERN_INTERPRETATIONS, key=lambda p: -pattern_counts[p]):
        nf = pattern_counts[pat]
        ne = pattern_emit[pat]
        lines.append(
            f"| `{pat}` | {PATTERN_INTERPRETATIONS[pat]} | {nf} | "
            f"{nf/max(n_total_faces,1):.1%} | {ne} | "
            f"{ne/max(n_total_emit,1):.1%} |"
        )
    lines.append("")

    # --- top-N divergent ---
    top = df.nlargest(top_n, "max_pair_jsd")
    lines.append(f"## Top-{top_n} most-divergent faces (by max pairwise JSD)")
    lines.append("")
    lines.append(
        "These are the diagnostic faces — where the use / read / act "
        "channels pull in different directions most. The agreement "
        "pattern column tells you which subset of channels GT-aligns; "
        "the per-channel modals tell you the specific disagreement."
    )
    lines.append("")
    lines.append(
        "| face | emit | pattern | gt | opus | haiku | bol | "
        "max-pair JSD | tightest pair |"
    )
    lines.append("|---|---:|---|---|---|---|---|---:|---|")
    for _, r in top.iterrows():
        # which pair has the smallest JSD = the agreeing pair
        pair_sims = [(a, b, float(r[f"sim_{a}_{b}"])) for a, b in PAIRS]
        pair_sims.sort(key=lambda kv: -kv[2])
        tight = f"{pair_sims[0][0]}↔{pair_sims[0][1]} (sim {pair_sims[0][2]:.2f})"
        lines.append(
            f"| `{r['first_word']}` | {int(r['emit_count'])} "
            f"| `{r['agreement_pattern']}` "
            f"| {r['modal_gt']} | {r['modal_opus']} | "
            f"{r['modal_haiku']} | {r['modal_bol']} "
            f"| {float(r['max_pair_jsd']):.3f} | {tight} |"
        )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `data/harness/three_way_per_face.tsv` — per-face data")
    lines.append("- `figures/harness/three_way_pairwise_heatmap.png`")
    lines.append("- `figures/harness/three_way_top_divergent.png`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-floor", type=int, default=3,
                    help="min Claude-GT emit count per face (default: 3)")
    ap.add_argument("--opus-tsv", type=Path,
                    default=HARNESS_DIR / "face_likelihood_opus_summary.tsv")
    ap.add_argument("--haiku-tsv", type=Path,
                    default=HARNESS_DIR / "face_likelihood_haiku_summary.tsv")
    ap.add_argument("--bol-parquet", type=Path,
                    default=CLAUDE_FACES_LEXICON_BAG_PATH)
    ap.add_argument("--top-n", type=int, default=12,
                    help="how many top-divergent faces to render (default: 12)")
    args = ap.parse_args()

    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _use_cjk_font()

    print(f"loading Claude-GT (floor={args.gt_floor})...")
    gt, gt_counts = _load_gt(args.gt_floor)
    print(f"  {len(gt)} faces in GT")

    print(f"loading Opus face_likelihood from {args.opus_tsv.name}...")
    opus = _load_summary_softmax(args.opus_tsv)
    print(f"  {len(opus)} faces")

    print(f"loading Haiku face_likelihood from {args.haiku_tsv.name}...")
    haiku = _load_summary_softmax(args.haiku_tsv)
    print(f"  {len(haiku)} faces")

    print(f"loading BoL parquet from {args.bol_parquet.name}...")
    bol = _load_bol(args.bol_parquet)
    print(f"  {len(bol)} faces with circumplex commitment")

    shared = sorted(set(gt) & set(opus) & set(haiku) & set(bol))
    n_emit = sum(gt_counts.get(f, 0) for f in shared)
    print(f"\ninner-join: {len(shared)} faces shared across all 4 channels")
    print(f"  {n_emit} GT emissions covered")
    if not shared:
        sys.exit("no shared faces — aborting")

    df = _per_face_table(shared, gt, opus, haiku, bol, gt_counts)
    out_tsv = HARNESS_DIR / "three_way_per_face.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    M_face, M_emit = _pairwise_means(df)
    per_q = _per_quadrant_breakdown(df)
    pattern_counts: Counter = Counter(df["agreement_pattern"])
    pattern_emit: Counter = Counter()
    for _, r in df.iterrows():
        pattern_emit[r["agreement_pattern"]] += int(r["emit_count"])

    print("\n=== pairwise similarity (face-uniform) ===")
    for a, b in PAIRS:
        s_face = float(M_face.loc[a, b])
        s_emit = float(M_emit.loc[a, b])
        print(f"  {a}↔{b}: face-uniform={s_face:.3f}  emit-weighted={s_emit:.3f}")

    print("\n=== modal-agreement patterns ===")
    for pat in sorted(PATTERN_INTERPRETATIONS, key=lambda p: -pattern_counts[p]):
        nf = pattern_counts[pat]
        ne = pattern_emit[pat]
        print(
            f"  {pat}  ({nf:3d} faces / {ne:5d} emit) — "
            f"{PATTERN_INTERPRETATIONS[pat]}"
        )

    fig_path = HARNESS_FIG_DIR / "three_way_pairwise_heatmap.png"
    _heatmap_figure(M_face, M_emit, fig_path, len(shared), n_emit)
    print(f"wrote {fig_path}")

    fig_path = HARNESS_FIG_DIR / "three_way_top_divergent.png"
    _top_divergent_figure(df, args.top_n, fig_path)
    print(f"wrote {fig_path}")

    md_path = HARNESS_DIR / "three_way_summary.md"
    _summary_md(
        df, M_face, M_emit, per_q,
        pattern_counts, pattern_emit,
        top_n=args.top_n, out_path=md_path,
        n_faces=len(shared), n_emit=n_emit,
        gt_floor=args.gt_floor,
    )


if __name__ == "__main__":
    main()
