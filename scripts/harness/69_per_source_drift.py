# pyright: reportArgumentType=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Per-source-model drift analysis: is the use/act BoL gap specific
to claude-opus-* or shared across providers?

Builds on script 68 (the 3-way GT/read/act analysis) by splitting
the BoL "act" channel by source model. Headline question: when BoL
disagrees with Claude-GT on a face like `(╯°□°)` (GT says HN-D,
pooled BoL says HP), is that disagreement concentrated in
claude-opus-4-7's deployment pattern, or do gpt-5.5 / codex-hook
synthesize the same face the same way?

Two structurally different findings live in the gap:
  - **shared across providers**: the symbol's deployment use has
    drifted from its denoted meaning generally (a kaomoji-vocabulary
    fact, not a Claude-specific behavior).
  - **claude-opus-4-7-only**: Claude specifically deploys the symbol
    differently from how other models do (a Claude-deployment fact —
    "Claude performs intensity through anger-coded faces" or similar).

Inputs:
  - data/harness/claude_faces_lexicon_bag_per_source.parquet (script 64)
  - data/harness/claude-runs*/ via load_claude_gt_distribution
  - data/harness/face_likelihood_{opus,haiku}_summary.tsv

Outputs:
  - data/harness/per_source_drift.tsv — per-(face, source_model)
    cell with quadrant distribution, modal Q, JSD vs GT, JSD vs
    pooled BoL, JSD vs each cross-source pair
  - data/harness/per_source_drift_summary.md — narrative writeup
    with: per-source mean similarity vs GT, cross-source-model
    disagreement table, the (╯°□°) and (´;ω;`) case files,
    headline call on shared-vs-Claude-specific
  - figures/harness/per_source_modal_heatmap.png — face × source_model
    grid with modal-quadrant cells colored by quadrant

Caveat: the synthesizer is Haiku for every source's per-face
synthesis. So cross-source variance reflects what Haiku reads from
the *contexts surrounding* the kaomoji when the kaomoji is in a
gpt-5.5 transcript vs a claude-opus-4-7 transcript. That's still
a deployment-pattern signal — the surrounding text is real
deployment evidence — but it isn't a clean comparison of "what each
provider's model thinks". We're measuring how Haiku *reads*
provider-conditioned context.

Usage:
    python scripts/harness/69_per_source_drift.py
    python scripts/harness/69_per_source_drift.py --gt-floor 1
    python scripts/harness/69_per_source_drift.py --min-cell-count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmoji_study.claude_faces import (
    load_bol_parquet,
    load_bol_parquet_per_source,
)
from llmoji_study.claude_gt import load_claude_gt_distribution
from llmoji_study.config import (
    CLAUDE_FACES_LEXICON_BAG_PATH,
    CLAUDE_FACES_LEXICON_BAG_PER_SOURCE_PATH,
    DATA_DIR,
    FIGURES_DIR,
)
from llmoji_study.emotional_analysis import QUADRANT_COLORS
from llmoji_study.jsd import js, normalize, similarity
from llmoji_study.lexicon import (
    QUADRANTS,
    bol_modal_quadrant,
    bol_to_quadrant_distribution,
)


HARNESS_DIR = DATA_DIR / "harness"
HARNESS_FIG_DIR = FIGURES_DIR / "harness"

# Source models that get pulled out as "interesting" for per-cell
# comparison. Anything below this stays in summaries only — single-
# source-only faces don't tell us about cross-source drift.
DEFAULT_FOCUS_SOURCES = (
    "claude-opus-4-7", "claude-opus-4-6",
    "codex-hook", "gpt-5.5", "gpt-5.4",
)


def _use_cjk_font() -> None:
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


def _modal(dist: list[float]) -> str:
    return QUADRANTS[int(np.argmax(dist))]


def _gt_dist_per_face(floor: int) -> tuple[dict[str, list[float]], dict[str, int]]:
    raw = load_claude_gt_distribution(floor=floor)
    out: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for face, qcounts in raw.items():
        out[face] = normalize(
            {q: float(qcounts.get(q, 0)) for q in QUADRANTS}, QUADRANTS,
        )
        counts[face] = int(sum(qcounts.values()))
    return out, counts


def _per_source_dists(path: Path) -> tuple[dict[tuple[str, str], list[float]],
                                            dict[tuple[str, str], int],
                                            list[str]]:
    """Load per-source BoL parquet → {(face, source_model):
    quadrant-distribution} + per-cell emit count + sorted source list.
    """
    faces, sms, counts, _, B = load_bol_parquet_per_source(path)
    out_d: dict[tuple[str, str], list[float]] = {}
    out_c: dict[tuple[str, str], int] = {}
    seen_sources: set[str] = set()
    for i, face in enumerate(faces):
        sm = sms[i]
        seen_sources.add(sm)
        dist = bol_to_quadrant_distribution(B[i])
        if dist.sum() <= 0:
            continue
        out_d[(face, sm)] = normalize(
            {q: float(dist[j]) for j, q in enumerate(QUADRANTS)}, QUADRANTS,
        )
        out_c[(face, sm)] = int(counts[i])
    return out_d, out_c, sorted(seen_sources)


def _pooled_bol_dists(path: Path) -> dict[str, list[float]]:
    """The single-pooled BoL from script 62, for reference."""
    fw, _, _, B = load_bol_parquet(path)
    out: dict[str, list[float]] = {}
    for i, face in enumerate(fw):
        dist = bol_to_quadrant_distribution(B[i])
        if dist.sum() <= 0:
            continue
        out[face] = normalize(
            {q: float(dist[j]) for j, q in enumerate(QUADRANTS)}, QUADRANTS,
        )
    return out


def _per_cell_table(
    per_src: dict[tuple[str, str], list[float]],
    cell_counts: dict[tuple[str, str], int],
    pooled: dict[str, list[float]],
    gt: dict[str, list[float]],
    gt_counts: dict[str, int],
    *,
    min_cell_count: int = 1,
) -> pd.DataFrame:
    """One row per (face, source_model). Columns: per-quadrant softmax,
    modal, JSD vs GT (when face is in GT), JSD vs pooled BoL."""
    rows = []
    for (face, sm), dist in per_src.items():
        cell_n = cell_counts.get((face, sm), 0)
        if cell_n < min_cell_count:
            continue
        modal = _modal(dist)
        row: dict = {
            "first_word": face,
            "source_model": sm,
            "cell_count": cell_n,
        }
        for j, q in enumerate(QUADRANTS):
            row[f"sm_{q}"] = round(dist[j], 4)
        row["modal_sm"] = modal
        if face in gt:
            j_gt = js(dist, gt[face])
            row["jsd_vs_gt"] = round(j_gt, 4)
            row["sim_vs_gt"] = round(similarity(j_gt), 4)
            row["modal_gt"] = _modal(gt[face])
            row["gt_emit"] = gt_counts.get(face, 0)
        else:
            row["jsd_vs_gt"] = float("nan")
            row["sim_vs_gt"] = float("nan")
            row["modal_gt"] = ""
            row["gt_emit"] = 0
        if face in pooled:
            j_pool = js(dist, pooled[face])
            row["jsd_vs_pooled"] = round(j_pool, 4)
            row["sim_vs_pooled"] = round(similarity(j_pool), 4)
        else:
            row["jsd_vs_pooled"] = float("nan")
            row["sim_vs_pooled"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _per_source_summary(
    df: pd.DataFrame, sources: list[str],
) -> pd.DataFrame:
    """Per-source-model: face count, emit count, mean sim vs GT
    (face-uniform + emit-weighted)."""
    rows = []
    for sm in sources:
        sub = df[df["source_model"] == sm]
        if sub.empty:
            continue
        n_cells = len(sub)
        n_emits = int(sub["cell_count"].sum())
        with_gt = sub.dropna(subset=["sim_vs_gt"])
        if with_gt.empty:
            row = {
                "source_model": sm,
                "n_cells": n_cells,
                "n_emits": n_emits,
                "n_with_gt": 0,
                "sim_vs_gt_face_unif": float("nan"),
                "sim_vs_gt_emit_wt": float("nan"),
                "modal_agree_rate": float("nan"),
            }
        else:
            sims = with_gt["sim_vs_gt"].astype(float).values
            weights = with_gt["cell_count"].astype(float).values
            face_unif = float(np.mean(sims))
            emit_wt = float(np.average(sims, weights=weights)) if weights.sum() > 0 else float("nan")
            agree = float((with_gt["modal_sm"] == with_gt["modal_gt"]).mean())
            row = {
                "source_model": sm,
                "n_cells": n_cells,
                "n_emits": n_emits,
                "n_with_gt": len(with_gt),
                "sim_vs_gt_face_unif": round(face_unif, 4),
                "sim_vs_gt_emit_wt": round(emit_wt, 4),
                "modal_agree_rate": round(agree, 4),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_source_table(
    per_src: dict[tuple[str, str], list[float]],
    sources: list[str],
    *,
    min_pair_overlap: int = 5,
) -> pd.DataFrame:
    """For each (sm_a, sm_b) pair, mean-pair-JSD over faces both
    sources synthesized. Symmetric square matrix flattened to long
    rows. Skip pairs with < min_pair_overlap shared faces — too noisy."""
    rows = []
    for i, a in enumerate(sources):
        for b in sources[i:]:
            shared = [
                face for (face, sm) in per_src
                if sm == a and (face, b) in per_src
            ]
            if len(shared) < min_pair_overlap and a != b:
                continue
            if a == b:
                rows.append({
                    "sm_a": a, "sm_b": b,
                    "n_shared": len(shared),
                    "mean_jsd": 0.0,
                    "mean_sim": 1.0,
                    "modal_agree_rate": 1.0,
                })
                continue
            jsds = []
            agrees = 0
            for face in shared:
                j_ab = js(per_src[(face, a)], per_src[(face, b)])
                jsds.append(j_ab)
                if _modal(per_src[(face, a)]) == _modal(per_src[(face, b)]):
                    agrees += 1
            rows.append({
                "sm_a": a, "sm_b": b,
                "n_shared": len(shared),
                "mean_jsd": round(float(np.mean(jsds)), 4),
                "mean_sim": round(similarity(float(np.mean(jsds))), 4),
                "modal_agree_rate": round(agrees / len(shared), 4),
            })
    return pd.DataFrame(rows)


def _case_file(
    face: str,
    per_src: dict[tuple[str, str], list[float]],
    cell_counts: dict[tuple[str, str], int],
    pooled: dict[str, list[float]],
    gt: dict[str, list[float]],
    gt_counts: dict[str, int],
    sources: list[str],
) -> list[str]:
    """Markdown lines for a single face's per-source-model breakdown.
    Used for the (╯°□°) and (´;ω;`) case files."""
    lines: list[str] = []
    if face not in pooled:
        lines.append(f"### `{face}` — not in BoL parquet, skipping")
        lines.append("")
        return lines
    lines.append(f"### `{face}` — case file")
    lines.append("")
    lines.append(
        "| channel | n | "
        + " | ".join(QUADRANTS) + " | modal |"
    )
    lines.append("|---|---:|" + "|".join(["---"] * len(QUADRANTS)) + "|---|")
    if face in gt:
        gt_n = gt_counts.get(face, 0)
        gt_dist = gt[face]
        cells = ["**GT (use)**", str(gt_n)] + [
            f"{gt_dist[j]:.2f}" for j in range(len(QUADRANTS))
        ] + [f"**{_modal(gt_dist)}**"]
        lines.append("| " + " | ".join(cells) + " |")
    pool_dist = pooled[face]
    cells = ["BoL pooled", "—"] + [
        f"{pool_dist[j]:.2f}" for j in range(len(QUADRANTS))
    ] + [_modal(pool_dist)]
    lines.append("| " + " | ".join(cells) + " |")
    for sm in sources:
        if (face, sm) not in per_src:
            continue
        d = per_src[(face, sm)]
        n = cell_counts.get((face, sm), 0)
        cells = [f"BoL · {sm}", str(n)] + [
            f"{d[j]:.2f}" for j in range(len(QUADRANTS))
        ] + [_modal(d)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# ------------------------------------------------------------- figures
def _modal_heatmap_figure(
    per_src: dict[tuple[str, str], list[float]],
    cell_counts: dict[tuple[str, str], int],
    sources: list[str],
    out_path: Path,
    *,
    top_faces_n: int = 30,
    min_total: int = 5,
) -> None:
    """face × source_model grid with cells colored by modal quadrant.
    Top-N faces by total cross-source emit; sources ordered by total
    emit volume descending."""
    # Pick faces with most cross-source coverage * total emit.
    by_face: dict[str, dict[str, str]] = {}
    by_face_total: dict[str, int] = {}
    for (face, sm), dist in per_src.items():
        n = cell_counts.get((face, sm), 0)
        by_face.setdefault(face, {})[sm] = _modal(dist)
        by_face_total[face] = by_face_total.get(face, 0) + n
    # Filter by min coverage and sort by emit*coverage proxy.
    eligible = [
        (face, by_face[face], by_face_total[face])
        for face in by_face
        if len(by_face[face]) >= 2 and by_face_total[face] >= min_total
    ]
    eligible.sort(key=lambda kv: (-kv[2], -len(kv[1])))
    eligible = eligible[:top_faces_n]
    if not eligible:
        print("  (no eligible faces for heatmap)")
        return
    faces = [f for f, _, _ in eligible]

    # Q index helper.
    q_to_color = {q: QUADRANT_COLORS[q] for q in QUADRANTS}
    n_rows = len(faces)
    n_cols = len(sources)

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_cols + 2.5), max(5, 0.32 * n_rows + 1.5)))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    for i, face in enumerate(faces):
        for j, sm in enumerate(sources):
            modal = by_face[face].get(sm)
            if modal is None:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color="#999")
                continue
            n = cell_counts.get((face, sm), 0)
            ax.add_patch(plt.Rectangle(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                facecolor=q_to_color[modal], edgecolor="white",
                linewidth=0.5, alpha=0.85,
            ))
            label = f"{modal}\n({n})"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(faces, fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(
        f"Per-(face, source_model) BoL modal quadrant\n"
        f"top {n_rows} faces by cross-source emit volume "
        f"(cell = modal Q + emit count)",
        fontsize=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Quadrant legend
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=QUADRANT_COLORS[q],
                   markeredgecolor="white", label=q)
        for q in QUADRANTS
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, title="modal Q")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_md(
    per_source: pd.DataFrame,
    cross_source: pd.DataFrame,
    df: pd.DataFrame,
    case_face_lines: list[list[str]],
    out_path: Path,
    *,
    n_total_cells: int,
    n_faces_with_gt: int,
    n_multi_source_faces: int,
    pooled_sim_vs_gt: float,
    gt_floor: int,
) -> None:
    lines: list[str] = []
    lines.append("# Per-source-model BoL drift")
    lines.append("")
    lines.append(
        "Splits the BoL channel from script 68's three-way analysis "
        "by source model. The pooled BoL aggregates across every "
        "source's per-face synthesis; here each source-model's "
        "BoL is kept separate. Headline question: when pooled BoL "
        "drifts from Claude-GT (e.g. on `(╯°□°)`), is the drift "
        "shared across providers (a kaomoji-vocabulary fact) or "
        "concentrated in claude-opus-4-7's deployment (a "
        "Claude-specific behavior)?"
    )
    lines.append("")
    lines.append(
        f"Coverage: **{n_total_cells} (face, source_model) cells** "
        f"across {len(per_source)} source models. "
        f"{n_multi_source_faces} faces appear under ≥2 sources. "
        f"Claude-GT (floor ≥ {gt_floor}) covers "
        f"{n_faces_with_gt} faces in this set."
    )
    lines.append("")
    lines.append(
        f"For reference: pooled-BoL solo similarity vs Claude-GT "
        f"(face-uniform across the same {n_faces_with_gt}-face set) "
        f"= **{pooled_sim_vs_gt:.3f}**."
    )
    lines.append("")

    # Per-source summary
    lines.append("## Per-source-model summary")
    lines.append("")
    lines.append(
        "Each row: how that source's per-face BoL stacks up against "
        "Claude-GT (Opus-4.7 elicitation). `modal_agree_rate` is the "
        "fraction of source-cells whose argmax quadrant matches GT's "
        "argmax."
    )
    lines.append("")
    lines.append(
        "| source_model | n cells | n emits | n with GT | "
        "sim vs GT (face-uniform) | sim vs GT (emit-weighted) | "
        "modal agree |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in per_source.sort_values("n_emits", ascending=False).iterrows():
        face_unif = r["sim_vs_gt_face_unif"]
        emit_wt = r["sim_vs_gt_emit_wt"]
        agree = r["modal_agree_rate"]
        lines.append(
            f"| `{r['source_model']}` | {int(r['n_cells'])} "
            f"| {int(r['n_emits'])} | {int(r['n_with_gt'])} "
            f"| {face_unif if pd.isna(face_unif) else f'{float(face_unif):.3f}'} "
            f"| {emit_wt if pd.isna(emit_wt) else f'{float(emit_wt):.3f}'} "
            f"| {agree if pd.isna(agree) else f'{float(agree):.0%}'} |"
        )
    lines.append("")

    # Cross-source pairwise
    lines.append("## Cross-source-model pairwise BoL similarity")
    lines.append("")
    lines.append(
        "On faces synthesized under both sources (≥5 shared faces), "
        "mean similarity (`1 − JSD/ln2`) of their per-face BoL "
        "distributions. High = sources synthesize the face the same "
        "way; low = sources read the face's deployment context "
        "differently."
    )
    lines.append("")
    lines.append("| sm_a | sm_b | n shared | mean sim | modal agree |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in cross_source[cross_source["sm_a"] != cross_source["sm_b"]] \
            .sort_values("n_shared", ascending=False).iterrows():
        lines.append(
            f"| `{r['sm_a']}` | `{r['sm_b']}` | {int(r['n_shared'])} "
            f"| {float(r['mean_sim']):.3f} "
            f"| {float(r['modal_agree_rate']):.0%} |"
        )
    lines.append("")

    # Case files
    lines.append("## Case files")
    lines.append("")
    lines.append(
        "Per-face breakdowns for the divergent faces from script 68's "
        "top-divergent table. The pattern to read: does GT's modal "
        "match more sources, or fewer? Where does the gap between "
        "GT and pooled-BoL come from?"
    )
    lines.append("")
    for case in case_face_lines:
        lines.extend(case)

    # Big-picture interpretive note
    lines.append("## Reading the result")
    lines.append("")
    lines.append(
        "Two diagnostic comparisons matter for the use/act gap "
        "interpretation:"
    )
    lines.append("")
    lines.append(
        "1. **claude-opus-4-7 vs claude-opus-4-6 BoL similarity** — "
        "if these two are very close to each other and both diverge "
        "from non-Claude sources, the pattern is Claude-deployment-"
        "specific (likely a Claude-deployment register fact, not a "
        "model-version fact)."
    )
    lines.append(
        "2. **claude-opus-4-7 vs gpt-5.5 / codex-hook BoL similarity** "
        "— if these are notably *lower* than Claude-vs-Claude, "
        "deployment patterns differ across providers on the same "
        "face vocabulary."
    )
    lines.append("")
    lines.append(
        "Caveat to internalize: every per-source BoL was synthesized "
        "by **the same Haiku model** reading provider-conditioned "
        "transcript context. So this measures how Haiku reads the "
        "context surrounding the kaomoji when the surrounding text "
        "is in each provider's style. That's still a real "
        "deployment-pattern signal — the surrounding text *is* "
        "deployment evidence — but it isn't a clean comparison of "
        "what each provider's model 'thinks'."
    )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-floor", type=int, default=3)
    ap.add_argument(
        "--min-cell-count", type=int, default=1,
        help="drop (face, source_model) cells with emit < this "
             "(default: 1; bump to 3 for tighter analysis)",
    )
    ap.add_argument(
        "--per-source-parquet", type=Path,
        default=CLAUDE_FACES_LEXICON_BAG_PER_SOURCE_PATH,
    )
    ap.add_argument(
        "--pooled-parquet", type=Path,
        default=CLAUDE_FACES_LEXICON_BAG_PATH,
    )
    ap.add_argument(
        "--case-faces", nargs="+",
        default=["(╯°□°)", "(´;ω;`)", "(╥﹏╥)", "(>∀<☆)", "(´-`)"],
        help="canonical kaomoji to render per-source case files for "
             "(default: top divergent from script 68)",
    )
    ap.add_argument("--top-faces-heatmap", type=int, default=30)
    args = ap.parse_args()

    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _use_cjk_font()

    print(f"loading Claude-GT (floor={args.gt_floor})...")
    gt, gt_counts = _gt_dist_per_face(args.gt_floor)
    print(f"  {len(gt)} faces in GT")

    print(f"loading per-source BoL from {args.per_source_parquet.name}...")
    per_src, cell_counts, sources = _per_source_dists(args.per_source_parquet)
    print(
        f"  {len(per_src)} (face, source_model) cells across {len(sources)} sources"
    )

    print(f"loading pooled BoL from {args.pooled_parquet.name}...")
    pooled = _pooled_bol_dists(args.pooled_parquet)
    print(f"  {len(pooled)} faces with pooled BoL")

    df = _per_cell_table(
        per_src, cell_counts, pooled, gt, gt_counts,
        min_cell_count=args.min_cell_count,
    )
    out_tsv = HARNESS_DIR / "per_source_drift.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    per_source = _per_source_summary(df, sources)
    cross_source = _cross_source_table(per_src, sources)

    print("\n=== per-source-model vs Claude-GT ===")
    print(per_source.sort_values("n_emits", ascending=False).to_string(index=False))
    print("\n=== cross-source-model BoL similarity (off-diagonal) ===")
    cs_off = cross_source[cross_source["sm_a"] != cross_source["sm_b"]] \
        .sort_values("n_shared", ascending=False)
    print(cs_off.to_string(index=False))

    # n_faces_with_gt = unique faces in df with non-null jsd_vs_gt
    n_faces_with_gt = df.dropna(subset=["jsd_vs_gt"])["first_word"].nunique()
    n_total_cells = len(df)
    n_multi_source = df["first_word"].value_counts()
    n_multi_source_faces = int((n_multi_source >= 2).sum())

    # Pooled-BoL solo sim vs GT for the same face set, for headline reference.
    pool_sims = []
    for face in set(df["first_word"]) & set(gt) & set(pooled):
        pool_sims.append(similarity(js(pooled[face], gt[face])))
    pooled_sim_vs_gt = float(np.mean(pool_sims)) if pool_sims else float("nan")

    # Case files for the user-supplied diagnostic faces.
    case_lines: list[list[str]] = []
    for face in args.case_faces:
        case_lines.append(
            _case_file(face, per_src, cell_counts, pooled, gt, gt_counts, sources)
        )

    fig_path = HARNESS_FIG_DIR / "per_source_modal_heatmap.png"
    _modal_heatmap_figure(
        per_src, cell_counts, sources, fig_path,
        top_faces_n=args.top_faces_heatmap,
    )
    print(f"wrote {fig_path}")

    md_path = HARNESS_DIR / "per_source_drift_summary.md"
    _summary_md(
        per_source, cross_source, df, case_lines, md_path,
        n_total_cells=n_total_cells,
        n_faces_with_gt=n_faces_with_gt,
        n_multi_source_faces=n_multi_source_faces,
        pooled_sim_vs_gt=pooled_sim_vs_gt,
        gt_floor=args.gt_floor,
    )


if __name__ == "__main__":
    main()
