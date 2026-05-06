"""Per-project + global Russell-quadrant histograms for the user's
actual Claude emissions. Three resolution modes:

  - ``--mode gt-priority`` (default): use Claude-GT (the Opus 4.7
    groundtruth runs, ``data/harness/claude-runs/run-*.jsonl``) for any
    face Claude itself emitted in any run, falling back to the
    bag-of-lexicon (BoL) inference for in-the-wild faces. Combines
    empirical Claude behavior on heavy-use faces with the
    synthesizer's structured commit on the long tail. Best for
    honest per-project reads.
  - ``--mode bol``: use the BoL-derived quadrant for every face,
    regardless of whether Claude emitted it in the pilot. The
    "synthesizer's eye" view — what Haiku's structured per-face
    synthesis says, with no Claude-emission validation.
  - ``--mode gt-only``: use Claude-GT for faces it covers, mark every
    other face unknown. No speculative grading. Strictest read,
    smallest sample (≈49% emission coverage on a9's journal).

Pre-2026-05-06 the BoL fallback was a "best-ensemble" prediction
read from ``face_likelihood_ensemble_predict.tsv`` (script 54). That
ensemble was a panel of local-LM-head encoders + Anthropic judges
voting on faces the encoders had never been calibrated against; the
BoL replacement is more principled (the synthesizer literally
committed to a structured pick over the locked v2 LEXICON, and 19
of those 48 lexicon words are explicit Russell-quadrant anchors)
and zero-cost (no model call). See
``llmoji_study.lexicon.bol_modal_quadrant``.

Source corpora (both):
  - ``~/.claude/kaomoji-journal.jsonl`` (Claude Code, has cwd → project)
  - claude.ai conversations.json export(s); accepts comma-separated
    list, multiple exports unioned by conversation UUID with the
    richer copy winning. Bucketed under ``"claude.ai"``.

Outputs (under ``data/harness/`` and ``figures/harness/``, suffixed
by mode so all three coexist on disk):
  - claude_per_project_<mode>.tsv  — per (project, quadrant) row
  - claude_per_project_<mode>.md   — readable per-project table
  - claude_per_project_unknown_<mode>.tsv — faces that didn't resolve
  - claude_per_project_<mode>.png  — stacked-bar chart

Usage:
  python scripts/66_per_project_quadrants.py
  python scripts/66_per_project_quadrants.py --mode bol
  python scripts/66_per_project_quadrants.py --mode gt-only
  python scripts/66_per_project_quadrants.py \\
      --mode gt-priority --claude-gt-floor 2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from llmoji_study.claude_faces import load_bol_parquet
from llmoji_study.claude_gt import load_claude_gt
from llmoji_study.config import (
    CLAUDE_FACES_LEXICON_BAG_PATH,
    DATA_DIR,
    FIGURES_DIR,
)
from llmoji_study.lexicon import bol_modal_quadrant
from llmoji_study.local_emissions import (
    DEFAULT_CLAUDE_EXPORTS,
    DEFAULT_CLAUDE_JOURNAL,
    iter_local_emissions,
)
from llmoji_study.per_project_charts import plot_per_project_quadrants

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
HARNESS_DIR = DATA_DIR / "harness"
FIGURES_HARNESS_DIR = FIGURES_DIR / "harness"

MODES = ("gt-priority", "bol", "gt-only")
MODE_TITLES = {
    "gt-priority": "GT-priority + BoL fallback",
    "bol": "BoL inference (synthesizer's structured commit)",
    "gt-only": "Claude-GT only (no speculative grading)",
}


def _load_bol_predictions(path: Path) -> dict[str, dict]:
    """Read the BoL parquet → per-face modal Russell quadrant.

    Each entry carries ``bol_pred`` (modal quadrant or "" when the
    face has no circumplex commitment, i.e. extension-only picks)
    and ``n_v2_descs`` for sparsity-aware downstream filtering.
    """
    if not path.exists():
        sys.exit(
            f"missing {path} — run scripts/harness/62_corpus_lexicon.py first"
        )
    fw, _n, n_v2, B = load_bol_parquet(path)
    out: dict[str, dict] = {}
    for i, face in enumerate(fw):
        modal = bol_modal_quadrant(B[i])
        out[face] = {
            "bol_pred": modal or "",
            "n_v2_descs": int(n_v2[i]),
        }
    return out


# Reader plumbing (DEFAULT_CLAUDE_JOURNAL, DEFAULT_CLAUDE_EXPORTS,
# format dispatch, journal + claude.ai readers) lives in
# `llmoji_study.local_emissions` so script 67 (wild residual marker
# semantics) can share the same code path. Per-emission rows from
# `iter_local_emissions` are `(face, source, project)` tuples; this
# script discards `source` and groups by `project`.


def _resolve(
    face: str,
    gt: dict[str, tuple[str, int]],
    preds: dict[str, dict],
    mode: str,
) -> tuple[str | None, str]:
    """Return ``(quadrant_or_None, source_label)``.

    source ∈ {"gt", "bol", "unknown"}. Mode determines which sources
    are consulted and in what priority. A BoL prediction can be the
    empty string when the face has no circumplex commitment
    (extension-only picks); we treat that as unresolved.
    """
    if mode == "bol":
        if face in preds and preds[face]["bol_pred"]:
            return preds[face]["bol_pred"], "bol"
        return None, "unknown"
    if mode == "gt-only":
        if face in gt:
            return gt[face][0], "gt"
        return None, "unknown"
    # gt-priority
    if face in gt:
        return gt[face][0], "gt"
    if face in preds and preds[face]["bol_pred"]:
        return preds[face]["bol_pred"], "bol"
    return None, "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="gt-priority",
                    help="resolution mode (default: gt-priority)")
    ap.add_argument("--claude-journal",
                    default=str(Path.home() / ".claude" / "kaomoji-journal.jsonl"),
                    help="Claude Code kaomoji-journal.jsonl")
    ap.add_argument("--claude-export",
                    default=",".join(str(p) for p in DEFAULT_CLAUDE_EXPORTS),
                    help="comma-separated list of claude.ai exports (each "
                         "either a conversations.json or its parent dir); "
                         "multiple exports are unioned by conversation UUID "
                         "with the richer copy winning. Empty string skips.")
    ap.add_argument("--bol-parquet",
                    default=str(CLAUDE_FACES_LEXICON_BAG_PATH),
                    help="BoL parquet (per-face 48-d lexicon vectors) "
                         "from scripts/harness/62_corpus_lexicon.py")
    ap.add_argument("--claude-gt-floor", type=int, default=1,
                    help="Claude-GT modal_n minimum (default 1; raise to 2 "
                         "for the sharper 22-face subset)")
    ap.add_argument("--min-per-project", type=int, default=5,
                    help="skip projects with fewer than N total emissions")
    args = ap.parse_args()
    mode = args.mode

    HARNESS_DIR.mkdir(parents=True, exist_ok=True)

    # GT and ensemble both load unconditionally so the markdown can
    # report counts for the inactive sources too.
    print(f"loading Claude-GT (floor={args.claude_gt_floor}) ...")
    gt = load_claude_gt(floor=args.claude_gt_floor)
    print(f"  {len(gt)} faces in Claude-GT")
    if mode != "bol":
        gt_quadrant_dist = Counter(q for q, _ in gt.values())
        for q in QUADRANTS:
            print(f"    {q:5s} {gt_quadrant_dist.get(q, 0):3d} faces")

    if mode == "gt-only":
        preds: dict[str, dict] = {}
        print("\nskipping BoL parquet (mode=gt-only)")
    else:
        print(f"\nloading BoL predictions from {args.bol_parquet} ...")
        preds = _load_bol_predictions(Path(args.bol_parquet))
        n_with_circumplex = sum(1 for v in preds.values() if v["bol_pred"])
        print(
            f"  {len(preds)} faces in BoL parquet "
            f"({n_with_circumplex} with circumplex commitment)"
        )

    emissions: list[tuple[str, str, str]] = []  # (project, face, src_corpus)
    export_paths = [Path(p.strip()) for p in args.claude_export.split(",")
                     if p.strip()]
    print("\nloading local emissions (Claude Code journal + claude.ai exports) ...")
    for face, src, proj in iter_local_emissions(
        Path(args.claude_journal), export_paths,
    ):
        emissions.append((proj or "(no_project)", face, src))

    n_total = max(len(emissions), 1)
    print(f"\ntotal emissions: {len(emissions)}")
    unique = {f for _, f, _ in emissions}
    print(f"unique kaomoji: {len(unique)}")

    # "Useful BoL" = face is in BoL parquet AND committed to a
    # circumplex quadrant. Empty-string bol_pred (extension-only
    # picks) doesn't count as resolvable.
    bol_useful = {f for f, v in preds.items() if v["bol_pred"]}
    in_gt = unique & set(gt)
    in_pred_only = (unique - set(gt)) & bol_useful
    in_both_pred_gt = unique & set(gt) & bol_useful
    in_unknown_for_mode: set[str] = set()
    for f in unique:
        q, _ = _resolve(f, gt, preds, mode)
        if q is None:
            in_unknown_for_mode.add(f)

    # Per-project counters.
    per_proj: dict[str, dict[str, int]] = defaultdict(
        lambda: {q: 0 for q in QUADRANTS}
    )
    per_proj_src: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "bol": 0, "unknown": 0}
    )
    per_proj_total: dict[str, int] = defaultdict(int)
    global_counts = {q: 0 for q in QUADRANTS}
    global_src = {"gt": 0, "bol": 0, "unknown": 0}

    for proj, face, _src in emissions:
        per_proj_total[proj] += 1
        q, src = _resolve(face, gt, preds, mode)
        per_proj_src[proj][src] += 1
        global_src[src] += 1
        if q is None:
            continue
        per_proj[proj][q] += 1
        global_counts[q] += 1

    n_gt = global_src["gt"]
    n_pred = global_src["bol"]
    n_unk = global_src["unknown"]
    print(f"resolved by Claude-GT: {n_gt}/{len(emissions)} "
          f"({n_gt/n_total*100:.1f}%)")
    print(f"resolved by BoL:       {n_pred}/{len(emissions)} "
          f"({n_pred/n_total*100:.1f}%)")
    print(f"unknown:               {n_unk}/{len(emissions)} "
          f"({n_unk/n_total*100:.1f}%)")

    suffix = mode.replace("-", "_")

    # Per-project TSV.
    rows = []
    for proj in sorted(per_proj_total):
        n = per_proj_total[proj]
        n_known = sum(per_proj[proj].values())
        for q in QUADRANTS:
            count = per_proj[proj][q]
            share = count / n_known if n_known > 0 else 0.0
            rows.append({
                "project": proj,
                "quadrant": q,
                "count": count,
                "share_of_known": round(share, 4),
                "n_total": n,
                "n_known": n_known,
                "n_gt": per_proj_src[proj]["gt"],
                "n_bol": per_proj_src[proj]["bol"],
                "n_unknown": per_proj_src[proj]["unknown"],
            })
    df = pd.DataFrame(rows)
    out_tsv = HARNESS_DIR / f"claude_per_project_{suffix}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    # Unknown-faces TSV (faces that didn't resolve under this mode).
    unk_counter: Counter[str] = Counter()
    unk_per_proj: dict[str, set[str]] = defaultdict(set)
    for proj, face, _src in emissions:
        if face in in_unknown_for_mode:
            unk_counter[face] += 1
            unk_per_proj[face].add(proj)
    unk_rows = [
        {
            "first_word": face,
            "count": n,
            "n_projects": len(unk_per_proj[face]),
            "sample_projects": ",".join(sorted(unk_per_proj[face])[:3]),
        }
        for face, n in sorted(unk_counter.items(), key=lambda kv: -kv[1])
    ]
    unk_tsv = HARNESS_DIR / f"claude_per_project_unknown_{suffix}.tsv"
    pd.DataFrame(unk_rows).to_csv(unk_tsv, sep="\t", index=False)
    print(f"wrote {unk_tsv}  ({len(unk_rows)} unique unknown faces)")

    # Markdown report.
    lines: list[str] = []
    lines.append(f"# Claude per-project quadrants — {MODE_TITLES[mode]}")
    lines.append("")
    lines.append(f"**Mode:** `{mode}`")
    lines.append(f"**Total emissions:** {len(emissions)}  "
                 f"(unique kaomoji: {len(unique)})")
    lines.append("")
    lines.append("**Resolution sources:**")
    lines.append("")
    lines.append("| source | unique faces | emissions | share of total |")
    lines.append("|---|---:|---:|---:|")
    if mode in ("gt-priority", "gt-only"):
        lines.append(f"| Claude-GT (modal_n ≥ {args.claude_gt_floor}) "
                     f"| {len(in_gt)} | {n_gt} | {n_gt/n_total:.1%} |")
    if mode in ("gt-priority", "bol"):
        if mode == "gt-priority":
            label = "BoL fallback"
            n_unique_pred = len(in_pred_only)
        else:
            label = "BoL"
            n_unique_pred = len(unique & bol_useful)
        lines.append(f"| {label} | {n_unique_pred} | {n_pred} "
                     f"| {n_pred/n_total:.1%} |")
    lines.append(f"| unknown | {len(in_unknown_for_mode)} | {n_unk} "
                 f"| {n_unk/n_total:.1%} |")
    lines.append("")

    # Cross-mode context paragraph (mode-specific).
    if mode == "gt-priority":
        lines.append(f"GT covers {len(in_both_pred_gt) + (len(in_gt) - len(in_both_pred_gt))} "
                     f"of {len(unique)} unique faces "
                     f"({len(in_gt)/max(len(unique),1):.1%}); BoL "
                     f"fallback adds {len(in_pred_only)} more.")
        lines.append("")
    elif mode == "gt-only":
        lines.append("Strict mode: only faces Claude itself emitted in the "
                     "Opus 4.7 pilot are scored. Anything in-the-wild gets "
                     "marked unknown rather than inferred from BoL. "
                     f"{n_pred + n_unk} of {len(emissions)} emissions "
                     f"({(n_pred+n_unk)/n_total:.1%}) are unscored.")
        lines.append("")

    n_known_global = sum(global_counts.values())
    if n_known_global > 0:
        lines.append("## Global distribution (all known emissions)")
        lines.append("")
        lines.append("| quadrant | count | share |")
        lines.append("|---|---:|---:|")
        for q in QUADRANTS:
            n = global_counts[q]
            lines.append(f"| {q} | {n} | {n/n_known_global:.1%} |")
        lines.append(f"| (unknown) | {n_unk} | "
                     f"{n_unk/(n_known_global+n_unk):.1%} of total |")
        lines.append("")

    lines.append(f"## Per project (≥{args.min_per_project} total emissions)")
    lines.append("")
    lines.append("Cells = % of known emissions in each quadrant. "
                 "Bold = modal quadrant. `gt` / `bol` / `?` columns count "
                 "emissions resolved by Claude-GT, BoL, and unknown "
                 "respectively (irrelevant columns stay 0 under the active "
                 "mode).")
    lines.append("")
    header = ["project", "n", "gt", "bol", "?"] + QUADRANTS + ["modal"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * (len(header) - 1)) + "|---|")

    for proj in sorted(per_proj_total, key=lambda p: -per_proj_total[p]):
        n = per_proj_total[proj]
        if n < args.min_per_project:
            continue
        n_known = sum(per_proj[proj].values())
        if n_known == 0:
            continue
        modal_q = max(QUADRANTS, key=lambda q: per_proj[proj][q])
        modal_share = per_proj[proj][modal_q] / n_known
        cells = [proj, str(n),
                 str(per_proj_src[proj]["gt"]),
                 str(per_proj_src[proj]["bol"]),
                 str(per_proj_src[proj]["unknown"])]
        for q in QUADRANTS:
            share = per_proj[proj][q] / n_known
            cell = f"{share:.0%}"
            if q == modal_q and per_proj[proj][modal_q] > 0:
                cell = f"**{cell}**"
            cells.append(cell)
        cells.append(f"{modal_q} ({modal_share:.0%})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Top contributors per quadrant (with source tag where meaningful).
    lines.append("## Top emitted kaomoji per quadrant")
    lines.append("")
    face_counts: Counter[str] = Counter()
    face_q: dict[str, str] = {}
    face_src_label: dict[str, str] = {}
    for _proj, face, _src in emissions:
        q, src = _resolve(face, gt, preds, mode)
        if q is None:
            continue
        face_counts[face] += 1
        face_q[face] = q
        face_src_label[face] = src
    by_quad: dict[str, list[tuple[str, int]]] = {q: [] for q in QUADRANTS}
    for face, n in face_counts.items():
        by_quad[face_q[face]].append((face, n))
    show_source_col = mode == "gt-priority"
    for q in QUADRANTS:
        items = sorted(by_quad[q], key=lambda kv: -kv[1])[:10]
        lines.append(f"### {q}")
        if not items:
            lines.append("(none)")
            lines.append("")
            continue
        if show_source_col:
            lines.append("| kaomoji | count | source |")
            lines.append("|---|---:|---|")
            for face, n in items:
                lines.append(f"| `{face}` | {n} | {face_src_label[face]} |")
        else:
            lines.append("| kaomoji | count |")
            lines.append("|---|---:|")
            for face, n in items:
                lines.append(f"| `{face}` | {n} |")
        lines.append("")

    out_md = HARNESS_DIR / f"claude_per_project_{suffix}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    # Stacked-bar figure.
    if mode == "gt-priority":
        subtitle = (f"{len(emissions)} emissions · "
                     f"Claude-GT {n_gt/n_total*100:.1f}% · "
                     f"BoL {n_pred/n_total*100:.1f}% · "
                     f"unknown {n_unk/n_total*100:.1f}%")
    elif mode == "bol":
        subtitle = (f"{len(emissions)} emissions · "
                     f"BoL covers {(n_pred)/n_total*100:.1f}% · "
                     f"unknown {n_unk/n_total*100:.1f}%")
    else:  # gt-only
        subtitle = (f"{len(emissions)} emissions · "
                     f"Claude-GT covers {n_gt/n_total*100:.1f}% · "
                     f"unknown {n_unk/n_total*100:.1f}% (no speculation)")

    fig_path = plot_per_project_quadrants(
        per_proj=per_proj,
        per_proj_total=per_proj_total,
        global_counts=global_counts,
        title=f"Claude per-project quadrants — {MODE_TITLES[mode]}",
        subtitle=subtitle,
        out_path=FIGURES_HARNESS_DIR / f"claude_per_project_{suffix}.png",
        min_per_project=args.min_per_project,
    )
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
