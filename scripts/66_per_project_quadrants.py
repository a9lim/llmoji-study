"""Per-project + global Russell-quadrant histograms for the user's
actual Claude emissions. Three resolution modes:

  - ``--mode gt-priority`` (default): use Claude-GT (the Opus 4.7
    groundtruth runs, ``data/harness/claude-runs/run-*.jsonl``) for any
    face Claude itself emitted in any run, falling back to the
    best-ensemble face_likelihood prediction for in-the-wild faces.
    Combines empirical Claude behavior on heavy-use faces with
    ensemble inference on the long tail. Best for honest per-project
    reads.
  - ``--mode ensemble``: use the best-ensemble face_likelihood
    prediction for every face, regardless of whether Claude emitted
    it in the pilot. The deployable-extension scenario — what a
    panel of open-weight models would infer about Claude's state from
    the kaomoji alone, with no Claude-side validation.
  - ``--mode gt-only``: use Claude-GT for faces it covers, mark every
    other face unknown. No speculative grading. Strictest read,
    smallest sample (≈49% emission coverage on a9's journal).

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
  python scripts/66_per_project_quadrants.py --mode ensemble
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

from llmoji.sources.journal import iter_journal
from llmoji.taxonomy import canonicalize_kaomoji
from llmoji_study.claude_gt import load_claude_gt
from llmoji_study.config import DATA_DIR, FIGURES_DIR
from llmoji_study.per_project_charts import plot_per_project_quadrants

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
HARNESS_DIR = DATA_DIR / "harness"
FIGURES_HARNESS_DIR = FIGURES_DIR / "harness"

MODES = ("gt-priority", "ensemble", "gt-only")
MODE_TITLES = {
    "gt-priority": "GT-priority + ensemble fallback",
    "ensemble": "ensemble predictions",
    "gt-only": "Claude-GT only (no speculative grading)",
}

DEFAULT_CLAUDE_EXPORTS = [
    Path("/Users/a9lim/Downloads/"
         "data-72de1230-b9fa-4c55-bc10-84a35b58d89c-1777763577-c21ac4ff-batch-0000/"
         "conversations.json"),
    Path("/Users/a9lim/Downloads/"
         "9cc23402cbb1e8aec9785eb0f750f1c442f1ba13e507bd6b04a727c627d64d08-"
         "2026-04-28-01-04-53-1d1e60e8c10441b1881c7e81834c3b26/"
         "conversations.json"),
]


def _to_float(v) -> float:
    """pandas-na-tolerant float cast; 0.0 on None / NaN / blank."""
    try:
        if v is None or (isinstance(v, str) and v == ""):
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _load_ensemble_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"missing {path} — run scripts/54_ensemble_predict.py first")
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


def _project_from_cwd(cwd: str | None) -> str:
    if not cwd:
        return "(no_project)"
    return Path(cwd).name or "(no_project)"


def _emissions_from_journal(path: Path, source: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
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
        rows.append((_project_from_cwd(sr.cwd), canon))
    print(f"  {path.name}: {len(rows)} emissions")
    return rows


def _emissions_from_claude_export(paths: list[Path]) -> list[tuple[str, str]]:
    """``iter_claude_export`` takes directories (it does
    ``dir / "conversations.json"`` internally), so each path may be
    either a ``conversations.json`` file or a directory. Multiple
    exports get unioned by conversation UUID with the richer copy
    winning per ``dedup_by_id_keep_richest``."""
    rows: list[tuple[str, str]] = []
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
        rows.append(("claude.ai", canon))
    print(f"  claude.ai: {len(rows)} emissions across {len(export_dirs)} export(s)")
    return rows


def _resolve(
    face: str,
    gt: dict[str, tuple[str, int]],
    preds: dict[str, dict],
    mode: str,
) -> tuple[str | None, str]:
    """Return ``(quadrant_or_None, source_label)``.

    source ∈ {"gt", "pred", "unknown"}. Mode determines which sources
    are consulted and in what priority.
    """
    if mode == "ensemble":
        if face in preds:
            return preds[face]["ensemble_pred"], "pred"
        return None, "unknown"
    if mode == "gt-only":
        if face in gt:
            return gt[face][0], "gt"
        return None, "unknown"
    # gt-priority
    if face in gt:
        return gt[face][0], "gt"
    if face in preds:
        return preds[face]["ensemble_pred"], "pred"
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
    ap.add_argument("--ensemble-tsv",
                    default=str(DATA_DIR / "local" / "face_likelihood_ensemble_predict.tsv"),
                    help="best-ensemble per-face prediction TSV from script 56")
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
    if mode != "ensemble":
        gt_quadrant_dist = Counter(q for q, _ in gt.values())
        for q in QUADRANTS:
            print(f"    {q:5s} {gt_quadrant_dist.get(q, 0):3d} faces")

    if mode == "gt-only":
        preds: dict[str, dict] = {}
        print("\nskipping ensemble TSV (mode=gt-only)")
    else:
        print(f"\nloading ensemble predictions from {args.ensemble_tsv} ...")
        preds = _load_ensemble_predictions(Path(args.ensemble_tsv))
        print(f"  {len(preds)} faces in ensemble TSV")

    emissions: list[tuple[str, str, str]] = []  # (project, face, src_corpus)
    print("\nloading Claude Code journal ...")
    for proj, face in _emissions_from_journal(Path(args.claude_journal),
                                              "claude_code"):
        emissions.append((proj, face, "claude_code"))

    export_paths = [Path(p.strip()) for p in args.claude_export.split(",")
                     if p.strip()]
    if export_paths:
        print(f"\nloading {len(export_paths)} claude.ai export(s) ...")
        for proj, face in _emissions_from_claude_export(export_paths):
            emissions.append((proj, face, "claude_ai"))

    n_total = max(len(emissions), 1)
    print(f"\ntotal emissions: {len(emissions)}")
    unique = {f for _, f, _ in emissions}
    print(f"unique kaomoji: {len(unique)}")

    in_gt = unique & set(gt)
    in_pred_only = (unique - set(gt)) & set(preds)
    in_both_pred_gt = unique & set(gt) & set(preds)
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
        lambda: {"gt": 0, "pred": 0, "unknown": 0}
    )
    per_proj_total: dict[str, int] = defaultdict(int)
    global_counts = {q: 0 for q in QUADRANTS}
    global_src = {"gt": 0, "pred": 0, "unknown": 0}

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
    n_pred = global_src["pred"]
    n_unk = global_src["unknown"]
    print(f"resolved by Claude-GT: {n_gt}/{len(emissions)} "
          f"({n_gt/n_total*100:.1f}%)")
    print(f"resolved by ensemble:  {n_pred}/{len(emissions)} "
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
                "n_pred": per_proj_src[proj]["pred"],
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
    if mode in ("gt-priority", "ensemble"):
        if mode == "gt-priority":
            label = "ensemble fallback"
            n_unique_pred = len(in_pred_only)
        else:
            label = "ensemble"
            n_unique_pred = len(unique & set(preds))
        lines.append(f"| {label} | {n_unique_pred} | {n_pred} "
                     f"| {n_pred/n_total:.1%} |")
    lines.append(f"| unknown | {len(in_unknown_for_mode)} | {n_unk} "
                 f"| {n_unk/n_total:.1%} |")
    lines.append("")

    # Cross-mode context paragraph (mode-specific).
    if mode == "gt-priority":
        lines.append(f"GT covers {len(in_both_pred_gt) + (len(in_gt) - len(in_both_pred_gt))} "
                     f"of {len(unique)} unique faces "
                     f"({len(in_gt)/max(len(unique),1):.1%}); ensemble "
                     f"fallback adds {len(in_pred_only)} more.")
        lines.append("")
    elif mode == "gt-only":
        lines.append("Strict mode: only faces Claude itself emitted in the "
                     "Opus 4.7 pilot are scored. Anything in-the-wild gets "
                     "marked unknown rather than guessed by the ensemble. "
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
                 "Bold = modal quadrant. `gt` / `pred` / `?` columns count "
                 "emissions resolved by Claude-GT, ensemble, and unknown "
                 "respectively (irrelevant columns stay 0 under the active "
                 "mode).")
    lines.append("")
    header = ["project", "n", "gt", "pred", "?"] + QUADRANTS + ["modal"]
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
                 str(per_proj_src[proj]["pred"]),
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
                     f"ensemble {n_pred/n_total*100:.1f}% · "
                     f"unknown {n_unk/n_total*100:.1f}%")
    elif mode == "ensemble":
        subtitle = (f"{len(emissions)} emissions · "
                     f"ensemble covers {(n_pred)/n_total*100:.1f}% · "
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
