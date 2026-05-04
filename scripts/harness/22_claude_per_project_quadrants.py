"""Per-project + global quadrant histograms for the user's actual Claude
emissions, using the face_likelihood ensemble's per-face predictions.

Sources:
  - ~/.claude/kaomoji-journal.jsonl (Claude Code, has project_slug)
  - claude.ai conversations.json export (no per-project; bucketed under
    "claude.ai" or per-conversation if --by-conversation)

For each kaomoji emission, look up the canonicalized first_word in the
ensemble's per-face prediction TSV (from script 56). Faces not in the
union are aggregated into an "unknown" bucket and listed separately —
they'd need ensemble re-scoring to get a quadrant prediction (deferred).

Outputs:
  data/harness/claude_per_project_quadrants.tsv  — per (project, quadrant)
  data/harness/claude_per_project_quadrants.md   — per-project bar table
  data/harness/claude_unknown_kaomoji.tsv        — kaomoji not in face union

Usage:
  python scripts/harness/22_claude_per_project_quadrants.py
  python scripts/harness/22_claude_per_project_quadrants.py \\
      --claude-export ~/Downloads/data-.../conversations.json
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
from llmoji_study.config import DATA_DIR

QUADRANTS = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
HARNESS_DIR = DATA_DIR / "harness"

DEFAULT_CLAUDE_EXPORT = (
    Path("/Users/a9lim/Downloads/"
         "data-72de1230-b9fa-4c55-bc10-84a35b58d89c-1777763577-c21ac4ff-batch-0000/"
         "conversations.json")
)


def _load_ensemble_predictions() -> dict[str, dict]:
    p = DATA_DIR / "face_likelihood_ensemble_predict.tsv"
    if not p.exists():
        sys.exit(f"missing {p} — run 56_ensemble_predict.py first")
    df = pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])
    out = {}
    for _, r in df.iterrows():
        f = str(r["first_word"])
        out[f] = {
            "ensemble_pred": str(r["ensemble_pred"]),
            "ensemble_conf": float(r.get("ensemble_conf", 0.0)),
            "majority_pred": str(r.get("majority_pred", "") or ""),
            **{f"p_{q}": float(r.get(f"ensemble_p_{q}", 0.0))
               for q in QUADRANTS},
        }
    return out


def _project_from_cwd(cwd: str | None) -> str:
    """Pre-v1.1 ScrapeRow had project_slug; v1.1+ only carries cwd. Derive
    a project name from the basename of cwd."""
    if not cwd:
        return "(no_project)"
    return Path(cwd).name or "(no_project)"


def _emissions_from_journal(path: Path, source: str) -> list[tuple[str, str]]:
    """Return (project_name, canonical_kaomoji) per emission."""
    rows = []
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


def _emissions_from_claude_export(path: Path) -> list[tuple[str, str]]:
    """Linear claude.ai conversations.json. Returns (conversation_name, face)."""
    rows = []
    if not path.exists():
        print(f"  skip {path} (missing)")
        return rows
    try:
        from llmoji.sources.claude_export import iter_claude_export
    except ImportError:
        sys.exit("could not import llmoji.sources.claude_export")
    for sr in iter_claude_export([path]):
        face = sr.first_word or ""
        if not face:
            continue
        canon = canonicalize_kaomoji(face)
        if not canon:
            continue
        # claude.ai exports don't carry cwd / project; bucket as "claude.ai".
        rows.append(("claude.ai", canon))
    print(f"  {path.name}: {len(rows)} emissions")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-journal",
                    default=str(Path.home() / ".claude" / "kaomoji-journal.jsonl"),
                    help="path to Claude Code kaomoji-journal.jsonl")
    ap.add_argument("--claude-export", default=str(DEFAULT_CLAUDE_EXPORT),
                    help="path to claude.ai conversations.json")
    ap.add_argument("--bucket-claude-ai-as-one", action="store_true", default=True,
                    help="put all claude.ai conversations under one bucket "
                         "rather than per-conversation (default: yes)")
    ap.add_argument("--min-per-project", type=int, default=5,
                    help="skip projects with fewer than N emissions")
    args = ap.parse_args()

    HARNESS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading ensemble predictions ...")
    preds = _load_ensemble_predictions()
    print(f"  {len(preds)} faces in face_likelihood_ensemble_predict.tsv")

    emissions: list[tuple[str, str, str]] = []  # (project, face, source)
    print("\nloading Claude Code journal ...")
    for proj, face in _emissions_from_journal(Path(args.claude_journal),
                                              "claude_code"):
        emissions.append((proj, face, "claude_code"))
    print("\nloading claude.ai export ...")
    for proj, face in _emissions_from_claude_export(Path(args.claude_export)):
        if args.bucket_claude_ai_as_one:
            proj = "claude.ai"
        emissions.append((proj, face, "claude_ai"))

    print(f"\ntotal emissions: {len(emissions)}")
    unique_faces = {e[1] for e in emissions}
    print(f"unique kaomoji: {len(unique_faces)}")
    in_ensemble = unique_faces & set(preds)
    not_in_ensemble = unique_faces - set(preds)
    print(f"  in ensemble face union: {len(in_ensemble)}")
    print(f"  NOT in ensemble (unknown): {len(not_in_ensemble)}")
    coverage_emissions = sum(1 for _, f, _ in emissions if f in preds)
    print(f"emissions covered by ensemble: {coverage_emissions}/{len(emissions)} "
          f"({coverage_emissions/max(len(emissions),1)*100:.1f}%)")

    # Per-project quadrant counts (only emissions covered by ensemble).
    per_proj: dict[str, dict[str, int]] = defaultdict(lambda: {q: 0 for q in QUADRANTS})
    per_proj_total: dict[str, int] = defaultdict(int)
    per_proj_unknown: dict[str, int] = defaultdict(int)
    global_counts = {q: 0 for q in QUADRANTS}
    global_unknown = 0

    for proj, face, _src in emissions:
        per_proj_total[proj] += 1
        if face not in preds:
            per_proj_unknown[proj] += 1
            global_unknown += 1
            continue
        q = preds[face]["ensemble_pred"]
        per_proj[proj][q] += 1
        global_counts[q] += 1

    # Per-project TSV.
    rows = []
    for proj in sorted(per_proj_total):
        n = per_proj_total[proj]
        n_known = sum(per_proj[proj].values())
        n_unk = per_proj_unknown[proj]
        for q in QUADRANTS:
            count = per_proj[proj][q]
            share = count / n_known if n_known > 0 else 0.0
            rows.append({
                "project": proj,
                "quadrant": q,
                "count": count,
                "share_of_known": share,
                "n_total": n,
                "n_known": n_known,
                "n_unknown": n_unk,
            })
    df = pd.DataFrame(rows)
    out_tsv = HARNESS_DIR / "claude_per_project_quadrants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    # Unknown faces TSV.
    unk_rows = []
    unk_counter: Counter[str] = Counter()
    unk_per_proj: dict[str, set[str]] = defaultdict(set)
    for proj, face, _src in emissions:
        if face not in preds:
            unk_counter[face] += 1
            unk_per_proj[face].add(proj)
    for face, n in sorted(unk_counter.items(), key=lambda kv: -kv[1]):
        unk_rows.append({
            "first_word": face,
            "count": n,
            "n_projects": len(unk_per_proj[face]),
            "sample_projects": ",".join(sorted(unk_per_proj[face])[:3]),
        })
    pd.DataFrame(unk_rows).to_csv(
        HARNESS_DIR / "claude_unknown_kaomoji.tsv", sep="\t", index=False
    )
    print(f"wrote {HARNESS_DIR / 'claude_unknown_kaomoji.tsv'}  "
          f"({len(unk_rows)} unique unknown faces)")

    # Markdown report.
    lines = []
    lines.append("# Claude per-project quadrant histograms")
    lines.append("")
    lines.append(f"**Source emissions:** {len(emissions)} "
                 f"(unique kaomoji: {len(unique_faces)})")
    lines.append(f"**Ensemble coverage:** {coverage_emissions}/"
                 f"{len(emissions)} "
                 f"({coverage_emissions/max(len(emissions),1)*100:.1f}%) "
                 f"of emissions, "
                 f"{len(in_ensemble)}/{len(unique_faces)} unique faces")
    lines.append(f"**Unknown faces** (not in ensemble's 306-face union): "
                 f"{len(not_in_ensemble)} unique, {global_unknown} emissions"
                 f" — see `claude_unknown_kaomoji.tsv`")
    lines.append("")

    # Global distribution
    n_total_global = sum(global_counts.values())
    if n_total_global > 0:
        lines.append("## Global distribution (all known emissions)")
        lines.append("")
        lines.append("| quadrant | count | share |")
        lines.append("|---|---:|---:|")
        for q in QUADRANTS:
            n = global_counts[q]
            lines.append(f"| {q} | {n} | {n/n_total_global:.1%} |")
        lines.append(f"| (unknown) | {global_unknown} | "
                     f"{global_unknown/(n_total_global+global_unknown):.1%} of total |")
        lines.append("")

    # Per-project (skipping low-N projects).
    lines.append(f"## Per project (≥{args.min_per_project} known emissions)")
    lines.append("")
    lines.append("Each row is one project. Cells = % of known emissions "
                 "in each quadrant. Bold = modal quadrant.")
    lines.append("")
    header = ["project", "n", "unknown"] + QUADRANTS + ["modal"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * (len(header)-1)) + "|---|")

    proj_modes = []
    for proj in sorted(per_proj_total, key=lambda p: -per_proj_total[p]):
        n_known = sum(per_proj[proj].values())
        if n_known < args.min_per_project:
            continue
        cells = [proj, str(per_proj_total[proj]),
                 str(per_proj_unknown[proj])]
        modal_q = max(QUADRANTS, key=lambda q: per_proj[proj][q])
        modal_share = per_proj[proj][modal_q] / n_known
        for q in QUADRANTS:
            share = per_proj[proj][q] / n_known
            cell = f"{share:.0%}"
            if q == modal_q:
                cell = f"**{cell}**"
            cells.append(cell)
        cells.append(f"{modal_q} ({modal_share:.0%})")
        lines.append("| " + " | ".join(cells) + " |")
        proj_modes.append((proj, modal_q, modal_share))
    lines.append("")

    # Top contributors per quadrant
    lines.append("## Top emitted kaomoji per quadrant")
    lines.append("")
    face_counts: Counter[str] = Counter(f for _, f, _ in emissions if f in preds)
    by_quad: dict[str, list[tuple[str, int]]] = {q: [] for q in QUADRANTS}
    for face, n in face_counts.items():
        q = preds[face]["ensemble_pred"]
        by_quad[q].append((face, n))
    for q in QUADRANTS:
        items = sorted(by_quad[q], key=lambda kv: -kv[1])[:10]
        lines.append(f"### {q}")
        if not items:
            lines.append("(none)")
            continue
        lines.append("| kaomoji | count | conf |")
        lines.append("|---|---:|---:|")
        for face, n in items:
            conf = preds[face]["ensemble_conf"]
            lines.append(f"| `{face}` | {n} | {conf:.2f} |")
        lines.append("")

    out_md = HARNESS_DIR / "claude_per_project_quadrants.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
