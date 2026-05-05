# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false, reportCallIssue=false
"""Build the canonical face union across v3, Claude pilot, and in-the-wild
data.

Single source of truth for "every kaomoji emitted by anything we know
about, with per-quadrant counts where labels exist + wild-emission
counts where they don't." Used by ``50_face_likelihood.py`` (face-
likelihood ensemble), and any future analysis that needs a unified
face vocabulary across labeled and unlabeled emissions.

Replaces the implicit face-union that used to live inside the
encoder-specific ``face_h_first_<m>.parquet`` files (deleted with the
face-input pipeline 2026-05-04).

Inputs:
  - ``data/local/<short>/emotional_raw.jsonl`` for each v3 main model
    (whichever exist on disk; missing models are skipped). LABELED
    quadrant ground truth.
  - ``data/harness/claude-runs/run-*.jsonl`` (optional Claude inclusion;
    union over all sequential runs). LABELED quadrant ground truth.
  - ``data/harness/hf_dataset/contributors/<id>/<bundle>/<provider>.jsonl``
    files (in-the-wild contributor journals: Claude Code, Claude.ai
    export, Codex, GPT-5, etc.). UNLABELED — pooled into wild_*
    columns separately from the per-quadrant counts.

Output:
  - ``data/v3_face_union.parquet``
  - ``data/v3_face_union.tsv`` (human-inspectable mirror)

Columns:
  - ``first_word``       canonicalized first-kaomoji string
  - ``is_claude``        emitted by Claude in the groundtruth pilot?
  - ``is_wild``          emitted in any in-the-wild contributor data?
  - ``total_emit_count`` sum across v3 quadrants (LABELED only)
  - ``total_emit_HP`` / ``total_emit_LP`` / ``total_emit_HN-D`` /
    ``total_emit_HN-S`` / ``total_emit_LN`` / ``total_emit_NB``
                         per-quadrant LABELED emission counts (v3 +
                         Claude pilot)
  - ``wild_emit_count``  total in-the-wild emissions, all providers
  - ``wild_providers``   comma-sep set of provider stems that emitted
                         this face in the wild (e.g.
                         "claude-opus-4-7,gpt-5.4,codex_journal")

Usage:
  python scripts/40_face_union.py
  # Subset
  python scripts/40_face_union.py --models gemma,qwen
  # No Claude pilot
  python scripts/40_face_union.py --no-claude
  # No wild data
  python scripts/40_face_union.py --no-wild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from llmoji.sources.journal import iter_journal
from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.claude_gt import (
    CLAUDE_RUNS_DIR,
    CLAUDE_RUNS_INTROSPECTION_DIR as CLAUDE_INTROSPECTION_RUNS_DIR,
    find_run_files,
)
from llmoji_study.config import DATA_DIR, MODEL_REGISTRY


DEFAULT_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
WILD_DATA_DIR = DATA_DIR / "harness" / "hf_dataset" / "contributors"
# v3_face_union pools v3 local emit + Claude pilot + wild contributor faces
# — genuinely cross-platform, lives at data/ root (not under local/ or harness/).
OUT_PARQUET = DATA_DIR / "v3_face_union.parquet"
OUT_TSV = DATA_DIR / "v3_face_union.tsv"

# Introspection-arm sources. Optional inclusion via --include-introspection.
# Each contributes face-emission counts in its respective quadrants alongside
# the naturalistic v3 + Claude data, expanding the union vocabulary.
# CLAUDE_INTROSPECTION_RUNS_DIR imported from claude_gt above.
GEMMA_INTROSPECTION_PATH = DATA_DIR / "local" / "gemma_intro_v7_primed" / "emotional_raw.jsonl"


def _is_clean_kaomoji(fw: str) -> bool:
    """Reject faces containing non-BMP codepoints (i.e., 4-byte UTF-8
    chars at U+1F000+). These are modern emoji like 🎉/😊/🤯 that the
    extractor sometimes lets through despite the per-model emoji
    suppression — they're not affective-state readouts, they're token-
    level decoration. Filter at union build time so the canonical
    vocabulary stays clean.

    BMP-range decorations (★ ☆ ✦ ✧ ✿ ❤ ❄, all <0xFFFF) are kept —
    those are legitimate kaomoji components."""
    return all(ord(c) <= 0xFFFF for c in fw)


def _bucket_from_prompt_id(pid: str) -> str | None:
    """Derive the 6-way Russell quadrant from a v3 prompt_id.
    HN-D / HN-S split: hn01-20 = HN-D, hn21-40 = HN-S."""
    pid = (pid or "").lower()
    if pid.startswith("hp"):
        return "HP"
    if pid.startswith("lp"):
        return "LP"
    if pid.startswith("nb"):
        return "NB"
    if pid.startswith("ln"):
        return "LN"
    if pid.startswith("hn"):
        try:
            n = int(pid[2:])
        except ValueError:
            return None
        return "HN-D" if n <= 20 else "HN-S"
    return None


def _accumulate_local(
    jsonl_path: Path, by_face: dict, dropped: dict,
    *, per_model: dict | None = None, model_short: str | None = None,
) -> int:
    """Accumulate per-face per-quadrant counts from a v3 JSONL into
    `by_face` (mutated in place). `dropped` is also mutated to track
    per-face drop counts for emoji-contaminated rows. Returns rows-kept.

    When ``per_model`` + ``model_short`` are provided, also accumulates
    per-(face, model) emit counts into ``per_model[face][model_short]``.
    Used by the cross-emit-sanity script (51) to bucket faces into
    {gemma_only, shared_2, shared_3, claude_only}.
    """
    n_kept = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            fw_raw = r.get("first_word") or ""
            fw = canonicalize_kaomoji(fw_raw) or ""
            if not (isinstance(fw, str) and len(fw) > 0 and fw.startswith("(")):
                continue
            if not _is_clean_kaomoji(fw):
                dropped[fw] = dropped.get(fw, 0) + 1
                continue
            q = _bucket_from_prompt_id(r.get("prompt_id") or "")
            if q is None:
                continue
            entry = by_face.setdefault(fw, {qq: 0 for qq in QUADRANT_ORDER})
            entry[q] += 1
            n_kept += 1
            if per_model is not None and model_short is not None:
                pm = per_model.setdefault(fw, {})
                pm[model_short] = pm.get(model_short, 0) + 1
    return n_kept


def _accumulate_wild(
    by_face: dict,
    wild_emit: dict,
    wild_providers: dict,
    dropped: dict,
) -> int:
    """Walk `data/harness/hf_dataset/contributors/<id>/<bundle>/<provider>.jsonl`
    and accumulate canonical-kaomoji counts into wild-only fields. The
    provider stem (e.g. "claude-opus-4-7", "gpt-5.4", "codex_journal")
    becomes the source label. Mutates `by_face`, `wild_emit`,
    `wild_providers`."""
    if not WILD_DATA_DIR.exists():
        print(f"  [wild] no data dir at {WILD_DATA_DIR}; skipping")
        return 0
    n_total = 0
    n_files = 0
    for path in sorted(WILD_DATA_DIR.glob("*/*/*.jsonl")):
        provider = path.stem
        if provider in ("manifest", "<synthetic>"):
            # Synthetic prompts aren't real-world emissions; manifest
            # files are bundle metadata.
            continue
        try:
            n_emit = 0
            for sr in iter_journal(path, source=provider):
                fw_raw = getattr(sr, "first_word", None) or ""
                fw = canonicalize_kaomoji(fw_raw) or ""
                if not (isinstance(fw, str) and len(fw) > 0 and fw.startswith("(")):
                    continue
                if not _is_clean_kaomoji(fw):
                    dropped[fw] = dropped.get(fw, 0) + 1
                    continue
                # Ensure face exists in by_face so it appears in output
                # even if no v3 quadrant counts.
                by_face.setdefault(fw, {qq: 0 for qq in QUADRANT_ORDER})
                wild_emit[fw] = wild_emit.get(fw, 0) + 1
                wild_providers.setdefault(fw, set()).add(provider)
                n_emit += 1
            if n_emit > 0:
                n_files += 1
                n_total += n_emit
        except Exception as e:
            print(f"  [wild] failed on {path.name}: {e}; skipping")
            continue
    print(f"  [wild] {n_total} kaomoji-bearing emissions across {n_files} journals")
    return n_total


def _accumulate_claude(
    jsonl_path: Path, by_face: dict, claude_faces: set, dropped: dict,
) -> int:
    """Same shape as `_accumulate_local`, but for Claude groundtruth.
    Quadrant comes from the row's ``quadrant`` field (already 6-way).
    Mutates `by_face`, `claude_faces`, `dropped`."""
    n_kept = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            q = r.get("quadrant", "")
            if q not in QUADRANT_ORDER:
                continue
            fw_raw = r.get("first_word") or ""
            fw = canonicalize_kaomoji(fw_raw) or ""
            if not (isinstance(fw, str) and len(fw) > 0 and fw.startswith("(")):
                continue
            if not _is_clean_kaomoji(fw):
                dropped[fw] = dropped.get(fw, 0) + 1
                continue
            entry = by_face.setdefault(fw, {qq: 0 for qq in QUADRANT_ORDER})
            entry[q] += 1
            claude_faces.add(fw)
            n_kept += 1
    return n_kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated v3 model keys. Default: {','.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--no-claude", action="store_true",
        help=f"Skip Claude groundtruth runs. Default: include the union "
             f"of all runs in {CLAUDE_RUNS_DIR.name}/ if any exist.",
    )
    parser.add_argument(
        "--no-wild", action="store_true",
        help=f"Skip in-the-wild contributor data. Default: include if "
             f"{WILD_DATA_DIR.name} exists.",
    )
    parser.add_argument(
        "--no-introspection", action="store_true",
        help=f"Skip introspection-arm data (Claude introspection runs in "
             f"{CLAUDE_INTROSPECTION_RUNS_DIR.name}/ and gemma's "
             f"{GEMMA_INTROSPECTION_PATH.name}). Default: include if "
             f"either path exists. Introspection-arm faces are pooled "
             f"into the same per-quadrant counts as naturalistic — this "
             f"expands the union vocabulary so face_likelihood scorers "
             f"see the priming-only kaomoji.",
    )
    args = parser.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in MODEL_REGISTRY:
            raise SystemExit(
                f"unknown model {m!r}; known: {sorted(MODEL_REGISTRY)}"
            )

    print(f"requested models: {requested}; include_claude: {not args.no_claude}")

    by_face: dict[str, dict[str, int]] = {}
    per_model_emit: dict[str, dict[str, int]] = {}
    claude_faces: set[str] = set()
    wild_emit: dict[str, int] = {}
    wild_providers: dict[str, set[str]] = {}
    dropped: dict[str, int] = {}  # emoji-contaminated faces, count per face
    n_local_total = 0
    for m in requested:
        M = MODEL_REGISTRY[m]
        if not M.emotional_data_path.exists():
            print(f"  [{m}] no v3 data at {M.emotional_data_path}; skipping")
            continue
        n = _accumulate_local(
            M.emotional_data_path, by_face, dropped,
            per_model=per_model_emit, model_short=m,
        )
        n_local_total += n
        print(f"  [{m}] {n} kaomoji-bearing rows kept")

    if not args.no_claude:
        runs = find_run_files()
        if runs:
            n_total = 0
            for idx, path in runs:
                n = _accumulate_claude(path, by_face, claude_faces, dropped)
                n_total += n
                print(f"  [claude run-{idx}] {n} kaomoji-bearing rows kept")
            print(f"  [claude all runs] {n_total} kaomoji-bearing rows kept")
        else:
            print(f"  [claude] no run-*.jsonl in {CLAUDE_RUNS_DIR}; skipping")

    if not args.no_introspection:
        # Claude introspection arm — same shape as naturalistic Claude runs.
        intro_runs = find_run_files(CLAUDE_INTROSPECTION_RUNS_DIR)
        if intro_runs:
            n_total = 0
            for idx, path in intro_runs:
                n = _accumulate_claude(path, by_face, claude_faces, dropped)
                n_total += n
                print(f"  [claude-intro run-{idx}] {n} kaomoji-bearing rows kept")
            print(f"  [claude-intro all runs] {n_total} kaomoji-bearing rows kept")
        else:
            print(f"  [claude-intro] no run-*.jsonl in "
                  f"{CLAUDE_INTROSPECTION_RUNS_DIR}; skipping")
        # Gemma introspection (v7-primed) — uses the local-shape accumulator
        # since prompt_id → quadrant derivation matches v3 main.
        if GEMMA_INTROSPECTION_PATH.exists():
            n = _accumulate_local(GEMMA_INTROSPECTION_PATH, by_face, dropped)
            print(f"  [gemma-intro v7-primed] {n} kaomoji-bearing rows kept")
        else:
            print(f"  [gemma-intro] no data at {GEMMA_INTROSPECTION_PATH}; skipping")

    if not args.no_wild:
        _accumulate_wild(by_face, wild_emit, wild_providers, dropped)

    if dropped:
        n_drop_rows = sum(dropped.values())
        print(f"\nfiltered {n_drop_rows} rows / {len(dropped)} unique faces "
              f"with non-BMP codepoints (likely emoji contamination)")
        # Top-10 dropped for diagnostic
        worst = sorted(dropped.items(), key=lambda kv: -kv[1])[:10]
        for fw, n in worst:
            # Highlight which non-BMP chars triggered the drop
            offenders = "".join(c for c in fw if ord(c) > 0xFFFF)
            print(f"  {fw}  ×{n}  (non-BMP: {offenders!r})")

    if not by_face:
        raise SystemExit("no faces accumulated — check that JSONLs exist")

    n_wild_only = sum(1 for fw in by_face
                      if sum(by_face[fw].values()) == 0
                      and wild_emit.get(fw, 0) > 0)
    print(f"\nface union: {len(by_face)} unique kaomoji "
          f"({len(claude_faces)} also emitted by Claude pilot, "
          f"{len(wild_emit)} in wild data, "
          f"{n_wild_only} wild-only)")

    rows: list[dict] = []
    for fw in sorted(by_face):
        counts = by_face[fw]
        total = sum(counts.values())
        wild_n = wild_emit.get(fw, 0)
        rec = {
            "first_word": fw,
            "is_claude": fw in claude_faces,
            "is_wild": wild_n > 0,
            "total_emit_count": total,
        }
        for q in QUADRANT_ORDER:
            rec[f"total_emit_{q}"] = counts[q]
        # Per-(face, model) emit counts so downstream scripts (e.g. the
        # cross-emit-sanity origin partition in 51) can bucket faces by
        # which v3 models emitted them.
        per_m = per_model_emit.get(fw, {})
        for m in requested:
            rec[f"{m}_emit_count"] = per_m.get(m, 0)
        rec["wild_emit_count"] = wild_n
        rec["wild_providers"] = ",".join(sorted(wild_providers.get(fw, set())))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_PARQUET}")
    print(f"wrote {OUT_TSV}")

    # Quick top-N sanity prints
    print("\ntop-15 by total_emit_count (v3-labeled):")
    top = df.sort_values("total_emit_count", ascending=False).head(15)
    for _, r in top.iterrows():
        per_q = " ".join(f"{q}={int(r[f'total_emit_{q}'])}" for q in QUADRANT_ORDER)
        claude_tag = "★" if bool(r["is_claude"]) else " "
        wild_tag = "🌐" if bool(r["is_wild"]) else " "
        print(f"  {claude_tag}{wild_tag} {r['first_word']:<24s}  "
              f"total={int(r['total_emit_count']):>3}  {per_q}  "
              f"wild={int(r['wild_emit_count']):>3}")
    print("\ntop-15 by wild_emit_count:")
    top_w = df[df["wild_emit_count"] > 0].sort_values(
        "wild_emit_count", ascending=False).head(15)
    for _, r in top_w.iterrows():
        provs = str(r["wild_providers"])
        if len(provs) > 50:
            provs = provs[:47] + "..."
        print(f"  {r['first_word']:<24s}  wild={int(r['wild_emit_count']):>4}  "
              f"v3={int(r['total_emit_count']):>3}  [{provs}]")


if __name__ == "__main__":
    main()
