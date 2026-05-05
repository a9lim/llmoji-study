"""Per-face Claude-modal-quadrant ground truth from the groundtruth runs.

Used by scripts 53 (subset search) and 56 (ensemble predict) when run
with ``--claude-gt``: replaces the pooled ``empirical_majority_quadrant``
(v3 + Claude + wild emit counts) with a Claude-only modal label.

Why: when the goal is to predict Claude's faces in production, GT
should be Claude's own modal quadrant — not a pooled measure that
mostly reflects v3 prompt distribution.

Run layout (post 2026-05-04 sequential-run scaling protocol):

    data/harness/claude-runs/
        run-0.jsonl        # original 120-gen pilot, block-staged
        run-0_summary.tsv
        run-1.jsonl        # subsequent runs: 120 gens, no block stage
        run-1_summary.tsv
        ...

Sequential runs are numbered; saturation is checked between runs by
``scripts/harness/10_emit_analysis.py``. The GT map is the
union over all runs on disk — every additional run can only add or
sharpen face-quadrant evidence.

Note on canonicalization: run-N.jsonl stores the raw extracted
``first_word`` (no canonicalization), so loading this map requires
running ``canonicalize_kaomoji`` to match the keys used in the
face_likelihood summary TSVs.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.config import DATA_DIR

CLAUDE_RUNS_DIR = DATA_DIR / "harness" / "claude-runs"
CLAUDE_RUNS_INTROSPECTION_DIR = DATA_DIR / "harness" / "claude-runs-introspection"

# Backward-compat alias. Prefer CLAUDE_RUNS_DIR / load_all_runs going
# forward; this points at run-0 specifically and exists only so older
# code paths that hardcoded the single-pilot file keep working until
# they're migrated.
DEFAULT_PILOT_PATH = CLAUDE_RUNS_DIR / "run-0.jsonl"

_RUN_FILENAME_RE = re.compile(r"^run-(\d+)\.jsonl$")


def find_run_files(claude_runs_dir: Path | None = None) -> list[tuple[int, Path]]:
    """Return ``[(run_index, jsonl_path), ...]`` sorted ascending by index.

    Empty list if the directory is missing or contains no run-N.jsonl
    files. Run indices need not be contiguous; missing runs are simply
    not surfaced.
    """
    d = claude_runs_dir or CLAUDE_RUNS_DIR
    if not d.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in d.iterdir():
        m = _RUN_FILENAME_RE.match(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def latest_run_index(claude_runs_dir: Path | None = None) -> int:
    """Highest run index on disk, or ``-1`` if no runs exist yet."""
    runs = find_run_files(claude_runs_dir)
    return runs[-1][0] if runs else -1


def load_run_rows(path: Path) -> list[dict]:
    """Read a single run-N.jsonl. Skips error rows."""
    out: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            out.append(r)
    return out


def load_all_run_rows(
    claude_runs_dir: Path | None = None,
    *,
    up_to_index: int | None = None,
) -> list[dict]:
    """Concatenate rows across all runs in the directory.

    ``up_to_index``: if given, include only runs with index ≤ this value.
    Useful for "what did we know before run N?" comparisons.
    """
    rows: list[dict] = []
    for idx, path in find_run_files(claude_runs_dir):
        if up_to_index is not None and idx > up_to_index:
            continue
        for r in load_run_rows(path):
            r.setdefault("run_index", idx)
            rows.append(r)
    return rows


def _load_face_per_quadrant_counts(
    pilot_path: Path | None = None,
    *,
    claude_runs_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, Counter[str]]:
    """Internal: build {canonical_face: Counter(quadrant -> emit_count)}
    from either a single pilot JSONL or the union of run-N.jsonls.

    ``include_introspection``: when True (default), also pool the
    introspection-arm runs in
    ``data/harness/claude-runs-introspection/``. The introspection arm uses a
    different preamble but the same affective prompts, so the per-face
    per-quadrant emission counts are validly poolable for
    distribution-shape estimation. Set False for a "naturalistic-only,
    deployment-shaped" GT.
    """
    if pilot_path is not None:
        rows: list[dict] = []
        with open(pilot_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                rows.append(r)
    else:
        rows = load_all_run_rows(claude_runs_dir, up_to_index=up_to_index)
        if include_introspection:
            intro_dir = (claude_runs_dir.parent / CLAUDE_RUNS_INTROSPECTION_DIR.name
                         if claude_runs_dir is not None
                         else CLAUDE_RUNS_INTROSPECTION_DIR)
            if intro_dir.exists():
                intro_rows = load_all_run_rows(intro_dir, up_to_index=up_to_index)
                # Tag intro rows so downstream can filter if needed.
                for r in intro_rows:
                    r.setdefault("preamble", r.get("preamble", "introspection"))
                rows.extend(intro_rows)

    counts: dict[str, Counter[str]] = {}
    for r in rows:
        f = r.get("first_word", "")
        q = r.get("quadrant", "")
        if not f or not q:
            continue
        f_canon = canonicalize_kaomoji(f)
        counts.setdefault(f_canon, Counter())[q] += 1
    return counts


def load_claude_gt(
    pilot_path: Path | None = None,
    *,
    floor: int = 1,
    claude_runs_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, tuple[str, int]]:
    """Return ``{canonical_face: (modal_quadrant, modal_n_emits)}``.

    Hard-modal mode. Kept for backwards compatibility and for production
    use where a deployed plugin emits a single quadrant call. For
    distribution-vs-distribution evaluation (the primary post-hoc
    metric since 2026-05-04) use ``load_claude_gt_distribution``.

    Sources rows by precedence:
      1. ``pilot_path`` if given (deprecated single-file mode — used
         only by callers that haven't migrated to the multi-run layout).
      2. Otherwise, the union of all runs in ``claude_runs_dir``
         (defaults to ``CLAUDE_RUNS_DIR``).

    ``floor``: faces with ``modal_n_emits < floor`` are excluded.
    Default ``floor=1`` includes every face Claude emitted at least
    once in the modal quadrant; ``floor=2`` requires ≥2 emits.

    ``up_to_index``: only meaningful in multi-run mode — pretend
    runs > N don't exist. For "what would the GT have looked like
    before adding run N+1?" backtesting.
    """
    counts = _load_face_per_quadrant_counts(
        pilot_path,
        claude_runs_dir=claude_runs_dir,
        up_to_index=up_to_index,
        include_introspection=include_introspection,
    )
    out: dict[str, tuple[str, int]] = {}
    for face, qmap in counts.items():
        modal_q, modal_n = qmap.most_common(1)[0]
        if modal_n >= floor:
            out[face] = (modal_q, modal_n)
    return out


def load_claude_gt_distribution(
    pilot_path: Path | None = None,
    *,
    floor: int = 3,
    claude_runs_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, dict[str, int]]:
    """Return ``{canonical_face: {quadrant: raw_emit_count}}``.

    Distribution mode — the primary post-hoc evaluation surface since
    the 2026-05-04 soft-everywhere methodology shift. Returns raw
    counts; the consumer normalizes (with smoothing as appropriate).
    Faces with total emit count < ``floor`` are excluded — sparse
    counts (1-2 emits) give very noisy distribution estimates.
    Default ``floor=3`` matches the modal-stability threshold used
    elsewhere in the project (script 25's ``min_emits=3``).

    Use with ``llmoji_study.jsd.normalize`` + ``js`` for the
    distribution-vs-distribution comparison.

    ``include_introspection``: when True (default), pools the
    introspection-arm runs (data/harness/claude-runs-introspection/) into the
    GT alongside naturalistic. Both arms emit kaomoji on the same
    affective prompts; pooling is honest for distribution-shape
    estimation. Set False if you want a deployment-shaped GT (no
    primed emissions).
    """
    counts = _load_face_per_quadrant_counts(
        pilot_path,
        claude_runs_dir=claude_runs_dir,
        up_to_index=up_to_index,
        include_introspection=include_introspection,
    )
    out: dict[str, dict[str, int]] = {}
    for face, qmap in counts.items():
        total = sum(qmap.values())
        if total < floor:
            continue
        out[face] = dict(qmap)
    return out
