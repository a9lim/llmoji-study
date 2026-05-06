"""Local-machine kaomoji emission readers for harness analyses.

Pulls a9's own emissions from two surfaces:

  - **claude_code**: ``~/.claude/kaomoji-journal.jsonl`` — the Stop-hook
    journal written by every Claude Code instance on this machine.
    Carries ``cwd``, so the per-project bucket is recoverable.
  - **claude_ai**: one or more claude.ai conversation exports. Two
    formats coexist:
      * legacy: a directory containing ``conversations.json`` (a single
        list of conversation objects, no per-row model). Read by
        :func:`llmoji.sources.claude_export.iter_claude_export`.
      * alt (post-2026-05): a directory containing one
        ``<title>.json`` per conversation plus an ``export_summary.json``
        sibling. Carries ``model`` per-conversation. Read by
        :func:`llmoji.sources.claude_export_alt.iter_claude_export_alt`.
    The format is auto-detected per directory via
    :func:`detect_claude_export_format`. Both formats union by
    conversation UUID with the richer copy winning, within their own
    reader; we don't cross-format dedup because the same conversation
    rarely lives in both shapes for one user.

Used by:
  - ``scripts/66_per_project_quadrants.py`` for per-project + global
    Russell-quadrant histograms.
  - ``scripts/67_wild_residual.py`` for marker-by-deployment-surface
    on the BoL PCA chart.

Shared module rather than copy-pasted helpers because the format
dispatch + DEFAULT_CLAUDE_EXPORTS list want to stay in lockstep across
both consumers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from llmoji.taxonomy import canonicalize_kaomoji


# ---------------------------------------------------------------- defaults
# Where the per-machine Claude Code journal lives. Single source of
# truth — both 66 and 67 read this path unless overridden.
DEFAULT_CLAUDE_JOURNAL = Path.home() / ".claude" / "kaomoji-journal.jsonl"

# Claude.ai exports a9 has on this machine. Mix of legacy + alt format;
# format is auto-detected per directory at iteration time. Empty paths
# are silently skipped so deleting an export doesn't break callers.
DEFAULT_CLAUDE_EXPORTS: list[Path] = [
    Path(
        "/Users/a9lim/Downloads/"
        "data-72de1230-b9fa-4c55-bc10-84a35b58d89c-1777763577-c21ac4ff-batch-0000/"
        "conversations.json"
    ),
    Path(
        "/Users/a9lim/Downloads/"
        "9cc23402cbb1e8aec9785eb0f750f1c442f1ba13e507bd6b04a727c627d64d08-"
        "2026-04-28-01-04-53-1d1e60e8c10441b1881c7e81834c3b26/"
        "conversations.json"
    ),
    # 2026-05-05 alt-format export: per-conversation .json files in a
    # directory + export_summary.json sibling.
    Path("/Users/a9lim/Downloads/claude-conversations-2026-05-05"),
]


# Source labels carried on every emission row. Stable across consumers
# so categorical logic in 66/67 can key off them safely.
SOURCE_CLAUDE_CODE = "claude_code"
SOURCE_CLAUDE_AI = "claude_ai"


# ---------------------------------------------------------------- format
def detect_claude_export_format(directory: Path) -> str:
    """Return 'legacy', 'alt', or 'unknown' for the given directory.

    'legacy' iff a single ``conversations.json`` is present.
    'alt' iff an ``export_summary.json`` sibling is present, or
    no ``conversations.json`` but multiple ``.json`` files (the
    likely shape for an alt-format export missing its summary).
    """
    if (directory / "conversations.json").is_file():
        return "legacy"
    if (directory / "export_summary.json").is_file():
        return "alt"
    json_files = [p for p in directory.glob("*.json") if p.is_file()]
    if len(json_files) >= 2:
        return "alt"
    return "unknown"


# ---------------------------------------------------------------- readers
def iter_local_emissions(
    journal_path: Path | None = None,
    claude_export_paths: list[Path] | None = None,
    *,
    quiet: bool = False,
) -> Iterator[tuple[str, str, str | None]]:
    """Yield ``(canonical_face, source, project_or_None)`` for every
    kaomoji-bearing emission in a9's local data.

    ``source`` is one of :data:`SOURCE_CLAUDE_CODE` or
    :data:`SOURCE_CLAUDE_AI`. ``project_or_None`` is the cwd basename
    for journal rows (Claude Code carries ``cwd``); the literal
    string ``"claude.ai"`` for export rows (no per-conversation
    project bucket exists). Both are used by script 66 for the
    per-project histograms.

    Defaults to :data:`DEFAULT_CLAUDE_JOURNAL` and
    :data:`DEFAULT_CLAUDE_EXPORTS`. Pass ``quiet=True`` to suppress
    per-source progress prints.
    """
    journal_path = journal_path or DEFAULT_CLAUDE_JOURNAL
    claude_export_paths = (
        claude_export_paths if claude_export_paths is not None
        else list(DEFAULT_CLAUDE_EXPORTS)
    )

    # Claude Code journal
    if journal_path.exists():
        from llmoji.sources.journal import iter_journal
        n_journal = 0
        for sr in iter_journal(journal_path, source=SOURCE_CLAUDE_CODE):
            face = (sr.first_word or "").strip()
            if not face:
                continue
            canon = canonicalize_kaomoji(face)
            if not canon:
                continue
            project = _project_from_cwd(getattr(sr, "cwd", None))
            yield (canon, SOURCE_CLAUDE_CODE, project)
            n_journal += 1
        if not quiet:
            print(f"  {journal_path.name}: {n_journal} emissions")
    elif not quiet:
        print(f"  skip {journal_path} (missing)")

    # Claude.ai exports — auto-dispatch by format
    legacy_dirs: list[Path] = []
    alt_dirs: list[Path] = []
    for path in claude_export_paths:
        if not path.exists():
            if not quiet:
                print(f"  skip {path} (missing)")
            continue
        directory = path.parent if path.is_file() else path
        fmt = detect_claude_export_format(directory)
        if fmt == "legacy":
            legacy_dirs.append(directory)
        elif fmt == "alt":
            alt_dirs.append(directory)
        elif not quiet:
            print(f"  skip {directory} (no recognized export format)")

    if legacy_dirs or alt_dirs:
        try:
            from llmoji.sources.claude_export import iter_claude_export
            from llmoji.sources.claude_export_alt import iter_claude_export_alt
        except ImportError:
            if not quiet:
                print("  skip claude.ai export (llmoji.sources.* not available)")
            return
        n_legacy = 0
        for sr in iter_claude_export(legacy_dirs):
            face = (sr.first_word or "").strip()
            if not face:
                continue
            canon = canonicalize_kaomoji(face)
            if not canon:
                continue
            yield (canon, SOURCE_CLAUDE_AI, "claude.ai")
            n_legacy += 1
        n_alt = 0
        for sr in iter_claude_export_alt(alt_dirs):
            face = (sr.first_word or "").strip()
            if not face:
                continue
            canon = canonicalize_kaomoji(face)
            if not canon:
                continue
            yield (canon, SOURCE_CLAUDE_AI, "claude.ai")
            n_alt += 1
        if not quiet:
            print(
                f"  claude.ai: {n_legacy + n_alt} emissions  "
                f"(legacy: {n_legacy} from {len(legacy_dirs)} export(s); "
                f"alt: {n_alt} from {len(alt_dirs)} export(s))"
            )


def _project_from_cwd(cwd: str | None) -> str:
    """Project bucket for a Claude Code emission. Mirrors the helper
    that used to live inline in script 66 — extracted here so 66 and
    67 share the same bucketing rule."""
    if not cwd:
        return "(no_project)"
    return Path(cwd).name or "(no_project)"


def load_face_source_counts(
    journal_path: Path | None = None,
    claude_export_paths: list[Path] | None = None,
    *,
    quiet: bool = False,
) -> dict[str, dict[str, int]]:
    """Per canonical face, return ``{source: emit_count}`` over the
    two surfaces (claude_code, claude_ai). Faces emitted only via one
    surface have a 0 count for the other (or the key is absent —
    callers should treat missing as 0).

    Used by script 67 to mark each face by deployment surface
    (Code vs chat) independent of the GT-overlap categorization.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {SOURCE_CLAUDE_CODE: 0, SOURCE_CLAUDE_AI: 0}
    )
    for face, source, _project in iter_local_emissions(
        journal_path, claude_export_paths, quiet=quiet,
    ):
        counts[face][source] += 1
    return dict(counts)


__all__ = [
    "DEFAULT_CLAUDE_EXPORTS",
    "DEFAULT_CLAUDE_JOURNAL",
    "SOURCE_CLAUDE_AI",
    "SOURCE_CLAUDE_CODE",
    "detect_claude_export_format",
    "iter_local_emissions",
    "load_face_source_counts",
]
