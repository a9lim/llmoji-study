"""Hook-journal source adapter.

Two cooperating Stop hooks log a JSONL line per assistant turn, one
for Claude Code (`~/.claude/hooks/kaomoji-log.sh`) and one for the
Codex CLI (`~/.codex/hooks/kaomoji-log.sh`). They share a single
schema (see either hook script for the field list), distinguished by
the `source` field (`"claude"` or `"codex"`).

Hook rows carry full `assistant_text` and `user_text`, so they're
first-class for the embed / Haiku-describe / eriskii consumers — no
metadata-only caveat. The adapter still emits ScrapeRow.source as
``"claude-hook"`` / ``"codex-hook"`` so per-source breakdowns stay
distinguishable from the transcript / export sources.

Cutoff handling: re-running the legacy transcript scrape is the only
way a turn ends up in both `_code.jsonl` AND `_hook.jsonl`. The agreed
plan is to bake `_code.jsonl` once at a fixed cutoff and never re-run
it, so hook rows logged after the bake are exclusively hook
territory. No dedup logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .claude_scrape import ScrapeRow
from .taxonomy import KAOMOJI_START_CHARS, extract


def _project_slug_from_cwd(cwd: str | None) -> str:
    if not cwd:
        return "(unknown)"
    name = Path(cwd).name
    return name or "(unknown)"


def _iter_journal(path: Path, *, source: str) -> Iterator[ScrapeRow]:
    """Yield ScrapeRow per kaomoji-bearing journal line.

    `source` is hardcoded by the caller (e.g. ``"claude"`` for
    `~/.claude/kaomoji-journal.jsonl`) since each journal file is
    dedicated to one agent — the field isn't redundantly stored on
    every row.

    The hook's `kaomoji` field is already the leading non-letter
    prefix; we still pipe it through `extract()` to (a) recover a
    clean balanced-paren `first_word` and (b) get a taxonomy match
    if any. Defensive guard against any legacy null-kaomoji rows
    or rows whose first char somehow slipped past the start-char
    filter at write time.
    """
    if not path.exists():
        return
    with path.open() as f:
        lines = f.read().splitlines()
    turn = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prefix = row.get("kaomoji")
        if not prefix:
            continue
        match = extract(str(prefix))
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            continue
        cwd = row.get("cwd")
        yield ScrapeRow(
            source=f"{source}-hook",
            session_id="",
            project_slug=_project_slug_from_cwd(cwd),
            assistant_uuid="",
            parent_uuid=None,
            model=str(row.get("model") or "") or None,
            timestamp=str(row.get("ts") or ""),
            cwd=str(cwd) if cwd else None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,
            assistant_text=str(row.get("assistant_text") or ""),
            first_word=match.first_word,
            kaomoji=match.kaomoji,
            kaomoji_label=match.label,
            surrounding_user=str(row.get("user_text") or ""),
        )
        turn += 1


def iter_claude_hook() -> Iterator[ScrapeRow]:
    """Yield kaomoji-bearing rows from both the Claude Code and Codex
    Stop-hook journals."""
    from .config import CLAUDE_HOOK_JOURNAL_CLAUDE, CLAUDE_HOOK_JOURNAL_CODEX
    yield from _iter_journal(CLAUDE_HOOK_JOURNAL_CLAUDE, source="claude")
    yield from _iter_journal(CLAUDE_HOOK_JOURNAL_CODEX,  source="codex")
