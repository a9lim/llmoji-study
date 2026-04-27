"""Unified kaomoji-scrape schema across Claude data sources.

Two concrete sources emit ScrapeRow instances:
  - claude_export_source.py: Claude.ai export
  - claude_hook_source.py:   ~/.claude + ~/.codex unified journal
                             (live Stop hooks + retroactive backfill)

Kaomoji extraction uses llmoji.taxonomy.extract (balanced-paren span
fallback; no dialect-specific dict required — unlike the gemma pilot,
the eriskii-style analysis clusters on unique strings, not pre-
registered labels).

`iter_all` chains both. Hook rows now carry full `assistant_text`
(after the unified-schema refactor), so they're text-rich and safe
for the embed / Haiku-describe / eriskii pipelines that key off
`assistant_text`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator


@dataclass
class ScrapeRow:
    """One kaomoji-bearing assistant message, source-agnostic."""

    # --- provenance ---
    source: str                 # "claude-code" | "claude-ai-export"
    session_id: str             # cc sessionId or webapp conversation uuid
    project_slug: str           # "-Users-a9lim-Work-llmoji" or conversation name
    assistant_uuid: str         # the assistant message uuid
    parent_uuid: str | None     # for cc: parentUuid; for webapp: parent_message_uuid

    # --- context ---
    model: str | None           # cc: message.model; webapp: None (not in export)
    timestamp: str              # ISO-8601
    cwd: str | None             # cc only
    git_branch: str | None      # cc only
    turn_index: int             # 0-based position in session
    had_thinking: bool          # did the assistant turn include a thinking block

    # --- content ---
    assistant_text: str         # full assistant message text
    first_word: str             # leading balanced-paren span or first whitespace-word
    kaomoji: str | None         # taxonomy-registered form (usually None for Claude)
    kaomoji_label: int          # +1/-1/0 (usually 0 since taxonomy is gemma-tuned)

    # --- upstream ---
    surrounding_user: str       # preceding user-authored text, resolved via parent chain

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Source adapters yield ScrapeRow. The unified iterator in
# llmoji.claude_scrape.iter_all chains the two sources together.


def iter_all() -> Iterator[ScrapeRow]:
    """Yield ScrapeRow from both the Claude.ai export and the unified
    Stop-hook journals (Claude + Codex)."""
    from .claude_export_source import iter_claude_export
    from .claude_hook_source import iter_claude_hook
    yield from iter_claude_export()
    yield from iter_claude_hook()
