"""Claude Code source adapter: ~/.claude/projects/**/*.jsonl.

Each JSONL file is one session. Events are threaded via parentUuid.
We emit one ScrapeRow per assistant message that contains at least one
text content block and whose first_word matches a kaomoji prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .claude_scrape import ScrapeRow
from .taxonomy import extract

# Same glyph set used by analysis.plot_kaomoji_heatmap and
# emotional_analysis._kaomoji_rows.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def _collect_text_and_thinking(message: dict[str, Any]) -> tuple[str, bool]:
    """Concatenate text blocks, return (assistant_text, had_thinking)."""
    parts: list[str] = []
    had_thinking = False
    for block in message.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            txt = block.get("text") or ""
            if txt:
                parts.append(txt)
        elif t == "thinking":
            had_thinking = True
    return "\n".join(parts), had_thinking


def _resolve_user_text(
    start_uuid: str | None,
    by_uuid: dict[str, dict[str, Any]],
    max_hops: int = 5,
) -> str:
    """Walk parentUuid backward to find the nearest user-authored text.

    Skips tool_result parents (which have type=='user' but content is
    machine-generated). Returns '' if nothing found within max_hops.
    """
    uuid = start_uuid
    for _ in range(max_hops):
        if uuid is None:
            return ""
        ev = by_uuid.get(uuid)
        if ev is None:
            return ""
        if ev.get("type") == "user":
            m = ev.get("message", {})
            content = m.get("content") if isinstance(m, dict) else None
            # user events come in two shapes: plain string content, or
            # list-of-blocks (tool_result). We want plain string.
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                # look for any "text"-type block; skip tool_result blocks
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        txt = b.get("text") or ""
                        if txt.strip():
                            return txt
        uuid = ev.get("parentUuid")
    return ""


def _iter_session(path: Path) -> Iterator[ScrapeRow]:
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    by_uuid: dict[str, dict[str, Any]] = {
        ev["uuid"]: ev for ev in events if isinstance(ev.get("uuid"), str)
    }
    # turn index: count assistant messages in order of appearance
    turn = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        m = ev.get("message", {})
        if not isinstance(m, dict):
            continue
        text, had_thinking = _collect_text_and_thinking(m)
        if not text.strip():
            continue
        match = extract(text)
        # only emit if first_word starts with a kaomoji-ish glyph
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            turn += 1
            continue
        user_text = _resolve_user_text(ev.get("parentUuid"), by_uuid)
        yield ScrapeRow(
            source="claude-code",
            session_id=str(ev.get("sessionId") or path.stem),
            project_slug=str(path.parent.name),
            assistant_uuid=str(ev.get("uuid") or ""),
            parent_uuid=ev.get("parentUuid"),
            model=str(m.get("model")) if m.get("model") else None,
            timestamp=str(ev.get("timestamp") or ""),
            cwd=str(ev.get("cwd")) if ev.get("cwd") else None,
            git_branch=str(ev.get("gitBranch")) if ev.get("gitBranch") else None,
            turn_index=turn,
            had_thinking=had_thinking,
            assistant_text=text,
            first_word=match.first_word,
            kaomoji=match.kaomoji,
            kaomoji_label=match.label,
            surrounding_user=user_text,
        )
        turn += 1


def iter_claude_code() -> Iterator[ScrapeRow]:
    """Yield every kaomoji-bearing assistant message from all Claude
    Code sessions on disk."""
    from .config import CLAUDE_CODE_PROJECTS_DIR
    root = Path(CLAUDE_CODE_PROJECTS_DIR)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        yield from _iter_session(path)
