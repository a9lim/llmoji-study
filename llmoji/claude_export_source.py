"""Claude.ai export source adapter: conversations.json.

The export is a single JSON file: list of conversation objects, each
with a chat_messages array. The top-level .text field on each message
is the canonical content (content-block .text is often empty in this
format). Model info isn't in the export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .claude_scrape import ScrapeRow
from .taxonomy import extract

KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def _message_text(msg: dict[str, Any]) -> str:
    """Prefer top-level .text; fall back to content[].text blocks."""
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    parts: list[str] = []
    for block in msg.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            bt = block.get("text") or ""
            if bt.strip():
                parts.append(bt)
    return "\n".join(parts)


def _iter_conversation(conv: dict[str, Any]) -> Iterator[ScrapeRow]:
    msgs = conv.get("chat_messages") or []
    if not msgs:
        return
    # index by uuid for parent_message_uuid walks
    by_uuid: dict[str, dict[str, Any]] = {
        m["uuid"]: m for m in msgs if isinstance(m.get("uuid"), str)
    }
    session_id = str(conv.get("uuid") or "")
    project_slug = str(conv.get("name") or "") or "(unnamed)"
    turn = 0
    for m in msgs:
        if m.get("sender") != "assistant":
            continue
        text = _message_text(m)
        if not text.strip():
            continue
        match = extract(text)
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            turn += 1
            continue
        # Resolve user-surrounding text by walking parent_message_uuid
        # back to the nearest human message with non-empty text.
        user_text = ""
        parent_uuid = m.get("parent_message_uuid")
        cur = parent_uuid
        for _ in range(5):
            if not cur:
                break
            pm = by_uuid.get(cur)
            if pm is None:
                break
            if pm.get("sender") == "human":
                pt = _message_text(pm)
                if pt.strip():
                    user_text = pt
                    break
            cur = pm.get("parent_message_uuid")
        yield ScrapeRow(
            source="claude-ai-export",
            session_id=session_id,
            project_slug=project_slug,
            assistant_uuid=str(m.get("uuid") or ""),
            parent_uuid=parent_uuid,
            model=None,  # not in export
            timestamp=str(m.get("created_at") or ""),
            cwd=None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,  # export doesn't include thinking blocks
            assistant_text=text,
            first_word=match.first_word,
            kaomoji=match.kaomoji,
            kaomoji_label=match.label,
            surrounding_user=user_text,
        )
        turn += 1


def iter_claude_export() -> Iterator[ScrapeRow]:
    """Yield every kaomoji-bearing assistant message from the
    Claude.ai export conversations.json."""
    from .config import CLAUDE_AI_EXPORT_DIR
    path = Path(CLAUDE_AI_EXPORT_DIR) / "conversations.json"
    if not path.exists():
        return
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        return
    for conv in data:
        if isinstance(conv, dict):
            yield from _iter_conversation(conv)
