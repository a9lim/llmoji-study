"""Pipeline primitives for the eriskii-replication experiment.

Three functional layers:

  - mask_kaomoji(text, first_word) — replace the leading kaomoji
    span with the literal token [FACE].
  - call_haiku(client, prompt, *, model_id, max_tokens) — single
    Haiku call returning the assistant text, stripped.
  - (later tasks) project_axes / label_clusters / weighted_group_stats /
    user_kaomoji_axis_correlation — analysis primitives consumed by
    scripts/16.
"""

from __future__ import annotations

from typing import Any


MASK_TOKEN = "[FACE]"


def mask_kaomoji(text: str, first_word: str) -> str:
    """Replace the leading kaomoji span with MASK_TOKEN.

    The leading kaomoji is identified by `first_word` (the value
    captured by llmoji.taxonomy.extract at scrape time). We strip
    leading whitespace, verify the text starts with first_word, and
    swap it. If the leading text doesn't match (e.g. the row had a
    kaomoji mid-line), we don't mutate — return the original text.
    """
    stripped = text.lstrip()
    if not first_word or not stripped.startswith(first_word):
        return text
    return MASK_TOKEN + stripped[len(first_word):]


def call_haiku(
    client: Any,
    prompt: str,
    *,
    model_id: str,
    max_tokens: int = 200,
) -> str:
    """Single Haiku call with a pre-formatted prompt. Returns the
    assistant's first text-block content, stripped. Raises on API
    error (caller's resume loop handles).

    `client` is an anthropic.Anthropic instance. We don't import the
    SDK here so this module is importable without anthropic being
    installed (matters for the smoke test in Step 2, which doesn't
    call Haiku)."""
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""
