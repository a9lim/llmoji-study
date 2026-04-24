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

import numpy as np


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


def compute_axis_vectors(
    embedder: Any,
    anchors: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    """For each axis name → (positive_anchor, negative_anchor),
    embed both, return the L2-normalized difference (positive − negative).

    `embedder` is a sentence_transformers.SentenceTransformer instance.
    """
    pos_texts = [pos for pos, _ in anchors.values()]
    neg_texts = [neg for _, neg in anchors.values()]
    # one batch call for all anchors at once
    pos_emb = embedder.encode(
        pos_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    neg_emb = embedder.encode(
        neg_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(anchors.keys()):
        diff = np.asarray(pos_emb[i]) - np.asarray(neg_emb[i])
        norm = float(np.linalg.norm(diff))
        if norm > 0:
            diff = diff / norm
        out[name] = diff
    return out


def project_onto_axes(
    E: np.ndarray,
    axis_vectors: dict[str, np.ndarray],
    axis_order: list[str],
) -> np.ndarray:
    """Return (n_kaomoji, n_axes) projection matrix.

    Rows of E are assumed already L2-normalized (matches what
    save_embeddings/load_embeddings produce). Axis vectors are
    L2-normalized by compute_axis_vectors. Cosine similarity collapses
    to dot product under that normalization, so result[i, j] is the
    cosine of kaomoji i's description-embedding with axis j.
    """
    A = np.stack([axis_vectors[name] for name in axis_order], axis=1)
    return E @ A
