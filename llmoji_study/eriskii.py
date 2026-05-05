"""Pipeline primitives for the eriskii-replication experiment.

Three primitives consumed by ``scripts/64_eriskii_replication.py``:

  - :func:`compute_axis_vectors` — embed each ``(positive, negative)``
    anchor pair, return the L2-normalized difference per axis.
  - :func:`project_onto_axes` — kaomoji embedding · axis matrix.
  - :func:`label_cluster_via_haiku` — single Haiku call returning a
    short cluster label given the cluster's members + descriptions.

Pre-2026-04-27 this module also carried ``mask_kaomoji`` /
``call_haiku`` (used by the local two-stage describe pipeline) and
``weighted_group_axis_stats`` / ``user_kaomoji_axis_correlation``
(used by the per-model / per-project / mechanistic-bridge sections of
the eriskii script). Both pipelines are gone in the HF-corpus refactor
— per-instance description happens contributor-side via the
``llmoji`` package, and the bridge / per-model breakdowns required
per-row metadata that the HF dataset doesn't carry.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_axis_vectors(
    embedder: Any,
    anchors: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    """For each axis name → ``(positive_anchor, negative_anchor)``,
    embed both, return the L2-normalized difference (positive − negative).

    ``embedder`` is a ``sentence_transformers.SentenceTransformer``.
    """
    pos_texts = [pos for pos, _ in anchors.values()]
    neg_texts = [neg for _, neg in anchors.values()]
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
    """Return ``(n_kaomoji, n_axes)`` projection matrix.

    Rows of ``E`` are assumed already L2-normalized (matches what
    ``save_embeddings`` / ``load_embeddings`` produce). Axis vectors
    are L2-normalized by :func:`compute_axis_vectors`. Cosine
    similarity collapses to dot product under that normalization, so
    ``result[i, j]`` is the cosine of kaomoji ``i``'s description-
    embedding with axis ``j``.
    """
    A = np.stack([axis_vectors[name] for name in axis_order], axis=1)
    return E @ A


def label_cluster_via_haiku(
    client: Any,
    members: list[tuple[str, str]],
    *,
    model_id: str,
    prompt_template: str,
    max_tokens: int = 60,
) -> str:
    """Given member ``[(first_word, description), ...]``, ask Haiku
    for a 3-5 word eriskii-style cluster label. Returns the stripped
    response text. Caller's resume loop handles errors."""
    members_str = "\n".join(f"- {fw}: {desc}" for fw, desc in members)
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": prompt_template.format(members=members_str),
        }],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""
