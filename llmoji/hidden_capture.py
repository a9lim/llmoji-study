# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Per-generated-token hidden-state capture by reading saklas's own
state after generation.

Saklas's built-in ``HiddenCapture`` accumulates the last-position
hidden state per forward pass during streaming generation — i.e. one
``(hidden_dim,)`` slice per generated token, at every probe layer.
That's exactly the signal we want for cosine-in-hidden-state-space
analysis, and it's already populated on ``session._capture._per_layer``
after a ``session.generate()`` call.

This module is thin: ``read_after_generate(session)`` pulls saklas's
per-layer buckets into a ``FullSequenceCapture`` with pre-computed
first / last / mean aggregates per layer and the full per-token trace.
No extra forward pass, no attention-implementation coupling, no
tokenizer chat-template reconstruction.

The three aggregates correspond to:
  h_first  — state that produced the first generated token (kaomoji
             under the kaomoji instruction). Most probative feature
             for "what state led the model to emit this kaomoji."
  h_last   — state after the full generation, carrying whatever
             accumulated from the entire response.
  h_mean   — mean over per-token hidden states. For linear probes,
             probe · h_mean == mean(probe · h_t) == saklas's
             ``probe_means`` aggregate — the identity we exploit in
             the smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class LayerCapture:
    """Per-layer capture: full per-token trace + three aggregates.

    Shapes are fp32 numpy:
      hidden_states : (n_tokens, hidden_dim)
      h_first       : (hidden_dim,) — hidden_states[0]
      h_last        : (hidden_dim,) — hidden_states[-1]
      h_mean        : (hidden_dim,) — hidden_states.mean(axis=0)
    """
    layer_idx: int
    hidden_states: np.ndarray
    h_first: np.ndarray
    h_last: np.ndarray
    h_mean: np.ndarray


@dataclass
class FullSequenceCapture:
    """All layer captures for one generation, plus bookkeeping."""
    layers: dict[int, LayerCapture]
    n_tokens: int  # number of generated tokens (= size of each layer's trace)


def read_after_generate(session: Any) -> FullSequenceCapture:
    """Read saklas's post-generation hidden-state buckets into a
    ``FullSequenceCapture``.

    Must be called after a ``session.generate()`` call and before
    anything else clears the capture. Raises if no probe layers were
    captured (which happens if ``session._begin_capture`` returned
    False because no probes are registered).

    When generation terminates on an EOS token, saklas's HiddenCapture
    records one extra bucket entry for the EOS step while
    ``session.last_per_token_scores`` is already trimmed to the
    generated-token count (see saklas ``score_per_token`` — it does
    ``h = h[:n]`` on captures that overshoot ``generated_ids``). We
    mirror that trim here so ``h_last`` aligns with the last scored
    generated token, not the EOS step.
    """
    buckets: dict[int, list[torch.Tensor]] = session._capture._per_layer
    if not buckets:
        raise RuntimeError(
            "session._capture._per_layer is empty — no probes registered, or "
            "generation didn't trigger a capture. Verify probes=... was "
            "passed to SaklasSession.from_pretrained."
        )

    # Determine the saklas-canonical generated-token count. Prefer the
    # trimmed per-token-scores length; fall back to the raw bucket
    # length if scores aren't populated for some reason.
    per_token_scores = getattr(session, "last_per_token_scores", None) or {}
    if per_token_scores:
        trim_n = len(next(iter(per_token_scores.values())))
    else:
        trim_n = min((len(b) for b in buckets.values() if b), default=0)

    layers: dict[int, LayerCapture] = {}
    n_tokens = 0
    for idx, bucket in buckets.items():
        if not bucket:
            continue
        # Trim to saklas's canonical length — drops trailing EOS capture
        # if generation terminated on EOS.
        trimmed = bucket[:trim_n] if trim_n > 0 else bucket
        if not trimmed:
            continue
        # Each element is a (hidden_dim,) tensor per step. Stack +
        # fp32 + CPU in one shot.
        stacked = torch.stack(trimmed).detach().to(torch.float32).cpu().numpy()
        if n_tokens == 0:
            n_tokens = stacked.shape[0]
        elif stacked.shape[0] != n_tokens:
            raise RuntimeError(
                f"inconsistent n_tokens across layers after trim: "
                f"layer {idx} has {stacked.shape[0]}, expected {n_tokens}"
            )
        layers[idx] = LayerCapture(
            layer_idx=idx,
            hidden_states=stacked,
            h_first=stacked[0].copy(),
            h_last=stacked[-1].copy(),
            h_mean=stacked.mean(axis=0),
        )

    return FullSequenceCapture(layers=layers, n_tokens=n_tokens)
