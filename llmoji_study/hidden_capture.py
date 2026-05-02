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
      hidden_states : (n_tokens, hidden_dim) when ``store_full_trace=True``;
                      (2, hidden_dim) fallback stack of (h_first, h_last)
                      when ``store_full_trace=False`` — matches the
                      ``load_hidden_states(full_trace=False)`` reconstruction
                      so downstream [0] / [-1] indexing still works.
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


def read_after_generate(
    session: Any,
    *,
    store_full_trace: bool = False,
) -> FullSequenceCapture:
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

    Performance: layers are stacked together on-device into a single
    ``(n_layers, n_tokens, hidden_dim)`` tensor and transferred to
    host with one ``.cpu()`` call. With ``store_full_trace=False`` (the
    hot path post-h_first cutover — see capture.run_sample), the full
    trace never leaves the GPU; only the three ``(n_layers, hidden_dim)``
    aggregates are transferred. ~55× fewer device→host syncs for a
    56-layer model.
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

    # Pass 1: per-layer trim + stack on-device. We hold off on the
    # device→host transfer until all kept layers are stacked together,
    # so we can do one .cpu() instead of N.
    kept_idxs: list[int] = []
    kept_stacks: list[torch.Tensor] = []
    n_tokens = 0
    for idx, bucket in buckets.items():
        if not bucket:
            continue
        # Trim to saklas's canonical length — drops trailing EOS capture
        # if generation terminated on EOS.
        trimmed = bucket[:trim_n] if trim_n > 0 else bucket
        if not trimmed:
            continue
        # (n_tokens, hidden_dim), still on-device, original dtype.
        stacked = torch.stack(trimmed)
        if n_tokens == 0:
            n_tokens = stacked.shape[0]
        elif stacked.shape[0] != n_tokens:
            raise RuntimeError(
                f"inconsistent n_tokens across layers after trim: "
                f"layer {idx} has {stacked.shape[0]}, expected {n_tokens}"
            )
        kept_idxs.append(idx)
        kept_stacks.append(stacked)

    if not kept_idxs:
        return FullSequenceCapture(layers={}, n_tokens=0)

    # All-layer batched stack: (n_layers, n_tokens, hidden_dim).
    # The existing inconsistency check above guarantees same n_tokens.
    all_stack = torch.stack(kept_stacks)  # device, original dtype

    layers: dict[int, LayerCapture] = {}
    if store_full_trace:
        # One device→host transfer for the entire (n_layers, n_tokens,
        # hidden_dim) tensor, then slice in numpy.
        full_np = all_stack.detach().to(torch.float32).cpu().numpy()
        for k, idx in enumerate(kept_idxs):
            stacked_np = full_np[k]
            layers[idx] = LayerCapture(
                layer_idx=idx,
                hidden_states=stacked_np,
                h_first=stacked_np[0].copy(),
                h_last=stacked_np[-1].copy(),
                h_mean=stacked_np.mean(axis=0),
            )
    else:
        # Compute aggregates on-device (still batched), then transfer
        # only the three small (n_layers, hidden_dim) tensors. The full
        # per-token trace never crosses the bus.
        h_first_dev = all_stack[:, 0, :]
        h_last_dev = all_stack[:, -1, :]
        h_mean_dev = all_stack.mean(dim=1)
        # Single combined transfer: (3, n_layers, hidden_dim).
        agg_np = (
            torch.stack([h_first_dev, h_last_dev, h_mean_dev])
            .detach().to(torch.float32).cpu().numpy()
        )
        h_first_np, h_last_np, h_mean_np = agg_np[0], agg_np[1], agg_np[2]
        for k, idx in enumerate(kept_idxs):
            hf, hl, hm = h_first_np[k], h_last_np[k], h_mean_np[k]
            # Length-2 stack of (h_first, h_last) — same shape that
            # load_hidden_states(full_trace=False) reconstructs.
            # Keeps downstream code that touches lc.hidden_states[0]
            # / [-1] working without branching.
            hidden_states_fallback = np.stack([hf, hl])
            layers[idx] = LayerCapture(
                layer_idx=idx,
                hidden_states=hidden_states_fallback,
                h_first=hf,
                h_last=hl,
                h_mean=hm,
            )

    return FullSequenceCapture(layers=layers, n_tokens=n_tokens)
