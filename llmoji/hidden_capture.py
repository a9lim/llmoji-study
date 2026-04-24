# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Full-sequence hidden-state + attention-weight capture.

Saklas's built-in ``HiddenCapture`` grabs only the last-position slice
per streaming forward pass — fine for per-generated-token probe scoring
but doesn't give us prompt-token hidden states and doesn't expose
attention weights. For cosine-in-hidden-state-space analysis we want
the full sequence at every probe layer, plus the final layer's last-
token attention distribution (for the "attention-weighted" aggregate).

Approach: after generation completes, run **one extra full-sequence
forward pass** of (prompt + generated) with hooks attached to every
probe layer and the final layer's ``self_attn``. Under causal masking,
position k's hidden state in a full-sequence forward equals what was
produced at step k during generation, so this is semantically
equivalent — just a cleaner way to gather everything in one shot.

Cost: ~1 extra forward pass per generation. For a ~60-token generation
on gemma-4-31b-it this is ~200 ms; negligible vs. the generation itself.

Returned per layer:
  h_first        = hidden_states[0]                            (first token)
  h_last         = hidden_states[-1]                           (last token)
  h_attn_weighted = sum_t attn_weights_last[t] * hidden_states[t]
                   where attn_weights_last comes from the FINAL
                   transformer layer's self_attn, last-token query,
                   averaged over heads.
  hidden_states  = full (seq_len, hidden_dim) tensor            (optional save)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class LayerCapture:
    """Per-layer capture results, fp32 on CPU as numpy arrays."""
    layer_idx: int
    hidden_states: np.ndarray          # (seq_len, hidden_dim)
    h_first: np.ndarray                # (hidden_dim,)
    h_last: np.ndarray                 # (hidden_dim,)
    h_attn_weighted: np.ndarray        # (hidden_dim,)


@dataclass
class FullSequenceCapture:
    """All capture results for one generation."""
    layers: dict[int, LayerCapture]
    attn_weights_last: np.ndarray      # (seq_len,) — final-layer last-token attention
    seq_len: int
    prompt_len: int                    # how many of the seq_len tokens are prompt
    input_ids: np.ndarray              # (seq_len,) — full input sequence


class _LayerHiddenHook:
    """Forward hook capturing a transformer block's output hidden states.

    Gemma-style transformer blocks return either a Tensor or a tuple
    whose first element is the hidden-state tensor of shape
    (batch, seq, dim). We take [0] (batch), cast to fp32 on CPU."""

    def __init__(self) -> None:
        self.hidden: Tensor | None = None

    # Underscore-prefixed args are part of PyTorch's forward-hook signature
    # (module, inputs, output) — names unused but required.
    def __call__(self, _m: Any, _i: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        # h shape: (batch, seq, dim)
        self.hidden = h[0].detach().to(torch.float32).cpu()


class _AttentionWeightsHook:
    """Forward hook capturing attention weights from a self_attn module.

    HuggingFace attention modules called with output_attentions=True
    return a tuple whose element [1] is attention weights of shape
    (batch, heads, seq_q, seq_k). Some modules return attn_weights=None
    if output_attentions wasn't threaded through — we handle that case
    by raising downstream.
    """

    def __init__(self) -> None:
        self.attn: Tensor | None = None

    def __call__(self, _m: Any, _i: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            self.attn = None
            return
        # attn shape: (batch, heads, seq_q, seq_k)
        self.attn = output[1][0].detach().to(torch.float32).cpu()


def capture_full_sequence(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    layer_idxs: list[int],
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    prompt_len: int = 0,
) -> FullSequenceCapture:
    """Run one forward pass with hooks on the specified layers + the
    final layer's self_attn; return per-layer hidden states and
    last-token attention weights.

    ``prompt_len`` is stored in the result for bookkeeping — lets
    downstream distinguish prompt vs generated token positions.
    """
    if not layer_idxs:
        raise ValueError("capture_full_sequence: must specify at least one layer index")

    hidden_hooks: dict[int, _LayerHiddenHook] = {}
    handles: list = []
    for idx in layer_idxs:
        hook = _LayerHiddenHook()
        hidden_hooks[idx] = hook
        handles.append(layers[idx].register_forward_hook(hook))

    final_layer = layers[-1]
    if not hasattr(final_layer, "self_attn"):
        for h in handles:
            h.remove()
        raise RuntimeError(
            f"final layer {type(final_layer).__name__} has no self_attn attribute; "
            f"attention-weighted capture unsupported on this architecture"
        )
    attn_hook = _AttentionWeightsHook()
    handles.append(final_layer.self_attn.register_forward_hook(attn_hook))

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    if attn_hook.attn is None:
        raise RuntimeError(
            "attention weights not captured — model's self_attn did not return "
            "attn_weights even with output_attentions=True. Check model config "
            "or switch to manual attention computation."
        )

    # Attention weights from final layer, last-token query, averaged over heads.
    # Shape: (heads, seq_q, seq_k) -> (seq_k,)
    attn_last = attn_hook.attn[:, -1, :].mean(dim=0).numpy()
    seq_len = int(input_ids.shape[1])

    per_layer: dict[int, LayerCapture] = {}
    for idx, hook in hidden_hooks.items():
        if hook.hidden is None:
            continue
        h = hook.hidden.numpy()  # (seq_len, hidden_dim)
        h_attn = (attn_last[:, None] * h).sum(axis=0)  # (hidden_dim,)
        per_layer[idx] = LayerCapture(
            layer_idx=idx,
            hidden_states=h,
            h_first=h[0],
            h_last=h[-1],
            h_attn_weighted=h_attn,
        )

    return FullSequenceCapture(
        layers=per_layer,
        attn_weights_last=attn_last,
        seq_len=seq_len,
        prompt_len=prompt_len,
        input_ids=input_ids[0].detach().cpu().numpy().astype(np.int64),
    )


def build_full_input_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    generated_text: str,
) -> tuple[Tensor, int]:
    """Reconstruct the full (prompt + generated) token sequence that
    was processed during generation, plus the prompt length.

    Apply the chat template to ``messages`` for the prompt half, then
    tokenize ``generated_text`` as the assistant continuation. Return
    a (1, seq_len) LongTensor and the prompt length.
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
    gen_ids = tokenizer(generated_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    prompt_len = int(prompt_ids.shape[1])
    full = torch.cat([prompt_ids, gen_ids], dim=1)
    return full, prompt_len
