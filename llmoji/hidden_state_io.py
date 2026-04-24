"""Per-row .npz sidecar I/O for saved hidden states.

Each captured generation writes one .npz at
``data/hidden/<experiment>/<row_uuid>.npz``. Keys in the archive:

  layer_idxs           (N,) int64   — captured layer indices
  seq_len              ()  int64    — total token count (prompt + gen)
  prompt_len           ()  int64    — prompt token count
  input_ids            (seq_len,) int64
  attn_weights_last    (seq_len,) float32
  h_first_L<idx>       (hidden_dim,) float32
  h_last_L<idx>        (hidden_dim,) float32
  h_attn_L<idx>        (hidden_dim,) float32
  hidden_L<idx>        (seq_len, hidden_dim) float32  — full per-token trace

One file per generation keeps writes incremental (safe to append to a
run without rewriting any existing file), random access via UUID
lookup, and the filesystem handles the indexing. Uses ``savez_compressed``
for ~2x space savings at minimal load-time cost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hidden_capture import FullSequenceCapture, LayerCapture


def save_hidden_states(
    capture: FullSequenceCapture,
    out_path: str | Path,
    *,
    store_full_trace: bool = True,
) -> None:
    """Write a ``FullSequenceCapture`` to an .npz sidecar.

    Pass ``store_full_trace=False`` to drop the (seq_len, hidden_dim)
    per-layer tensor and keep only the three aggregates — 60x smaller
    for typical sequences, useful when you know post-hoc probe
    computation only needs endpoints + attention-weighted pool."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "layer_idxs": np.array(sorted(capture.layers.keys()), dtype=np.int64),
        "seq_len": np.int64(capture.seq_len),
        "prompt_len": np.int64(capture.prompt_len),
        "input_ids": capture.input_ids,
        "attn_weights_last": capture.attn_weights_last.astype(np.float32),
    }
    for idx, lc in capture.layers.items():
        payload[f"h_first_L{idx}"] = lc.h_first.astype(np.float32)
        payload[f"h_last_L{idx}"] = lc.h_last.astype(np.float32)
        payload[f"h_attn_L{idx}"] = lc.h_attn_weighted.astype(np.float32)
        if store_full_trace:
            payload[f"hidden_L{idx}"] = lc.hidden_states.astype(np.float32)

    np.savez_compressed(out_path, **payload)


def load_hidden_states(
    in_path: str | Path,
    *,
    full_trace: bool = True,
) -> FullSequenceCapture:
    """Load a ``FullSequenceCapture`` from an .npz sidecar.

    If ``full_trace=False`` or the file was saved without full traces,
    ``LayerCapture.hidden_states`` is reconstructed as a length-2 stack
    of (h_first, h_last) so downstream code can still index [0] and
    [-1]; any mid-sequence indexing will silently return the wrong
    thing, so check ``seq_len == 2`` if you care."""
    data = np.load(in_path)
    layer_idxs = [int(x) for x in data["layer_idxs"]]
    seq_len = int(data["seq_len"])
    prompt_len = int(data["prompt_len"])

    layers: dict[int, LayerCapture] = {}
    for idx in layer_idxs:
        h_first = data[f"h_first_L{idx}"]
        h_last = data[f"h_last_L{idx}"]
        h_attn = data[f"h_attn_L{idx}"]
        full_key = f"hidden_L{idx}"
        if full_trace and full_key in data.files:
            hidden_states = data[full_key]
        else:
            hidden_states = np.stack([h_first, h_last])
        layers[idx] = LayerCapture(
            layer_idx=idx,
            hidden_states=hidden_states,
            h_first=h_first,
            h_last=h_last,
            h_attn_weighted=h_attn,
        )

    return FullSequenceCapture(
        layers=layers,
        attn_weights_last=data["attn_weights_last"],
        seq_len=seq_len,
        prompt_len=prompt_len,
        input_ids=data["input_ids"],
    )


def hidden_state_path(data_dir: Path, experiment: str, row_uuid: str) -> Path:
    """Canonical sidecar path: ``<data_dir>/hidden/<experiment>/<uuid>.npz``."""
    return data_dir / "hidden" / experiment / f"{row_uuid}.npz"
