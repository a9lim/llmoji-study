"""Analysis over saved hidden states: post-hoc probe computation and
cosine similarity in hidden-state space.

The motivating argument for the whole refactor is that saklas's 5
bipolar probes collapse to ~1 valence direction (PC1 = 89-95% of
variance across all our data), so probe-score-cosine throws away most
of the structure in the underlying representations. Cosine on raw
hidden states (~4096-dim in gemma-4-31b-it) preserves the full
activation signature — whatever structure exists in the model's
internal state is available for analysis.

This module provides the basic primitives:

  load_row_hidden(row, data_dir) -> FullSequenceCapture
      load the sidecar for a row (looked up by row_uuid).

  recompute_probe_scores(capture, session) -> dict[probe_name, float]
      apply current probes to saved hidden states. Used by the smoke
      test to verify round-trip vs. on-the-fly scores.

  pooled_kaomoji_cosine_from_hidden(df, data_dir, *, which='h_last',
                                    layer=None) -> DataFrame
      per-(kaomoji, source) cosine similarity matrix, computed from
      saved hidden states rather than probe scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hidden_capture import FullSequenceCapture
from .hidden_state_io import hidden_state_path, load_hidden_states


WHICH_SNAPSHOTS = ("h_first", "h_last", "h_mean")


def load_row_hidden(
    row: dict[str, Any],
    data_dir: Path,
    experiment: str,
) -> FullSequenceCapture:
    """Load hidden-state sidecar for a JSONL row. Expects ``row_uuid`` key."""
    uuid = row.get("row_uuid")
    if not uuid:
        raise KeyError("row has no row_uuid; probably pre-refactor data")
    path = hidden_state_path(data_dir, experiment, uuid)
    if not path.exists():
        raise FileNotFoundError(f"no sidecar at {path}")
    return load_hidden_states(path)


def recompute_probe_scores(
    capture: FullSequenceCapture,
    session: Any,
    *,
    which: str = "h_last",
) -> dict[str, float]:
    """Run saklas's own probe-scoring on saved hidden states for one
    snapshot (h_first / h_last / h_attn_weighted). Used to verify the
    sidecars faithfully reproduce on-the-fly probe scores.

    The snapshot must be at (or above) the layers saklas's probes
    target; ``capture.layers`` covers those by construction."""
    import torch

    if which not in WHICH_SNAPSHOTS:
        raise ValueError(f"which must be one of {WHICH_SNAPSHOTS}")

    # Build the {layer_idx: Tensor(hidden_dim,)} dict saklas expects.
    hidden_dict: dict[int, Any] = {}
    for idx, lc in capture.layers.items():
        arr = getattr(lc, which)
        hidden_dict[idx] = torch.from_numpy(arr)

    monitor = session._monitor
    scores = monitor.score_single_token(hidden_dict)
    return {str(name): float(v) for name, v in scores.items()}


def stack_snapshot(
    captures: list[FullSequenceCapture],
    *,
    which: str = "h_last",
    layer: int | None = None,
) -> np.ndarray:
    """Stack one snapshot across rows into (n_rows, hidden_dim).

    ``layer`` selects which probe layer to read from. Default picks
    the highest layer index in the first capture (closest to the
    output) which is usually where affect probes live."""
    if not captures:
        return np.zeros((0, 0), dtype=np.float32)
    if layer is None:
        layer = max(captures[0].layers.keys())
    out = []
    for cap in captures:
        if layer not in cap.layers:
            raise KeyError(
                f"layer {layer} missing from a capture; "
                f"available: {sorted(cap.layers.keys())}"
            )
        out.append(getattr(cap.layers[layer], which))
    return np.asarray(out, dtype=np.float32)


def cosine_similarity_matrix(
    X: np.ndarray, *, center: bool = True,
) -> np.ndarray:
    """Pairwise cosine similarity of rows in X, optionally centered
    on the column mean first. Same recipe we use on probe vectors —
    centering removes the shared-baseline direction that dominates
    uncentered cosine across activations."""
    if center and len(X) > 0:
        X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Xn = X / norms
    return Xn @ Xn.T
