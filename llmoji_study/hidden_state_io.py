"""Per-row .npz sidecar I/O for saved hidden states.

Each captured generation writes one .npz at
``data/local/hidden/<experiment>/<row_uuid>.npz``. Keys in the archive:

  layer_idxs       (N,) int64         — captured layer indices
  n_tokens         ()  int64          — number of generated tokens
  h_first_L<idx>   (hidden_dim,) float32
  h_last_L<idx>    (hidden_dim,) float32
  h_mean_L<idx>    (hidden_dim,) float32
  hidden_L<idx>    (n_tokens, hidden_dim) float32  — full per-token trace

One file per generation keeps writes incremental (safe to append to a
run without rewriting any existing file), random access via UUID
lookup, and the filesystem handles the indexing. Uses
``savez_compressed`` for ~2x space savings at minimal load-time cost.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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

    Pass ``store_full_trace=False`` to drop the (n_tokens, hidden_dim)
    per-layer tensor and keep only the three aggregates — 60x smaller
    for typical sequences, useful when you know post-hoc probe
    computation only needs endpoints + mean-pool."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "layer_idxs": np.array(sorted(capture.layers.keys()), dtype=np.int64),
        "n_tokens": np.int64(capture.n_tokens),
    }
    for idx, lc in capture.layers.items():
        payload[f"h_first_L{idx}"] = lc.h_first.astype(np.float32)
        payload[f"h_last_L{idx}"] = lc.h_last.astype(np.float32)
        payload[f"h_mean_L{idx}"] = lc.h_mean.astype(np.float32)
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
    thing, so check ``n_tokens == 2`` if you care."""
    data = np.load(in_path)
    layer_idxs = [int(x) for x in data["layer_idxs"]]
    n_tokens = int(data["n_tokens"])

    layers: dict[int, LayerCapture] = {}
    for idx in layer_idxs:
        h_first = data[f"h_first_L{idx}"]
        h_last = data[f"h_last_L{idx}"]
        h_mean = data[f"h_mean_L{idx}"]
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
            h_mean=h_mean,
        )

    return FullSequenceCapture(layers=layers, n_tokens=n_tokens)


def hidden_state_path(data_dir: Path, experiment: str, row_uuid: str) -> Path:
    """Canonical sidecar path: ``<data_dir>/local/hidden/<experiment>/<uuid>.npz``.

    Post-2026-05-05 layout refactor: hidden states are unambiguously
    local-side (no harness equivalent), so they live under
    ``data/local/hidden/`` rather than ``data/local/hidden/``.
    """
    return data_dir / "local" / "hidden" / experiment / f"{row_uuid}.npz"


class SidecarWriter:
    """Single-thread background pool for ``save_hidden_states`` calls.

    ``savez_compressed`` is GIL-releasing CPU work that currently blocks
    the next row's prefill. Offloading it to one worker overlaps the
    write of row N with the generation of row N+1; serializing on a
    single thread avoids contention with main-thread numpy work.

    The capture object handed to ``submit`` is already a self-contained
    CPU numpy snapshot (``read_after_generate`` finished the device→host
    transfer before returning), so no GPU state crosses the thread
    boundary. With ``store_full_trace=False`` (the default), the
    snapshot is also small.

    Lifecycle: create one ``SidecarWriter`` per run; pass it to
    ``run_sample(..., sidecar_writer=writer)``; ``writer.close()`` (or
    use as a context manager) at run end to drain pending writes and
    re-raise the first exception. Wrap the run loop in ``try/finally``
    so the writer drains even on SIGINT.
    """

    def __init__(self, max_pending: int = 0) -> None:
        # max_workers=1: one worker is enough — savez_compressed is
        # GIL-releasing for the I/O part, and serializing avoids
        # contention. max_pending=0 means unbounded; in practice the
        # runner generates one row at a time so the queue stays small.
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidecar")
        self._pending: deque[Future] = deque()
        self._max_pending = max_pending
        self._closed = False

    def submit(
        self,
        capture: FullSequenceCapture,
        out_path: str | Path,
        *,
        store_full_trace: bool = True,
    ) -> Future:
        """Enqueue a sidecar write. Non-blocking. Returns the Future
        for callers that want it; otherwise the writer tracks it
        internally for drain-on-close.

        Periodically drains completed futures so the deque doesn't
        grow without bound — also surfaces exceptions early instead
        of holding them all until close().
        """
        if self._closed:
            raise RuntimeError("SidecarWriter is closed")

        # Drain any already-completed futures, raising the first error.
        # This makes failures visible promptly (next submit) rather
        # than only at run-end close.
        while self._pending and self._pending[0].done():
            f = self._pending.popleft()
            f.result()  # re-raises if the worker failed

        fut = self._exec.submit(
            save_hidden_states,
            capture, out_path,
            store_full_trace=store_full_trace,
        )
        self._pending.append(fut)
        return fut

    def close(self) -> None:
        """Drain all pending writes and shut down the executor.

        Re-raises the first exception encountered. Safe to call
        multiple times; subsequent calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        first_exc: BaseException | None = None
        try:
            while self._pending:
                f = self._pending.popleft()
                try:
                    f.result()
                except BaseException as e:
                    if first_exc is None:
                        first_exc = e
        finally:
            self._exec.shutdown(wait=True)
        if first_exc is not None:
            raise first_exc

    def __enter__(self) -> "SidecarWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
