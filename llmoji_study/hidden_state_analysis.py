# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
"""Analysis over saved hidden states: feature loading, group-mean
aggregation, cosine heatmaps, PCA scatters, and per-(kaomoji, source)
primitives used by analysis.py / emotional_analysis.py.

Motivating argument for the whole refactor: saklas's 5 bipolar probes
collapse to ~1 valence direction (PC1 = 89-95% of variance across
v1/v2/v3). Cosine on raw hidden states (~4096-dim in gemma-4-31b-it)
preserves the full activation signature — whatever structure exists
in the model's internal state is available for analysis.

Primitives are generic: they take a DataFrame of metadata + an
(n_rows, hidden_dim) feature matrix, and a groupby column. The
specific analysis modules compose these into their figures.

Layer selection: default is the highest captured probe layer
(closest to output; "the deepest affect-readable representation").
Override via ``layer=`` where needed.

Snapshot selection: default is ``h_last`` (matches saklas's
aggregate readout, which scores the last non-special generated token).
Use ``h_first`` for "state that produced the kaomoji" under kaomoji-
prompted conditions, or ``h_mean`` for a whole-generation summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hidden_capture import FullSequenceCapture
from .hidden_state_io import hidden_state_path, load_hidden_states


WHICH_SNAPSHOTS = ("h_first", "h_last", "h_mean")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_hidden_features(
    jsonl_path: str | Path,
    data_dir: Path,
    experiment: str,
    *,
    which: str = "h_first",
    layer: int | None = None,
    drop_errors: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a JSONL + its hidden-state sidecars into a (metadata df,
    feature matrix) pair.

    Rows without a ``row_uuid`` or missing sidecar files are dropped.
    ``layer=None`` picks the highest layer index present in the first
    loaded sidecar (typically the probe-set max, closest to output).

    Returns
    -------
    df : pd.DataFrame
        Metadata columns only; no probe-score columns loaded. Row order
        matches X row order.
    X : np.ndarray
        (n_rows, hidden_dim) fp32 matrix of the chosen snapshot at the
        chosen layer.

    The ``$LLMOJI_WHICH`` environment variable (h_first|h_last|h_mean)
    overrides ``which`` if set. Project-wide aggregate sweep hook.
    """
    import os
    env_which = os.environ.get("LLMOJI_WHICH")
    if env_which:
        if env_which not in WHICH_SNAPSHOTS:
            raise ValueError(
                f"LLMOJI_WHICH must be one of {WHICH_SNAPSHOTS}, got {env_which!r}"
            )
        which = env_which

    if which not in WHICH_SNAPSHOTS:
        raise ValueError(f"which must be one of {WHICH_SNAPSHOTS}")

    jsonl_path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_errors and "error" in r:
                continue
            rows.append(r)

    chosen_layer = layer
    features: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    missing = 0
    for r in rows:
        uuid = r.get("row_uuid", "")
        if not uuid:
            missing += 1
            continue
        sidecar = hidden_state_path(data_dir, experiment, uuid)
        if not sidecar.exists():
            missing += 1
            continue
        cap = load_hidden_states(sidecar, full_trace=False)
        if chosen_layer is None:
            chosen_layer = max(cap.layers.keys())
        lc = cap.layers.get(chosen_layer)
        if lc is None:
            missing += 1
            continue
        features.append(getattr(lc, which).astype(np.float32))
        kept.append(r)

    if missing:
        print(f"  [load_hidden_features] dropped {missing} rows with "
              f"no sidecar / missing layer")

    if not kept:
        return pd.DataFrame(), np.zeros((0, 0), dtype=np.float32)

    df = pd.DataFrame(kept)
    X = np.asarray(features, dtype=np.float32)
    return df.reset_index(drop=True), X


# ---------------------------------------------------------------------------
# Group aggregation
# ---------------------------------------------------------------------------


def group_mean_vectors(
    df: pd.DataFrame,
    X: np.ndarray,
    group_by: str | list[str],
    *,
    min_count: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Group rows of X by one or more DataFrame columns, return
    (keys_df, per-group-mean matrix, counts).

    keys_df has one row per surviving group with the group-by columns,
    aligned with M's row order. Groups with n < min_count are dropped.
    """
    if isinstance(group_by, str):
        group_cols = [group_by]
    else:
        group_cols = list(group_by)

    counts = df.groupby(group_cols).size()
    keep_idx = counts[counts >= min_count].index
    if len(keep_idx) == 0:
        empty_keys = pd.DataFrame({c: [] for c in group_cols})
        return empty_keys, np.zeros((0, X.shape[1] if X.size else 0), dtype=np.float32), pd.Series(dtype=int)

    # Build a single Series for group-key comparisons.
    if len(group_cols) == 1:
        row_keys = df[group_cols[0]]
    else:
        row_keys = pd.MultiIndex.from_frame(df[group_cols])

    keys_out: list[Any] = []
    means: list[np.ndarray] = []
    counts_kept: list[int] = []
    for key in keep_idx:
        if len(group_cols) == 1:
            mask = (df[group_cols[0]] == key).to_numpy()
        else:
            mask = np.asarray([row_keys[i] == key for i in range(len(df))])
        if not mask.any():
            continue
        keys_out.append(key)
        means.append(X[mask].mean(axis=0))
        counts_kept.append(int(mask.sum()))

    if len(group_cols) == 1:
        keys_df = pd.DataFrame({group_cols[0]: keys_out})
    else:
        # keys are tuples; expand
        rows = [dict(zip(group_cols, k)) for k in keys_out]
        keys_df = pd.DataFrame(rows)

    M = np.asarray(means, dtype=np.float32)
    counts_series = pd.Series(counts_kept, index=range(len(keys_out)), name="n")
    return keys_df, M, counts_series


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(
    X: np.ndarray, *, center: bool = True,
) -> np.ndarray:
    """Pairwise cosine similarity of rows in X. ``center=True``
    subtracts the column mean first — in hidden-state space there's
    still a shared response-baseline direction that uncentered cosine
    would be dominated by (same reasoning that applied to probe
    vectors, only more so given 4096 dims of room for a shared mean).
    """
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Xn = X / norms
    return Xn @ Xn.T


def cosine_to_mean(X: np.ndarray) -> np.ndarray:
    """For each row in X, its cosine similarity to the mean across
    rows. Used for within-group consistency distributions."""
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    mean_norm = np.linalg.norm(mean, axis=1)
    row_norms = np.linalg.norm(X, axis=1)
    denom = row_norms * mean_norm
    dots = (X * mean).sum(axis=1)
    out = np.divide(dots, denom, out=np.zeros_like(dots, dtype=np.float32), where=denom > 0)
    return out


# ---------------------------------------------------------------------------
# Probe recomputation (retained for validation / cross-check)
# ---------------------------------------------------------------------------


def recompute_probe_scores(
    capture: FullSequenceCapture,
    session: Any,
    *,
    which: str = "h_first",
) -> dict[str, float]:
    """Run saklas's own probe-scoring on saved hidden states for one
    snapshot. Used by the smoke test to verify sidecars faithfully
    reproduce on-the-fly probe scores."""
    import torch

    if which not in WHICH_SNAPSHOTS:
        raise ValueError(f"which must be one of {WHICH_SNAPSHOTS}")

    hidden_dict: dict[int, Any] = {}
    for idx, lc in capture.layers.items():
        arr = getattr(lc, which)
        hidden_dict[idx] = torch.from_numpy(arr)

    monitor = session._monitor
    scores = monitor.score_single_token(hidden_dict)
    return {str(name): float(v) for name, v in scores.items()}


# ---------------------------------------------------------------------------
# Multi-layer loading (for layer-wise emergence + per-layer CKA scripts)
# ---------------------------------------------------------------------------


def load_hidden_features_all_layers(
    jsonl_path: str | Path,
    data_dir: Path,
    experiment: str,
    *,
    which: str = "h_first",
    layers: list[int] | None = None,
    drop_errors: bool = True,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Load JSONL + sidecars and stack one snapshot per layer into a
    3D feature tensor. Returns ``(df, X3, layer_idxs)`` where
    ``X3.shape == (n_rows, n_layers, hidden_dim)``.

    The slow path is one ``np.load`` per row (44k for an 800-row run
    × 56 layers if we used ``load_hidden_features`` once per layer);
    this helper opens each sidecar exactly once. Optionally writes the
    result to ``cache_path`` (an .npz of the X3 tensor + a JSONL of
    row metadata) and reads it back on subsequent calls.

    ``layers=None`` uses every layer present in the first sidecar.
    Pass an explicit list to subset (e.g. every 4th layer).
    """
    import os
    env_which = os.environ.get("LLMOJI_WHICH")
    if env_which:
        if env_which not in WHICH_SNAPSHOTS:
            raise ValueError(
                f"LLMOJI_WHICH must be one of {WHICH_SNAPSHOTS}, got {env_which!r}"
            )
        which = env_which
        # Auto-rename cache path so we don't clobber the h_mean cache.
        # Convention: callers pass paths with "h_mean" in the filename;
        # we swap that to the active `which` so each aggregate gets its
        # own cache. Falls through if the substring isn't present.
        if cache_path is not None and "h_mean" in str(cache_path):
            cache_path = Path(str(cache_path).replace("h_mean", which))

    if which not in WHICH_SNAPSHOTS:
        raise ValueError(f"which must be one of {WHICH_SNAPSHOTS}")

    if cache_path is not None and cache_path.exists():
        # Stale-cache guard: if the source JSONL has more rows than the
        # cache (a generation run finished or extended after the cache
        # was built), invalidate. Matched-or-fewer counts are fine —
        # `drop_errors` and missing sidecars can legitimately reduce
        # cache row count below jsonl count, but the cache will never
        # exceed jsonl. See docs/gotchas.md "stale all-layers cache".
        n_jsonl = sum(1 for line in Path(jsonl_path).open() if line.strip())
        cached = np.load(cache_path)
        n_cache = int(cached["X3"].shape[0]) if "X3" in cached.files else 0
        if n_cache < n_jsonl:
            cache_path.unlink()
            meta_path = cache_path.with_suffix(".meta.jsonl")
            if meta_path.exists():
                meta_path.unlink()
            print(
                f"  [load_hidden_features_all_layers] cache stale "
                f"({n_cache} cached vs {n_jsonl} jsonl rows); rebuilding"
            )
        else:
            meta_path = cache_path.with_suffix(".meta.jsonl")
            df = pd.read_json(meta_path, lines=True) if meta_path.exists() else pd.DataFrame()
            layer_idxs = [int(x) for x in cached["layer_idxs"]]
            X3 = cached["X3"]
            return df, X3, layer_idxs

    jsonl_path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_errors and "error" in r:
                continue
            rows.append(r)

    chosen_layers: list[int] | None = layers
    per_row: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    missing = 0
    for r in rows:
        uuid = r.get("row_uuid", "")
        if not uuid:
            missing += 1
            continue
        sidecar = hidden_state_path(data_dir, experiment, uuid)
        if not sidecar.exists():
            missing += 1
            continue
        cap = load_hidden_states(sidecar, full_trace=False)
        if chosen_layers is None:
            chosen_layers = sorted(cap.layers.keys())
        try:
            stack = np.stack([
                getattr(cap.layers[L], which).astype(np.float32)
                for L in chosen_layers
            ])
        except KeyError:
            missing += 1
            continue
        per_row.append(stack)
        kept.append(r)

    if missing:
        print(f"  [load_hidden_features_all_layers] dropped {missing} rows "
              f"with no sidecar / missing layer")

    if not kept:
        return pd.DataFrame(), np.zeros((0, 0, 0), dtype=np.float32), []

    df = pd.DataFrame(kept).reset_index(drop=True)
    X3 = np.asarray(per_row, dtype=np.float32)
    layer_idxs = list(chosen_layers or [])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X3=X3,
            layer_idxs=np.array(layer_idxs, dtype=np.int64),
        )
        df.to_json(cache_path.with_suffix(".meta.jsonl"),
                   orient="records", lines=True)

    return df, X3, layer_idxs
