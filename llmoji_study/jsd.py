"""Information-theoretic helpers for distribution-vs-distribution comparison.

Lifted from `scripts/harness/25_groundtruth_compare_runs.py` and promoted
to a shared module so the post-hoc face_likelihood scripts (52-56) can
use the same JSD machinery for soft-everywhere evaluation.

Methodology shift (2026-05-04 late evening): the post-hoc evaluation
metric is JSD between the predictor's per-quadrant distribution and
Claude's empirical per-quadrant emission distribution per face. Hard
accuracy + κ stay as supplementary informational metrics, but the
*primary* metric is mean-JSD-vs-empirical.

Why: Claude's GT is itself a distribution (e.g. face emitted 8x HP /
7x LP / 0 elsewhere). Treating it as one-hot via the modal-extraction
step is information loss; the predictor's softmax is a distribution
on the same space; comparing distribution-to-distribution is the
honest metric. See script 56's writeup for the full argument.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

QUADRANT_ORDER = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
LN2 = math.log(2.0)


def normalize(
    counts: dict[str, float],
    vocab: Iterable[str] = QUADRANT_ORDER,
    eps: float = 1e-6,
) -> list[float]:
    """Smoothed prob distribution over ``vocab`` from ``counts``.

    Adds ``eps`` to every vocab entry, then renormalizes — keeps JS finite
    even when one side has zero mass on a label that the other side has.
    Default vocab is the 6-way Russell quadrant order; pass a custom
    iterable for face-vocab use cases.
    """
    raw = [float(counts.get(v, 0)) + eps for v in vocab]
    total = sum(raw)
    return [x / total for x in raw]


def kl(p: list[float], q: list[float]) -> float:
    """KL(P || Q) in nats. Both must be > 0 (use ``normalize`` first)."""
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def js(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in nats.

    Bounded above by ln(2) ≈ 0.693. 0 means identical distributions.
    Symmetric: js(p, q) == js(q, p).
    """
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_quadrant(
    pred_dist: dict[str, float] | list[float],
    gt_dist: dict[str, float] | list[float],
    *,
    eps: float = 1e-6,
) -> float:
    """Convenience: JS in nats between two per-quadrant distributions.

    Each input may be a {quadrant: prob} dict (re-normalized via
    ``normalize`` over QUADRANT_ORDER) or a length-6 list aligned to
    QUADRANT_ORDER (used as-is, but smoothed via ``normalize`` to
    handle exact-zero entries).
    """
    if isinstance(pred_dist, dict):
        p = normalize(pred_dist, QUADRANT_ORDER, eps=eps)
    else:
        p = normalize(
            {q: float(pred_dist[i]) for i, q in enumerate(QUADRANT_ORDER)},
            QUADRANT_ORDER, eps=eps,
        )
    if isinstance(gt_dist, dict):
        g = normalize(gt_dist, QUADRANT_ORDER, eps=eps)
    else:
        g = normalize(
            {q: float(gt_dist[i]) for i, q in enumerate(QUADRANT_ORDER)},
            QUADRANT_ORDER, eps=eps,
        )
    return js(p, g)


def similarity(jsd_nats: float) -> float:
    """Normalized distribution-similarity score in [0, 1].

    ``1 - JSD/ln2``. 1.0 = identical distributions; 0.0 = maximally
    divergent. Useful as a human-readable companion to raw JSD.
    """
    return max(0.0, min(1.0, 1.0 - jsd_nats / LN2))
