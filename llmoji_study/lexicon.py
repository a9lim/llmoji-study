"""Bag-of-lexicon (BoL) representation of v2+ ``synthesis`` rows.

The ``llmoji`` v2 bundle replaced free-form prose synthesis with a
structured pick from a locked 48-word LEXICON. Each lexicon word
carries two tags: a Russell quadrant (HP/LP/HN-D/HN-S/LN/NB or
``None``) and a family (``circumplex``/``stance``/``modality``/
``functional``/``confidence``).

This module is the canonical research-side accessor for that
structure. Helpers here:

  - :data:`LEXICON_WORDS` — sorted list of all 48 words (stable
    column order for the BoL parquet).
  - :data:`WORD_TO_INDEX` / :data:`WORD_TO_QUADRANT` /
    :data:`WORD_TO_FAMILY` — lookups.
  - :data:`QUADRANT_INDICES` — quadrant → indices of its anchor words
    in :data:`LEXICON_WORDS`. Drives :func:`bol_to_quadrant_distribution`.
  - :func:`bol_from_synthesis` — turn a single per-bundle synthesis
    dict into a 48-d weighted vector.
  - :func:`pool_bol` — count-weighted mean across per-bundle vectors
    for a single canonical face.
  - :func:`bol_to_quadrant_distribution` — collapse a 48-d BoL onto
    the 6 Russell quadrants using only circumplex slots.
  - :func:`assert_lexicon_v1` — refuse to mix lexicon versions.

Why this matters: 19 of the 48 lexicon words are explicit Russell-
quadrant anchors, so the synthesizer's structured commit *is* a
6-d quadrant distribution per face — no encoder, no projection, no
post-hoc inference. That replaces the MiniLM → 21-axis eriskii
projection that previously stood in for "what does this face mean".

When ``LEXICON_VERSION`` rotates (v3+), this module needs the new
lexicon's quadrant tags. Consumers should call :func:`assert_lexicon_v1`
on every parquet read so silent mixing fails loud.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from llmoji.synth_prompts import (
    CIRCUMPLEX_ANCHORS,
    EXTENSION_AXES,
    LEXICON,
    LEXICON_VERSION,
)


# Sorted stable column order. The (word, quadrant, family) tuples in
# llmoji.synth_prompts.LEXICON are unordered; we sort by word so the
# BoL parquet's column ordering is deterministic across rebuilds.
LEXICON_WORDS: list[str] = sorted(item[0] for item in LEXICON)
N_LEXICON = len(LEXICON_WORDS)
assert N_LEXICON == 48, f"unexpected lexicon size: {N_LEXICON} (v1 has 48)"

WORD_TO_INDEX: dict[str, int] = {w: i for i, w in enumerate(LEXICON_WORDS)}

# Quadrant tag per word; ``None`` for the 29 extension words.
WORD_TO_QUADRANT: dict[str, str | None] = {
    item[0]: item[1] for item in LEXICON
}

# Family tag per word ('circumplex' / 'stance' / 'modality' /
# 'functional' / 'confidence'). All 48 words have a family tag.
WORD_TO_FAMILY: dict[str, str] = {item[0]: item[2] for item in LEXICON}

# Sanity: ``CIRCUMPLEX_ANCHORS`` (the 19 quadrant-tagged words) and
# ``EXTENSION_AXES`` (the 29 untagged stance/modality/etc) should
# partition the lexicon. Catch a future drift loud.
_circumplex_set = set(CIRCUMPLEX_ANCHORS)
_extension_set = set(EXTENSION_AXES)
assert _circumplex_set | _extension_set == set(LEXICON_WORDS), (
    "CIRCUMPLEX_ANCHORS ∪ EXTENSION_AXES != LEXICON_WORDS — lexicon drift"
)
assert _circumplex_set & _extension_set == set(), (
    "CIRCUMPLEX_ANCHORS ∩ EXTENSION_AXES non-empty — overlapping families"
)
assert all(WORD_TO_QUADRANT[w] is not None for w in _circumplex_set)
assert all(WORD_TO_QUADRANT[w] is None for w in _extension_set)

# Quadrant → list of indices into LEXICON_WORDS for the words that
# anchor that quadrant. Drives bol_to_quadrant_distribution() and the
# script-55 BoL encoder.
QUADRANTS: tuple[str, ...] = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
QUADRANT_INDICES: dict[str, list[int]] = {q: [] for q in QUADRANTS}
for w in _circumplex_set:
    q = WORD_TO_QUADRANT[w]
    assert q in QUADRANT_INDICES, f"unknown quadrant tag {q!r} for {w!r}"
    QUADRANT_INDICES[q].append(WORD_TO_INDEX[w])
for q in QUADRANTS:
    QUADRANT_INDICES[q].sort()
    # The v1 lexicon has 3 / 5 / 3 / 3 / 3 / 2 anchor words per
    # quadrant (19 total). NB is the smallest at 2 (`neutral`,
    # `detached`); the per-quadrant prior in the BoL→quadrant softmax
    # accounts for this.
assert sum(len(v) for v in QUADRANT_INDICES.values()) == len(_circumplex_set)


def assert_lexicon_v1(version: int | None) -> None:
    """Refuse to consume a parquet stamped with a non-v1 lexicon.

    BoL columns are ordered + interpreted under v1 quadrant tags. If
    a future v2+ lexicon rotates the vocabulary or shuffles anchor
    sets, every consumer downstream needs re-validation. Hard-fail
    on read so we never silently mix.
    """
    if version is None:
        raise ValueError(
            "BoL parquet is missing 'lexicon_version' — refuse to load "
            "(produced by a pre-BoL pipeline)"
        )
    if version != LEXICON_VERSION:
        raise ValueError(
            f"BoL parquet stamped lexicon_version={version} but the "
            f"current llmoji.synth_prompts is {LEXICON_VERSION}; refuse "
            "to mix lexicon versions in one analysis"
        )


def bol_from_synthesis(
    synthesis: dict | None,
    *,
    primary_weight: float = 1.0,
    extension_weight: float = 0.5,
) -> np.ndarray:
    """Turn one per-bundle synthesis dict into a 48-d weighted indicator.

    The synthesizer commits 1-3 ``primary_affect`` words and 3-5
    ``stance_modality_function`` words per face per bundle. We treat
    primaries as twice as informative as extensions by default
    (``primary_weight=1.0`` vs ``extension_weight=0.5``) — the
    ratio is what matters; absolute scale is normalized away
    downstream.

    Returns a zero vector iff ``synthesis`` is missing or non-v2.
    """
    bag = np.zeros(N_LEXICON, dtype=float)
    if not isinstance(synthesis, dict):
        return bag
    for w in synthesis.get("primary_affect") or []:
        idx = WORD_TO_INDEX.get(w)
        if idx is None:
            # Unknown word — synthesizer drift outside the locked
            # lexicon. Skip silently; caller can warn if it cares.
            continue
        bag[idx] += primary_weight
    for w in synthesis.get("stance_modality_function") or []:
        idx = WORD_TO_INDEX.get(w)
        if idx is None:
            continue
        bag[idx] += extension_weight
    return bag


def pool_bol(
    per_bundle_bols: Iterable[np.ndarray],
    weights: Iterable[float] | None = None,
    *,
    l1_normalize: bool = True,
) -> np.ndarray:
    """Count-weighted pool of per-bundle BoL vectors → one 48-d face vector.

    Default weights = 1 per bundle (caller usually wants
    ``weights=[bundle_count_i]`` from the corpus). With
    ``l1_normalize=True``, the pooled vector is L1-normalized so it
    reads as a soft distribution over the lexicon.
    """
    bols = [np.asarray(b, dtype=float) for b in per_bundle_bols]
    if not bols:
        return np.zeros(N_LEXICON, dtype=float)
    stacked = np.stack(bols, axis=0)
    if weights is None:
        w = np.ones(len(bols), dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float)
        if w.shape[0] != stacked.shape[0]:
            raise ValueError(
                f"weights length {w.shape[0]} != n_bols {stacked.shape[0]}"
            )
    if w.sum() <= 0:
        return np.zeros(N_LEXICON, dtype=float)
    pooled = (stacked * w[:, None]).sum(axis=0)
    if l1_normalize:
        s = float(pooled.sum())
        if s > 0:
            pooled = pooled / s
    return pooled


def bol_to_quadrant_distribution(
    bol: np.ndarray,
    *,
    smooth: float = 0.0,
) -> np.ndarray:
    """Collapse a BoL vector onto the 6 Russell quadrants.

    For each quadrant, sum the BoL mass on its anchor words; then
    L1-normalize across quadrants. Extension words don't contribute —
    they're stance/modality/etc, not Russell-circumplex.

    ``smooth`` adds a uniform prior (Dirichlet-like) before
    normalization. Useful when many faces have only 1-2 primary
    picks (common in the long tail) and the resulting hard-one-hot
    distribution overstates confidence. Set to e.g. 0.05 to round
    edges; default 0 keeps the synthesizer's commit literal.

    Returns a 6-d vector in the order of :data:`QUADRANTS`. Zero
    vector iff the BoL has no circumplex mass.
    """
    bol = np.asarray(bol, dtype=float)
    out = np.zeros(len(QUADRANTS), dtype=float)
    for j, q in enumerate(QUADRANTS):
        out[j] = float(bol[QUADRANT_INDICES[q]].sum())
    if smooth > 0:
        out = out + float(smooth)
    s = float(out.sum())
    if s <= 0:
        return np.zeros(len(QUADRANTS), dtype=float)
    return out / s


def bol_modal_quadrant(
    bol: np.ndarray,
    *,
    smooth: float = 0.0,
) -> str | None:
    """argmax over :func:`bol_to_quadrant_distribution`. ``None`` iff
    the face has no circumplex commitment at all."""
    dist = bol_to_quadrant_distribution(bol, smooth=smooth)
    if dist.sum() <= 0:
        return None
    return QUADRANTS[int(np.argmax(dist))]


def top_lexicon_words(
    bol: np.ndarray,
    *,
    k: int = 5,
    min_weight: float = 0.0,
) -> list[tuple[str, float]]:
    """Return the top-k highest-weight lexicon words from a BoL.
    Used as deterministic cluster signatures (no Haiku call needed)."""
    bol = np.asarray(bol, dtype=float)
    order = np.argsort(-bol)
    out: list[tuple[str, float]] = []
    for i in order[:k]:
        w = float(bol[i])
        if w <= min_weight:
            break
        out.append((LEXICON_WORDS[i], w))
    return out


__all__ = [
    "LEXICON_VERSION",
    "LEXICON_WORDS",
    "N_LEXICON",
    "QUADRANTS",
    "QUADRANT_INDICES",
    "WORD_TO_FAMILY",
    "WORD_TO_INDEX",
    "WORD_TO_QUADRANT",
    "assert_lexicon_v1",
    "bol_from_synthesis",
    "bol_modal_quadrant",
    "bol_to_quadrant_distribution",
    "pool_bol",
    "top_lexicon_words",
]
