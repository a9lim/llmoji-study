"""Per-face Claude-modal-quadrant ground truth from the groundtruth pilot.

Used by scripts 53 (subset search) and 56 (ensemble predict) when run
with ``--claude-gt``: replaces the pooled ``empirical_majority_quadrant``
(v3 + Claude + wild emit counts) with a Claude-only modal label.

Why: when the goal is to predict Claude's faces in production, GT
should be Claude's own modal quadrant — not a pooled measure that
mostly reflects v3 prompt distribution. Cuts the GT subset to
~25-51 faces (depending on canonicalization match + floor).

Note on canonicalization: ``claude_groundtruth_pilot.jsonl`` stores
the raw extracted ``first_word`` (no canonicalization), so loading
this map requires running ``canonicalize_kaomoji`` to match the
keys used in the face_likelihood summary TSVs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.config import DATA_DIR

DEFAULT_PILOT_PATH = DATA_DIR / "claude_groundtruth_pilot.jsonl"


def load_claude_gt(
    pilot_path: Path | None = None,
    *,
    floor: int = 1,
) -> dict[str, tuple[str, int]]:
    """Return ``{canonical_face: (modal_quadrant, modal_n_emits)}``.

    Faces with ``modal_n_emits < floor`` are excluded from the map.
    Default ``floor=1`` includes every face Claude emitted at least
    once; ``floor=2`` requires at least two emits in the modal
    quadrant (sharper labels but smaller N).
    """
    p = pilot_path or DEFAULT_PILOT_PATH
    counts: dict[str, Counter[str]] = {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            f = r.get("first_word", "")
            q = r.get("quadrant", "")
            if not f or not q:
                continue
            f_canon = canonicalize_kaomoji(f)
            counts.setdefault(f_canon, Counter())[q] += 1
    out: dict[str, tuple[str, int]] = {}
    for face, qmap in counts.items():
        modal_q, modal_n = qmap.most_common(1)[0]
        if modal_n >= floor:
            out[face] = (modal_q, modal_n)
    return out
