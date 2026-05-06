"""Per-canonical-kaomoji bag-of-lexicon (BoL) vectors.

Reads ``data/harness/claude_descriptions.jsonl`` (the flat HF-corpus
output of ``scripts/harness/60_corpus_pull.py``). For each canonical
kaomoji, builds a 48-d weighted indicator over the locked llmoji v2
LEXICON by count-weighted pooling of every per-bundle ``synthesis``
pick, L1-normalized so each row reads as a soft distribution over the
lexicon.

Pre-2026-05-06 this script (``62_corpus_embed.py``) ran every
synthesized prose description through MiniLM and pooled the resulting
384-d embeddings — a noisy reconstruction of structure that was
already present in the v2 ``synthesis`` object. Post-refactor we
consume the structured commit directly. See
``llmoji_study.lexicon`` for the canonical 48-word index + Russell-
quadrant tags, and ``llmoji_study.claude_faces.embed_lexicon_bags``
for the pooler.

Faces with zero v2 descriptions (legacy v1.x bundles only) are
dropped — BoL is undefined for free-form prose. The resulting parquet
is the input for: scripts/harness/55_bol_encoder.py (face_likelihood
TSV), scripts/harness/63_corpus_pca.py (PCA + clustering),
scripts/66_per_project_quadrants.py (BoL fallback for in-the-wild
faces), scripts/67_wild_residual.py (BoL geometry).

Usage:
    python scripts/harness/62_corpus_lexicon.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.claude_faces import (
    embed_lexicon_bags,
    load_descriptions,
    save_bol_parquet,
)
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_LEXICON_BAG_PATH,
    DATA_DIR,
)
from llmoji_study.lexicon import (
    LEXICON_VERSION,
    bol_modal_quadrant,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--primary-weight", type=float, default=1.0,
        help="weight per primary_affect pick (default: 1.0)",
    )
    ap.add_argument(
        "--extension-weight", type=float, default=0.5,
        help=(
            "weight per stance_modality_function pick (default: 0.5; "
            "primary picks are 1-3 per cell, extensions are 3-5, so the "
            "ratio defines how much primaries dominate the pool)"
        ),
    )
    args = ap.parse_args()

    if not CLAUDE_DESCRIPTIONS_PATH.exists():
        print(
            f"no corpus at {CLAUDE_DESCRIPTIONS_PATH}; "
            "run scripts/harness/60_corpus_pull.py first"
        )
        sys.exit(1)

    print(f"loading corpus from {CLAUDE_DESCRIPTIONS_PATH}...")
    rows = load_descriptions(CLAUDE_DESCRIPTIONS_PATH)
    n_descs = sum(len(r.get("descriptions", [])) for r in rows)
    n_v2_descs = sum(
        sum(1 for d in r.get("descriptions", []) if isinstance(d.get("synthesis"), dict))
        for r in rows
    )
    print(
        f"  {len(rows)} canonical kaomoji, {n_descs} per-bundle descriptions "
        f"({n_v2_descs} v2)"
    )
    if not rows:
        print("nothing to bag.")
        return

    print(
        f"building BoL vectors (lexicon_version={LEXICON_VERSION}, "
        f"primary_weight={args.primary_weight}, "
        f"extension_weight={args.extension_weight})..."
    )
    fw, n, n_v2_per_face, B = embed_lexicon_bags(
        rows,
        primary_weight=args.primary_weight,
        extension_weight=args.extension_weight,
    )
    print(f"  {len(fw)} kaomoji bagged, dim={B.shape[1]}")
    if not len(fw):
        print("nothing kept — every face is v1-only legacy.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_bol_parquet(fw, n, n_v2_per_face, B, CLAUDE_FACES_LEXICON_BAG_PATH)
    print(f"wrote {CLAUDE_FACES_LEXICON_BAG_PATH}")

    # Quick sanity: top-10 by emit count + their inferred Russell quadrant.
    print("\ntop 10 by emit count (face → modal quadrant from BoL):")
    order = sorted(
        range(len(fw)), key=lambda i: -int(n[i]),
    )
    for i in order[:10]:
        q = bol_modal_quadrant(B[i]) or "-"
        print(
            f"  {int(n[i]):5d}  {fw[i]:<14}  "
            f"({int(n_v2_per_face[i])} v2 descs -> {q})"
        )

    # Coverage diagnostic — what fraction of v2 description rows
    # made it into the bag (i.e. weren't dropped for synthesizer
    # drift or empty picks).
    n_kept_v2 = int(n_v2_per_face.sum())
    if n_v2_descs > 0:
        print(
            f"\nv2 description coverage: {n_kept_v2}/{n_v2_descs} "
            f"({100 * n_kept_v2 / n_v2_descs:.1f}%) "
            "- skipped rows had no in-lexicon picks"
        )


if __name__ == "__main__":
    main()
