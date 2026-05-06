"""Per-(canonical-kaomoji, source-model) bag-of-lexicon (BoL) vectors.

Long-format counterpart to ``scripts/harness/62_corpus_lexicon.py``.
Where 62 pools every face's per-bundle synthesis picks across source
models into one 48-d vector, this script keeps each face × source-model
cell separate. The same canonical face appears once per source model
that synthesized it.

Reads ``data/harness/claude_descriptions.jsonl`` (the flat HF-corpus
output of ``scripts/harness/60_corpus_pull.py``). Output:
``data/harness/claude_faces_lexicon_bag_per_source.parquet``.

Used by ``scripts/harness/69_per_source_drift.py`` to ask whether the
synthesis-vs-elicitation gap on faces like `(╯°□°)` is specific to
claude-opus-* deployment patterns or shared across providers
(gpt-5.x, codex-hook, etc).

Usage:
    python scripts/harness/64_corpus_lexicon_per_source.py
    python scripts/harness/64_corpus_lexicon_per_source.py --min-count 2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.claude_faces import (
    embed_lexicon_bags_per_source,
    load_descriptions,
    save_bol_parquet_per_source,
)
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_LEXICON_BAG_PER_SOURCE_PATH,
    DATA_DIR,
)
from llmoji_study.lexicon import (
    LEXICON_VERSION,
    bol_modal_quadrant,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary-weight", type=float, default=1.0)
    ap.add_argument("--extension-weight", type=float, default=0.5)
    ap.add_argument(
        "--min-count", type=int, default=1,
        help="drop (face, source_model) cells with total emit count "
             "below this (default: 1 — keep singletons; bump to 2+ "
             "for tighter cross-source comparisons)",
    )
    args = ap.parse_args()

    if not CLAUDE_DESCRIPTIONS_PATH.exists():
        sys.exit(
            f"no corpus at {CLAUDE_DESCRIPTIONS_PATH}; "
            "run scripts/harness/60_corpus_pull.py first"
        )

    print(f"loading corpus from {CLAUDE_DESCRIPTIONS_PATH}...")
    rows = load_descriptions(CLAUDE_DESCRIPTIONS_PATH)
    print(f"  {len(rows)} canonical kaomoji")

    print(
        f"building per-(face, source_model) BoL "
        f"(lexicon_version={LEXICON_VERSION}, "
        f"primary_weight={args.primary_weight}, "
        f"extension_weight={args.extension_weight}, "
        f"min_count={args.min_count})..."
    )
    faces, sms, counts, n_descs, B = embed_lexicon_bags_per_source(
        rows,
        primary_weight=args.primary_weight,
        extension_weight=args.extension_weight,
        min_count=args.min_count,
    )
    print(f"  {len(faces)} (face, source_model) cells, dim={B.shape[1]}")
    if not len(faces):
        print("nothing kept.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_bol_parquet_per_source(
        faces, sms, counts, n_descs, B,
        CLAUDE_FACES_LEXICON_BAG_PER_SOURCE_PATH,
    )
    print(f"wrote {CLAUDE_FACES_LEXICON_BAG_PER_SOURCE_PATH}")

    # Per-source-model coverage diagnostic.
    sm_counts: Counter = Counter()
    sm_emit: Counter = Counter()
    for i, sm in enumerate(sms):
        sm_counts[sm] += 1
        sm_emit[sm] += int(counts[i])

    print("\nper-source-model coverage:")
    print(f"  {'source_model':<32s}  {'cells':>6s}  {'emits':>6s}")
    for sm, n in sm_counts.most_common():
        print(f"  {sm:<32s}  {n:6d}  {sm_emit[sm]:6d}")

    # Sample: faces with the most source-model coverage (cross-source
    # candidates) plus their per-source modal quadrants.
    by_face: dict[str, list[tuple[str, int, str | None]]] = {}
    for i, face in enumerate(faces):
        modal = bol_modal_quadrant(B[i])
        by_face.setdefault(face, []).append((sms[i], int(counts[i]), modal))
    multi_source = sorted(
        ((f, items) for f, items in by_face.items() if len(items) >= 3),
        key=lambda kv: -sum(c for _, c, _ in kv[1]),
    )
    if multi_source:
        print(
            f"\ntop 10 faces by total emit across multi-source coverage "
            f"(face -> per-source modal Q):"
        )
        for face, items in multi_source[:10]:
            items_sorted = sorted(items, key=lambda t: -t[1])
            shown = ", ".join(
                f"{sm}:{q or '-'}({c})" for sm, c, q in items_sorted
            )
            print(f"  {face:<14}  {shown}")


if __name__ == "__main__":
    main()
