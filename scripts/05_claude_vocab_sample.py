"""Frequency sweep over kaomoji-bearing first_words from both sources.

Writes data/claude_vocab_sample.tsv with columns
  first_word, count, example_user_snippet

Informational: the downstream scrape (06) and cluster plot (08-09)
don't depend on any manual taxonomy curation.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_scrape import iter_all
from llmoji.config import CLAUDE_VOCAB_SAMPLE_PATH, DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    total = 0
    for row in iter_all():
        counts[row.first_word] += 1
        total += 1
        if row.first_word not in examples and row.surrounding_user:
            examples[row.first_word] = row.surrounding_user[:120].replace("\n", " ")
    print(f"scanned {total} kaomoji-bearing assistant messages")
    print(f"{len(counts)} distinct first_words")

    with CLAUDE_VOCAB_SAMPLE_PATH.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["first_word", "count", "example_user_snippet"])
        for fw, n in counts.most_common():
            w.writerow([fw, n, examples.get(fw, "")])
    print(f"wrote {CLAUDE_VOCAB_SAMPLE_PATH}")

    print("\ntop 25:")
    for fw, n in counts.most_common(25):
        print(f"  {n:5d}  {fw}")


if __name__ == "__main__":
    main()
