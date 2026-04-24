"""Unified scrape: Claude Code sessions + Claude.ai export → JSONL.

One line per kaomoji-bearing assistant message. Not incremental —
re-runs overwrite. Cheap: ~2000 files × a few ms = under a minute.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_scrape import iter_all
from llmoji.config import CLAUDE_KAOMOJI_PATH, DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_total = 0
    n_by_source: dict[str, int] = {}
    with CLAUDE_KAOMOJI_PATH.open("w") as f:
        for row in iter_all():
            f.write(json.dumps(row.to_dict()) + "\n")
            n_total += 1
            n_by_source[row.source] = n_by_source.get(row.source, 0) + 1
            if n_total % 2000 == 0:
                print(f"  ... {n_total} rows, {time.time() - t0:.1f}s elapsed")
    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s; wrote {n_total} rows to {CLAUDE_KAOMOJI_PATH}")
    for src, n in sorted(n_by_source.items()):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
