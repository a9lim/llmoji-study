"""Basic descriptive stats on data/claude_kaomoji.jsonl.

Prints:
  - Total rows, by source
  - Top-N first_words overall and per-source
  - Breakdown by model (Claude Code only — export has no model info)
  - Per-project top kaomoji (Claude Code)
  - Per-month emission timeline

Purely informational; doesn't write files. Saves an eyeballable
summary before we commit to the clustering pipeline.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import CLAUDE_KAOMOJI_PATH


def main() -> None:
    if not CLAUDE_KAOMOJI_PATH.exists():
        print(f"no data at {CLAUDE_KAOMOJI_PATH}; run scripts/06_claude_scrape.py first")
        return
    rows: list[dict] = [
        json.loads(line) for line in CLAUDE_KAOMOJI_PATH.read_text().splitlines()
        if line.strip()
    ]
    print(f"loaded {len(rows)} rows")

    by_src: Counter[str] = Counter(r["source"] for r in rows)
    print("\nby source:")
    for s, n in by_src.most_common():
        print(f"  {s}: {n}")

    print("\ntop 20 first_words overall:")
    overall = Counter(r["first_word"] for r in rows)
    for fw, n in overall.most_common(20):
        print(f"  {n:5d}  {fw}")

    print("\ntop 10 per source:")
    for src in ("claude-code", "claude-ai-export"):
        c = Counter(r["first_word"] for r in rows if r["source"] == src)
        if not c:
            continue
        print(f"  --- {src} ---")
        for fw, n in c.most_common(10):
            print(f"    {n:5d}  {fw}")

    print("\nby model (claude-code only):")
    m_counts: Counter[str] = Counter(
        r["model"] or "(unknown)" for r in rows if r["source"] == "claude-code"
    )
    for m, n in m_counts.most_common():
        print(f"  {m}: {n}")

    print("\nper-model top-5 kaomoji:")
    per_model: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r["source"] != "claude-code":
            continue
        per_model[r["model"] or "(unknown)"][r["first_word"]] += 1
    for m, c in sorted(per_model.items(), key=lambda kv: -sum(kv[1].values())):
        if sum(c.values()) < 20:
            continue
        print(f"  --- {m} ---")
        for fw, n in c.most_common(5):
            print(f"    {n:4d}  {fw}")

    print("\nper-month emission (yyyy-mm):")
    per_month: Counter[str] = Counter()
    for r in rows:
        ts = r.get("timestamp") or ""
        if len(ts) >= 7:
            per_month[ts[:7]] += 1
    for mm in sorted(per_month):
        print(f"  {mm}: {per_month[mm]}")


if __name__ == "__main__":
    main()
