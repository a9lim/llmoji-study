"""Per-source kaomoji scrape → JSONL, plus a merged view.

Two sources, two files; merged `claude_kaomoji.jsonl` is the
concatenation. Hook rows now carry full `assistant_text` (the unified
hook + retroactive backfill — see `scripts/21_backfill_journals.py`),
so the merge is no longer text-rich-only and downstream embed /
Haiku-describe / eriskii consumers see the full picture.

    data/claude_kaomoji_export.jsonl   Claude.ai export
    data/claude_kaomoji_hook.jsonl     ~/.claude + ~/.codex unified journal
    data/claude_kaomoji.jsonl          export + hook (merged)

Sources are independently regeneratable; default reruns both
(they're cheap). The journal is the single source of truth for every
agent assistant turn — historical transcripts are replayed into it
once via the backfill script, then live hooks append from there.

Post v1.0 split: source adapters live in the `llmoji` PyPI package
under `llmoji.sources.*`. The hook source used to be a Claude+Codex-
hardcoded `iter_claude_hook`; now it's a generic `iter_journal(path,
source=...)` that we call once per provider here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.scrape import ScrapeRow
from llmoji.sources.claude_export import iter_claude_export
from llmoji.sources.journal import iter_journal
from llmoji_study.config import (
    CLAUDE_AI_EXPORT_DIRS,
    CLAUDE_HOOK_JOURNAL_CLAUDE,
    CLAUDE_HOOK_JOURNAL_CODEX,
    CLAUDE_KAOMOJI_EXPORT_PATH,
    CLAUDE_KAOMOJI_HOOK_PATH,
    CLAUDE_KAOMOJI_PATH,
    DATA_DIR,
)


def _iter_export() -> Iterator[ScrapeRow]:
    return iter_claude_export(CLAUDE_AI_EXPORT_DIRS)


def _iter_hook() -> Iterator[ScrapeRow]:
    """Concatenate the two configured hook journals (Claude + Codex)
    using the package's generic journal iterator."""
    yield from iter_journal(CLAUDE_HOOK_JOURNAL_CLAUDE, source="claude")
    yield from iter_journal(CLAUDE_HOOK_JOURNAL_CODEX, source="codex")


ALL_SOURCES = ("export", "hook")

SOURCES: dict[str, tuple[Path, Callable[[], Iterator[ScrapeRow]]]] = {
    "export": (CLAUDE_KAOMOJI_EXPORT_PATH, _iter_export),
    "hook": (CLAUDE_KAOMOJI_HOOK_PATH, _iter_hook),
}


def _scrape_one(name: str) -> int:
    """Write data/claude_kaomoji_<name>.jsonl. Returns row count."""
    out_path, iter_fn = SOURCES[name]
    t0 = time.time()
    n = 0
    by_subsource: dict[str, int] = {}
    with out_path.open("w") as f:
        for row in iter_fn():
            f.write(json.dumps(row.to_dict()) + "\n")
            n += 1
            by_subsource[row.source] = by_subsource.get(row.source, 0) + 1
            if n % 2000 == 0:
                print(f"  {name}: {n} rows, {time.time() - t0:.1f}s elapsed")
    dt = time.time() - t0
    print(f"  {name}: wrote {n} rows to {out_path.name} in {dt:.1f}s")
    for sub, k in sorted(by_subsource.items()):
        if sub != name:
            print(f"    {sub}: {k}")
    return n


def _merge(merged: Path) -> int:
    """Concatenate every per-source file into the merged view."""
    n = 0
    with merged.open("w") as out:
        for src in ALL_SOURCES:
            path, _ = SOURCES[src]
            if not path.exists():
                continue
            for line in path.open():
                line = line.rstrip("\n")
                if not line:
                    continue
                out.write(line + "\n")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sources",
        nargs="*",
        choices=ALL_SOURCES + ("all",),
        default=list(ALL_SOURCES),
        help=(
            f"sources to (re)scrape. default: {' '.join(ALL_SOURCES)}. "
            "`all` is a synonym."
        ),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="skip rewriting claude_kaomoji.jsonl (the merged view).",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    requested = list(dict.fromkeys(  # dedup, preserve order
        ALL_SOURCES if "all" in args.sources else args.sources
    ))

    print(f"scraping: {' '.join(requested)}")
    counts: dict[str, int] = {}
    for name in requested:
        counts[name] = _scrape_one(name)

    if not args.no_merge:
        n_merged = _merge(CLAUDE_KAOMOJI_PATH)
        print(f"merged: wrote {n_merged} rows to {CLAUDE_KAOMOJI_PATH.name}")

    total = sum(counts.values())
    print(f"\ndone. {total} per-source rows across {len(counts)} sources:")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
