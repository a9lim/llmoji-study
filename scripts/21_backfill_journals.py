"""One-shot: replay Claude Code transcripts + Codex rollouts into the
unified kaomoji journals.

After this runs, ~/.claude/kaomoji-journal.jsonl and
~/.codex/kaomoji-journal.jsonl carry every assistant turn from
history alongside whatever the live Stop hooks log going forward —
single source of truth, no separate transcript scrape.

Re-runs OVERWRITE the journals. Pause active Claude / Codex sessions
before running; an in-flight turn could otherwise land in both the
backfill (via transcript) and the live hook (via Stop) within the
same second.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.backfill import backfill_claude_code, backfill_codex
from llmoji_study.config import CLAUDE_HOOK_JOURNAL_CLAUDE, CLAUDE_HOOK_JOURNAL_CODEX

CLAUDE_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"
CODEX_ROLLOUT_ROOT = Path.home() / ".codex" / "sessions"


def main() -> int:
    print(f"backfilling {CLAUDE_HOOK_JOURNAL_CLAUDE} from {CLAUDE_TRANSCRIPT_ROOT}")
    t0 = time.time()
    n_claude = backfill_claude_code(CLAUDE_TRANSCRIPT_ROOT, CLAUDE_HOOK_JOURNAL_CLAUDE)
    dt = time.time() - t0
    print(f"  claude: {n_claude} rows in {dt:.1f}s")

    print(f"backfilling {CLAUDE_HOOK_JOURNAL_CODEX} from {CODEX_ROLLOUT_ROOT}")
    t0 = time.time()
    n_codex = backfill_codex(CODEX_ROLLOUT_ROOT, CLAUDE_HOOK_JOURNAL_CODEX)
    dt = time.time() - t0
    print(f"  codex: {n_codex} rows in {dt:.1f}s")

    print(f"\ndone. {n_claude + n_codex} historical rows backfilled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
