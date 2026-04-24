# Claude Faces — Scrape + Eriskii-Style Cluster Plot

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scrape every kaomoji-bearing assistant message from (a) local Claude Code sessions in `~/.claude/projects/` and (b) the Claude.ai desktop/web export at `/Users/a9lim/Downloads/data-72de1230-b9fa-4c55-bc10-84a35b58d89c-1776479747-1b0e6bd8-batch-0000/`, unify them into one JSONL, then produce an eriskii.net-style plot: t-SNE scatter of unique kaomoji, colored by cluster, with cluster labels, sized by frequency.

**Architecture:** Two stages. Stage 1 (Tasks 1–8) is data-only: flat modules for two source adapters, a unified iterator, and a scrape driver. Stage 2 (Tasks 9–13) is ML: response-based per-kaomoji embeddings via sentence-transformers, t-SNE reduction, HDBSCAN + k-means clustering, static matplotlib scatter. Completely independent from the emotional-battery experiment — no shared files except `llmoji/taxonomy.py::extract` (reused for balanced-paren span detection, no taxonomy labels needed since eriskii didn't use labels either).

**Tech stack:** Python 3, `sentence-transformers` (new dep, pulls torch which is already installed via saklas), sklearn (`TSNE`, `HDBSCAN`, `KMeans`), matplotlib. Optional: plotly for an interactive hover version. No API keys required — all local.

**Testing convention:** Same as the emotional-battery plan — sanity-check-and-smoke-run in lieu of unit tests, per CLAUDE.md's no-tests stance.

---

## File Structure

**Create:**

- `llmoji/llmoji/claude_scrape.py` — `ScrapeRow` dataclass, format-neutral types
- `llmoji/llmoji/claude_code_source.py` — walk `~/.claude/projects`, emit `ScrapeRow`
- `llmoji/llmoji/claude_export_source.py` — parse `conversations.json`, emit `ScrapeRow`
- `llmoji/llmoji/claude_faces.py` — embedding + t-SNE + clustering + plot
- `llmoji/scripts/05_claude_vocab_sample.py` — first-word frequency sweep
- `llmoji/scripts/06_claude_scrape.py` — unified scrape → `data/claude_kaomoji.jsonl`
- `llmoji/scripts/07_claude_kaomoji_basics.py` — frequency tables + per-source/model/project splits
- `llmoji/scripts/08_claude_faces_embed.py` — per-kaomoji embeddings → `data/claude_faces_embed.parquet`
- `llmoji/scripts/09_claude_faces_plot.py` — t-SNE + clustering + scatter → `figures/claude_faces_tsne.png`
- `llmoji/data/claude_kaomoji.jsonl` — gitignored
- `llmoji/data/claude_vocab_sample.tsv` — gitignored
- `llmoji/data/claude_faces_embed.parquet` — gitignored
- `llmoji/figures/claude_faces_tsne.png`
- `llmoji/figures/claude_faces_interactive.html` — optional

**Modify:**

- `llmoji/llmoji/config.py` — add path constants for Claude data sources and outputs
- `llmoji/pyproject.toml` — add `sentence-transformers` dependency

**Do not touch:**
Everything from the emotional-battery experiment. No shared data, no shared modules.

---

## Task 1: Config + dependency

**Files:**
- Modify: `llmoji/pyproject.toml`
- Modify: `llmoji/llmoji/config.py`

- [ ] **Step 1: Add `sentence-transformers` to dependencies**

In `llmoji/pyproject.toml`, add to the `dependencies` list:

```toml
dependencies = [
    "saklas>=1.4.6",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "sentence-transformers",
]
```

- [ ] **Step 2: Install the new dep**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && pip install -e . 2>&1 | tail -5`

Expected: `Successfully installed sentence-transformers-...` (or `Requirement already satisfied` if already present).

- [ ] **Step 3: Add config constants**

Append to `llmoji/config.py`:

```python
# --- claude-faces experiment (scrape + t-SNE cluster plot) ---
CLAUDE_CODE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
CLAUDE_AI_EXPORT_DIR = Path(
    "/Users/a9lim/Downloads/data-72de1230-b9fa-4c55-bc10-84a35b58d89c"
    "-1776479747-1b0e6bd8-batch-0000"
)
CLAUDE_KAOMOJI_PATH = DATA_DIR / "claude_kaomoji.jsonl"
CLAUDE_VOCAB_SAMPLE_PATH = DATA_DIR / "claude_vocab_sample.tsv"
CLAUDE_FACES_EMBED_PATH = DATA_DIR / "claude_faces_embed.parquet"
```

- [ ] **Step 4: Verify imports**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "from llmoji.config import CLAUDE_CODE_PROJECTS_DIR, CLAUDE_AI_EXPORT_DIR, CLAUDE_KAOMOJI_PATH; print(CLAUDE_CODE_PROJECTS_DIR.exists(), CLAUDE_AI_EXPORT_DIR.exists())"`

Expected: `True True`.

- [ ] **Step 5: Commit**

```bash
git add llmoji/config.py pyproject.toml
git commit -m "config: add claude-faces paths + sentence-transformers dep"
```

---

## Task 2: Unified `ScrapeRow` schema

**Files:**
- Create: `llmoji/llmoji/claude_scrape.py`

- [ ] **Step 1: Create the module with the shared dataclass only (adapters come in Tasks 3–4, iterator in Task 5)**

```python
"""Unified kaomoji-scrape schema across Claude data sources.

Two concrete sources emit ScrapeRow instances:
  - claude_code_source.py: ~/.claude/projects/**/*.jsonl
  - claude_export_source.py: Claude.ai export conversations.json

Kaomoji extraction uses llmoji.taxonomy.extract (balanced-paren span
fallback; no dialect-specific dict required — unlike the gemma pilot,
the eriskii-style analysis clusters on unique strings, not pre-
registered labels).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator


@dataclass
class ScrapeRow:
    """One kaomoji-bearing assistant message, source-agnostic."""

    # --- provenance ---
    source: str                 # "claude-code" | "claude-ai-export"
    session_id: str             # cc sessionId or webapp conversation uuid
    project_slug: str           # "-Users-a9lim-Work-llmoji" or conversation name
    assistant_uuid: str         # the assistant message uuid
    parent_uuid: str | None     # for cc: parentUuid; for webapp: parent_message_uuid

    # --- context ---
    model: str | None           # cc: message.model; webapp: None (not in export)
    timestamp: str              # ISO-8601
    cwd: str | None             # cc only
    git_branch: str | None      # cc only
    turn_index: int             # 0-based position in session
    had_thinking: bool          # did the assistant turn include a thinking block

    # --- content ---
    assistant_text: str         # full assistant message text
    first_word: str             # leading balanced-paren span or first whitespace-word
    kaomoji: str | None         # taxonomy-registered form (usually None for Claude)
    kaomoji_label: int          # +1/-1/0 (usually 0 since taxonomy is gemma-tuned)

    # --- upstream ---
    surrounding_user: str       # preceding user-authored text, resolved via parent chain

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Source adapters yield ScrapeRow. The unified iterator in
# llmoji.claude_scrape.iter_all chains the two sources together.


def iter_all() -> Iterator[ScrapeRow]:
    """Yield ScrapeRow from both Claude Code and Claude.ai export."""
    from .claude_code_source import iter_claude_code
    from .claude_export_source import iter_claude_export
    yield from iter_claude_code()
    yield from iter_claude_export()
```

- [ ] **Step 2: Verify the module imports**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "from llmoji.claude_scrape import ScrapeRow; import dataclasses; print([f.name for f in dataclasses.fields(ScrapeRow)])"`

Expected: prints the field list including `source`, `session_id`, `first_word`, `surrounding_user`.

- [ ] **Step 3: Commit**

```bash
git add llmoji/claude_scrape.py
git commit -m "claude-scrape: ScrapeRow schema + iter_all entry point"
```

---

## Task 3: Claude Code source adapter

**Files:**
- Create: `llmoji/llmoji/claude_code_source.py`

- [ ] **Step 1: Create the adapter**

Design notes:
- Walk every `*.jsonl` under `CLAUDE_CODE_PROJECTS_DIR`.
- Each file is one session. Read the whole file into memory first (sessions are small, tens to hundreds of KB typically) so we can build a uuid→event index for parentUuid resolution.
- For each `type=="assistant"` event: iterate `message.content` blocks, concatenate all `{type: "text"}` blocks' `text` fields into `assistant_text`; set `had_thinking = any(b.type == "thinking")`.
- If `assistant_text` is empty, skip (pure tool-use turn).
- `surrounding_user` resolution: walk `parentUuid` chain backward. A useful "user-authored" parent is a `type=="user"` event whose `message.content` has a `text` block (not a tool_result). Fall back to empty string if nothing found in 5 hops.
- Project slug = `file.parent.name`.

```python
"""Claude Code source adapter: ~/.claude/projects/**/*.jsonl.

Each JSONL file is one session. Events are threaded via parentUuid.
We emit one ScrapeRow per assistant message that contains at least one
text content block and whose first_word matches a kaomoji prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .claude_scrape import ScrapeRow
from .taxonomy import extract

# Same glyph set used by analysis.plot_kaomoji_heatmap and
# emotional_analysis._kaomoji_rows.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def _collect_text_and_thinking(message: dict[str, Any]) -> tuple[str, bool]:
    """Concatenate text blocks, return (assistant_text, had_thinking)."""
    parts: list[str] = []
    had_thinking = False
    for block in message.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            txt = block.get("text") or ""
            if txt:
                parts.append(txt)
        elif t == "thinking":
            had_thinking = True
    return "\n".join(parts), had_thinking


def _resolve_user_text(
    start_uuid: str | None,
    by_uuid: dict[str, dict[str, Any]],
    max_hops: int = 5,
) -> str:
    """Walk parentUuid backward to find the nearest user-authored text.

    Skips tool_result parents (which have type=='user' but content is
    machine-generated). Returns '' if nothing found within max_hops.
    """
    uuid = start_uuid
    for _ in range(max_hops):
        if uuid is None:
            return ""
        ev = by_uuid.get(uuid)
        if ev is None:
            return ""
        if ev.get("type") == "user":
            m = ev.get("message", {})
            content = m.get("content") if isinstance(m, dict) else None
            # user events come in two shapes: plain string content, or
            # list-of-blocks (tool_result). We want plain string.
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                # look for any "text"-type block; skip tool_result blocks
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        txt = b.get("text") or ""
                        if txt.strip():
                            return txt
        uuid = ev.get("parentUuid")
    return ""


def _iter_session(path: Path) -> Iterator[ScrapeRow]:
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    by_uuid: dict[str, dict[str, Any]] = {
        ev["uuid"]: ev for ev in events if isinstance(ev.get("uuid"), str)
    }
    # turn index: count assistant messages in order of appearance
    turn = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        m = ev.get("message", {})
        if not isinstance(m, dict):
            continue
        text, had_thinking = _collect_text_and_thinking(m)
        if not text.strip():
            continue
        match = extract(text)
        # only emit if first_word starts with a kaomoji-ish glyph
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            turn += 1
            continue
        user_text = _resolve_user_text(ev.get("parentUuid"), by_uuid)
        yield ScrapeRow(
            source="claude-code",
            session_id=str(ev.get("sessionId") or path.stem),
            project_slug=str(path.parent.name),
            assistant_uuid=str(ev.get("uuid") or ""),
            parent_uuid=ev.get("parentUuid"),
            model=str(m.get("model")) if m.get("model") else None,
            timestamp=str(ev.get("timestamp") or ""),
            cwd=str(ev.get("cwd")) if ev.get("cwd") else None,
            git_branch=str(ev.get("gitBranch")) if ev.get("gitBranch") else None,
            turn_index=turn,
            had_thinking=had_thinking,
            assistant_text=text,
            first_word=match.first_word,
            kaomoji=match.kaomoji,
            kaomoji_label=match.label,
            surrounding_user=user_text,
        )
        turn += 1


def iter_claude_code() -> Iterator[ScrapeRow]:
    """Yield every kaomoji-bearing assistant message from all Claude
    Code sessions on disk."""
    from .config import CLAUDE_CODE_PROJECTS_DIR
    root = Path(CLAUDE_CODE_PROJECTS_DIR)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        yield from _iter_session(path)
```

- [ ] **Step 2: Smoke-test on one real session file**

Run:
```bash
cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
from llmoji.claude_code_source import _iter_session
from pathlib import Path
import glob
files = sorted(glob.glob(str(Path.home() / '.claude/projects/*/*.jsonl')))
print(f'{len(files)} session files found')
# sample 3 random ones
import random
random.seed(0)
sample = random.sample(files, min(3, len(files)))
for f in sample:
    rows = list(_iter_session(Path(f)))
    print(f'  {Path(f).parent.name}/{Path(f).name}: {len(rows)} kaomoji rows')
    for r in rows[:2]:
        print(f'    turn={r.turn_index} model={r.model} first_word={r.first_word!r}')
        print(f'    user[:80]={r.surrounding_user[:80]!r}')
        print(f'    assistant[:80]={r.assistant_text[:80]!r}')
"
```

Expected: prints ~1800 files, sample of 3 yields at least some kaomoji rows (this depends on whether Claude was using kaomoji in those sessions; near-certain given your global instruction).

- [ ] **Step 3: Commit**

```bash
git add llmoji/claude_code_source.py
git commit -m "claude-scrape: Claude Code JSONL source adapter"
```

---

## Task 4: Claude.ai export source adapter

**Files:**
- Create: `llmoji/llmoji/claude_export_source.py`

Design notes from verified format:
- Top-level of `conversations.json`: list of conversation dicts.
- Each conversation: `{uuid, name, summary, created_at, chat_messages, ...}`.
- Each message: `{uuid, text, content, sender, created_at, parent_message_uuid, ...}`. `sender ∈ {"human", "assistant"}`.
- `text` (top-level on the message) is the **canonical text field** — content blocks often have empty `text` in this export format, but the top-level `text` has the full concatenated message.
- No model info in the export → `model=None`.
- Thinking blocks: not exposed in the export format — set `had_thinking=False` always.

- [ ] **Step 1: Create the adapter**

```python
"""Claude.ai export source adapter: conversations.json.

The export is a single JSON file: list of conversation objects, each
with a chat_messages array. The top-level .text field on each message
is the canonical content (content-block .text is often empty in this
format). Model info isn't in the export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .claude_scrape import ScrapeRow
from .taxonomy import extract

KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def _message_text(msg: dict[str, Any]) -> str:
    """Prefer top-level .text; fall back to content[].text blocks."""
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    parts: list[str] = []
    for block in msg.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            bt = block.get("text") or ""
            if bt.strip():
                parts.append(bt)
    return "\n".join(parts)


def _iter_conversation(conv: dict[str, Any]) -> Iterator[ScrapeRow]:
    msgs = conv.get("chat_messages") or []
    if not msgs:
        return
    # index by uuid for parent_message_uuid walks
    by_uuid: dict[str, dict[str, Any]] = {
        m["uuid"]: m for m in msgs if isinstance(m.get("uuid"), str)
    }
    session_id = str(conv.get("uuid") or "")
    project_slug = str(conv.get("name") or "") or "(unnamed)"
    turn = 0
    for m in msgs:
        if m.get("sender") != "assistant":
            continue
        text = _message_text(m)
        if not text.strip():
            continue
        match = extract(text)
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            turn += 1
            continue
        # Resolve user-surrounding text by walking parent_message_uuid
        # back to the nearest human message with non-empty text.
        user_text = ""
        parent_uuid = m.get("parent_message_uuid")
        cur = parent_uuid
        for _ in range(5):
            if not cur:
                break
            pm = by_uuid.get(cur)
            if pm is None:
                break
            if pm.get("sender") == "human":
                pt = _message_text(pm)
                if pt.strip():
                    user_text = pt
                    break
            cur = pm.get("parent_message_uuid")
        yield ScrapeRow(
            source="claude-ai-export",
            session_id=session_id,
            project_slug=project_slug,
            assistant_uuid=str(m.get("uuid") or ""),
            parent_uuid=parent_uuid,
            model=None,  # not in export
            timestamp=str(m.get("created_at") or ""),
            cwd=None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,  # export doesn't include thinking blocks
            assistant_text=text,
            first_word=match.first_word,
            kaomoji=match.kaomoji,
            kaomoji_label=match.label,
            surrounding_user=user_text,
        )
        turn += 1


def iter_claude_export() -> Iterator[ScrapeRow]:
    """Yield every kaomoji-bearing assistant message from the
    Claude.ai export conversations.json."""
    from .config import CLAUDE_AI_EXPORT_DIR
    path = Path(CLAUDE_AI_EXPORT_DIR) / "conversations.json"
    if not path.exists():
        return
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        return
    for conv in data:
        if isinstance(conv, dict):
            yield from _iter_conversation(conv)
```

- [ ] **Step 2: Smoke-test on the export**

Run:
```bash
cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
from llmoji.claude_export_source import iter_claude_export
from itertools import islice
rows = list(islice(iter_claude_export(), 10))
print(f'first 10 rows:')
for r in rows:
    print(f'  sess={r.session_id[:8]} first_word={r.first_word!r} user[:60]={r.surrounding_user[:60]!r}')
# full count
total = sum(1 for _ in iter_claude_export())
print(f'total: {total} kaomoji rows from claude.ai export')
"
```

Expected: prints 10 sample rows with kaomoji-looking first_words, then a total count (probably 50–1000 range depending on your history).

- [ ] **Step 3: Commit**

```bash
git add llmoji/claude_export_source.py
git commit -m "claude-scrape: Claude.ai export conversations.json adapter"
```

---

## Task 5: Vocab-sample script (reporting only, no manual taxonomy)

**Files:**
- Create: `llmoji/scripts/05_claude_vocab_sample.py`

Purpose: print the top-N first-word frequencies across both sources so you can eyeball the dialect. Unlike the gemma pilot's `00_vocab_sample.py`, this does NOT require you to lock in a taxonomy before scraping — eriskii clusters on unique strings, not taxonomy labels. Output is informational.

- [ ] **Step 1: Create the script**

```python
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
```

- [ ] **Step 2: Run it**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/05_claude_vocab_sample.py`

Expected: prints total count + top 25 first_words. Writes `data/claude_vocab_sample.tsv`. Eyeball the top 25 for anything weird (markdown bolding, plain English words slipping in, etc).

- [ ] **Step 3: Commit**

```bash
git add scripts/05_claude_vocab_sample.py
git commit -m "scripts: add 05_claude_vocab_sample"
```

---

## Task 6: Full scrape script

**Files:**
- Create: `llmoji/scripts/06_claude_scrape.py`

- [ ] **Step 1: Create the script**

```python
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
```

- [ ] **Step 2: Run the scrape**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/06_claude_scrape.py`

Expected: runs in under a minute, prints per-source counts. Verify `data/claude_kaomoji.jsonl` exists and is non-empty: `wc -l data/claude_kaomoji.jsonl`.

- [ ] **Step 3: Commit**

```bash
git add scripts/06_claude_scrape.py
git commit -m "scripts: add 06_claude_scrape (unified Claude Code + export → JSONL)"
```

---

## Task 7: Basic-stats script

**Files:**
- Create: `llmoji/scripts/07_claude_kaomoji_basics.py`

- [ ] **Step 1: Create the script**

```python
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

from llmoji.config import CLAUDE_KAOMOJI_PATH


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
```

- [ ] **Step 2: Run it**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/07_claude_kaomoji_basics.py`

Expected: prints totals, per-source counts, top-20 overall, per-model top-5, per-month timeline. Gives you a feel for the data before clustering.

- [ ] **Step 3: Commit**

```bash
git add scripts/07_claude_kaomoji_basics.py
git commit -m "scripts: add 07_claude_kaomoji_basics (descriptive stats)"
```

---

## Task 8: `claude_faces.py` — aggregation + embedding

**Files:**
- Create: `llmoji/llmoji/claude_faces.py`

Design: per unique `first_word`, aggregate all its assistant-text occurrences (with the kaomoji glyph removed to avoid trivial self-similarity) and compute a single mean sentence-embedding. Persist to parquet so the plot stage (Task 9) is cheap to iterate.

- [ ] **Step 1: Create the module**

```python
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportPrivateImportUsage=false
"""Per-kaomoji response-based embeddings.

For each unique first_word with count >= min_count, embed each
occurrence's assistant_text (with the leading kaomoji stripped) using
sentence-transformers/all-MiniLM-L6-v2, then mean-pool to a single
384-dim vector per kaomoji. Eriskii-style analysis consumes these.

Rationale for response-based (vs user-based): the user message is
short and varied; the assistant's own text around the kaomoji is
longer and carries the tonal context that drives which face was
chosen. This matches option (B) in the design sketch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass
class KaomojiEmbed:
    first_word: str
    n: int
    mean_embedding: np.ndarray  # shape (EMBED_DIM,)


def _strip_leading(text: str, kaomoji: str) -> str:
    """Remove a leading kaomoji occurrence from text so embeddings don't
    collapse on the literal face string."""
    stripped = text.lstrip()
    if stripped.startswith(kaomoji):
        return stripped[len(kaomoji):].lstrip()
    return stripped


def load_rows(path: Path) -> pd.DataFrame:
    """Load data/claude_kaomoji.jsonl into a DataFrame."""
    df: pd.DataFrame = pd.read_json(path, lines=True)
    return df


def compute_embeddings(
    df: pd.DataFrame,
    *,
    min_count: int = 3,
    batch_size: int = 64,
    device: str | None = None,
    progress: bool = True,
) -> list[KaomojiEmbed]:
    """For each first_word with >= min_count rows, mean-pool an
    embedding over its assistant_text occurrences (kaomoji stripped).

    Uses sentence-transformers on CPU by default. Pass device="mps"
    on Apple Silicon for speed.
    """
    from sentence_transformers import SentenceTransformer

    counts = df["first_word"].value_counts()
    keep = counts[counts >= min_count].index.tolist()
    if not keep:
        return []

    sub = df[df["first_word"].isin(keep)].copy()
    # strip leading kaomoji in place
    sub["stripped"] = [
        _strip_leading(str(t), str(fw))
        for t, fw in zip(sub["assistant_text"], sub["first_word"])
    ]

    model = SentenceTransformer(EMBED_MODEL, device=device)
    texts: list[str] = sub["stripped"].tolist()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        normalize_embeddings=True,
    )

    sub = sub.assign(emb_idx=range(len(sub)))
    out: list[KaomojiEmbed] = []
    for fw in keep:
        idx = sub.loc[sub["first_word"] == fw, "emb_idx"].to_numpy()
        if len(idx) == 0:
            continue
        mean = np.asarray(embs)[idx].mean(axis=0)
        # renormalize the mean so cosine comparisons are well-behaved
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        out.append(KaomojiEmbed(first_word=str(fw), n=int(len(idx)), mean_embedding=mean))
    return out


def save_embeddings(embeds: Iterable[KaomojiEmbed], path: Path) -> None:
    """Persist per-kaomoji embeddings to parquet."""
    rows = []
    for e in embeds:
        row = {"first_word": e.first_word, "n": e.n}
        for i, v in enumerate(e.mean_embedding.tolist()):
            row[f"e{i:03d}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (first_words, counts, embedding matrix)."""
    df: pd.DataFrame = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    E = df[emb_cols].to_numpy(dtype=float)
    return df["first_word"].tolist(), df["n"].to_numpy(dtype=int), E
```

- [ ] **Step 2: Smoke-test the strip + small-model call on a tiny slice**

Run:
```bash
cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
import pandas as pd
from llmoji.claude_faces import compute_embeddings
# fabricate a 9-row frame: 3 kaomoji × 3 texts each
rows = []
for fw in ['(◕‿◕)', '(｡•̀ᴗ-)', '(T_T)']:
    for txt in ['this is a short response.', 'another response of moderate length.', 'final one.']:
        rows.append({'first_word': fw, 'assistant_text': fw + ' ' + txt})
df = pd.DataFrame(rows)
embs = compute_embeddings(df, min_count=2, progress=False)
print(f'got {len(embs)} per-kaomoji embeddings')
for e in embs:
    print(f'  {e.first_word} n={e.n} shape={e.mean_embedding.shape} norm={float((e.mean_embedding**2).sum()**0.5):.3f}')
"
```

Expected: downloads `all-MiniLM-L6-v2` on first run (~80MB, happens once), then prints 3 embeddings with shape `(384,)` and norm ≈ 1.0. First run may take a minute for the download; subsequent runs are seconds.

- [ ] **Step 3: Commit**

```bash
git add llmoji/claude_faces.py
git commit -m "claude-faces: response-based per-kaomoji embeddings"
```

---

## Task 9: Embedding driver script

**Files:**
- Create: `llmoji/scripts/08_claude_faces_embed.py`

- [ ] **Step 1: Create the script**

```python
"""Driver: compute per-kaomoji response-based embeddings from the
scrape, persist to parquet.

Usage:
  python scripts/08_claude_faces_embed.py [--min-count N] [--device mps]

Defaults: min_count=5 (aggressive; eriskii needed 519 kaomoji from a
larger corpus, we'll likely see fewer), device="mps" if on Apple
Silicon else CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_faces import compute_embeddings, load_rows, save_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_PATH, CLAUDE_KAOMOJI_PATH, DATA_DIR


def _default_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--device", type=str, default=_default_device())
    args = ap.parse_args()

    if not CLAUDE_KAOMOJI_PATH.exists():
        print(f"no scrape at {CLAUDE_KAOMOJI_PATH}; run scripts/06_claude_scrape.py first")
        return
    print(f"loading {CLAUDE_KAOMOJI_PATH}")
    df = load_rows(CLAUDE_KAOMOJI_PATH)
    print(f"  {len(df)} rows; {df['first_word'].nunique()} distinct first_words")

    print(f"computing embeddings (device={args.device}, min_count={args.min_count})...")
    embeds = compute_embeddings(df, min_count=args.min_count, device=args.device)
    print(f"  {len(embeds)} kaomoji embedded")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeds, CLAUDE_FACES_EMBED_PATH)
    print(f"wrote {CLAUDE_FACES_EMBED_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on real scrape output**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/08_claude_faces_embed.py`

Expected: loads the scrape, prints row count + distinct-kaomoji count, embeds, writes parquet. On M-series CPU: a minute or two for a few thousand kaomoji occurrences.

- [ ] **Step 3: Commit**

```bash
git add scripts/08_claude_faces_embed.py
git commit -m "scripts: add 08_claude_faces_embed"
```

---

## Task 10: t-SNE + clustering + scatter plot

**Files:**
- Create: `llmoji/scripts/09_claude_faces_plot.py`

Design:
- Load per-kaomoji embeddings (parquet, 384-dim).
- t-SNE to 2D (`sklearn.manifold.TSNE`, `perplexity=min(30, N/4)` for robustness on small N).
- Two clusterings side-by-side panels:
  - **HDBSCAN** — no k, finds natural clusters, noise points shown in gray.
  - **KMeans(k=15)** — eriskii-parity.
- Scatter: point size ∝ log(count); color by cluster; annotate top-30 points with their kaomoji strings; cluster-centroid text labels show cluster id.
- Save a static PNG and (optional) an interactive HTML via plotly (for hover-tooltips like eriskii's).

- [ ] **Step 1: Create the script**

```python
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false
"""Eriskii-style Claude-faces plot.

Panel A: t-SNE + HDBSCAN auto-clustering (noise in gray).
Panel B: t-SNE + KMeans(k=15) for eriskii parity.
Both panels: top-30 most-frequent kaomoji annotated; point size ~
log(count); cluster-centroid id labels.

Also writes figures/claude_faces_interactive.html (plotly) with
hover-tooltips showing kaomoji + count + cluster id, matching eriskii's
interactive explorer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_faces import load_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_PATH, FIGURES_DIR


def _use_cjk_font() -> None:
    import matplotlib
    import matplotlib.font_manager as fm
    preferred = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Maru Gothic ProN",
        "Apple Color Emoji", "Noto Sans CJK JP", "Yu Gothic", "MS Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return


def _tsne_2d(E: np.ndarray, *, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = len(E)
    perplexity = max(5, min(30, (n - 1) // 4))
    model = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity,
        init="pca", learning_rate="auto", random_state=seed,
    )
    return model.fit_transform(E)


def _hdbscan(E: np.ndarray) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    model = HDBSCAN(metric="cosine", min_cluster_size=3, min_samples=2)
    return model.fit_predict(E)


def _kmeans(E: np.ndarray, *, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    if len(E) <= k:
        # degenerate: one cluster per point
        return np.arange(len(E))
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return model.fit_predict(E)


def _plot_panel(
    ax,
    xy: np.ndarray,
    labels: list[str],
    counts: np.ndarray,
    clusters: np.ndarray,
    *,
    title: str,
    annotate_top: int = 30,
):
    import matplotlib.pyplot as plt

    uniq = sorted(set(int(c) for c in clusters))
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors  # 40 colors
    cluster_color = {c: ("#bbbbbb" if c == -1 else palette[i % len(palette)])
                     for i, c in enumerate(uniq)}
    colors = [cluster_color[int(c)] for c in clusters]

    # size by log(count), floor at 15, cap at 250
    sizes = 15 + 60 * np.log1p(counts)
    sizes = np.clip(sizes, 15, 250)

    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    # annotate top-N most frequent
    top_idx = np.argsort(-counts)[:annotate_top]
    for i in top_idx:
        ax.annotate(
            labels[i], xy=(xy[i, 0], xy[i, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#222",
        )

    # cluster id at each cluster centroid (skip -1 noise)
    for c in uniq:
        if c == -1:
            continue
        mask = clusters == c
        cx = float(xy[mask, 0].mean())
        cy = float(xy[mask, 1].mean())
        ax.text(cx, cy, str(c), fontsize=11, fontweight="bold",
                color="#111", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=cluster_color[c], alpha=0.9))

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])


def _write_interactive(
    xy: np.ndarray, labels: list[str], counts: np.ndarray, clusters: np.ndarray,
    out_path: Path,
) -> None:
    """Optional plotly HTML. Falls back silently if plotly isn't installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  (plotly not installed; skipping {out_path.name})")
        return
    hover_text = [
        f"{lab}  n={n}  cluster={c}"
        for lab, n, c in zip(labels, counts, clusters)
    ]
    fig = go.Figure(data=[
        go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode="markers",
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                size=np.clip(6 + 4 * np.log1p(counts), 6, 30).tolist(),
                color=clusters.tolist(),
                colorscale="Turbo",
                showscale=True,
                line=dict(color="white", width=0.4),
            ),
        )
    ])
    fig.update_layout(
        title="Claude faces — t-SNE (hover for kaomoji + cluster)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000, height=800,
    )
    fig.write_html(str(out_path))
    print(f"  wrote {out_path}")


def main() -> None:
    if not CLAUDE_FACES_EMBED_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_PATH}; run scripts/08_claude_faces_embed.py first")
        return
    _use_cjk_font()
    labels, counts, E = load_embeddings(CLAUDE_FACES_EMBED_PATH)
    print(f"loaded {len(labels)} kaomoji embeddings, dim={E.shape[1]}")

    if len(labels) < 3:
        print("need at least 3 kaomoji to plot; exiting")
        return

    import matplotlib.pyplot as plt

    print("computing t-SNE...")
    xy = _tsne_2d(E)
    print("computing HDBSCAN...")
    clusters_hdb = _hdbscan(E)
    print("computing KMeans(k=15)...")
    clusters_km = _kmeans(E, k=15)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _plot_panel(
        axes[0], xy, labels, counts, clusters_hdb,
        title=f"HDBSCAN (auto-k; {len(set(int(c) for c in clusters_hdb) - {-1})} clusters + noise)",
    )
    _plot_panel(
        axes[1], xy, labels, counts, clusters_km,
        title="KMeans (k=15, eriskii parity)",
    )
    fig.suptitle(
        f"Claude-faces t-SNE ({len(labels)} kaomoji, response-based embedding)",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = FIGURES_DIR / "claude_faces_tsne.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    out_html = FIGURES_DIR / "claude_faces_interactive.html"
    _write_interactive(xy, labels, counts, clusters_km, out_html)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/09_claude_faces_plot.py`

Expected: loads parquet, prints t-SNE/clustering progress, writes `figures/claude_faces_tsne.png` (two-panel) and optionally the plotly HTML. Eyeball: look for clear cluster structure in the KMeans panel, noise points in gray in the HDBSCAN panel.

- [ ] **Step 3: Commit**

```bash
git add scripts/09_claude_faces_plot.py
git commit -m "scripts: add 09_claude_faces_plot (t-SNE + HDBSCAN + KMeans panels)"
```

---

## Task 11: Commit the plan doc

- [ ] **Step 1: Commit**

```bash
git add docs/superpowers/plans/2026-04-23-claude-faces-scrape-and-cluster.md
git commit -m "plan: claude faces scrape + eriskii-style cluster plot"
```

---

## Self-review notes

**Spec coverage:**
- Scope B (both sources) → Tasks 3 (Claude Code) + 4 (Claude.ai export) + unified iterator in Task 2
- Response-based embedding → Task 8 `_strip_leading` + `compute_embeddings` on `assistant_text`
- Eriskii-style plot → Task 10 two-panel t-SNE figure, plus optional plotly HTML
- Scrape-first, explore-later → Tasks 1–7 fully independent of Stage 2; you could stop after Task 7 and still have a useful dataset
- Full plan (Stage 1 + Stage 2) → 11 tasks total

**Placeholder scan:** No TBDs, no "implement appropriate X", no "similar to Task N." Every task has full code or explicit shell commands.

**Type consistency:** `ScrapeRow` fields are referenced by name in both source adapters; `assistant_uuid`/`parent_uuid`/`first_word` spellings match across Tasks 2–4 and downstream in Tasks 8–10. `CLAUDE_FACES_EMBED_PATH` / `CLAUDE_KAOMOJI_PATH` / `CLAUDE_VOCAB_SAMPLE_PATH` defined once in Task 1 and referenced everywhere.

**Conventions:** Flat modules under `llmoji/` (matching the pilot's layout), numbered scripts (`05`–`09` continuing the existing `00`–`04` series), pyright pragma at the top of files that handle pandas/sklearn for the same stubs-noise reason `analysis.py` does.

**Known approximations:**
- Claude.ai export has no per-message model info → `model=None` for those rows (Task 4). Per-model analysis in Task 7 explicitly filters to `source == "claude-code"`.
- Claude.ai export has no thinking-block markers → `had_thinking=False` for those rows. Fine; just a data-source caveat to record in the readme if you want to write this up.
- Eriskii's cluster labels ("Warm reassuring support" etc) are NOT generated by this plan — only numeric cluster ids. Adding LLM-generated labels is a sensible follow-up but not in scope here.
