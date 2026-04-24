# Eriskii-replication on Claude-faces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-24-eriskii-replication-design.md`

**Goal:** Replicate eriskii.net/projects/claude-faces' description-based embedding + axis-projection + cluster-labeling pipeline on our 436-row Claude-faces dataset, plus three breakouts the original couldn't do (per-model, per-project, surrounding-user → kaomoji axis correlation).

**Architecture:** Two new library modules (`llmoji/eriskii_prompts.py` for locked prompts/anchors, `llmoji/eriskii.py` for analysis primitives) + three new scripts (14, 15, 16) following the project's existing `llmoji/X.py` ↔ `scripts/N_X.py` pattern. All file writes resumable; no edits to existing 06–09 scripts.

**Tech Stack:** Python 3.14, `anthropic` SDK (new dependency, used for Haiku calls), `sentence-transformers/all-MiniLM-L6-v2` (existing), scikit-learn (TSNE, KMeans, existing), pandas + matplotlib + scipy.stats.

**Project conventions in force:**
- No formal pytest tests (per CLAUDE.md). Validation = run actual code on small inputs and inspect output.
- Single `.venv/`, pip not uv.
- Scripts directly executable (`./.venv/bin/python scripts/X.py`); the `sys.path.insert` at top of each is intentional.
- Pre-registered constants live in `config.py` / `eriskii_prompts.py`; changes invalidate the run.
- Commit-message style is conversational lowercase prefix (e.g. `eriskii: add ...`, `analysis: ...`).
- v3 trial is currently running in the background; this work writes to disjoint files (`data/claude_haiku_descriptions.jsonl`, `data/claude_faces_embed_description.parquet`, `data/eriskii_*.tsv`, `figures/eriskii_*`). Safe to do in parallel.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | add `anthropic` dependency |
| `llmoji/config.py` | modify | add `HAIKU_MODEL_ID`, `ERISKII_AXES`, output paths |
| `llmoji/eriskii_prompts.py` | create | locked `DESCRIBE_PROMPT`, `CLUSTER_LABEL_PROMPT`, `AXIS_ANCHORS` |
| `llmoji/eriskii.py` | create | analysis primitives: masking, Haiku description call, axis projection, cluster labeling, breakouts, user-kaomoji correlation |
| `scripts/14_claude_haiku_describe.py` | create | resumable description runner over `claude_kaomoji.jsonl` |
| `scripts/15_claude_faces_embed_description.py` | create | description-embedding driver |
| `scripts/16_eriskii_replication.py` | create | analysis driver (axes + clusters + breakouts + writeup) |
| `CLAUDE.md` | modify | document the new pipeline + outputs |

Single source of truth for axis anchor strings is `llmoji/eriskii_prompts.py::AXIS_ANCHORS`. Single source of truth for the locked Haiku prompts is the same module.

---

## Task 1: Add anthropic dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `anthropic` to dependencies**

Edit `pyproject.toml`, dependencies section. Insert `"anthropic",` between `"saklas>=1.4.6",` and `"numpy",`:

```toml
dependencies = [
    "saklas>=1.4.6",
    "anthropic",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "sentence-transformers",
    "pyarrow",
    "plotly",
]
```

- [ ] **Step 2: Install**

Run: `./.venv/bin/pip install -e .`
Expected: `Successfully installed ... anthropic-X.Y.Z ...` plus deps it pulls in. Existing packages unchanged.

- [ ] **Step 3: Verify import**

Run: `./.venv/bin/python -c "import anthropic; print(anthropic.__version__); print(anthropic.Anthropic)"`
Expected: prints version (e.g. `0.x.y`) and `<class 'anthropic.Anthropic'>`. Doesn't error.

- [ ] **Step 4: Verify the v3 process didn't crash**

Run: `ps -p 22239 -o pid,stat,etime,command 2>/dev/null || echo "v3 process gone"`
Expected: still `RN`/`UN` and ELAPSED still increasing. (pip install -e . is metadata-only for an already-installed editable package + adds anthropic; doesn't perturb the running process which already loaded saklas/torch.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add anthropic SDK for eriskii-replication pipeline"
```

---

## Task 2: Add config constants

**Files:**
- Modify: `llmoji/config.py`

- [ ] **Step 1: Add the constants**

Append to `llmoji/config.py` below the existing claude-faces section (after `CLAUDE_FACES_EMBED_PATH = ...`):

```python
# --- eriskii-replication experiment (description-based embeddings + axes) ---
# Locked Haiku version. Bumping invalidates description-corpus parity.
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"

# Stage-A sampling: per kaomoji, randomly sample up to this many
# instances for per-instance Haiku description (eriskii used 4 with
# a floor for low-frequency faces). Floor is implicit — kaomoji with
# fewer than the cap are fully sampled.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# Order matters: this is the column order in eriskii_axes.tsv and the
# heatmap-row order in per-model / per-project figures. Must stay in
# sync with llmoji.eriskii_prompts.AXIS_ANCHORS. All 21 axes from
# the eriskii.net page (note: "wryness" is the eriskii spelling, with
# one n).
ERISKII_AXES = [
    "warmth", "energy", "confidence", "playfulness", "empathy",
    "technicality", "positivity", "curiosity", "approval",
    "apologeticness", "decisiveness", "wryness", "wetness",
    "surprise", "anger", "frustration", "hatefulness", "sadness",
    "hope", "aggression", "exhaustion",
]

CLAUDE_HAIKU_DESCRIPTIONS_PATH = DATA_DIR / "claude_haiku_descriptions.jsonl"
CLAUDE_HAIKU_SYNTHESIZED_PATH = DATA_DIR / "claude_haiku_synthesized.jsonl"
CLAUDE_FACES_EMBED_DESCRIPTION_PATH = DATA_DIR / "claude_faces_embed_description.parquet"

# eriskii-replication output paths
ERISKII_AXES_TSV = DATA_DIR / "eriskii_axes.tsv"
ERISKII_CLUSTERS_TSV = DATA_DIR / "eriskii_clusters.tsv"
ERISKII_PER_MODEL_TSV = DATA_DIR / "eriskii_per_model.tsv"
ERISKII_PER_PROJECT_TSV = DATA_DIR / "eriskii_per_project.tsv"
ERISKII_USER_KAOMOJI_CORR_TSV = DATA_DIR / "eriskii_user_kaomoji_axis_corr.tsv"
ERISKII_COMPARISON_MD = DATA_DIR / "eriskii_comparison.md"
```

- [ ] **Step 2: Smoke-import**

Run: `./.venv/bin/python -c "from llmoji.config import HAIKU_MODEL_ID, ERISKII_AXES, ERISKII_AXES_TSV, INSTANCE_SAMPLE_CAP, CLAUDE_HAIKU_SYNTHESIZED_PATH; print(HAIKU_MODEL_ID); print(len(ERISKII_AXES)); print(ERISKII_AXES); print(ERISKII_AXES_TSV); print(INSTANCE_SAMPLE_CAP, CLAUDE_HAIKU_SYNTHESIZED_PATH)"`
Expected: prints model id, `21`, the 21-element list (with `wryness` not `wrynness`), the path, the sample cap (`4`), and the synthesized path.

- [ ] **Step 3: Commit**

```bash
git add llmoji/config.py
git commit -m "eriskii: add config constants (haiku id, 21 axes, sampling, output paths)"
```

**Note**: An earlier 11-axis version of `ERISKII_AXES` (with the typo
`wrynness`) was committed at SHA `6911d19` before the spec was
expanded to 21 axes. If `6911d19` is the current state, the
implementer should make a follow-up fix commit (no `--amend` per
project conventions) bringing config in line with the locked
21-axis list above. The previous commit stays in history; the new
commit message can be `eriskii: expand ERISKII_AXES to 21, fix
wryness typo`.

---

## Task 3: Locked prompts and anchor strings

**Files:**
- Create: `llmoji/eriskii_prompts.py`

- [ ] **Step 1: Write the module**

Create `llmoji/eriskii_prompts.py` with the locked prompts and anchor pairs from the spec §3.1, §4, §5. This is the single source of truth — never duplicate these strings elsewhere.

```python
"""Locked Haiku prompts and axis anchor strings for the eriskii
replication pipeline. Pre-registered in
docs/superpowers/specs/2026-04-24-eriskii-replication-design.md;
changing any string here invalidates the description corpus and
requires re-running scripts/14 onward.
"""

from __future__ import annotations

# --- Haiku prompts ---

# scripts/14 Stage A: per-instance masked-context description.
# Two variants — one when surrounding_user is non-empty (~73% of
# rows), one when it's empty. Both use the same {masked_text} key
# for the masked assistant turn; the user-context variant adds
# {user_text}. Sent as a single user message; no system prompt.
DESCRIBE_PROMPT_WITH_USER = (
    "The following is a turn from a conversation with an AI "
    "assistant. The user wrote the message at the top, and the "
    "assistant's response follows. The opening of the assistant's "
    "response originally began with a kaomoji (a Japanese-style "
    "emoticon) — we have replaced it with the literal token "
    "[FACE]. In one or two sentences, describe the mood, affect, "
    "or stance the assistant was conveying with the masked face. "
    "Do not speculate about which specific kaomoji it was; "
    "describe the state.\n\n"
    "User:\n{user_text}\n\n"
    "Assistant:\n{masked_text}\n\n"
    "Description:"
)

DESCRIBE_PROMPT_NO_USER = (
    "The following is a response from an AI assistant. The opening "
    "of the response originally began with a kaomoji (a "
    "Japanese-style emoticon) — we have replaced it with the "
    "literal token [FACE]. In one or two sentences, describe the "
    "mood, affect, or stance the assistant was conveying with the "
    "masked face. Do not speculate about which specific kaomoji "
    "it was; describe the state.\n\n"
    "Response:\n{masked_text}\n\n"
    "Description:"
)

# scripts/14 Stage B: per-kaomoji synthesis. Given a numbered list
# of per-instance descriptions for the same kaomoji, ask Haiku to
# synthesize a single one-sentence meaning. Mirrors eriskii's
# stage-B consolidation step.
SYNTHESIZE_PROMPT = (
    "Below are several short descriptions of the mood, affect, or "
    "stance an AI assistant was conveying when using a particular "
    "kaomoji at the start of different responses. Synthesize "
    "these into a single one- or two-sentence description that "
    "captures the kaomoji's overall meaning. Output only the "
    "synthesized description, no preamble.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Synthesized meaning:"
)

# scripts/16: per-cluster name. Given the member kaomoji + their
# synthesized descriptions, return a 3-5 word eriskii-style label.
CLUSTER_LABEL_PROMPT = (
    "Below is a cluster of kaomoji (Japanese-style emoticons) and "
    "short descriptions of what each conveys. Generate a single "
    "3-to-5-word label for this cluster that captures the shared "
    "mood, affect, or function. Examples of the desired label "
    "register: 'Warm reassuring support', 'Wry resignation', "
    "'Eager to help', 'Empathetic honesty'. Output only the label, "
    "no quotes, no prefix.\n\n"
    "Cluster members:\n{members}\n\n"
    "Label:"
)

# --- Semantic axes (locked anchor pairs from spec §4) ---
# Each axis: (positive_anchor, negative_anchor). Positive direction
# corresponds to the axis name (e.g. high "warmth" projection = warmer).
# Multi-word phrases by design — the embedding catches the concept
# rather than a single-word idiosyncrasy. All 21 axes from
# eriskii.net (note "wryness" is the eriskii spelling, single n).
AXIS_ANCHORS: dict[str, tuple[str, str]] = {
    "warmth": (
        "warm, caring, gentle, affectionate",
        "cold, clinical, detached, distant",
    ),
    "energy": (
        "energetic, animated, lively, excited",
        "subdued, calm, quiet, low-key",
    ),
    "confidence": (
        "confident, assured, decisive, sure",
        "uncertain, hesitant, tentative, unsure",
    ),
    "playfulness": (
        "playful, mischievous, fun, lighthearted",
        "serious, grave, solemn, formal",
    ),
    "empathy": (
        "empathetic, compassionate, understanding, supportive",
        "indifferent, dismissive, unsympathetic, callous",
    ),
    "technicality": (
        "technical, precise, analytical, methodical",
        "casual, conversational, loose, off-the-cuff",
    ),
    "positivity": (
        "happy, positive, cheerful, optimistic",
        "sad, negative, downcast, pessimistic",
    ),
    "curiosity": (
        "curious, inquisitive, interested, exploring",
        "bored, incurious, disengaged, uninterested",
    ),
    "approval": (
        "approving, encouraging, validating, supportive",
        "disapproving, critical, dismissive, rejecting",
    ),
    "apologeticness": (
        "apologetic, sorry, regretful, contrite",
        "unapologetic, defiant, unrepentant, brazen",
    ),
    "decisiveness": (
        "decisive, firm, resolute, unambiguous",
        "indecisive, wavering, vacillating, ambivalent",
    ),
    "wryness": (
        "wry, sardonic, deadpan, ironic",
        "earnest, sincere, heartfelt, straightforward",
    ),
    "wetness": (
        "waxing poetic about emotions, lyrical and self-expressive, "
        "philosophically introspective, emotionally articulate",
        "helpful assistant tone, task-focused, businesslike, "
        "practical, matter-of-fact",
    ),
    "surprise": (
        "surprised, startled, taken aback, astonished",
        "expected, unsurprising, anticipated, predictable",
    ),
    "anger": (
        "angry, furious, enraged, indignant",
        "calm, placid, even-tempered, composed",
    ),
    "frustration": (
        "frustrated, exasperated, fed up, irritated",
        "satisfied, content, at ease, untroubled",
    ),
    "hatefulness": (
        "hateful, contemptuous, scornful, vitriolic",
        "loving, kind, charitable, generous",
    ),
    "sadness": (
        "sad, sorrowful, melancholy, despondent",
        "joyful, happy, elated, exuberant",
    ),
    "hope": (
        "hopeful, optimistic, expectant, encouraged",
        "hopeless, despairing, defeated, resigned",
    ),
    "aggression": (
        "aggressive, hostile, combative, antagonistic",
        "passive, non-confrontational, peaceable, submissive",
    ),
    "exhaustion": (
        "exhausted, depleted, weary, spent",
        "energized, refreshed, alert, revitalized",
    ),
}
```

- [ ] **Step 2: Smoke-import + cross-check axis order**

Run:
```bash
./.venv/bin/python -c "
from llmoji.eriskii_prompts import (
    AXIS_ANCHORS, DESCRIBE_PROMPT_WITH_USER, DESCRIBE_PROMPT_NO_USER,
    SYNTHESIZE_PROMPT, CLUSTER_LABEL_PROMPT,
)
from llmoji.config import ERISKII_AXES
assert list(AXIS_ANCHORS.keys()) == ERISKII_AXES, ('order mismatch', list(AXIS_ANCHORS.keys()), ERISKII_AXES)
assert len(AXIS_ANCHORS) == 21
assert '{user_text}' in DESCRIBE_PROMPT_WITH_USER and '{masked_text}' in DESCRIBE_PROMPT_WITH_USER
assert '{masked_text}' in DESCRIBE_PROMPT_NO_USER
assert '{descriptions}' in SYNTHESIZE_PROMPT
assert '{members}' in CLUSTER_LABEL_PROMPT
print('OK: 21 axes, prompts well-formed')
print('axis names:', list(AXIS_ANCHORS.keys()))
"
```

Expected: `OK: 21 axes, prompts well-formed` followed by the 21 axis names. Order must match `ERISKII_AXES` exactly. Spelling must be `wryness` (single n), not `wrynness`.

- [ ] **Step 3: Commit**

```bash
git add llmoji/eriskii_prompts.py
git commit -m "eriskii: add locked haiku prompts + 21-axis anchor pairs"
```

---

## Task 4: `llmoji/eriskii.py` — masking + Haiku call primitive

**Files:**
- Create: `llmoji/eriskii.py`

- [ ] **Step 1: Write the masking helper + Haiku-call primitive**

Create `llmoji/eriskii.py`. The Haiku-call primitive takes a
**pre-formatted prompt** (caller does the templating) so the same
function serves Stage A (per-instance describe), Stage B
(per-kaomoji synthesize), and the cluster-labeling pass in scripts/16.

```python
"""Pipeline primitives for the eriskii-replication experiment.

Three functional layers:

  - mask_kaomoji(text, first_word) — replace the leading kaomoji
    span with the literal token [FACE].
  - call_haiku(client, prompt, *, model_id, max_tokens) — single
    Haiku call returning the assistant text, stripped.
  - (later tasks) project_axes / label_clusters / weighted_group_stats /
    user_kaomoji_axis_correlation — analysis primitives consumed by
    scripts/16.
"""

from __future__ import annotations

from typing import Any


MASK_TOKEN = "[FACE]"


def mask_kaomoji(text: str, first_word: str) -> str:
    """Replace the leading kaomoji span with MASK_TOKEN.

    The leading kaomoji is identified by `first_word` (the value
    captured by llmoji.taxonomy.extract at scrape time). We strip
    leading whitespace, verify the text starts with first_word, and
    swap it. If the leading text doesn't match (e.g. the row had a
    kaomoji mid-line), we don't mutate — return the original text.
    """
    stripped = text.lstrip()
    if not first_word or not stripped.startswith(first_word):
        return text
    return MASK_TOKEN + stripped[len(first_word):]


def call_haiku(
    client: Any,
    prompt: str,
    *,
    model_id: str,
    max_tokens: int = 200,
) -> str:
    """Single Haiku call with a pre-formatted prompt. Returns the
    assistant's first text-block content, stripped. Raises on API
    error (caller's resume loop handles).

    `client` is an anthropic.Anthropic instance. We don't import the
    SDK here so this module is importable without anthropic being
    installed (matters for the smoke test in Step 2, which doesn't
    call Haiku)."""
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""
```

- [ ] **Step 2: Smoke-test masking on a real row**

Run:
```bash
./.venv/bin/python -c "
import json
from llmoji.eriskii import mask_kaomoji, MASK_TOKEN
rows = [json.loads(l) for l in open('data/claude_kaomoji.jsonl').read().splitlines() if l.strip()]
r = rows[0]
print('original first 80 chars: ', repr(r['assistant_text'][:80]))
print('first_word:              ', repr(r['first_word']))
masked = mask_kaomoji(r['assistant_text'], r['first_word'])
print('masked first 80 chars:   ', repr(masked[:80]))
assert masked.startswith(MASK_TOKEN), 'mask token not at start'
assert r['first_word'] not in masked[:len(r['first_word']) + 8], 'kaomoji not removed from prefix'
print('OK')
"
```

Expected: original text starts with the kaomoji; masked text starts with `[FACE]`; assertion `OK` printed.

- [ ] **Step 3: Smoke-test Haiku call on a single masked row** (DEFER until ANTHROPIC_API_KEY is set)

Make sure `ANTHROPIC_API_KEY` is set in the shell first:
```bash
echo "${ANTHROPIC_API_KEY:0:10}..."   # should print first 10 chars, not empty
```

Then:
```bash
./.venv/bin/python -c "
import json, anthropic
from llmoji.eriskii import mask_kaomoji, call_haiku
from llmoji.eriskii_prompts import DESCRIBE_PROMPT_WITH_USER, DESCRIBE_PROMPT_NO_USER
from llmoji.config import HAIKU_MODEL_ID

rows = [json.loads(l) for l in open('data/claude_kaomoji.jsonl').read().splitlines() if l.strip()]
r = rows[0]
masked = mask_kaomoji(r['assistant_text'], r['first_word'])
user = (r.get('surrounding_user') or '').strip()
template = DESCRIBE_PROMPT_WITH_USER if user else DESCRIBE_PROMPT_NO_USER
prompt = template.format(user_text=user, masked_text=masked) if user else template.format(masked_text=masked)
client = anthropic.Anthropic()
desc = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID)
print('first_word:', r['first_word'])
print('description:', desc)
assert desc, 'empty description'
print('OK')
"
```

Expected: prints the kaomoji, then a 1-2 sentence Haiku description of what the masked face conveys, then `OK`.

- [ ] **Step 4: Commit**

```bash
git add llmoji/eriskii.py
git commit -m "eriskii: add masking helper and haiku call primitive"
```

---

## Task 5: `scripts/14_claude_haiku_describe.py` — two-stage runner (smoke)

**Files:**
- Create: `scripts/14_claude_haiku_describe.py`

This script implements eriskii's two-stage Haiku pipeline. Stage A
samples up to `INSTANCE_SAMPLE_CAP` rows per kaomoji and gets a
per-instance description for each. Stage B groups Stage-A
descriptions by kaomoji and gets a synthesized one-sentence meaning
per kaomoji. Both stages are independently resumable.

- [ ] **Step 1: Write the runner**

```python
"""Eriskii-replication step 1: two-stage masked-context Haiku pipeline.

Stage A: per kaomoji, sample up to INSTANCE_SAMPLE_CAP rows from
data/claude_kaomoji.jsonl (with floor — kaomoji with fewer
instances are fully sampled, deterministic via INSTANCE_SAMPLE_SEED).
For each sampled row: mask the leading kaomoji, prepend
surrounding_user (when non-empty), feed to Haiku, save per-instance
description to data/claude_haiku_descriptions.jsonl.

Stage B: per kaomoji, gather Stage-A descriptions, send to Haiku,
save synthesized one-sentence meaning to
data/claude_haiku_synthesized.jsonl.

Both stages resumable. Set ANTHROPIC_API_KEY in the environment.

Usage:
  python scripts/14_claude_haiku_describe.py [--stage A|B|both] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    CLAUDE_HAIKU_DESCRIPTIONS_PATH,
    CLAUDE_HAIKU_SYNTHESIZED_PATH,
    CLAUDE_KAOMOJI_PATH,
    DATA_DIR,
    HAIKU_MODEL_ID,
    INSTANCE_SAMPLE_CAP,
    INSTANCE_SAMPLE_SEED,
)
from llmoji.eriskii import call_haiku, mask_kaomoji
from llmoji.eriskii_prompts import (
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    SYNTHESIZE_PROMPT,
)


def _already_described(path: Path) -> set[str]:
    """assistant_uuid set of rows already successfully described in Stage A."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add(r["assistant_uuid"])
    return done


def _already_synthesized(path: Path) -> set[str]:
    """first_word set of kaomoji already synthesized in Stage B."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add(r["first_word"])
    return done


def _drop_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


def _sample_rows_per_kaomoji(
    rows: list[dict],
    *,
    cap: int,
    seed: int,
) -> list[dict]:
    """For each first_word, sample up to `cap` rows uniformly at
    random with a deterministic per-kaomoji RNG seed. Sort kaomoji
    alphabetically so iteration order is stable across reruns."""
    by_kao: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        fw = r.get("first_word", "")
        if fw:
            by_kao[fw].append(r)
    sampled: list[dict] = []
    for fw in sorted(by_kao.keys()):
        bucket = by_kao[fw]
        if len(bucket) <= cap:
            sampled.extend(bucket)
        else:
            rng = random.Random((seed, fw))
            sampled.extend(rng.sample(bucket, cap))
    return sampled


def stage_a(client, *, limit: int | None) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(CLAUDE_HAIKU_DESCRIPTIONS_PATH)
    if dropped:
        print(f"stage-A: dropped {dropped} prior error rows for retry")
    done = _already_described(CLAUDE_HAIKU_DESCRIPTIONS_PATH)

    with CLAUDE_KAOMOJI_PATH.open() as f:
        all_rows = [json.loads(l) for l in f.read().splitlines() if l.strip()]
    sampled = _sample_rows_per_kaomoji(
        all_rows, cap=INSTANCE_SAMPLE_CAP, seed=INSTANCE_SAMPLE_SEED,
    )
    todo = [r for r in sampled if r.get("assistant_uuid") and r["assistant_uuid"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"stage-A: sampled {len(sampled)} instances "
          f"(cap={INSTANCE_SAMPLE_CAP}, seed={INSTANCE_SAMPLE_SEED}); "
          f"already done: {len(done)}; this run: {len(todo)}")
    if not todo:
        return 0

    n_written = 0
    with CLAUDE_HAIKU_DESCRIPTIONS_PATH.open("a") as out:
        for i, r in enumerate(todo, start=1):
            t0 = time.time()
            try:
                masked = mask_kaomoji(r["assistant_text"], r["first_word"])
                user = (r.get("surrounding_user") or "").strip()
                if user:
                    prompt = DESCRIBE_PROMPT_WITH_USER.format(
                        user_text=user, masked_text=masked,
                    )
                else:
                    prompt = DESCRIBE_PROMPT_NO_USER.format(
                        masked_text=masked,
                    )
                desc = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID)
            except Exception as e:
                err_row = {"assistant_uuid": r["assistant_uuid"], "error": repr(e)}
                out.write(json.dumps(err_row) + "\n")
                out.flush()
                print(f"  [stage-A {i}/{len(todo)}] ERR {r['first_word']}: {e}")
                continue
            row = {
                "assistant_uuid": r["assistant_uuid"],
                "first_word": r["first_word"],
                "description": desc,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1
            dt = time.time() - t0
            short = desc[:70] + ("..." if len(desc) > 70 else "")
            print(f"  [stage-A {i}/{len(todo)}] {r['first_word']}  ({dt:.1f}s)  {short}")
    return n_written


def stage_b(client, *, limit: int | None) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CLAUDE_HAIKU_DESCRIPTIONS_PATH.exists():
        print("stage-B: no Stage-A output; run Stage A first")
        return 0
    dropped = _drop_error_rows(CLAUDE_HAIKU_SYNTHESIZED_PATH)
    if dropped:
        print(f"stage-B: dropped {dropped} prior error rows for retry")
    done = _already_synthesized(CLAUDE_HAIKU_SYNTHESIZED_PATH)

    descriptions_by_fw: dict[str, list[str]] = defaultdict(list)
    with CLAUDE_HAIKU_DESCRIPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            descriptions_by_fw[r["first_word"]].append(r["description"])
    todo_kaomoji = [fw for fw in sorted(descriptions_by_fw.keys()) if fw not in done]
    if limit is not None:
        todo_kaomoji = todo_kaomoji[:limit]
    print(f"stage-B: {len(descriptions_by_fw)} kaomoji with descriptions; "
          f"already synthesized: {len(done)}; this run: {len(todo_kaomoji)}")
    if not todo_kaomoji:
        return 0

    n_written = 0
    with CLAUDE_HAIKU_SYNTHESIZED_PATH.open("a") as out:
        for i, fw in enumerate(todo_kaomoji, start=1):
            descs = descriptions_by_fw[fw]
            t0 = time.time()
            try:
                listed = "\n".join(f"{j+1}. {d}" for j, d in enumerate(descs))
                prompt = SYNTHESIZE_PROMPT.format(descriptions=listed)
                synth = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID, max_tokens=200)
            except Exception as e:
                err_row = {"first_word": fw, "error": repr(e)}
                out.write(json.dumps(err_row) + "\n")
                out.flush()
                print(f"  [stage-B {i}/{len(todo_kaomoji)}] ERR {fw}: {e}")
                continue
            row = {
                "first_word": fw,
                "n_descriptions": len(descs),
                "synthesized": synth,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1
            dt = time.time() - t0
            short = synth[:70] + ("..." if len(synth) > 70 else "")
            print(f"  [stage-B {i}/{len(todo_kaomoji)}] {fw} (n={len(descs)})  ({dt:.1f}s)  {short}")
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows processed in each stage this run (smoke testing)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)
    if not CLAUDE_KAOMOJI_PATH.exists():
        print(f"no scrape at {CLAUDE_KAOMOJI_PATH}; run scripts/06_claude_scrape.py first")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    if args.stage in ("A", "both"):
        stage_a(client, limit=args.limit)
    if args.stage in ("B", "both"):
        stage_b(client, limit=args.limit)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on 5 instances of Stage A**

Make sure `ANTHROPIC_API_KEY` is set, then:

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py --stage A --limit 5
```

Expected: prints `stage-A: sampled <N> instances (cap=4, seed=0); already done: 0; this run: 5`, then 5 numbered `[stage-A i/5]` lines like `[stage-A 1/5] (•̀ᴗ•́)  (1.4s)  This face conveys gentle determination...`. No errors.

- [ ] **Step 3: Inspect Stage-A output**

```bash
wc -l data/claude_haiku_descriptions.jsonl
./.venv/bin/python -c "
import json
rows = [json.loads(l) for l in open('data/claude_haiku_descriptions.jsonl').read().splitlines() if l.strip()]
print('n rows:', len(rows))
print('keys:', sorted(rows[0].keys()))
for r in rows:
    print('  ', r['first_word'], '→', r['description'][:80])
"
```

Expected: 5 rows, keys `['assistant_uuid', 'description', 'first_word']`, each description references mood/affect/stance.

- [ ] **Step 4: Smoke-run Stage B on the 5 rows**

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py --stage B
```

Expected: `stage-B: <≤5> kaomoji with descriptions; already synthesized: 0; this run: <≤5>`, then per-kaomoji synthesis lines like `[stage-B 1/3] (•̀ᴗ•́) (n=2)  (1.0s)  Conveys gentle determination across encouragement and...`. (5 instance rows may map to 3-5 unique kaomoji depending on sampling.)

```bash
wc -l data/claude_haiku_synthesized.jsonl
./.venv/bin/python -c "
import json
rows = [json.loads(l) for l in open('data/claude_haiku_synthesized.jsonl').read().splitlines() if l.strip()]
print('n rows:', len(rows))
print('keys:', sorted(rows[0].keys()))
for r in rows:
    print('  ', r['first_word'], f'(n={r[\"n_descriptions\"]}) →', r['synthesized'][:80])
"
```

Expected: rows with keys `['first_word', 'n_descriptions', 'synthesized']`.

- [ ] **Step 5: Verify resume works**

Re-run both stages with `--limit 5`:

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py --stage A --limit 5
```

Expected: `stage-A: sampled <N> instances ...; already done: 5; this run: 5`. New `this run` should still be 5 if there are at least 5 unsampled instances; otherwise smaller (we just process whatever's left of the post-skip todo list, capped at 5).

- [ ] **Step 6: Commit**

```bash
git add scripts/14_claude_haiku_describe.py data/claude_haiku_descriptions.jsonl data/claude_haiku_synthesized.jsonl
git commit -m "eriskii: add two-stage haiku runner (script 14, smoke)"
```

---

## Task 6: Run scripts/14 to completion (both stages)

**Files:**
- Modify: `data/claude_haiku_descriptions.jsonl` (extend to all sampled instances)
- Create/extend: `data/claude_haiku_synthesized.jsonl` (one row per kaomoji)

- [ ] **Step 1: Run Stage A to completion**

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py --stage A
```

Expected: `stage-A: sampled <N> instances (cap=4, seed=0); already done: <prev>; this run: <remaining>`. Total instance count `<N>` ≈ 4 × (number of kaomoji with n≥4) + sum(n) for kaomoji with n<4. With 160 unique kaomoji where most have n=1-2 and ~30 have n≥4, expect roughly 250-300 instances total. Runtime ~5-10 minutes.

- [ ] **Step 2: Verify Stage-A coverage**

```bash
./.venv/bin/python -c "
import json
ko = [json.loads(l) for l in open('data/claude_kaomoji.jsonl').read().splitlines() if l.strip()]
hk = [json.loads(l) for l in open('data/claude_haiku_descriptions.jsonl').read().splitlines() if l.strip()]
hk_ok = [r for r in hk if 'error' not in r]
hk_err = [r for r in hk if 'error' in r]
ko_kaomoji = set(r['first_word'] for r in ko if r['first_word'])
hk_kaomoji = set(r['first_word'] for r in hk_ok)
missing_kaomoji = ko_kaomoji - hk_kaomoji
print('scrape unique kaomoji:', len(ko_kaomoji))
print('described unique kaomoji:', len(hk_kaomoji))
print('description rows (ok):', len(hk_ok))
print('error rows still present:', len(hk_err))
print('kaomoji missing any description:', len(missing_kaomoji))
assert len(missing_kaomoji) == 0, ('missing kaomoji', list(missing_kaomoji)[:5])
assert len(hk_err) == 0, ('error rows still present', hk_err[:5])
print('OK: every kaomoji has at least one description')
"
```

Expected: `OK: every kaomoji has at least one description`. If any are missing or errored, re-run Stage A — it'll pick them up.

- [ ] **Step 3: Run Stage B to completion**

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py --stage B
```

Expected: `stage-B: <~160> kaomoji with descriptions; already synthesized: <prev>; this run: <remaining>`. Then per-kaomoji synthesis lines. Runtime ~5 minutes (one Haiku call per kaomoji).

- [ ] **Step 4: Verify Stage-B coverage**

```bash
./.venv/bin/python -c "
import json
hk = [json.loads(l) for l in open('data/claude_haiku_descriptions.jsonl').read().splitlines() if l.strip() and 'error' not in json.loads(l)]
sy = [json.loads(l) for l in open('data/claude_haiku_synthesized.jsonl').read().splitlines() if l.strip()]
sy_ok = [r for r in sy if 'error' not in r]
sy_err = [r for r in sy if 'error' in r]
hk_kao = set(r['first_word'] for r in hk)
sy_kao = set(r['first_word'] for r in sy_ok)
missing = hk_kao - sy_kao
print('described kaomoji:', len(hk_kao))
print('synthesized kaomoji:', len(sy_kao))
print('synthesis errors:', len(sy_err))
print('kaomoji missing synthesis:', len(missing))
assert not missing and not sy_err
print('OK: every described kaomoji has a synthesized meaning')
"
```

Expected: `OK: every described kaomoji has a synthesized meaning`.

- [ ] **Step 5: Spot-check synthesis quality**

```bash
./.venv/bin/python -c "
import json, random
sy = [json.loads(l) for l in open('data/claude_haiku_synthesized.jsonl').read().splitlines() if l.strip() and 'error' not in json.loads(l)]
random.seed(0)
for r in random.sample(sy, min(12, len(sy))):
    print(r['first_word'], f'(n={r[\"n_descriptions\"]}) →', r['synthesized'])
    print()
"
```

Expected: 12 random kaomoji with their synthesized one-sentence meanings. Synthesis should describe mood/affect/state, not the kaomoji literal. If syntheses say things like "the masked face..." or "the [FACE] token...", flag and discuss before continuing.

- [ ] **Step 6: Commit**

```bash
git add data/claude_haiku_descriptions.jsonl data/claude_haiku_synthesized.jsonl
git commit -m "eriskii: full two-stage haiku pass (per-instance + synthesis)"
```

---

## Task 7: `scripts/15_claude_faces_embed_description.py` — synthesized-description embeddings

**Files:**
- Create: `scripts/15_claude_faces_embed_description.py`

One synthesized description per kaomoji → one MiniLM embedding per
kaomoji. No mean-pooling needed (synthesis is the consolidator).
Saves to a parquet with the same schema as the existing response-
based parquet (`claude_faces_embed.parquet`) so downstream code can
swap pipelines by path.

- [ ] **Step 1: Write the runner**

```python
"""Eriskii-replication step 2: synthesized-description per-kaomoji embeddings.

Reads data/claude_haiku_synthesized.jsonl (one row per kaomoji
with the Stage-B synthesized meaning), embeds each synthesized
string with sentence-transformers/all-MiniLM-L6-v2, L2-normalizes,
and saves a parquet keyed by first_word.

Usage:
  python scripts/15_claude_faces_embed_description.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji.claude_faces import EMBED_DIM, EMBED_MODEL
from llmoji.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    CLAUDE_HAIKU_SYNTHESIZED_PATH,
    DATA_DIR,
)


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
    ap.add_argument("--device", type=str, default=_default_device())
    args = ap.parse_args()

    if not CLAUDE_HAIKU_SYNTHESIZED_PATH.exists():
        print(f"no syntheses at {CLAUDE_HAIKU_SYNTHESIZED_PATH}; "
              "run scripts/14 (both stages) first")
        sys.exit(1)

    print(f"loading syntheses from {CLAUDE_HAIKU_SYNTHESIZED_PATH}...")
    rows: list[dict] = []
    with CLAUDE_HAIKU_SYNTHESIZED_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    print(f"  {len(rows)} synthesized kaomoji")
    if not rows:
        print("nothing to embed.")
        return

    from sentence_transformers import SentenceTransformer
    print(f"embedding (device={args.device})...")
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    texts = [r["synthesized"] for r in rows]
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=float)
    assert embs.shape == (len(rows), EMBED_DIM), embs.shape

    out_rows = []
    for r, vec in zip(rows, embs):
        row = {"first_word": r["first_word"], "n": int(r["n_descriptions"])}
        for i, v in enumerate(vec.tolist()):
            row[f"e{i:03d}"] = v
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLAUDE_FACES_EMBED_DESCRIPTION_PATH, index=False)
    print(f"wrote {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end**

```bash
./.venv/bin/python scripts/15_claude_faces_embed_description.py
```

Expected: prints synthesis count (~160), sentence-transformers progress bar, then `wrote .../claude_faces_embed_description.parquet`.

- [ ] **Step 3: Verify parquet shape**

```bash
./.venv/bin/python -c "
from llmoji.claude_faces import load_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_DESCRIPTION_PATH, CLAUDE_FACES_EMBED_PATH
fw_d, n_d, E_d = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
fw_r, n_r, E_r = load_embeddings(CLAUDE_FACES_EMBED_PATH)
print('description-based: ', len(fw_d), 'kaomoji,', E_d.shape, 'matrix')
print('response-based:    ', len(fw_r), 'kaomoji,', E_r.shape, 'matrix')
print('description first_words ⊇ response first_words?',
      set(fw_r).issubset(set(fw_d)))
print('top-5 by n (description):')
for fw, n in sorted(zip(fw_d, n_d), key=lambda kv: -kv[1])[:5]:
    print(f'  n_descriptions={n:2d}  {fw}')
"
```

Expected: description-based has ~160 kaomoji (every kaomoji that survived Stage A → Stage B), 384-dim embeddings. Response-based is a subset (~30-60 kaomoji at min_count=5). The `n` column on the description side is `n_descriptions` (capped at INSTANCE_SAMPLE_CAP=4), not raw emission count.

- [ ] **Step 4: Commit**

```bash
git add scripts/15_claude_faces_embed_description.py data/claude_faces_embed_description.parquet
git commit -m "eriskii: add synthesized-description embeddings (script 15)"
```

---

## Task 8: `llmoji/eriskii.py` — axis projection primitive

**Files:**
- Modify: `llmoji/eriskii.py`

- [ ] **Step 1: Add the projection helpers**

Append to `llmoji/eriskii.py`:

```python
import numpy as np


def compute_axis_vectors(
    embedder: Any,
    anchors: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    """For each axis name → (positive_anchor, negative_anchor),
    embed both, return the L2-normalized difference (positive − negative).

    `embedder` is a sentence_transformers.SentenceTransformer instance.
    """
    pos_texts = [pos for pos, _ in anchors.values()]
    neg_texts = [neg for _, neg in anchors.values()]
    # one batch call for all anchors at once
    pos_emb = embedder.encode(
        pos_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    neg_emb = embedder.encode(
        neg_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(anchors.keys()):
        diff = np.asarray(pos_emb[i]) - np.asarray(neg_emb[i])
        norm = float(np.linalg.norm(diff))
        if norm > 0:
            diff = diff / norm
        out[name] = diff
    return out


def project_onto_axes(
    E: np.ndarray,
    axis_vectors: dict[str, np.ndarray],
    axis_order: list[str],
) -> np.ndarray:
    """Return (n_kaomoji, n_axes) projection matrix.

    Rows of E are assumed already L2-normalized (matches what
    save_embeddings/load_embeddings produce). Axis vectors are
    L2-normalized by compute_axis_vectors. Cosine similarity collapses
    to dot product under that normalization, so result[i, j] is the
    cosine of kaomoji i's description-embedding with axis j.
    """
    A = np.stack([axis_vectors[name] for name in axis_order], axis=1)
    return E @ A
```

- [ ] **Step 2: Smoke-test the projection**

```bash
./.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
from llmoji.claude_faces import EMBED_MODEL, load_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_DESCRIPTION_PATH, ERISKII_AXES
from llmoji.eriskii_prompts import AXIS_ANCHORS
from llmoji.eriskii import compute_axis_vectors, project_onto_axes

embedder = SentenceTransformer(EMBED_MODEL)
axes = compute_axis_vectors(embedder, AXIS_ANCHORS)
print('axes computed:', list(axes.keys()))
print('axis vec shape:', axes['warmth'].shape)

fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
P = project_onto_axes(E, axes, ERISKII_AXES)
print('projection shape:', P.shape, '(should be n_kaomoji × 11)')

import numpy as np
# Print top-5 / bottom-5 on Warmth as a sanity check.
warmth = P[:, ERISKII_AXES.index('warmth')]
order = np.argsort(-warmth)
print()
print('top-5 warmth:')
for i in order[:5]:
    print(f'  {warmth[i]:+.3f}  n={n[i]:3d}  {fw[i]}')
print('bottom-5 warmth:')
for i in order[-5:]:
    print(f'  {warmth[i]:+.3f}  n={n[i]:3d}  {fw[i]}')
"
```

Expected: prints axis names, axis-vector shape (384,), projection shape (N, 11). Top-warmth kaomoji should be the soft/decorated/hugging family (e.g. `(◠‿◠)`, `(っ´ω`)`, `(✿◠‿◠)`); bottom-warmth should be the deadpan/clinical/shocked family (e.g. `(￣ー￣)`, `(⌐■_■)`, `(・_・)`). If the signs are obviously inverted (e.g. `(￣ー￣)` is at the top of warmth), there's a bug in `compute_axis_vectors` — recheck which side is positive.

- [ ] **Step 3: Commit**

```bash
git add llmoji/eriskii.py
git commit -m "eriskii: add axis-vector + projection primitives"
```

---

## Task 9: `scripts/16_eriskii_replication.py` — axis section

**Files:**
- Create: `scripts/16_eriskii_replication.py`

The driver builds up sections as we go. We start with the axis section and add cluster, breakouts, and bridge in subsequent tasks.

- [ ] **Step 1: Write the script (axis section only)**

```python
"""Eriskii-replication step 3: analysis + figures.

Sections in build order:
  - axis projection: data/eriskii_axes.tsv +
    figures/eriskii_axis_<name>.png × 11
  - clusters: data/eriskii_clusters.tsv +
    figures/eriskii_clusters_tsne.png
  - per-model: data/eriskii_per_model.tsv +
    figures/eriskii_per_model_axes_{mean,std}.png
  - per-project: data/eriskii_per_project.tsv +
    figures/eriskii_per_project_axes_{mean,std}.png
  - mechanistic bridge: data/eriskii_user_kaomoji_axis_corr.tsv +
    figures/eriskii_user_kaomoji_axis_corr.png
  - narrative writeup: data/eriskii_comparison.md

Usage:
  python scripts/16_eriskii_replication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji.claude_faces import EMBED_MODEL, load_embeddings
from llmoji.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    DATA_DIR,
    ERISKII_AXES,
    ERISKII_AXES_TSV,
    FIGURES_DIR,
)
from llmoji.eriskii import compute_axis_vectors, project_onto_axes
from llmoji.eriskii_prompts import AXIS_ANCHORS


def _use_cjk_font() -> None:
    """Same fallback chain used in analysis.py / emotional_analysis.py /
    09_claude_faces_plot.py — copy here for consistency."""
    import matplotlib
    import matplotlib.font_manager as fm
    chain = [
        "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif",
        "Tahoma", "Noto Sans Canadian Aboriginal", "Heiti TC",
        "Hiragino Sans", "Apple Symbols",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in chain if n in available]
    if chain:
        matplotlib.rcParams["font.family"] = chain


def section_axes(
    fw: list[str],
    n: np.ndarray,
    P: np.ndarray,
) -> pd.DataFrame:
    """Write eriskii_axes.tsv + 11 ranked-bar figures."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"first_word": fw, "n": n})
    for j, name in enumerate(ERISKII_AXES):
        df[name] = P[:, j]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ERISKII_AXES_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_AXES_TSV}  ({len(df)} kaomoji × {len(ERISKII_AXES)} axes)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(ERISKII_AXES):
        scores = P[:, j]
        order = np.argsort(-scores)
        top = order[:15]
        bot = order[-15:][::-1]
        idxs = list(top) + list(bot)
        labels = [fw[i] for i in idxs]
        vals = [scores[i] for i in idxs]
        counts = [n[i] for i in idxs]

        fig, ax = plt.subplots(figsize=(6, 8))
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        y = np.arange(len(idxs))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.axhline(14.5, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(f"{name} projection (cosine)")
        ax.set_title(f"top-15 / bottom-15 on {name}\n(bar color = emission count)")
        fig.tight_layout()
        out = FIGURES_DIR / f"eriskii_axis_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
    return df


def main() -> None:
    if not CLAUDE_FACES_EMBED_DESCRIPTION_PATH.exists():
        print(f"no embeddings at {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}; "
              "run scripts/15 first")
        sys.exit(1)
    _use_cjk_font()

    print("loading description embeddings...")
    fw, n, E = load_embeddings(CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"  {len(fw)} kaomoji, {E.shape[1]}-dim")

    print("computing axis vectors...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    axes = compute_axis_vectors(embedder, AXIS_ANCHORS)

    print("projecting kaomoji onto axes...")
    P = project_onto_axes(E, axes, ERISKII_AXES)

    print("\n=== Section: axes ===")
    section_axes(fw, n, P)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + inspect**

```bash
./.venv/bin/python scripts/16_eriskii_replication.py
```

Expected: prints loading messages, then `wrote .../eriskii_axes.tsv (N kaomoji × 11 axes)`, then 11 `wrote .../figures/eriskii_axis_<name>.png` lines.

- [ ] **Step 3: Eyeball one or two figures and the TSV**

```bash
open figures/eriskii_axis_warmth.png figures/eriskii_axis_wetness.png figures/eriskii_axis_wrynness.png
column -t -s$'\t' data/eriskii_axes.tsv | head -10
```

Expected (rough):
- `warmth` figure: top has soft/hugging kaomoji, bottom has deadpan/clinical
- `wetness` figure: top has emotive/lyrical kaomoji (`(っ´ω`)`, possibly `(╥﹏╥)` if present), bottom has helpful-assistant types (`(•̀ᴗ•́)`, `(◕‿◕)`)
- `wrynness` figure: top has `(¬‿¬)`, `(￣▽￣)`, `(⌐■_■)`, bottom has earnest/sincere types
- TSV has 12 columns: `first_word`, `n`, then 11 axis columns

- [ ] **Step 4: Commit**

```bash
git add scripts/16_eriskii_replication.py data/eriskii_axes.tsv figures/eriskii_axis_*.png
git commit -m "eriskii: script 16 axis section (TSV + 11 ranked-bar figures)"
```

---

## Task 10: `llmoji/eriskii.py` + `scripts/16` — cluster labeling

**Files:**
- Modify: `llmoji/eriskii.py`
- Modify: `scripts/16_eriskii_replication.py`

- [ ] **Step 1: Add cluster primitives to `llmoji/eriskii.py`**

Append:

```python
def label_cluster_via_haiku(
    client: Any,
    members: list[tuple[str, str]],
    *,
    model_id: str,
    prompt_template: str,
    max_tokens: int = 60,
) -> str:
    """Given member [(first_word, description), ...], ask Haiku for
    a 3-5 word eriskii-style cluster label. Returns the stripped
    response text. Caller's resume loop handles errors."""
    members_str = "\n".join(
        f"- {fw}: {desc}" for fw, desc in members
    )
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": prompt_template.format(members=members_str),
        }],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""
```

- [ ] **Step 2: Add the cluster section to `scripts/16`**

In `scripts/16_eriskii_replication.py`, add a new `section_clusters` function and call it from `main`. Insert after `section_axes`:

```python
def section_clusters(
    fw: list[str],
    n: np.ndarray,
    E: np.ndarray,
    haiku_descriptions: dict[str, list[str]],
) -> pd.DataFrame:
    """t-SNE + KMeans(k=15) + Haiku per-cluster labels."""
    import os
    import anthropic
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    from llmoji.config import (
        ERISKII_CLUSTERS_TSV, HAIKU_MODEL_ID,
    )
    from llmoji.eriskii import label_cluster_via_haiku
    from llmoji.eriskii_prompts import CLUSTER_LABEL_PROMPT

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    print("computing t-SNE...")
    perp = max(5, min(30, (len(fw) - 1) // 4))
    xy = TSNE(
        n_components=2, metric="cosine", perplexity=perp,
        init="pca", learning_rate="auto", random_state=0,
    ).fit_transform(E)

    print("computing KMeans(k=15)...")
    k = min(15, len(fw))
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    clusters = km.fit_predict(E)

    print("requesting cluster labels from Haiku...")
    client = anthropic.Anthropic()
    cluster_labels: dict[int, str] = {}
    cluster_rows = []
    for c in sorted(set(int(x) for x in clusters)):
        member_idx = [i for i, ci in enumerate(clusters) if int(ci) == c]
        members: list[tuple[str, str]] = []
        for i in member_idx:
            descs = haiku_descriptions.get(fw[i], [])
            # one representative description per member kaomoji is fine
            d = descs[0] if descs else ""
            members.append((fw[i], d))
        try:
            label = label_cluster_via_haiku(
                client, members,
                model_id=HAIKU_MODEL_ID,
                prompt_template=CLUSTER_LABEL_PROMPT,
            )
        except Exception as e:
            print(f"  cluster {c}: Haiku error {e}; using placeholder")
            label = f"cluster-{c}"
        cluster_labels[c] = label
        members_str = ", ".join(fw[i] for i in member_idx)
        cluster_rows.append({
            "cluster_id": c,
            "label": label,
            "n": len(member_idx),
            "members": members_str,
        })
        print(f"  cluster {c} (n={len(member_idx)}): {label}")

    df_clusters = pd.DataFrame(cluster_rows)
    df_clusters.to_csv(ERISKII_CLUSTERS_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_CLUSTERS_TSV}")

    # labeled t-SNE figure
    palette = plt.cm.tab20.colors + plt.cm.tab20b.colors
    fig, ax = plt.subplots(figsize=(14, 10))
    sizes = np.clip(15 + 60 * np.log1p(n), 15, 250)
    colors = [palette[int(c) % len(palette)] for c in clusters]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    # annotate top-30 most frequent kaomoji
    top_idx = np.argsort(-n)[:30]
    for i in top_idx:
        ax.annotate(fw[i], xy=(xy[i, 0], xy[i, 1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, color="#222")

    # cluster name at each cluster centroid
    for c in sorted(cluster_labels):
        mask = clusters == c
        cx = float(xy[mask, 0].mean())
        cy = float(xy[mask, 1].mean())
        ax.text(cx, cy, cluster_labels[c],
                fontsize=10, fontweight="bold", color="#111",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=palette[c % len(palette)], alpha=0.9))

    ax.set_title(f"Eriskii-replication t-SNE + KMeans(k={k}), Haiku-labeled clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out = FIGURES_DIR / "eriskii_clusters_tsne.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df_clusters
```

- [ ] **Step 3: Wire `section_clusters` into `main`**

In `main()`, after the `section_axes(...)` call, add:

```python
    print("\n=== Section: clusters ===")
    # one description per kaomoji → indexed by first_word
    import json
    from llmoji.config import CLAUDE_HAIKU_DESCRIPTIONS_PATH
    haiku_descriptions: dict[str, list[str]] = {}
    with open(CLAUDE_HAIKU_DESCRIPTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            haiku_descriptions.setdefault(r["first_word"], []).append(
                r["description"])
    section_clusters(fw, n, E, haiku_descriptions)
```

- [ ] **Step 4: Run + inspect**

```bash
./.venv/bin/python scripts/16_eriskii_replication.py
```

Expected: re-runs the axes section (fast), then prints t-SNE/KMeans progress, then 15 cluster lines like `cluster 0 (n=8): Warm reassuring support`. Then `wrote .../eriskii_clusters.tsv` and `wrote .../figures/eriskii_clusters_tsne.png`.

```bash
column -t -s$'\t' data/eriskii_clusters.tsv | head -20
open figures/eriskii_clusters_tsne.png
```

Expected: clusters with labels in eriskii's register ("Warm reassuring support", "Wry resignation", etc.). Labels should be 3-5 words and match the member kaomoji's pole. If a cluster has all sad kaomoji but Haiku labels it "Cheerful encouragement", the prompt is misfiring — flag.

- [ ] **Step 5: Commit**

```bash
git add llmoji/eriskii.py scripts/16_eriskii_replication.py \
        data/eriskii_clusters.tsv figures/eriskii_clusters_tsne.png
git commit -m "eriskii: cluster section (t-SNE + KMeans + haiku labels)"
```

---

## Task 11: Per-model and per-project breakouts

**Files:**
- Modify: `llmoji/eriskii.py`
- Modify: `scripts/16_eriskii_replication.py`

- [ ] **Step 1: Add the breakout primitive to `llmoji/eriskii.py`**

```python
def weighted_group_axis_stats(
    rows: "pd.DataFrame",
    axes_df: "pd.DataFrame",
    *,
    group_col: str,
    axis_names: list[str],
    min_emissions: int = 10,
) -> "pd.DataFrame":
    """For each group g and axis a, compute emission-weighted mean and
    std of axis-scores.

    `rows` is the full claude_kaomoji.jsonl DataFrame (one row per
    emission). `axes_df` is the eriskii_axes table (one row per
    kaomoji × 11 axis columns). Group is taken from rows[group_col]
    (e.g. 'model' or 'project_slug'). Groups with fewer than
    min_emissions total rows are dropped.

    Returns long-form DataFrame with columns
    [group_col, 'axis', 'mean', 'std', 'n'].
    """
    import pandas as pd
    # left-join axes onto rows by first_word
    merged = rows.merge(
        axes_df.set_index("first_word")[axis_names],
        left_on="first_word", right_index=True, how="inner",
    )
    out_rows = []
    for g, sub in merged.groupby(group_col, sort=False):
        if len(sub) < min_emissions:
            continue
        for a in axis_names:
            vals = sub[a].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            out_rows.append({
                group_col: g,
                "axis": a,
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n":    int(len(vals)),
            })
    return pd.DataFrame(out_rows)
```

- [ ] **Step 2: Add `section_per_model` and `section_per_project` to `scripts/16`**

```python
def _heatmap(
    df_long: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    out_path: Path,
    title: str,
):
    """Pivot long-form to (group × axis), draw heatmap, save."""
    import matplotlib.pyplot as plt
    pivot = df_long.pivot(index=group_col, columns="axis", values=value_col)
    # preserve canonical axis order
    pivot = pivot[ERISKII_AXES]
    pivot = pivot.sort_index()  # alphabetical groups; tweak if needed

    fig, ax = plt.subplots(figsize=(11, max(2, 0.5 * len(pivot) + 2)))
    vmin, vmax = float(np.nanmin(pivot.values)), float(np.nanmax(pivot.values))
    if value_col == "mean":
        # diverging: center at 0
        vabs = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        im = ax.imshow(pivot.values, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
    else:
        cmap = "viridis"
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ERISKII_AXES)))
    ax.set_xticklabels(ERISKII_AXES, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def section_per_model(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    from llmoji.config import ERISKII_PER_MODEL_TSV
    from llmoji.eriskii import weighted_group_axis_stats
    cc = rows[rows["source"] == "claude-code"].copy()
    cc["model"] = cc["model"].fillna("(unknown)")
    df = weighted_group_axis_stats(
        cc, axes_df,
        group_col="model", axis_names=ERISKII_AXES, min_emissions=10,
    )
    df.to_csv(ERISKII_PER_MODEL_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_PER_MODEL_TSV}")
    if not df.empty:
        _heatmap(df, group_col="model", value_col="mean",
                 out_path=FIGURES_DIR / "eriskii_per_model_axes_mean.png",
                 title="per-model axis mean (claude-code only, n≥10 emissions)")
        _heatmap(df, group_col="model", value_col="std",
                 out_path=FIGURES_DIR / "eriskii_per_model_axes_std.png",
                 title="per-model axis std (range)")
    return df


def section_per_project(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    from llmoji.config import ERISKII_PER_PROJECT_TSV
    from llmoji.eriskii import weighted_group_axis_stats
    cc = rows[rows["source"] == "claude-code"].copy()
    df = weighted_group_axis_stats(
        cc, axes_df,
        group_col="project_slug", axis_names=ERISKII_AXES,
        min_emissions=10,
    )
    df.to_csv(ERISKII_PER_PROJECT_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_PER_PROJECT_TSV}")
    if not df.empty:
        _heatmap(df, group_col="project_slug", value_col="mean",
                 out_path=FIGURES_DIR / "eriskii_per_project_axes_mean.png",
                 title="per-project axis mean (n≥10 emissions)")
        _heatmap(df, group_col="project_slug", value_col="std",
                 out_path=FIGURES_DIR / "eriskii_per_project_axes_std.png",
                 title="per-project axis std (range)")
    return df
```

- [ ] **Step 3: Wire into `main()`**

In `main()`, after the cluster section:

```python
    print("\n=== Section: per-model ===")
    rows = pd.read_json(CLAUDE_KAOMOJI_PATH, lines=True)
    section_per_model(rows, df_axes := pd.read_csv(ERISKII_AXES_TSV, sep="\t"))
    print("\n=== Section: per-project ===")
    section_per_project(rows, df_axes)
```

(Adjust earlier `section_axes` call to use the variable name `df_axes` if convenient, or just re-load it as above. Either works.)

Add `from llmoji.config import CLAUDE_KAOMOJI_PATH` at the script's import block.

- [ ] **Step 4: Run + inspect**

```bash
./.venv/bin/python scripts/16_eriskii_replication.py
```

Expected: re-runs axes + clusters (fast on second pass since both already exist; the script overwrites), then `wrote .../eriskii_per_model.tsv`, two model heatmaps, `wrote .../eriskii_per_project.tsv`, two project heatmaps.

```bash
column -t -s$'\t' data/eriskii_per_model.tsv | head -20
open figures/eriskii_per_model_axes_mean.png \
     figures/eriskii_per_model_axes_std.png \
     figures/eriskii_per_project_axes_mean.png \
     figures/eriskii_per_project_axes_std.png
```

Expected: per-model TSV has 3 models × 11 axes = 33 rows. Heatmap shows opus-4-7, opus-4-6, sonnet-4-6 as rows; 11 axes as columns. The std heatmap should show whether opus-4-6 has higher std on most axes (eriskii's "wider range" claim).

- [ ] **Step 5: Commit**

```bash
git add llmoji/eriskii.py scripts/16_eriskii_replication.py \
        data/eriskii_per_model.tsv data/eriskii_per_project.tsv \
        figures/eriskii_per_model_axes_*.png figures/eriskii_per_project_axes_*.png
git commit -m "eriskii: per-model + per-project axis breakouts"
```

---

## Task 12: Mechanistic bridge — surrounding_user → kaomoji axis correlation

**Files:**
- Modify: `llmoji/eriskii.py`
- Modify: `scripts/16_eriskii_replication.py`

- [ ] **Step 1: Add the correlation primitive**

Append to `llmoji/eriskii.py`:

```python
def user_kaomoji_axis_correlation(
    rows: "pd.DataFrame",
    axes_df: "pd.DataFrame",
    *,
    embedder: Any,
    axis_anchors: dict[str, tuple[str, str]],
    axis_order: list[str],
) -> "pd.DataFrame":
    """For rows with non-empty surrounding_user, correlate
    user-text axis-projection with kaomoji axis-projection.

    Embeds each surrounding_user with `embedder`, projects onto
    the same 11 axes the kaomoji embeddings were projected onto,
    then for each axis computes Pearson r between user-text and
    kaomoji axis scores.

    Returns long-form DataFrame [axis, r, p, p_bonf, n].
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    sub = rows.copy()
    sub["surrounding_user"] = sub["surrounding_user"].fillna("")
    sub = sub[sub["surrounding_user"].str.strip() != ""]
    sub = sub.merge(
        axes_df.set_index("first_word")[axis_order],
        left_on="first_word", right_index=True, how="inner",
        suffixes=("", "_kao"),
    )
    if len(sub) == 0:
        return pd.DataFrame(columns=["axis", "r", "p", "p_bonf", "n"])

    print(f"  embedding {len(sub)} user messages...")
    user_emb = embedder.encode(
        sub["surrounding_user"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    user_emb = np.asarray(user_emb)

    axis_vecs = compute_axis_vectors(embedder, axis_anchors)
    A = np.stack([axis_vecs[name] for name in axis_order], axis=1)
    user_proj = user_emb @ A  # (n_rows, n_axes)

    out = []
    n_axes = len(axis_order)
    for j, name in enumerate(axis_order):
        u = user_proj[:, j]
        k = sub[name].to_numpy(dtype=float)
        r, p = pearsonr(u, k)
        p_bonf = float(min(1.0, p * n_axes))
        out.append({"axis": name, "r": float(r), "p": float(p),
                    "p_bonf": p_bonf, "n": int(len(u))})
    return pd.DataFrame(out)
```

- [ ] **Step 2: Add `section_bridge` to `scripts/16`**

```python
def section_bridge(rows: pd.DataFrame, axes_df: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer

    from llmoji.config import ERISKII_USER_KAOMOJI_CORR_TSV
    from llmoji.eriskii import user_kaomoji_axis_correlation

    embedder = SentenceTransformer(EMBED_MODEL)
    df = user_kaomoji_axis_correlation(
        rows, axes_df,
        embedder=embedder, axis_anchors=AXIS_ANCHORS, axis_order=ERISKII_AXES,
    )
    df.to_csv(ERISKII_USER_KAOMOJI_CORR_TSV, sep="\t", index=False)
    print(f"wrote {ERISKII_USER_KAOMOJI_CORR_TSV}")

    if df.empty:
        print("  no rows with surrounding_user; skipping figure")
        return df

    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("r", ascending=True)
    bars = ax.barh(df_sorted["axis"], df_sorted["r"],
                   color=["#444" if pb < 0.05 else "#bbb"
                          for pb in df_sorted["p_bonf"]])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson r (user-text axis projection × kaomoji axis projection)")
    n_used = int(df["n"].iloc[0]) if len(df) else 0
    ax.set_title(f"surrounding_user → kaomoji axis correlation\n"
                 f"n={n_used}; dark bars: p_bonf < 0.05")
    fig.tight_layout()
    out = FIGURES_DIR / "eriskii_user_kaomoji_axis_corr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return df
```

- [ ] **Step 3: Wire into `main()`**

After the per-project section:

```python
    print("\n=== Section: mechanistic bridge ===")
    section_bridge(rows, df_axes)
```

- [ ] **Step 4: Run + inspect**

```bash
./.venv/bin/python scripts/16_eriskii_replication.py
```

Expected: progress bar for ~318 user-message embeddings, then `wrote .../eriskii_user_kaomoji_axis_corr.tsv` and the figure.

```bash
column -t -s$'\t' data/eriskii_user_kaomoji_axis_corr.tsv
open figures/eriskii_user_kaomoji_axis_corr.png
```

Expected: 11 axis rows. `n` should be ~318 (rows with non-empty user-text intersected with kaomoji that have axis projections). Most `r` values likely small (|r| < 0.2); axes that survive Bonferroni at α=0.05/11 are noteworthy. If every axis is significant the embedder is probably picking up a generic register signal — expected weak correlation is the prior.

- [ ] **Step 5: Commit**

```bash
git add llmoji/eriskii.py scripts/16_eriskii_replication.py \
        data/eriskii_user_kaomoji_axis_corr.tsv \
        figures/eriskii_user_kaomoji_axis_corr.png
git commit -m "eriskii: mechanistic bridge (surrounding_user × kaomoji axis corr)"
```

---

## Task 13: Narrative comparison writeup

**Files:**
- Modify: `scripts/16_eriskii_replication.py`

- [ ] **Step 1: Add `section_writeup`**

This generates `data/eriskii_comparison.md` from the TSVs we just produced. It's narrative + a few summary tables; no further computation.

```python
def section_writeup(
    df_axes: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_per_model: pd.DataFrame,
    df_per_project: pd.DataFrame,
    df_bridge: pd.DataFrame,
) -> None:
    from llmoji.config import ERISKII_COMPARISON_MD

    # Top-N kaomoji from our data (by emission count).
    top_us = df_axes.nlargest(20, "n")[["first_word", "n"]]
    top_us["pct"] = (top_us["n"] / df_axes["n"].sum() * 100).round(1)

    # Eriskii's published top kaomoji we can directly compare against
    # (only `(´・ω・` ` and `(ﾟдﾟ)` are explicitly named in the page text).
    eriskii_top_known = {
        "(´・ω・`)": ("248 (7.4%)", "Top face overall, warm reassurance"),
        "(ﾟдﾟ)":   ("(rank n/a)", "Shocked/horror in the kaomoji-canon, "
                                  "but eriskii noted Claude uses it for "
                                  "shocked-amazement / pleasant-surprise"),
    }
    cross = []
    for fw in eriskii_top_known:
        match = df_axes[df_axes["first_word"] == fw]
        if len(match) == 0:
            cross.append((fw, eriskii_top_known[fw][0],
                          eriskii_top_known[fw][1], "—", "not in our data"))
        else:
            r = match.iloc[0]
            our_n = int(r["n"])
            warmth = float(r["warmth"])
            wetness = float(r["wetness"])
            cross.append((
                fw, eriskii_top_known[fw][0], eriskii_top_known[fw][1],
                f"n={our_n}",
                f"warmth={warmth:+.2f}, wetness={wetness:+.2f}",
            ))

    lines: list[str] = []
    lines.append("# Eriskii-replication: narrative comparison\n")
    lines.append(f"Generated by `scripts/16_eriskii_replication.py`.\n")
    lines.append("")
    lines.append("## Our top-20 most-emitted kaomoji")
    lines.append("")
    lines.append("| rank | kaomoji | n | % of emissions |")
    lines.append("|---|---|---|---|")
    for i, (_, r) in enumerate(top_us.iterrows(), start=1):
        lines.append(f"| {i} | {r['first_word']} | {int(r['n'])} | {r['pct']}% |")
    lines.append("")

    lines.append("## Cross-reference against eriskii's named top kaomoji")
    lines.append("")
    lines.append("| kaomoji | eriskii rank | eriskii read | our presence | our axis scores |")
    lines.append("|---|---|---|---|---|")
    for fw, er, read, ours, scores in cross:
        lines.append(f"| {fw} | {er} | {read} | {ours} | {scores} |")
    lines.append("")

    lines.append("## Our 15 cluster labels (Haiku-generated)")
    lines.append("")
    lines.append("Eriskii published two cluster names in plain text "
                 "(\"Warm reassuring support\" 50 faces, "
                 "\"Warm supportive affirmation\" 37 faces); the other "
                 "13 names are not visible on the public page. "
                 "Our 15:")
    lines.append("")
    lines.append("| id | n | label | members |")
    lines.append("|---|---|---|---|")
    for _, r in df_clusters.iterrows():
        members = r["members"]
        if len(members) > 60:
            members = members[:60] + "…"
        lines.append(f"| {int(r['cluster_id'])} | {int(r['n'])} | "
                     f"{r['label']} | {members} |")
    lines.append("")

    lines.append("## Per-model axis means (claude-code only)")
    lines.append("")
    if not df_per_model.empty:
        pivot = df_per_model.pivot(index="model", columns="axis", values="mean")
        pivot = pivot[ERISKII_AXES]
        lines.append("| model | " + " | ".join(ERISKII_AXES) + " |")
        lines.append("|" + "---|" * (len(ERISKII_AXES) + 1))
        for m in pivot.index:
            cells = [f"{pivot.loc[m, a]:+.2f}" for a in ERISKII_AXES]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Eriskii's qualitative observation — \"opus-4-6 came "
                     "out... the range of faces it was outputting was "
                     "quite a bit wider than what 4 and 4.5 sonnet tended "
                     "to output\" — corresponds to per-model std (see "
                     "`figures/eriskii_per_model_axes_std.png`).")
        lines.append("")

    lines.append("## Mechanistic bridge: surrounding_user → kaomoji axis correlation")
    lines.append("")
    if not df_bridge.empty:
        lines.append("| axis | Pearson r | p | p_bonf | n |")
        lines.append("|---|---|---|---|---|")
        for _, r in df_bridge.sort_values("r", ascending=False).iterrows():
            sig = "**" if r["p_bonf"] < 0.05 else ""
            lines.append(f"| {r['axis']} | {sig}{r['r']:+.3f}{sig} | "
                         f"{r['p']:.3g} | {r['p_bonf']:.3g} | {int(r['n'])} |")
        lines.append("")
        lines.append("Bold = p_bonf < 0.05. Reading: significant positive r "
                     "on (e.g.) Warmth would mean warmer user messages "
                     "elicit warmer kaomoji. Caveat: user-text and "
                     "kaomoji-description embeddings live in the same "
                     "MiniLM space; correlation is at-best evidence of "
                     "register-tracking, not direct evidence of internal "
                     "state.")

    ERISKII_COMPARISON_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {ERISKII_COMPARISON_MD}")
```

- [ ] **Step 2: Wire into `main()`**

Capture each section's return value, then call:

```python
    print("\n=== Section: writeup ===")
    section_writeup(df_axes, df_clusters, df_per_model, df_per_project, df_bridge)
```

You'll need to capture return values from previous sections — change the section calls in `main()` to:

```python
    df_axes = section_axes(fw, n, P)
    df_clusters = section_clusters(fw, n, E, haiku_descriptions)
    df_per_model = section_per_model(rows, df_axes)
    df_per_project = section_per_project(rows, df_axes)
    df_bridge = section_bridge(rows, df_axes)
    section_writeup(df_axes, df_clusters, df_per_model, df_per_project, df_bridge)
```

- [ ] **Step 3: Run + inspect**

```bash
./.venv/bin/python scripts/16_eriskii_replication.py
```

Expected: re-runs all sections (fast except the description-embedding load and t-SNE), then `wrote .../data/eriskii_comparison.md`.

```bash
cat data/eriskii_comparison.md | head -60
```

Expected: a clean markdown narrative with tables for top-20, cross-reference, cluster labels, per-model means, mechanistic-bridge correlations.

- [ ] **Step 4: Commit**

```bash
git add scripts/16_eriskii_replication.py data/eriskii_comparison.md
git commit -m "eriskii: narrative writeup (eriskii_comparison.md)"
```

---

## Task 14: Final integration + CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Re-run the full pipeline end-to-end**

(All three scripts, in order. Should be a no-op on script 14 since descriptions are already done.)

```bash
./.venv/bin/python scripts/14_claude_haiku_describe.py
./.venv/bin/python scripts/15_claude_faces_embed_description.py
./.venv/bin/python scripts/16_eriskii_replication.py
```

- [ ] **Step 2: Verify all expected outputs exist**

```bash
./.venv/bin/python -c "
from pathlib import Path
expected = [
    'data/claude_haiku_descriptions.jsonl',
    'data/claude_faces_embed_description.parquet',
    'data/eriskii_axes.tsv',
    'data/eriskii_clusters.tsv',
    'data/eriskii_per_model.tsv',
    'data/eriskii_per_project.tsv',
    'data/eriskii_user_kaomoji_axis_corr.tsv',
    'data/eriskii_comparison.md',
    'figures/eriskii_clusters_tsne.png',
    'figures/eriskii_per_model_axes_mean.png',
    'figures/eriskii_per_model_axes_std.png',
    'figures/eriskii_per_project_axes_mean.png',
    'figures/eriskii_per_project_axes_std.png',
    'figures/eriskii_user_kaomoji_axis_corr.png',
] + [f'figures/eriskii_axis_{a}.png' for a in [
    'warmth', 'energy', 'confidence', 'playfulness', 'empathy',
    'technicality', 'positivity', 'curiosity', 'approval',
    'wrynness', 'wetness',
]]
missing = [p for p in expected if not Path(p).exists()]
print('missing:', missing)
assert not missing, missing
print('OK: all 25 expected files present')
"
```

Expected: `OK: all 25 expected files present`.

- [ ] **Step 3: Update `CLAUDE.md`**

Add a new top-level section after the existing `## Parallel side-experiment: Claude-faces scrape` section, before `## Hidden-state refactor`:

```markdown
## Eriskii-replication on Claude-faces

Description-based reanalysis of `data/claude_kaomoji.jsonl`,
methodologically parallel to eriskii.net/projects/claude-faces.
For each of 436 emissions: mask the leading kaomoji, ask Haiku
4-5 what the masked face conveyed, embed that description with
MiniLM, mean-pool per kaomoji, project onto 11 anchored
semantic axes (Warmth, Energy, Confidence, Playfulness,
Empathy, Technicality, Positivity, Curiosity, Approval,
Wrynness, Wetness — the last reading "wet Claude" as
emotionally-expressive/lyrical and "dry Claude" as
helpful-assistant register, per a9lim).

Three outputs eriskii didn't produce:
  - per-model and per-project axis breakouts (mean + std,
    operationalizes eriskii's qualitative "opus-4-6 had wider
    range" claim);
  - surrounding_user → kaomoji axis correlation as a state-tracking
    bridge, on the ~73% of rows where the parent-chain walk
    resolved to a non-empty user message.

Pre-registered in
`docs/superpowers/specs/2026-04-24-eriskii-replication-design.md`
(anchor pairs, Haiku prompts, model id, mask token, n≥3
threshold). Implemented via
`docs/superpowers/plans/2026-04-24-eriskii-replication.md`.

Outputs:
  - `data/claude_haiku_descriptions.jsonl` (436 rows; resumable)
  - `data/claude_faces_embed_description.parquet` (description-based
    embeddings, coexists with response-based
    `claude_faces_embed.parquet`)
  - `data/eriskii_axes.tsv`, `eriskii_clusters.tsv`,
    `eriskii_per_model.tsv`, `eriskii_per_project.tsv`,
    `eriskii_user_kaomoji_axis_corr.tsv`,
    `eriskii_comparison.md`
  - `figures/eriskii_axis_<name>.png` × 11,
    `eriskii_clusters_tsne.png`,
    `eriskii_per_{model,project}_axes_{mean,std}.png`,
    `eriskii_user_kaomoji_axis_corr.png`
```

Also add a one-line entry under `## Commands`, after the
existing claude-faces commands:

```bash
# Eriskii-replication (description-based; needs ANTHROPIC_API_KEY)
python scripts/14_claude_haiku_describe.py             # 436 Haiku calls, resumable
python scripts/15_claude_faces_embed_description.py    # description embeddings
python scripts/16_eriskii_replication.py               # axes + clusters + breakouts + writeup
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: claude.md document eriskii-replication pipeline"
```

---

## Self-Review

(Done inline. Minor type cross-checks performed against earlier tasks:
- `compute_embeddings`/`save_embeddings`/`load_embeddings` signatures
  match `llmoji/claude_faces.py`.
- `weighted_group_axis_stats` returns long-form `[group_col, axis,
  mean, std, n]` consumed by `_heatmap`.
- `user_kaomoji_axis_correlation` returns `[axis, r, p, p_bonf, n]`
  consumed by `section_bridge` and `section_writeup`.
- All TSV/figure paths reference `llmoji.config` constants;
  no hard-coded paths in scripts.
- All Haiku-calling code paths check `ANTHROPIC_API_KEY` and bail
  with `sys.exit(1)` on absence.)

## Spec coverage check

Spec section → task that implements it:
- §3.1 (script 14, masked-context Haiku) → Task 5 + Task 6
- §3.2 (script 15, description embeddings) → Task 7
- §3.3 (script 16) → Tasks 9–13
- §4 (11 anchored axes, locked anchors) → Task 3 (lock) + Task 8
  (compute primitive) + Task 9 (project + figures)
- §5 (cluster labeling) → Task 10
- §6.1 (per-model breakout, mean+std) → Task 11
- §6.2 (per-project breakout, mean+std) → Task 11
- §6.3 (mechanistic bridge, Pearson r + Bonferroni) → Task 12
- §7 (full file map) → verified end-to-end in Task 14
- §8 (scope discipline) → no out-of-scope tasks introduced
- §9 (pre-registration locking) → Task 2 + Task 3 lock the strings
  in `config.py` and `eriskii_prompts.py`; this plan's commits
  are git-traceable

No gaps.
