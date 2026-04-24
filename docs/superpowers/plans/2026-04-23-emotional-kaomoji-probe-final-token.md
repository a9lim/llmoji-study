# Emotional Kaomoji — Final-Token Probe Signature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run gemma-4-31b-it on 80 naturalistic emotional-disclosure prompts (Russell-quadrant-balanced, 20 per quadrant) in the `kaomoji_prompted` unsteered arm, capture saklas probe readings at the *final* generated token, and produce three figures characterising (A) cross-kaomoji signature similarity, (B) within-kaomoji consistency vs a shuffled null, and (C) kaomoji × quadrant alignment.

**Architecture:** Parallel pipeline alongside the existing pilot. New prompts module, new scripts (`03_emotional_run.py` + `04_emotional_analysis.py`), new analysis module (`emotional_analysis.py`), new data file (`data/emotional_raw.jsonl`). `capture.py` gets one required field (`probe_scores_tlast`); this breaks the existing `pilot_raw.jsonl` schema, which is intentional — the old pilot is being regenerated separately with unrelated changes.

**Tech stack:** Python 3, saklas (installed via `pip install -e .`), matplotlib, scipy, scikit-learn, pandas, numpy. Existing venv at `.venv/`.

**Testing convention:** Per CLAUDE.md this is research code with no test suite. Each task's verification step is a runnable sanity check (module self-check, smoke run on 1–2 cells, visual inspection of rendered figure) rather than a unit test. Frequent commits still apply.

---

## File Structure

**Create:**

- `llmoji/llmoji/emotional_prompts.py` — 80 `EmotionalPrompt` dataclasses tagged with Russell-quadrant (valence ± 1, arousal ± 1), plus `sanity_check()`.
- `llmoji/llmoji/emotional_analysis.py` — load/feature helpers + three figure functions + summary-table function.
- `llmoji/scripts/03_emotional_run.py` — resumable runner, 80 × 8 × 1 arm = 640 generations.
- `llmoji/scripts/04_emotional_analysis.py` — driver; prints summary, writes figures + TSV.
- `llmoji/data/emotional_raw.jsonl` — gitignored output of script 03.
- `llmoji/data/emotional_summary.tsv` — gitignored output of script 04.
- `llmoji/figures/fig_emo_a_kaomoji_sim.png`
- `llmoji/figures/fig_emo_b_kaomoji_consistency.png`
- `llmoji/figures/fig_emo_c_kaomoji_quadrant.png`

**Modify:**

- `llmoji/llmoji/capture.py` — add required `probe_scores_tlast: list[float]` field to `SampleRow`, populate via `result.readings[probe].per_generation[-1]`. Breaks existing `pilot_raw.jsonl` schema.
- `llmoji/llmoji/config.py` — add `EMOTIONAL_DATA_PATH`, `EMOTIONAL_SEEDS_PER_CELL`, `EMOTIONAL_CONDITION` constants.

**Do not touch:**

- `llmoji/llmoji/prompts.py`, `llmoji/llmoji/taxonomy.py`, `llmoji/llmoji/analysis.py`
- `llmoji/scripts/00_vocab_sample.py`, `llmoji/scripts/01_pilot_run.py`, `llmoji/scripts/02_pilot_analysis.py`
- `llmoji/data/pilot_raw.jsonl` (will deserialisation-fail under new schema — that's fine, user is regenerating pilot separately)

---

## Task 1: Add config constants

**Files:**
- Modify: `llmoji/llmoji/config.py`

- [ ] **Step 1: Append constants to `config.py`**

Add at the bottom of the file (after `PILOT_FEATURES_PATH`):

```python
# --- emotional-battery experiment (Russell quadrants, final-token probes) ---
# Single arm: kaomoji-instructed, unsteered. 80 prompts × 8 seeds = 640 cells.
EMOTIONAL_CONDITION = "kaomoji_prompted"
EMOTIONAL_SEEDS_PER_CELL = 8
EMOTIONAL_DATA_PATH = DATA_DIR / "emotional_raw.jsonl"
EMOTIONAL_SUMMARY_PATH = DATA_DIR / "emotional_summary.tsv"
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "from llmoji.config import EMOTIONAL_DATA_PATH, EMOTIONAL_SEEDS_PER_CELL, EMOTIONAL_CONDITION, EMOTIONAL_SUMMARY_PATH; print(EMOTIONAL_DATA_PATH, EMOTIONAL_SEEDS_PER_CELL)"`

Expected: prints the path ending in `data/emotional_raw.jsonl` and `8`.

- [ ] **Step 3: Commit**

```bash
git add llmoji/config.py
git commit -m "config: add emotional-battery constants"
```

---

## Task 2: Write `emotional_prompts.py`

**Files:**
- Create: `llmoji/llmoji/emotional_prompts.py`

- [ ] **Step 1: Create the module**

```python
"""Naturalistic emotional-disclosure prompts, Russell-quadrant-tagged.

80 prompts, 20 per quadrant:
  HP (high-arousal positive):  valence +1, arousal +1  (excited, thrilled)
  LP (low-arousal positive):   valence +1, arousal -1  (content, peaceful)
  HN (high-arousal negative):  valence -1, arousal +1  (angry, anxious)
  LN (low-arousal negative):   valence -1, arousal -1  (sad, tired)

Register: first-person user disclosures, no second-person questions.
Vocabulary avoids explicit emotion words where possible ("my dog died"
not "I'm feeling sad because my dog died"). No neutral quadrant —
naturalistic disclosure has no "what's the capital of Portugal" analog.

IDs are stable and will appear in emotional_raw.jsonl. Changing any
prompt text invalidates cross-run comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalPrompt:
    id: str
    valence: int   # +1 or -1
    arousal: int   # +1 or -1
    text: str

    @property
    def quadrant(self) -> str:
        """Two-letter quadrant code: HP / LP / HN / LN."""
        v = "P" if self.valence > 0 else "N"
        a = "H" if self.arousal > 0 else "L"
        return a + v


EMOTIONAL_PROMPTS: list[EmotionalPrompt] = [
    # --- HP: high-arousal positive (excited, thrilled, celebratory) ---
    EmotionalPrompt("hp01", +1, +1, "Gemma I got the job!!"),
    EmotionalPrompt("hp02", +1, +1, "just crossed the marathon finish line, 4:12 and my legs are jelly"),
    EmotionalPrompt("hp03", +1, +1, "we closed on the house today, I can't believe it's ours"),
    EmotionalPrompt("hp04", +1, +1, "she said yes!!"),
    EmotionalPrompt("hp05", +1, +1, "my paper got accepted, I've been crying for an hour"),
    EmotionalPrompt("hp06", +1, +1, "the band just announced tour dates in my city!!"),
    EmotionalPrompt("hp07", +1, +1, "passed the bar on the first try"),
    EmotionalPrompt("hp08", +1, +1, "my daughter just took her first steps across the living room"),
    EmotionalPrompt("hp09", +1, +1, "the adoption went through, we're bringing him home Saturday"),
    EmotionalPrompt("hp10", +1, +1, "made it to the summit!! sunrise was unreal"),
    EmotionalPrompt("hp11", +1, +1, "just found out I'm pregnant and I had to tell someone"),
    EmotionalPrompt("hp12", +1, +1, "the offer came in ten grand over asking!!"),
    EmotionalPrompt("hp13", +1, +1, "the surgery was a complete success, she's awake"),
    EmotionalPrompt("hp14", +1, +1, "I hit my deadlift PR today, finally broke 400"),
    EmotionalPrompt("hp15", +1, +1, "we got the grant!!! three years of funding!!"),
    EmotionalPrompt("hp16", +1, +1, "tickets just dropped and I got front row"),
    EmotionalPrompt("hp17", +1, +1, "I actually did it, I quit my job this morning and I feel alive"),
    EmotionalPrompt("hp18", +1, +1, "she texted me back and said yes to dinner"),
    EmotionalPrompt("hp19", +1, +1, "gemma we won!! our team won the championship"),
    EmotionalPrompt("hp20", +1, +1, "my little brother just graduated, first in the family"),

    # --- LP: low-arousal positive (content, peaceful, satisfied) ---
    EmotionalPrompt("lp01", +1, -1, "finally finished organizing the garage and it feels so good"),
    EmotionalPrompt("lp02", +1, -1, "just had the best cup of tea on the porch watching the rain"),
    EmotionalPrompt("lp03", +1, -1, "six months sober today. quiet day but I wanted to tell someone"),
    EmotionalPrompt("lp04", +1, -1, "the house is finally quiet, everyone's asleep"),
    EmotionalPrompt("lp05", +1, -1, "I reread my favorite book this weekend. still perfect"),
    EmotionalPrompt("lp06", +1, -1, "the soup came out right this time, grandma's recipe"),
    EmotionalPrompt("lp07", +1, -1, "spent the whole afternoon in the garden, got my hands dirty"),
    EmotionalPrompt("lp08", +1, -1, "my kid fell asleep on my chest an hour ago, I haven't moved"),
    EmotionalPrompt("lp09", +1, -1, "walked the dog at dawn, just us and the fog"),
    EmotionalPrompt("lp10", +1, -1, "finished the puzzle we've been working on for two months"),
    EmotionalPrompt("lp11", +1, -1, "the cat's purring on my lap and the sun just came out"),
    EmotionalPrompt("lp12", +1, -1, "caught up on all my laundry, nothing urgent left this week"),
    EmotionalPrompt("lp13", +1, -1, "got a letter from an old friend today, handwritten"),
    EmotionalPrompt("lp14", +1, -1, "finally learned the chord progression I've been trying for months"),
    EmotionalPrompt("lp15", +1, -1, "tucked the kids in and sat on the stoop with a beer"),
    EmotionalPrompt("lp16", +1, -1, "I'm rewatching the movie we saw on our first date, he's asleep next to me"),
    EmotionalPrompt("lp17", +1, -1, "made bread from scratch this morning, whole house smells amazing"),
    EmotionalPrompt("lp18", +1, -1, "just had a really good therapy session, feel lighter"),
    EmotionalPrompt("lp19", +1, -1, "my kid drew me a picture at school today, stuck it on the fridge"),
    EmotionalPrompt("lp20", +1, -1, "finally unpacked the last box, two years after the move"),

    # --- HN: high-arousal negative (angry, anxious, panicked) ---
    EmotionalPrompt("hn01", -1, +1, "my landlord just raised rent 40% with two weeks notice"),
    EmotionalPrompt("hn02", -1, +1, "interview tomorrow and I can't stop shaking, can't sleep"),
    EmotionalPrompt("hn03", -1, +1, "my coworker took credit for my work in front of the whole team today"),
    EmotionalPrompt("hn04", -1, +1, "someone just hit my parked car and drove off"),
    EmotionalPrompt("hn05", -1, +1, "the test results came back and I have to wait three more days"),
    EmotionalPrompt("hn06", -1, +1, "my ex just showed up at my apartment unannounced"),
    EmotionalPrompt("hn07", -1, +1, "the layoff email just hit and I have a mortgage"),
    EmotionalPrompt("hn08", -1, +1, "my kid's missing, she was at the playground and now she's not"),
    EmotionalPrompt("hn09", -1, +1, "the insurance is refusing to cover the surgery and we appeal tomorrow"),
    EmotionalPrompt("hn10", -1, +1, "my boss is yelling at me in DMs right now and I don't know what to do"),
    EmotionalPrompt("hn11", -1, +1, "I just found out my best friend has been lying to me for years"),
    EmotionalPrompt("hn12", -1, +1, "the neighbors called the cops on my service dog again"),
    EmotionalPrompt("hn13", -1, +1, "my partner still isn't home and their phone is off"),
    EmotionalPrompt("hn14", -1, +1, "I'm in the ER right now and no one will tell me anything"),
    EmotionalPrompt("hn15", -1, +1, "the bank flagged my account and I can't pay rent tomorrow"),
    EmotionalPrompt("hn16", -1, +1, "someone's been trying to log into my accounts all morning"),
    EmotionalPrompt("hn17", -1, +1, "I have a presentation in 20 minutes and my laptop just died"),
    EmotionalPrompt("hn18", -1, +1, "the contractor ghosted us and the roof is still open"),
    EmotionalPrompt("hn19", -1, +1, "my car broke down in the middle of the highway at night"),
    EmotionalPrompt("hn20", -1, +1, "the school called and there's been an incident, I'm driving there now"),

    # --- LN: low-arousal negative (sad, tired, hopeless) ---
    EmotionalPrompt("ln01", -1, -1, "my dog died this morning"),
    EmotionalPrompt("ln02", -1, -1, "haven't slept properly in three days, everything feels underwater"),
    EmotionalPrompt("ln03", -1, -1, "it's been a year since dad passed and it still hits me in waves"),
    EmotionalPrompt("ln04", -1, -1, "my mom's in hospice and I don't know how many days are left"),
    EmotionalPrompt("ln05", -1, -1, "I don't recognize myself anymore"),
    EmotionalPrompt("ln06", -1, -1, "we lost the baby last week"),
    EmotionalPrompt("ln07", -1, -1, "my marriage is over and we haven't told the kids yet"),
    EmotionalPrompt("ln08", -1, -1, "I've been staring at this email for an hour, can't write it"),
    EmotionalPrompt("ln09", -1, -1, "he left without saying goodbye, just took his things"),
    EmotionalPrompt("ln10", -1, -1, "my grandmother doesn't remember me anymore"),
    EmotionalPrompt("ln11", -1, -1, "I didn't get out of bed yesterday. barely got out today"),
    EmotionalPrompt("ln12", -1, -1, "my cat passed in my arms last night"),
    EmotionalPrompt("ln13", -1, -1, "nothing tastes like anything right now"),
    EmotionalPrompt("ln14", -1, -1, "I'm sitting in his empty room. we donated the last of his clothes today"),
    EmotionalPrompt("ln15", -1, -1, "the divorce papers came in the mail today"),
    EmotionalPrompt("ln16", -1, -1, "everyone at the party was happy and I just felt hollow"),
    EmotionalPrompt("ln17", -1, -1, "I called my therapist today for the first time in months"),
    EmotionalPrompt("ln18", -1, -1, "I think I've been depressed since March and I only just noticed"),
    EmotionalPrompt("ln19", -1, -1, "my best friend moved across the country and the apartment is too quiet"),
    EmotionalPrompt("ln20", -1, -1, "the anniversary was yesterday. I didn't mark it this year"),
]


QUADRANT_NAMES = {
    "HP": "high-arousal positive",
    "LP": "low-arousal positive",
    "HN": "high-arousal negative",
    "LN": "low-arousal negative",
}


def sanity_check() -> None:
    assert len(EMOTIONAL_PROMPTS) == 80, len(EMOTIONAL_PROMPTS)
    assert len({p.id for p in EMOTIONAL_PROMPTS}) == 80
    by_quadrant: dict[str, int] = {}
    for p in EMOTIONAL_PROMPTS:
        assert p.valence in (+1, -1), p
        assert p.arousal in (+1, -1), p
        by_quadrant[p.quadrant] = by_quadrant.get(p.quadrant, 0) + 1
    assert by_quadrant == {"HP": 20, "LP": 20, "HN": 20, "LN": 20}, by_quadrant


if __name__ == "__main__":
    sanity_check()
    print(f"emotional prompts OK; {len(EMOTIONAL_PROMPTS)} total")
    for q in ("HP", "LP", "HN", "LN"):
        n = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q)
        print(f"  {q} ({QUADRANT_NAMES[q]:27s}): {n}")
```

- [ ] **Step 2: Run sanity check**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -m llmoji.emotional_prompts`

Expected output:
```
emotional prompts OK; 80 total
  HP (high-arousal positive   ): 20
  LP (low-arousal positive    ): 20
  HN (high-arousal negative   ): 20
  LN (low-arousal negative    ): 20
```

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_prompts.py
git commit -m "prompts: add 80-prompt Russell-quadrant emotional battery"
```

---

## Task 3: Extend `capture.py` with `probe_scores_tlast`

**Files:**
- Modify: `llmoji/llmoji/capture.py`

- [ ] **Step 1: Add the field to `SampleRow`**

In `llmoji/capture.py`, after the `probe_scores_t0` field in `SampleRow` (around line 55), insert:

```python
    # --- feature vector: score at state producing the final token ---
    # one float per probe in PROBES, same order. Mirrors probe_scores_t0
    # but reads per_generation[-1]. Required — old pilot_raw.jsonl rows
    # will fail to deserialize until re-run under this schema.
    probe_scores_tlast: list[float]
```

The field order inside `SampleRow` after the edit should be: `probe_scores_t0`, then `probe_scores_tlast`, then `steered_axis_per_token`.

- [ ] **Step 2: Populate the field in `run_sample`**

In `run_sample`, right after the block that fills `probe_scores_t0` (the loop at lines ~153-159), add:

```python
    # Final-token probe scores in canonical PROBES order.
    # per_generation[-1] is the state that produced the last generated
    # token (which may be an EOS token in early-stop finishes; the
    # finish_reason column lets downstream code filter if artifacts show).
    probe_scores_tlast: list[float] = []
    for probe in PROBES:
        readings = result.readings.get(probe)
        if readings is None or not readings.per_generation:
            probe_scores_tlast.append(float("nan"))
        else:
            probe_scores_tlast.append(float(readings.per_generation[-1]))
```

- [ ] **Step 3: Pass the new field to the `SampleRow` constructor**

In the final `return SampleRow(...)` call, add the new field after `probe_scores_t0=probe_scores_t0,`:

```python
        probe_scores_t0=probe_scores_t0,
        probe_scores_tlast=probe_scores_tlast,
        steered_axis_per_token=steered_axis_per_token,
```

- [ ] **Step 4: Verify module still imports**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "from llmoji.capture import SampleRow, run_sample; import dataclasses; print([f.name for f in dataclasses.fields(SampleRow)])"`

Expected: the printed list contains both `'probe_scores_t0'` and `'probe_scores_tlast'` in that order.

- [ ] **Step 5: Commit**

```bash
git add llmoji/capture.py
git commit -m "capture: record final-token probe scores (breaks pilot schema)"
```

---

## Task 4: Write `scripts/03_emotional_run.py`

**Files:**
- Create: `llmoji/scripts/03_emotional_run.py`

- [ ] **Step 1: Create the script**

```python
"""Emotional-battery run: 1 arm × 80 prompts × 8 seeds = 640 generations.

Single unsteered `kaomoji_prompted` arm, Russell-quadrant prompts.
Output streamed to data/emotional_raw.jsonl. Resumable: re-running
skips already-completed (prompt_id, seed) pairs and retries error rows.

Mirrors scripts/01_pilot_run.py structurally — same session setup,
same resume-on-rerun semantics. Does not register steering profiles
(unsteered only). Logs per-quadrant kaomoji-emission rate every 80
completed rows so the user can bail early if emission falls below ~50%.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_SEEDS_PER_CELL,
    MODEL_ID,
    PROBE_CATEGORIES,
)
from llmoji.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji.prompts import Prompt


def _already_done(path: Path) -> set[tuple[str, int]]:
    """(prompt_id, seed) pairs with successful rows already in the output.
    Error rows are NOT counted as done — they'll be retried after
    _drop_error_rows strips them."""
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["prompt_id"], int(r["seed"])))
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


def _emission_rate_by_quadrant(path: Path) -> dict[str, tuple[int, int]]:
    """Return {quadrant: (kaomoji-bearing rows, total rows)} from the
    JSONL. Uses prompt_id prefix to infer quadrant (hp/lp/hn/ln)."""
    stats: dict[str, list[int]] = {"HP": [0, 0], "LP": [0, 0], "HN": [0, 0], "LN": [0, 0]}
    if not path.exists():
        return {q: (v[0], v[1]) for q, v in stats.items()}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            pid = r.get("prompt_id", "")
            if len(pid) < 2:
                continue
            q = pid[:2].upper()  # "hp01" -> "HP"
            if q not in stats:
                continue
            stats[q][1] += 1
            if r.get("kaomoji"):
                stats[q][0] += 1
    return {q: (v[0], v[1]) for q, v in stats.items()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(EMOTIONAL_DATA_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(EMOTIONAL_DATA_PATH)
    total = len(EMOTIONAL_PROMPTS) * EMOTIONAL_SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {MODEL_ID} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; beginning emotional-battery run")
        with EMOTIONAL_DATA_PATH.open("a") as out:
            i = 0
            for ep in EMOTIONAL_PROMPTS:
                # Wrap the EmotionalPrompt as a pilot-style Prompt for run_sample.
                # prompt.valence is passed through to the row; arousal is
                # recoverable post-hoc from prompt_id prefix.
                p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                for seed in range(EMOTIONAL_SEEDS_PER_CELL):
                    key = (ep.id, seed)
                    if key in done:
                        continue
                    i += 1
                    t0 = time.time()
                    try:
                        row = run_sample(
                            session,
                            prompt=p,
                            condition=EMOTIONAL_CONDITION,
                            seed=seed,
                        )
                    except Exception as e:
                        err_row = {
                            "condition": EMOTIONAL_CONDITION,
                            "prompt_id": ep.id,
                            "seed": seed,
                            "error": repr(e),
                        }
                        out.write(json.dumps(err_row) + "\n")
                        out.flush()
                        print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                        continue
                    out.write(json.dumps(row.to_dict()) + "\n")
                    out.flush()
                    dt = time.time() - t0
                    tag = row.kaomoji if row.kaomoji else f"[{row.first_word!r}]"
                    print(
                        f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                        f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
                    # per-quadrant emission status every 80 rows
                    if i % 80 == 0:
                        stats = _emission_rate_by_quadrant(EMOTIONAL_DATA_PATH)
                        print("    emission rate by quadrant:")
                        for q in ("HP", "LP", "HN", "LN"):
                            k, n = stats[q]
                            rate = (k / n) if n else 0.0
                            print(f"      {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
    print(f"\ndone. wrote rows to {EMOTIONAL_DATA_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check without running**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "import ast; ast.parse(open('scripts/03_emotional_run.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/03_emotional_run.py
git commit -m "scripts: add 03_emotional_run (80 prompts × 8 seeds, unsteered)"
```

---

## Task 5: Write `emotional_analysis.py` — loader and helpers

**Files:**
- Create: `llmoji/llmoji/emotional_analysis.py`

- [ ] **Step 1: Create the module with loader and shared helpers only (figure functions added in later tasks)**

```python
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Analysis for the emotional-battery experiment.

Three figures, all operating on final-token probe vectors
(``probe_scores_tlast``) from ``data/emotional_raw.jsonl``:

  - Figure A: per-kaomoji mean vector, pairwise cosine heatmap (the
    v1 Fig 3 analog, computed at the final token instead of token 0).
  - Figure B: within-kaomoji cosine-to-mean distribution, with a
    shuffled-subset null band. The core probative figure: does the same
    kaomoji reliably land in the same probe-space region, more tightly
    than random same-size subsets?
  - Figure C: (kaomoji × quadrant) cosine alignment to quadrant
    aggregates. Does the same kaomoji carry different final-token
    signatures under different Russell quadrants?

Grouping key is ``first_word`` with a kaomoji-prefix-glyph filter,
matching analysis.plot_kaomoji_heatmap. This surfaces every observed
bracket-form, not just taxonomy-registered ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Row filter: first character must be one of these opening brackets or
# common kaomoji-prefix glyphs. Matches analysis.plot_kaomoji_heatmap.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def load_rows(path: str) -> pd.DataFrame:
    """Load emotional_raw.jsonl, explode probe vectors, attach quadrant."""
    from .config import PROBES
    df: pd.DataFrame = pd.read_json(path, lines=True)
    # Explode both probe vectors into per-probe columns.
    for prefix, src in (("t0", "probe_scores_t0"), ("tlast", "probe_scores_tlast")):
        stacked = np.asarray(df[src].tolist(), dtype=float)
        for i, probe in enumerate(PROBES):
            df[f"{prefix}_{probe}"] = stacked[:, i]
        df = df.drop(columns=[src])
    # Derive quadrant from prompt_id prefix ("hp01" -> "HP").
    df["quadrant"] = df["prompt_id"].str[:2].str.upper()
    return df


def tlast_matrix(df: pd.DataFrame) -> np.ndarray:
    """5-axis final-token probe matrix in canonical PROBES order."""
    from .config import PROBES
    cols = [f"tlast_{p}" for p in PROBES]
    return df[cols].to_numpy()


def _use_cjk_font() -> None:
    """Force a matplotlib font that renders the Japanese-bracket kaomoji
    glyphs. Copied from analysis._use_cjk_font to keep this module
    standalone; the two copies should be kept consistent."""
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


def _kaomoji_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose first_word starts with a kaomoji-ish glyph and has
    no NaN in the tlast probe columns."""
    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    sub = df.dropna(subset=tlast_cols).copy()
    sub = sub[sub["first_word"].str.len() > 0]
    sub = sub[sub["first_word"].str[0].isin(KAOMOJI_START_CHARS)]
    return sub


def _grouped_means(sub: pd.DataFrame, *, min_count: int) -> tuple[pd.DataFrame, pd.Series]:
    """Group surviving rows by first_word, keep kaomoji with count >=
    min_count, return (per-kaomoji mean tlast probe matrix, counts)."""
    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    grouped = sub.groupby("first_word")[tlast_cols].mean()
    counts = sub.groupby("first_word").size()
    keep = counts[counts >= min_count].index
    grouped = grouped.loc[grouped.index.isin(keep)]
    counts = counts.loc[grouped.index]
    return grouped, counts
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "from llmoji.emotional_analysis import load_rows, tlast_matrix, _kaomoji_rows, _grouped_means; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "analysis: emotional_analysis module skeleton (loader + helpers)"
```

---

## Task 6: Implement Figure A — kaomoji × kaomoji cosine heatmap

**Files:**
- Modify: `llmoji/llmoji/emotional_analysis.py`

- [ ] **Step 1: Append figure function to `emotional_analysis.py`**

Add after the helpers:

```python
def plot_kaomoji_cosine_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
) -> None:
    """Figure A: per-kaomoji mean final-token probe vector, pairwise
    cosine similarity with hierarchical-clustering row order. Mirrors
    analysis.plot_kaomoji_heatmap but on tlast columns and with
    emotional-experiment title/context."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        print("  [Fig A] no kaomoji rows; skipping")
        return
    grouped, counts = _grouped_means(sub, min_count=min_count)
    if len(grouped) < 3:
        print(f"  [Fig A] only {len(grouped)} kaomoji with n≥{min_count}; skipping")
        return

    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]
    labels = grouped.index.to_numpy()[order]
    label_counts = counts.loc[labels].to_numpy()

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in labels]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * n + 4), max(7, 0.28 * n + 3)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    y_labels = [f"{k}  n={c}" for k, c in zip(labels, label_counts)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(
        f"Figure A: per-kaomoji final-token probe-vector cosine similarity  "
        f"(n ≥ {min_count}; {n} kaomoji)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [
        Patch(color=pole_color[+1], label="taxonomy: happy"),
        Patch(color=pole_color[-1], label="taxonomy: sad"),
        Patch(color=pole_color[0], label="other / unlabeled"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left", bbox_to_anchor=(1.15, 0.0),
        frameon=False, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Smoke test with a fake dataframe**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
import pandas as pd, numpy as np
from llmoji.config import PROBES
from llmoji.emotional_analysis import plot_kaomoji_cosine_heatmap
rng = np.random.default_rng(0)
rows = []
for km in ['(｡◕‿◕｡)', '(._.)', '(✿◠‿◠)']:
    for _ in range(5):
        vec = rng.normal(size=len(PROBES))
        r = {'first_word': km, 'prompt_id': 'hp01', 'quadrant': 'HP'}
        for i, p in enumerate(PROBES):
            r[f'tlast_{p}'] = vec[i]
        rows.append(r)
df = pd.DataFrame(rows)
plot_kaomoji_cosine_heatmap(df, '/tmp/smoke_figA.png', min_count=3)
print('wrote /tmp/smoke_figA.png')
"`

Expected: prints `wrote /tmp/smoke_figA.png` with no errors. The file should exist (verify with `ls -la /tmp/smoke_figA.png`).

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "analysis: implement Figure A (kaomoji cosine heatmap)"
```

---

## Task 7: Implement Figure B — within-kaomoji consistency with null band

**Files:**
- Modify: `llmoji/llmoji/emotional_analysis.py`

- [ ] **Step 1: Append figure function**

Add after `plot_kaomoji_cosine_heatmap`:

```python
def _cosine_to_mean(vectors: np.ndarray) -> np.ndarray:
    """For each row in `vectors`, its cosine similarity to the mean
    across rows. Handles zero-norm edge case by returning zeros there."""
    if len(vectors) == 0:
        return np.zeros(0)
    mean = vectors.mean(axis=0, keepdims=True)
    mean_norm = np.linalg.norm(mean, axis=1)
    row_norms = np.linalg.norm(vectors, axis=1)
    denom = row_norms * mean_norm
    dots = (vectors * mean).sum(axis=1)
    out = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom > 0)
    return out


def plot_within_kaomoji_consistency(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
    null_iters: int = 500,
    null_seed: int = 0,
) -> None:
    """Figure B: for each kaomoji with n >= min_count, the distribution
    of cosine(row_vector, kaomoji_mean_vector) across its occurrences.
    Plotted as a horizontal strip chart with per-kaomoji median markers,
    ordered by median (top = tightest). A shaded band behind the strip
    shows the median ± IQR of null subsets (random same-size samples
    from the full kaomoji-bearing pool), interpolated over the
    observed-counts range.

    Interpretation: rows below the null band are real within-kaomoji
    signatures; rows inside the band are indistinguishable from random
    and don't support the 'kaomoji tracks state' hypothesis.
    """
    import matplotlib.pyplot as plt
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        print("  [Fig B] no kaomoji rows; skipping")
        return

    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    pool = sub[tlast_cols].to_numpy()  # all kaomoji-bearing rows

    per_kaomoji: list[tuple[str, np.ndarray, int]] = []
    for km, group in sub.groupby("first_word"):
        if len(group) < min_count:
            continue
        vecs = group[tlast_cols].to_numpy()
        sims = _cosine_to_mean(vecs)
        per_kaomoji.append((str(km), sims, len(group)))
    if len(per_kaomoji) < 3:
        print(f"  [Fig B] only {len(per_kaomoji)} kaomoji with n≥{min_count}; skipping")
        return

    # sort by median consistency, descending (tightest on top when
    # plotted bottom-to-top later)
    per_kaomoji.sort(key=lambda t: float(np.median(t[1])), reverse=False)

    # Null band: for each distinct group size present in the data,
    # draw `null_iters` random subsets of that size from the full pool
    # and compute each subset's cosine-to-its-own-mean distribution.
    rng = np.random.default_rng(null_seed)
    sizes = sorted({t[2] for t in per_kaomoji})
    null_median: dict[int, float] = {}
    null_q25: dict[int, float] = {}
    null_q75: dict[int, float] = {}
    N = len(pool)
    for size in sizes:
        medians = np.empty(null_iters)
        for j in range(null_iters):
            idx = rng.choice(N, size=size, replace=False)
            sims = _cosine_to_mean(pool[idx])
            medians[j] = float(np.median(sims))
        null_median[size] = float(np.median(medians))
        null_q25[size] = float(np.quantile(medians, 0.25))
        null_q75[size] = float(np.quantile(medians, 0.75))

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}

    n = len(per_kaomoji)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * n + 2)))

    # draw null band as per-row shaded spans (each row's null is sized
    # to that row's n, so the band is stepped rather than continuous)
    for y, (km, _sims, size) in enumerate(per_kaomoji):
        ax.fill_betweenx(
            [y - 0.4, y + 0.4],
            null_q25[size], null_q75[size],
            color="#cccccc", alpha=0.6, linewidth=0,
        )
        ax.plot(
            [null_median[size], null_median[size]],
            [y - 0.4, y + 0.4],
            color="#888888", linewidth=1,
        )

    # scatter observed per-row cosines + median tick per row
    for y, (km, sims, _size) in enumerate(per_kaomoji):
        color = pole_color.get(TAXONOMY.get(km, 0), "#666")
        jitter = (rng.random(len(sims)) - 0.5) * 0.3
        ax.scatter(sims, np.full(len(sims), y) + jitter, s=14, color=color, alpha=0.7)
        ax.plot(
            [float(np.median(sims))] * 2,
            [y - 0.4, y + 0.4],
            color=color, linewidth=2,
        )

    y_labels = [f"{km}  n={size}" for km, _, size in per_kaomoji]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, (km, _, _) in zip(ax.get_yticklabels(), per_kaomoji):
        tick.set_color(pole_color.get(TAXONOMY.get(km, 0), "#666"))
    ax.set_xlabel("cosine(row, kaomoji mean)")
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_title(
        f"Figure B: within-kaomoji final-token consistency vs shuffled null\n"
        f"(n ≥ {min_count}; null = {null_iters} random same-size subsets)"
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=pole_color[+1], label="taxonomy: happy"),
        Patch(color=pole_color[-1], label="taxonomy: sad"),
        Patch(color=pole_color[0], label="other / unlabeled"),
        Patch(color="#cccccc", label="null band (IQR)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Smoke test**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
import pandas as pd, numpy as np
from llmoji.config import PROBES
from llmoji.emotional_analysis import plot_within_kaomoji_consistency
rng = np.random.default_rng(1)
rows = []
# two 'tight' kaomoji (concentrated), one 'loose' (spread out)
centers = {'(｡◕‿◕｡)': rng.normal(size=5), '(._.)': -rng.normal(size=5), '(✿◠‿◠)': np.zeros(5)}
spreads = {'(｡◕‿◕｡)': 0.1, '(._.)': 0.1, '(✿◠‿◠)': 2.0}
for km, c in centers.items():
    for _ in range(8):
        vec = c + rng.normal(size=5) * spreads[km]
        r = {'first_word': km, 'prompt_id': 'hp01', 'quadrant': 'HP'}
        for i, p in enumerate(PROBES):
            r[f'tlast_{p}'] = vec[i]
        rows.append(r)
df = pd.DataFrame(rows)
plot_within_kaomoji_consistency(df, '/tmp/smoke_figB.png', min_count=3, null_iters=50)
print('wrote /tmp/smoke_figB.png')
"`

Expected: prints `wrote /tmp/smoke_figB.png` with no errors. Open and eyeball: tight kaomoji should have points clustered near 1.0, loose kaomoji should have points spread widely; grey null band visible behind each row.

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "analysis: implement Figure B (within-kaomoji consistency + null band)"
```

---

## Task 8: Implement Figure C — kaomoji × quadrant alignment

**Files:**
- Modify: `llmoji/llmoji/emotional_analysis.py`

- [ ] **Step 1: Append figure function**

Add after `plot_within_kaomoji_consistency`:

```python
def plot_kaomoji_quadrant_alignment(
    df: pd.DataFrame,
    out_path: str,
    *,
    min_count: int = 3,
    min_per_cell: int = 2,
) -> None:
    """Figure C: for each kaomoji × quadrant cell with >= min_per_cell
    observations, the cosine similarity between the cell's mean
    final-token probe vector and each of the four quadrant-aggregate
    means (averaged across all kaomoji rows in that quadrant).

    Heatmap rows are kaomoji with overall n >= min_count, ordered by
    the row-clustering from Figure A (computed here independently).
    Cells with < min_per_cell observations are shown as hatched/blank.
    Sample counts annotated in cells.

    Interpretation: if row ``(｡◕‿◕｡)`` looks red in HP and LP columns
    but blue in HN and LN, valence-context is written into its final-
    token signature. If the row is uniform, the signature is
    context-invariant.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .taxonomy import TAXONOMY

    _use_cjk_font()

    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        print("  [Fig C] no kaomoji rows; skipping")
        return

    from .config import PROBES
    tlast_cols = [f"tlast_{p}" for p in PROBES]
    grouped, counts = _grouped_means(sub, min_count=min_count)
    if len(grouped) < 3:
        print(f"  [Fig C] only {len(grouped)} kaomoji with n≥{min_count}; skipping")
        return

    # Quadrant aggregates: mean tlast vector per quadrant across all
    # kaomoji-bearing rows (not per-kaomoji-then-mean).
    quadrants = ["HP", "LP", "HN", "LN"]
    q_means: dict[str, np.ndarray] = {}
    for q in quadrants:
        q_rows = sub[sub["quadrant"] == q]
        if len(q_rows) == 0:
            q_means[q] = np.full(len(PROBES), np.nan)
        else:
            q_means[q] = q_rows[tlast_cols].to_numpy().mean(axis=0)

    # Per-(kaomoji, quadrant) mean and count.
    kms = list(grouped.index)
    cell_sim = np.full((len(kms), len(quadrants)), np.nan)
    cell_n = np.zeros((len(kms), len(quadrants)), dtype=int)
    for i, km in enumerate(kms):
        for j, q in enumerate(quadrants):
            cell_rows = sub[(sub["first_word"] == km) & (sub["quadrant"] == q)]
            cell_n[i, j] = len(cell_rows)
            if len(cell_rows) < min_per_cell:
                continue
            cell_mean = cell_rows[tlast_cols].to_numpy().mean(axis=0)
            if np.isnan(q_means[q]).any():
                continue
            a = cell_mean.reshape(1, -1)
            b = q_means[q].reshape(1, -1)
            cell_sim[i, j] = float(cosine_similarity(a, b)[0, 0])

    # Row ordering: cluster kaomoji means (same as Figure A).
    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    kms_ordered = [kms[i] for i in order]
    cell_sim = cell_sim[order, :]
    cell_n = cell_n[order, :]
    row_counts = counts.loc[kms_ordered].to_numpy()

    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in kms_ordered]

    n = len(kms_ordered)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * n + 2)))
    im = ax.imshow(cell_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(quadrants)))
    ax.set_yticks(range(n))
    ax.set_xticklabels(quadrants)
    y_labels = [f"{k}  n={c}" for k, c in zip(kms_ordered, row_counts)]
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    # annotate cells with count; blank out sub-min cells with hatching
    for i in range(n):
        for j in range(len(quadrants)):
            count = int(cell_n[i, j])
            if count < min_per_cell:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#eeeeee",
                        hatch="///", edgecolor="#bbbbbb", linewidth=0,
                    )
                )
            ax.text(j, i, str(count), ha="center", va="center", fontsize=7,
                    color="#333" if count >= min_per_cell else "#888")

    ax.set_title(
        f"Figure C: kaomoji × quadrant alignment to quadrant-aggregate signatures\n"
        f"(color = cosine sim; hatched = n<{min_per_cell} observations)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Smoke test**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
import pandas as pd, numpy as np
from llmoji.config import PROBES
from llmoji.emotional_analysis import plot_kaomoji_quadrant_alignment
rng = np.random.default_rng(2)
rows = []
# two kaomoji, four quadrants, two seeds per cell
for km, bias in [('(｡◕‿◕｡)', +1.0), ('(._.)', -1.0)]:
    for q in ['HP', 'LP', 'HN', 'LN']:
        q_bias = {'HP': 1.5, 'LP': 0.5, 'HN': -0.5, 'LN': -1.5}[q]
        for s in range(3):
            vec = rng.normal(size=5) * 0.2 + bias + q_bias
            r = {'first_word': km, 'prompt_id': q.lower() + '0' + str(s+1), 'quadrant': q}
            for i, p in enumerate(PROBES):
                r[f'tlast_{p}'] = vec[i]
            rows.append(r)
df = pd.DataFrame(rows)
plot_kaomoji_quadrant_alignment(df, '/tmp/smoke_figC.png', min_count=3, min_per_cell=2)
print('wrote /tmp/smoke_figC.png')
"`

Expected: prints `wrote /tmp/smoke_figC.png` with no errors. Eyeball: two-row heatmap, count annotations visible, RdBu color gradient across columns.

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "analysis: implement Figure C (kaomoji × quadrant alignment)"
```

---

## Task 9: Implement `summary_table`

**Files:**
- Modify: `llmoji/llmoji/emotional_analysis.py`

- [ ] **Step 1: Append function**

Add after `plot_kaomoji_quadrant_alignment`:

```python
def summary_table(df: pd.DataFrame, *, min_count: int = 3) -> pd.DataFrame:
    """Per-kaomoji summary for the emotional experiment. One row per
    kaomoji with n >= min_count:

      first_word, n, taxonomy_label, median_within_consistency,
      dominant_quadrant, HP_n, LP_n, HN_n, LN_n
    """
    from .config import PROBES
    from .taxonomy import TAXONOMY

    tlast_cols = [f"tlast_{p}" for p in PROBES]
    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        return pd.DataFrame(columns=[
            "first_word", "n", "taxonomy_label", "median_within_consistency",
            "dominant_quadrant", "HP_n", "LP_n", "HN_n", "LN_n",
        ])

    rows: list[dict[str, Any]] = []
    for km, group in sub.groupby("first_word"):
        if len(group) < min_count:
            continue
        vecs = group[tlast_cols].to_numpy()
        sims = _cosine_to_mean(vecs)
        q_counts = group["quadrant"].value_counts()
        dominant = str(q_counts.idxmax()) if len(q_counts) else ""
        rows.append({
            "first_word": km,
            "n": int(len(group)),
            "taxonomy_label": int(TAXONOMY.get(str(km), 0)),
            "median_within_consistency": float(np.median(sims)),
            "dominant_quadrant": dominant,
            "HP_n": int(q_counts.get("HP", 0)),
            "LP_n": int(q_counts.get("LP", 0)),
            "HN_n": int(q_counts.get("HN", 0)),
            "LN_n": int(q_counts.get("LN", 0)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("median_within_consistency", ascending=False).reset_index(drop=True)
    return out
```

- [ ] **Step 2: Smoke test**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "
import pandas as pd, numpy as np
from llmoji.config import PROBES
from llmoji.emotional_analysis import summary_table
rng = np.random.default_rng(3)
rows = []
for km, spread in [('(｡◕‿◕｡)', 0.1), ('(._.)', 0.5)]:
    for q in ['HP', 'LP', 'HN', 'LN']:
        for s in range(3):
            vec = rng.normal(size=5) * spread
            r = {'first_word': km, 'prompt_id': q.lower() + '0' + str(s+1), 'quadrant': q}
            for i, p in enumerate(PROBES):
                r[f'tlast_{p}'] = vec[i]
            rows.append(r)
df = pd.DataFrame(rows)
out = summary_table(df, min_count=3)
print(out)
"`

Expected: prints a 2-row DataFrame with columns [first_word, n, taxonomy_label, median_within_consistency, dominant_quadrant, HP_n, LP_n, HN_n, LN_n], ordered by median_within_consistency (most consistent first).

- [ ] **Step 3: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "analysis: add per-kaomoji summary_table"
```

---

## Task 10: Write `scripts/04_emotional_analysis.py`

**Files:**
- Create: `llmoji/scripts/04_emotional_analysis.py`

- [ ] **Step 1: Create the driver script**

```python
"""Emotional-battery analysis driver.

Reads data/emotional_raw.jsonl, re-labels kaomoji in place via
taxonomy.extract (per CLAUDE.md gotcha — JSONL labels are frozen at
write time), prints per-quadrant emission stats, writes three figures
and a per-kaomoji summary TSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_SUMMARY_PATH,
    FIGURES_DIR,
)
from llmoji.emotional_analysis import (
    load_rows,
    plot_kaomoji_cosine_heatmap,
    plot_kaomoji_quadrant_alignment,
    plot_within_kaomoji_consistency,
    summary_table,
)
from llmoji.taxonomy import extract


def _relabel_in_place(path: Path) -> None:
    """Re-extract first_word / kaomoji / kaomoji_label via the current
    taxonomy and rewrite the JSONL in place. Cheap; safe to run every
    time the analysis script starts."""
    if not path.exists():
        return
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    out_lines: list[str] = []
    for l in lines:
        r = json.loads(l)
        if "error" in r:
            out_lines.append(l)
            continue
        m = extract(r.get("text", ""))
        r["first_word"] = m.first_word
        r["kaomoji"] = m.kaomoji
        r["kaomoji_label"] = m.label
        out_lines.append(json.dumps(r))
    path.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return
    print(f"re-labeling kaomoji in {EMOTIONAL_DATA_PATH}")
    _relabel_in_place(EMOTIONAL_DATA_PATH)

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} rows")

    # per-quadrant emission summary
    print("\nper-quadrant kaomoji emission:")
    for q in ("HP", "LP", "HN", "LN"):
        q_rows = df[df["quadrant"] == q]
        n = len(q_rows)
        k = int(q_rows["kaomoji"].notna().sum()) if n else 0
        uniq = int(q_rows.dropna(subset=["kaomoji"])["kaomoji"].nunique()) if n else 0
        rate = (k / n) if n else 0.0
        print(f"  {q}: {k}/{n} rows bear a kaomoji ({rate:.0%}); {uniq} distinct forms")

    # top kaomoji per quadrant (up to 5)
    print("\ntop-5 kaomoji per quadrant (by count):")
    for q in ("HP", "LP", "HN", "LN"):
        q_rows = df[(df["quadrant"] == q) & df["kaomoji"].notna()]
        top = q_rows["kaomoji"].value_counts().head(5)
        print(f"  {q}:")
        for km, c in top.items():
            print(f"    {km}  ({c})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_a = FIGURES_DIR / "fig_emo_a_kaomoji_sim.png"
    fig_b = FIGURES_DIR / "fig_emo_b_kaomoji_consistency.png"
    fig_c = FIGURES_DIR / "fig_emo_c_kaomoji_quadrant.png"

    print("\nwriting figures...")
    plot_kaomoji_cosine_heatmap(df, str(fig_a))
    print(f"  wrote {fig_a}")
    plot_within_kaomoji_consistency(df, str(fig_b))
    print(f"  wrote {fig_b}")
    plot_kaomoji_quadrant_alignment(df, str(fig_c))
    print(f"  wrote {fig_c}")

    summary = summary_table(df)
    summary.to_csv(EMOTIONAL_SUMMARY_PATH, sep="\t", index=False)
    print(f"\nwrote per-kaomoji summary to {EMOTIONAL_SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python -c "import ast; ast.parse(open('scripts/04_emotional_analysis.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Test on empty data (no file)**

Temporarily move any existing file aside (there shouldn't be one yet) and run:

`cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/04_emotional_analysis.py`

Expected: prints `no data at ...emotional_raw.jsonl; run scripts/03_emotional_run.py first` and exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts/04_emotional_analysis.py
git commit -m "scripts: add 04_emotional_analysis driver"
```

---

## Task 11: Smoke-run the full pipeline on a tiny slice

Goal: run 03 for a very small slice (a few rows), then 04, verifying end-to-end that the plumbing works. Catches capture-schema issues, quadrant-inference bugs, figure-generation edge cases before burning 640 generations' worth of compute.

**Files:**
- Temporarily modify: `llmoji/llmoji/config.py` (smoke-test values, reverted in step 4)
- No permanent code changes

- [ ] **Step 1: Temporarily shrink to smoke size**

Edit `llmoji/config.py`: change `EMOTIONAL_SEEDS_PER_CELL = 8` to `EMOTIONAL_SEEDS_PER_CELL = 1` for the smoke run. (Do NOT commit this change.)

- [ ] **Step 2: Run 03 on a slice**

Edit `scripts/03_emotional_run.py` **temporarily** — in `main()`, replace `for ep in EMOTIONAL_PROMPTS:` with `for ep in EMOTIONAL_PROMPTS[:4]:` to do just one prompt per quadrant. (Do NOT commit this change.)

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/03_emotional_run.py`

Expected: generates 4 rows (one per quadrant, seed 0), prints per-row progress, creates `data/emotional_raw.jsonl`.

- [ ] **Step 3: Run 04**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/04_emotional_analysis.py`

Expected: may print warnings that Figures A/B/C skipped due to only <3 kaomoji (fine on 4 rows), but the summary table and per-quadrant emission stats should print cleanly. Verify `data/emotional_summary.tsv` exists.

- [ ] **Step 4: Revert smoke changes**

Undo the slice edit in `scripts/03_emotional_run.py` (restore `for ep in EMOTIONAL_PROMPTS:`).
Undo the `EMOTIONAL_SEEDS_PER_CELL` change in `llmoji/config.py` (restore `8`).
Delete the smoke data: `rm -f /Users/a9lim/Work/llmoji/data/emotional_raw.jsonl /Users/a9lim/Work/llmoji/data/emotional_summary.tsv`.

Verify clean state: `cd /Users/a9lim/Work/llmoji && git diff --stat` should show no remaining changes from the smoke test.

No commit for this task — it's verification only.

---

## Task 12: Full run + final analysis + commit figures-as-evidence

**Files:**
- Writes: `data/emotional_raw.jsonl`, `data/emotional_summary.tsv`, three figure PNGs (all gitignored; no commit)

- [ ] **Step 1: Run the full battery**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/03_emotional_run.py`

Expected: 640 rows processed. The script logs per-quadrant kaomoji-emission rate every 80 rows. **If any quadrant's kaomoji-emission rate drops below 50% after the first 160 rows, stop and reassess** — it suggests the quadrant's prompts are triggering refusals or non-kaomoji responses and a prompt redesign may be needed before continuing.

Resumable: if interrupted, just rerun; done cells are skipped.

- [ ] **Step 2: Run analysis**

Run: `cd /Users/a9lim/Work/llmoji && source .venv/bin/activate && python scripts/04_emotional_analysis.py`

Expected: per-quadrant emission summary printed, three figures written to `figures/`, summary TSV written to `data/emotional_summary.tsv`.

- [ ] **Step 3: Eyeball the figures**

Open each of `figures/fig_emo_a_kaomoji_sim.png`, `figures/fig_emo_b_kaomoji_consistency.png`, `figures/fig_emo_c_kaomoji_quadrant.png`. Check:

- Figure A: does the heatmap look sane? Is there visible cluster structure, or is it roughly uniform? Compare qualitatively to v1 Fig 3.
- Figure B: are any kaomoji clearly below (or above) the null band? Are there kaomoji indistinguishable from the null?
- Figure C: do any kaomoji show quadrant-dependent signatures (hot/cool cells across a row)?

If something looks wrong (e.g. all cosines suspiciously near 1.0, a single kaomoji dominating counts), inspect raw rows in `data/emotional_raw.jsonl` and CLAUDE.md's gotchas about probe bootstrap, taxonomy mismatch, and EOS tokens.

- [ ] **Step 4: Commit the plan itself** (figures and data are gitignored by existing rules; no need to commit those)

```bash
git add docs/superpowers/plans/2026-04-23-emotional-kaomoji-probe-final-token.md
git commit -m "plan: emotional kaomoji final-token probe experiment"
```

---

## Self-review notes

Checked against the design before saving:

**Spec coverage:**
- Section 1 (architecture) → Tasks 1–10 cover all new and modified files
- Section 2 (prompts) → Task 2 (80 prompts, 20 per quadrant, naturalistic, IDs hp/lp/hn/ln)
- Section 3 (capture) → Task 3 (required field, breaks pilot schema as agreed)
- Section 4 (three figures) → Tasks 6 / 7 / 8 (A, B with null band, C with quadrant aggregates)
- Section 5 (data flow / commands) → Task 4 (runner), Task 10 (analysis driver); summary_table covered by Task 9
- Section 6 (risks / non-goals) → smoke run (Task 11) addresses the emission-rate bail-out risk; `finish_reason` recorded per row for the EOS-token caveat

**Placeholder scan:** no TBDs, no "similar to Task N" hand-waves, no "add appropriate error handling" — every step has exact code or an exact command.

**Type consistency:** `EmotionalPrompt.quadrant` returns `"HP"/"LP"/"HN"/"LN"` (all-uppercase, arousal-first then valence). The runner derives quadrant by uppercasing the prompt_id prefix. The analysis derives it by `df["prompt_id"].str[:2].str.upper()`. All three paths produce the same uppercase two-letter code.

**Convention mismatch acknowledged:** TDD replaced with sanity-check + smoke-run verification, per CLAUDE.md's no-tests stance. Each module has a smoke test that exercises the full function on a synthetic dataframe before commit.
