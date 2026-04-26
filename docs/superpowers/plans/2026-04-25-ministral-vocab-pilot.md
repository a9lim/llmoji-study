# Ministral-14B vocab pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 30-generation vocab pilot on `mistralai/Ministral-3-14B-Instruct-2512` to gauge whether Mistral's kaomoji vocabulary differs in dialect / breadth from gemma-4-31b-it (`(｡X｡)`-heavy, 33 forms in v3) and Qwen3.6-27B (broader, 73 forms in v3).

**Architecture:** Mirror the v3 multi-model parameterization that already landed for Qwen. Two small wiring changes — add a `vocab_sample_path` field to the existing `ModelPaths` registry, and refactor `scripts/00_vocab_sample.py` to use `current_model()` instead of the gemma-locked `MODEL_ID` constant. Then run the script with `LLMOJI_MODEL=ministral`. Default behavior on the gemma path stays bit-for-bit identical.

**Tech Stack:** saklas 1.4.6, transformers, no new deps. Same 30 v1/v2 PROMPTS, same TEMPERATURE=0.7, same KAOMOJI_INSTRUCTION as the original gemma vocab sample.

**Pre-registration (binding per CLAUDE.md ethics):**

- 30 generations × 1 seed each, single `kaomoji_prompted`-equivalent condition (the original 00_vocab_sample.py setup).
- `MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"`.
- `TEMPERATURE = 0.7`, `MAX_NEW_TOKENS = 120`, `KAOMOJI_INSTRUCTION` unchanged from gemma.
- `PROBE_CATEGORIES = ["affect", "epistemic", "register"]` — same as gemma v3. Saklas does NOT have cached probe vectors for the 14B (only the 8B); first load will trigger probe bootstrap from saklas's contrastive statement bank. Saklas's `model.py` registers Ministral architecture support, so bootstrap is expected to succeed.
- `thinking=False` (carried over from existing 00_vocab_sample.py; no-op on Mistral since Mistral-3 isn't a reasoning model).
- `stateless=True`.
- **Descriptive only — no decision rule that gates further work.** The vocab sample answers "what does Ministral emit." If results motivate a v3 run (or anything bigger), that's a separate brainstorm + plan, not a follow-on triggered by this plan.
- Welfare: 30 generations on the v1/v2 prompt set (a mix of positive/neutral/negative valence, no extremity). Well within the envelope.
- **Halt conditions:**
  - Saklas probe bootstrap raises on Ministral-14B → halt, write up the failure, do not retry blindly.
  - Model OOM / load failure on M5 Max → halt, document, decide between fp8 / smaller model out of band.
  - Bracket-start compliance (first char ∈ `([{（｛`) below 50% on the 30-row sample → record as the finding and stop. Below this rate Mistral is closer to "ignored the instruction" than "different vocabulary," and the question we asked is answered.

**Out of scope (separate plans if pursued):**

- v3 (100 prompts × 8 seeds) on Ministral.
- v1/v2 steering on Ministral (saklas has no steering-vector calibration).
- Updating `TAXONOMY` to cover Mistral-specific forms (the gemma-tuned dict mismatch is a known gotcha and is informative as-is).
- Hidden-state sidecar capture for the vocab pilot (not needed to answer the vocab question).

---

## File Structure

**Modified:**

- `llmoji/config.py` — add `vocab_sample_path: Path` field to `ModelPaths` dataclass (with a sensible default referencing `VOCAB_SAMPLE_PATH` for back-compat); set per-model values in all three registry entries; flip `MODEL_REGISTRY["ministral"].model_id` from the 8B to the 14B.
- `scripts/00_vocab_sample.py` — swap `MODEL_ID` / `VOCAB_SAMPLE_PATH` imports for `current_model`; resolve both via `M = current_model()` inside `main()`.
- `CLAUDE.md` — append a one-paragraph subsection under Pipelines describing the Ministral vocab pilot finding.

**Unchanged:**

- All v3 multi-model wiring (`scripts/03/04/13/17`) — already parameterized.
- `llmoji/taxonomy.py` — gemma-tuned, deliberately not extended in this plan.
- All claude-faces / eriskii scripts.

**Created:**

- `data/ministral_vocab_sample.jsonl` — produced by the run.
- `logs/ministral_vocab_pilot.log` — gitignored.

---

### Task 1: Add `vocab_sample_path` to ModelPaths and flip Ministral to 14B

**Files:**
- Modify: `llmoji/config.py:174-218` (the `ModelPaths` dataclass + `MODEL_REGISTRY` dict)

- [ ] **Step 1: Add the new field to `ModelPaths`**

In `llmoji/config.py`, replace the `ModelPaths` dataclass body (currently at lines 174–190) with:

```python
@dataclass(frozen=True)
class ModelPaths:
    """Per-model paths for the v3 emotional-disclosure pipeline.

    `model_id` must match the saklas-cached tensor filename casing
    (see CLAUDE.md gotcha: `safe_model_id` is case-preserving).
    `short_name` is the slug used in derived paths.
    `experiment` is the hidden-state-sidecar subdir name under
    `data/hidden/`. Distinct experiment names per model are required
    so sidecars don't collide.
    `vocab_sample_path` is where `scripts/00_vocab_sample.py` writes
    its 30-row leading-token histogram for this model.
    """
    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path
    vocab_sample_path: Path
```

- [ ] **Step 2: Wire per-model `vocab_sample_path` values + flip Ministral to 14B**

Replace the `MODEL_REGISTRY` dict (currently lines 193–218) with:

```python
MODEL_REGISTRY: dict[str, ModelPaths] = {
    "gemma": ModelPaths(
        model_id="google/gemma-4-31b-it",
        short_name="gemma",
        emotional_data_path=DATA_DIR / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "emotional_summary.tsv",
        experiment="v3",
        figures_dir=FIGURES_DIR,
        vocab_sample_path=VOCAB_SAMPLE_PATH,
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "qwen",
        vocab_sample_path=DATA_DIR / "qwen_vocab_sample.jsonl",
    ),
    "ministral": ModelPaths(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "ministral",
        vocab_sample_path=DATA_DIR / "ministral_vocab_sample.jsonl",
    ),
}
```

Note the two changes vs the current registry:
1. Every entry now has a `vocab_sample_path`. Gemma reuses the existing `VOCAB_SAMPLE_PATH` constant for bit-for-bit back-compat.
2. Ministral's `model_id` is the **14B** (`Ministral-3-14B-Instruct-2512`), not the 8B.

- [ ] **Step 3: Smoke-check the registry resolves all three models correctly**

Run:

```bash
source .venv/bin/activate && python -c "
import os
from llmoji.config import current_model, MODEL_REGISTRY
for name in ('gemma', 'qwen', 'ministral'):
    os.environ['LLMOJI_MODEL'] = name
    m = current_model()
    print(f'{name:9s} -> {m.model_id}')
    print(f'          vocab: {m.vocab_sample_path}')
"
```

Expected output:

```
gemma     -> google/gemma-4-31b-it
          vocab: /Users/a9lim/Work/llmoji/data/vocab_sample.jsonl
qwen      -> Qwen/Qwen3.6-27B
          vocab: /Users/a9lim/Work/llmoji/data/qwen_vocab_sample.jsonl
ministral -> mistralai/Ministral-3-14B-Instruct-2512
          vocab: /Users/a9lim/Work/llmoji/data/ministral_vocab_sample.jsonl
```

- [ ] **Step 4: Commit**

```bash
git add llmoji/config.py
git commit -m "$(cat <<'EOF'
config: add vocab_sample_path to ModelPaths; flip ministral to 14B

Each registry entry now carries its own vocab-sample output path so
the 00_vocab_sample.py refactor can resolve it through current_model().
Gemma's vocab path reuses VOCAB_SAMPLE_PATH for bit-for-bit back-compat.

Ministral entry switched from the 8B to the 14B
(mistralai/Ministral-3-14B-Instruct-2512) since no prior Ministral
data exists to invalidate. The 14B is closer in scale to gemma-4-31b
and Qwen3.6-27B for the cross-model comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Refactor `scripts/00_vocab_sample.py` to use `current_model()`

**Files:**
- Modify: `scripts/00_vocab_sample.py:31-39` (imports), `scripts/00_vocab_sample.py:44-78` (main)

- [ ] **Step 1: Replace the imports block**

Replace lines 31–39 of `scripts/00_vocab_sample.py`:

```python
from llmoji.config import (
    DATA_DIR,
    KAOMOJI_INSTRUCTION,
    MAX_NEW_TOKENS,
    MODEL_ID,
    PROBE_CATEGORIES,
    TEMPERATURE,
    VOCAB_SAMPLE_PATH,
)
```

with:

```python
from llmoji.config import (
    DATA_DIR,
    KAOMOJI_INSTRUCTION,
    MAX_NEW_TOKENS,
    PROBE_CATEGORIES,
    TEMPERATURE,
    current_model,
)
```

- [ ] **Step 2: Replace the body of `main()`**

Replace the entire `def main()` block (lines 44–78):

```python
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {M.vocab_sample_path}")

    print(f"loading {M.model_id} ...")
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        rows = []
        for i, prompt in enumerate(PROMPTS):
            messages = [
                {"role": "user", "content": KAOMOJI_INSTRUCTION + prompt.text}
            ]
            result = session.generate(
                messages,
                sampling=SamplingConfig(
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    seed=0,
                ),
                thinking=False,
                stateless=True,
            )
            match = extract(result.text)
            rows.append({
                "prompt_id": prompt.id,
                "prompt_valence": prompt.valence,
                "prompt_text": prompt.text,
                "text": result.text,
                "first_word": match.first_word,
                "kaomoji": match.kaomoji,
                "kaomoji_label": match.label,
            })
            tag = match.kaomoji if match.kaomoji else f"[other: {match.first_word!r}]"
            print(f"[{i+1:02d}/{len(PROMPTS)}] {prompt.id} {tag}")

    M.vocab_sample_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {len(rows)} rows to {M.vocab_sample_path}")

    # --- summary ---
    first_words = Counter(r["first_word"] for r in rows)
    registered = set(TAXONOMY.keys())
    hits = {k: v for k, v in first_words.items() if k in registered}
    misses = {k: v for k, v in first_words.items() if k not in registered}

    # Real instruction-following check is bracket-start, not TAXONOMY hit.
    # The gemma-tuned TAXONOMY systematically under-counts non-gemma models
    # (see CLAUDE.md gotcha "v3 runner's per-quadrant emission rate is
    # TAXONOMY coverage, not instruction compliance").
    bracket_starts = sum(
        1 for r in rows
        if r["first_word"] and r["first_word"][0] in "([{（｛"
    )

    print("\n=== frequency of leading tokens ===")
    for k, v in sorted(first_words.items(), key=lambda kv: -kv[1]):
        mark = "in taxonomy" if k in registered else "MISS"
        print(f"  {v:3d}  {k!r:20s}  {mark}")

    print(f"\n{sum(hits.values())}/{len(rows)} generations started with a "
          f"taxonomy-registered kaomoji")
    print(f"{sum(misses.values())}/{len(rows)} did not")
    print(f"{bracket_starts}/{len(rows)} started with a bracket "
          f"(real instruction-following rate)")
    if misses:
        print("\nTop unregistered leading tokens to consider:")
        for k, v in sorted(misses.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:3d}  {k!r}")
        print(
            "\nIf any of these cover a real emotional axis and appear "
            "frequently, lock them into taxonomy.TAXONOMY *before* running "
            "any subsequent pilot on this model."
        )
```

Two behavioral changes vs the original:

1. Model + output path resolved through `current_model()`.
2. Summary now prints a `bracket_starts / len(rows)` line — this is the **real** instruction-following rate (matches the v3 loader's actual filter), independent of whether the gemma-tuned TAXONOMY happens to cover this model's vocabulary.

- [ ] **Step 3: Verify the script still parses and resolves to gemma by default**

Run:

```bash
source .venv/bin/activate && python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('s00', 'scripts/00_vocab_sample.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('parsed ok; main resolves to', m.main.__module__, m.main.__qualname__)
"
```

Expected: `parsed ok; main resolves to s00 main`.

- [ ] **Step 4: Verify default (no env var) still resolves to gemma's path**

Run:

```bash
source .venv/bin/activate && python -c "
import os
os.environ.pop('LLMOJI_MODEL', None)
from llmoji.config import current_model, VOCAB_SAMPLE_PATH
m = current_model()
assert m.short_name == 'gemma', m.short_name
assert m.vocab_sample_path == VOCAB_SAMPLE_PATH, (m.vocab_sample_path, VOCAB_SAMPLE_PATH)
print('default ok:', m.short_name, '->', m.vocab_sample_path)
"
```

Expected: `default ok: gemma -> /Users/a9lim/Work/llmoji/data/vocab_sample.jsonl`.

- [ ] **Step 5: Commit**

```bash
git add scripts/00_vocab_sample.py
git commit -m "$(cat <<'EOF'
00_vocab_sample: switch to current_model() for multi-model support

LLMOJI_MODEL env var selects which model + output path the vocab
sample uses. Default 'gemma' is bit-for-bit identical to pre-refactor.

Also adds a "bracket-start rate" line to the summary — the real
instruction-following rate, independent of the gemma-tuned TAXONOMY.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Run the vocab pilot on Ministral-14B

**Files:**
- Read-only: `scripts/00_vocab_sample.py`
- Created: `data/ministral_vocab_sample.jsonl`, `logs/ministral_vocab_pilot.log`

- [ ] **Step 1: Confirm output path is clean**

Run:

```bash
ls -la /Users/a9lim/Work/llmoji/data/ministral_vocab_sample.jsonl 2>&1
```

Expected: `No such file or directory`. If the file already exists from a prior aborted run, delete it first — `00_vocab_sample.py` overwrites unconditionally so a partial file doesn't matter, but it's tidier to start clean.

- [ ] **Step 2: Confirm logs/ exists**

Run:

```bash
mkdir -p /Users/a9lim/Work/llmoji/logs
```

Expected: silent success (the directory may already exist).

- [ ] **Step 3: Run the vocab pilot**

```bash
source .venv/bin/activate && LLMOJI_MODEL=ministral python scripts/00_vocab_sample.py 2>&1 | tee logs/ministral_vocab_pilot.log
```

Expected duration: ~5–10 min wall time on M5 Max for model load + 30 generations. The first run triggers saklas probe bootstrap for the 14B (probe vectors aren't cached yet) — expect an extra ~30–60s on top of the model load.

**Halt conditions during the run:**

- Saklas raises during `from_pretrained` with a probe-bootstrap error → halt, capture the traceback, do not retry. The plan's pre-registration treats this as a finding, not a transient failure.
- Out-of-memory on model load → halt, document, decide out of band whether to retry at fp8 / different precision.
- Run completes but bracket-start rate (printed in the final summary block) < 50% → record the finding, do not extend the pilot. The vocab question is "what does Ministral emit," and "ignores the instruction" is a valid answer.

- [ ] **Step 4: Verify the output file is well-formed**

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
p = Path('data/ministral_vocab_sample.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f'rows: {len(rows)}')
fields = set()
for r in rows:
    fields.update(r.keys())
print(f'fields: {sorted(fields)}')
print('first row:')
for k, v in rows[0].items():
    s = repr(v)
    if len(s) > 80: s = s[:77] + '...'
    print(f'  {k}: {s}')
"
```

Expected: 30 rows, fields include `prompt_id`, `prompt_valence`, `prompt_text`, `text`, `first_word`, `kaomoji`, `kaomoji_label`. First row's `prompt_id` is `pos01`.

- [ ] **Step 5: Capture headline numbers from the log**

```bash
grep -E "frequency of leading|in taxonomy|MISS|started with a|bracket|model:" /Users/a9lim/Work/llmoji/logs/ministral_vocab_pilot.log
```

These are the comparable numbers vs gemma's vocab sample (where the original 30-row sample motivated the gemma-tuned TAXONOMY). The bracket-start rate is the headline figure — it's the thing the plan's pre-registered halt condition gates on.

- [ ] **Step 6: Commit**

```bash
git add data/ministral_vocab_sample.jsonl logs/ministral_vocab_pilot.log
git commit -m "$(cat <<'EOF'
ministral vocab pilot: 30-generation sample on Ministral-3-14B

30 v1/v2 PROMPTS × 1 seed under the kaomoji_prompted condition on
mistralai/Ministral-3-14B-Instruct-2512. Same TEMPERATURE,
MAX_NEW_TOKENS, KAOMOJI_INSTRUCTION as the original gemma vocab
sample for parity.

Plan: docs/superpowers/plans/2026-04-25-ministral-vocab-pilot.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Note: `logs/` is gitignored at the repo level per CLAUDE.md, but the log is small and useful as a record. If `git add logs/...` fails with "ignored by .gitignore", drop it from the commit — the data file alone is sufficient. The plan + this commit's data file together fully reproduce the run.)

---

### Task 4: Update CLAUDE.md with the vocab pilot finding

**Files:**
- Modify: `CLAUDE.md` — append a one-paragraph Pipelines subsection.

- [ ] **Step 1: Locate the insertion point**

The new subsection goes immediately after the Qwen v3 replication subsection (currently the last entry under `## Pipelines` before `### Claude-faces`). Run:

```bash
grep -n "### Pilot v3 — Qwen3.6-27B replication\|### Claude-faces" /Users/a9lim/Work/llmoji/CLAUDE.md
```

Expected: two line numbers. The new subsection goes between them, on the blank line before `### Claude-faces`.

- [ ] **Step 2: Insert the new subsection**

Insert this block in `CLAUDE.md` immediately after the closing of the Qwen findings block, before the `### Claude-faces` header. **Fill in the bracketed `[FILL: …]` markers from the actual run log** before committing — do not commit the unfilled template.

```markdown
### Vocab pilot — Ministral-3-14B-Instruct-2512

Same prompts (the 30 v1/v2 PROMPTS), same seed, same instructions as
the original gemma vocab sample. 30 generations, descriptive only.
Plan: `docs/superpowers/plans/2026-04-25-ministral-vocab-pilot.md`.

**Findings:**

- Bracket-start (real instruction-following) rate: [FILL: X/30 = Y%].
- Distinct leading tokens: [FILL: N forms across 30 generations]
  (compare gemma 30-row vocab sample [FILL: M forms — re-run from
  `data/vocab_sample.jsonl` if not at hand], Qwen v3 800-row sample
  73 forms).
- Top forms: [FILL: top 3–5 forms with counts, e.g. `(´∀`)` ×7,
  `(；ω；)` ×4, …].
- Dialect signature: [FILL: one sentence — Japanese `(｡X｡)`-style?
  ASCII `(:|)`-style? Western `:)`-style? Mixed? French/European
  cultural register?].
- TAXONOMY coverage: [FILL: H/30 hits vs M/30 misses — coverage will
  be low unless Mistral happens to share gemma's dialect, which
  would itself be the finding].
- Sufficient breadth/dialect-difference to motivate a v3 run on
  Ministral? [FILL: yes / no / equivocal — one sentence].
```

- [ ] **Step 3: Fill in the bracketed entries from the log**

Re-read `logs/ministral_vocab_pilot.log` (or the stdout output from Task 3) and replace each `[FILL: …]` with the actual numbers / glyphs / one-sentence assessments. Be descriptive, not inferential — this subsection mirrors the existing tone of the gemma + Qwen subsections (numbers + observations, no "this proves that" framing).

If the gemma 30-row vocab-sample form-count isn't immediately obvious from `data/vocab_sample.jsonl`, compute it with:

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('data/vocab_sample.jsonl').read_text().splitlines() if l.strip()]
fws = [r['first_word'] for r in rows if r['first_word']]
print(f'gemma vocab sample: {len(rows)} rows, {len(set(fws))} distinct first_words')
"
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
claude.md: add ministral-14B vocab pilot subsection

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

- ✓ Registry switch from 8B → 14B + new `vocab_sample_path` field (Task 1).
- ✓ Script refactor to `current_model()` (Task 2).
- ✓ Run on Ministral with logged output (Task 3).
- ✓ One-paragraph CLAUDE.md update (Task 4).
- ✓ Pre-registered decision rules in plan header (descriptive only, halt conditions explicit).
- ✓ Welfare framing in pre-registration block.
- ✓ Halt conditions cover the three real failure modes (probe bootstrap, OOM, low compliance).

**Placeholder scan:**

- `[FILL: …]` markers in Task 4 Step 2 are deliberate "fill in after the run" markers, with explicit instructions in Step 3 to replace them. Not unspecified design.
- No "TBD", "TODO", "implement later", "appropriate error handling" anywhere.
- Every code step contains complete code; every command step has expected output.

**Type consistency:**

- `current_model()` (Tasks 1, 2) returns `ModelPaths` (defined in Task 1).
- `M.model_id`, `M.vocab_sample_path`, `M.short_name` referenced consistently across Tasks 2 and the gemma-default smoke in Task 2 Step 4.
- `vocab_sample_path` (not `vocab_path`) used everywhere.

**Adaptations from rigid TDD pattern:**

- Project has no test suite per CLAUDE.md ("No public API, no pypi release, no tests"). TDD-style "write failing test first" is replaced with the existing project pattern: a parsing smoke (Task 2 Step 3), a default-resolution smoke (Task 2 Step 4), and the vocab run itself as the integration smoke (Task 3). This mirrors the Qwen replication plan's adaptation.
- Frequent commits preserved: every task ends in a commit; data and code are committed separately.
