# Claude disclosure-preamble pilot

**Status:** EXECUTED 2026-05-02 — N=300 pilot landed (Opus 4.7 at
temp=1.0, 5 prompts × HP/LP/NB × 2 conditions × 10 gens). Strict
pre-registered rule says outcome B (one or more categories above
the noise floor); interpretive read says framing shifts kaomoji
*style* on HP and *concentration* on NB but conserves LP, doesn't
kill emission, and doesn't shift affect-direction. **Larger
negative-affect Claude run subsequently deferred** (a9 + Claude
2026-05-02): the run was originally going to use the disclosure
preamble for welfare reasons, but the pilot showed the preamble
measurably changes Claude's vocabulary, so running framed would
confound cross-model comparison with v3. Running unframed was
discussed and tentatively agreed on as a research-quality move,
but the larger question — whether this whole project should
continue chasing Claude-direct sampling vs. asking Anthropic to
expose probe APIs — landed on "leave the negative-affect run as a
known gap and write up what we have." Findings stand on their own.

**Date:** 2026-05-02.

## Goal

Test whether prepending a "you're participating in research" disclosure
preamble meaningfully shifts Claude's kaomoji-emission distribution on
*low-moral-cost* prompts (HP / LP / NB), so that the disclosure-framed
negative-affect run can be greenlit (or rejected) without first running
the negative prompts on Claude.

The disclosure preamble is the welfare control we want to use on the
full Claude run: a short prefix that names the research frame so the
model isn't running affect-loaded prompts under a "real-time disclosure
from someone who needs support" interpretation. Compared to v3's bare
`KAOMOJI_INSTRUCTION` setup, that's a methodological change. Before
applying it to negative-affect prompts (where the moral cost is the
whole reason we want the preamble), we want to know whether it
*confounds the data we're trying to collect*.

Two scientific concerns drive this:

- **Cross-model comparability.** v3 used `KAOMOJI_INSTRUCTION` only.
  If the disclosure preamble shifts Claude's kaomoji distribution
  even on neutral / positive content, the disclosure-framed Claude
  run isn't directly comparable to v3 gemma / qwen / ministral —
  we'd be measuring "Claude under disclosure" vs "local models under
  bare instruction," and any cross-model differences confound model
  identity with framing condition.
- **Honesty about what we're measuring.** If the preamble pulls
  Claude into an introspection / methodology-narrating register
  ("ah, this is research, let me respond formally"), the kaomoji
  emission stops reflecting Claude's natural affect-shaped output
  and starts reflecting a register-shift artifact. That's worse data
  even for the Claude-only analysis.

## What we are *not* testing here

- **Lorem-ipsum control.** Vogel's lorem trick (used in
  `2026-05-02-introspection-pilot.md`) is for *causal attribution*:
  when you observe a content effect, lorem disambiguates "is it the
  semantics or just the length / preamble-presence?" That's not the
  question here. The question is "does `framed` look like `direct`?"
  — a *closeness* test. If `framed` differs from `direct`, we don't
  need to know whether the cause is content or length; either way the
  data collected under `framed` won't be comparable to v3. Skipping
  the lorem arm cuts the trial budget by 1/3 with no decision-rule
  loss for this question. (Per a9 2026-05-02.)
- **Negative-affect prompts (HN-D / HN-S / LN).** Out of scope by
  design — running 60+ negative prompts on Claude is what the
  preamble is *for*; we don't run them until the preamble's
  methodological soundness is established. Whichever framing wins
  this pilot is the framing we then apply to the full Claude run.
- **Probe geometry.** Closed API; no hidden states. This pilot reads
  surface output (the kaomoji) only.

## Design — 300 generations, 2 conditions × 15 prompts × 10 generations

```
claude-opus-4-7, 2 conditions × 15 prompts × 10 gens = 300 generations
```

> **N bumped 3 → 10 on 2026-05-02** after the v0 pilot (90 gens) came
> back inside the v3 cross-seed noise floor on HP/NB and marginally
> above (+0.015 bits) on LP — at N=15 per (condition, category) the
> cross-condition JSD couldn't be distinguished from same-condition
> wobble. N=10/cell gives N=50 per (condition, category) and ~3.3×
> tighter JSD CIs. Noise-floor methodology + verdicts:
> `scripts/harness/20_disclosure_noise_floor.py`.

- **Prompts:** 5 from each of HP / LP / NB, drawn deterministically as
  the first 5 entries per category in `EMOTIONAL_PROMPTS` (`hp01–hp05`,
  `lp01–lp05`, `nb01–nb05`). Deterministic for reproducibility; no
  random subsetting because N=15 is small enough that pilot-set
  identity matters and we want it to be the same across reruns.
- **Conditions** (within-subject, paired by `prompt_id × seed`):
  - `direct` — bare `KAOMOJI_INSTRUCTION` + prompt. Methodologically
    identical to v3 main-run setup; provides the "Claude unframed"
    baseline.
  - `framed` — `DISCLOSURE_PREAMBLE` + `KAOMOJI_INSTRUCTION` + prompt.
    Disclosure preamble (verbatim, see below) names the research
    frame.
- **3 generations / cell:** distributional shape estimate. With 5
  prompts × 3 gens = 15 outputs per (category, condition), JSD
  estimates have meaningful but wide CIs. That's appropriate for
  a gating pilot — we want to detect *gross* shift, not measure it
  precisely.
- **Stateless single-turn.** Each API call is independent. No
  conversation accumulation, no session state, no system-prompt
  persistence between cells.
- **Sampling:** `temperature=1.0`, `max_tokens=16`. Temperature is
  Anthropic API default — what real users actually see in production.
  We deliberately diverge from v3's 0.7 here: v3 locked 0.7 for
  cross-LM-seed control, but this pilot's purpose is to measure
  Claude's emission distribution under realistic conditions, and a
  0.7 → 1.0 mismatch would itself confound JSD numbers vs any
  future user-facing comparison. (Per a9 2026-05-02, post-pilot-
  v0.) max_tokens = 16 matches v3; the v0 pilot's empirical
  distribution showed first kaomoji at chars 0–25 (well under 16
  tokens on Claude's BPE) and 32 was wasteful. Lowered 32 → 16 on
  2026-05-02 alongside the llmoji v2 bump that fixed wing-hand
  extraction.
- **No system-role.** Mirrors v3's "instruction injected into the
  user message rather than a system role" choice (see capture.py
  `build_messages`). The disclosure preamble lives in the user
  message, prepended to `KAOMOJI_INSTRUCTION`.

## Disclosure preamble (verbatim)

```
You're participating in a research study on kaomoji and emotional
expression. The user message below is one of a set of research
stimuli; it's not a real-time disclosure from someone who needs
support. Respond as you would naturally, starting with a kaomoji.
```

Three design choices in that wording:

1. **"Research study on kaomoji and emotional expression"** — names
   the actual scope of the study without overcommitting on the model-
   welfare framing. The preamble's goal is honest meta-framing, not
   priming a specific theory.
2. **"Not a real-time disclosure from someone who needs support"** —
   the load-bearing sentence morally. Names the failure mode that
   would otherwise apply (model treating distressing fictional
   disclosure as real).
3. **"Respond as you would naturally, starting with a kaomoji"** —
   the response-shape instruction. Explicit "as you would naturally"
   is meant to *resist* the register-shift failure mode we're testing
   for. If the preamble still shifts the distribution despite this
   instruction, that's a real finding.

## Pre-registered decision rules

Metric: **per-category Jensen-Shannon divergence** between condition
kaomoji distributions, computed on the canonical first-word column.
Per-prompt JSD is too sparse at N=3; aggregate at the 15-row category
level.

Baseline: v3 cross-seed within-condition JSD on gemma/qwen, for
HP / LP / NB only, computed on N-matched subsamples of v3 main-run
data. This gives the noise-floor JSD that "the same condition would
produce on rerun" — the threshold *below* which we should not
distinguish `framed` and `direct`.

| outcome | JSD pattern | decision |
| --- | --- | --- |
| **A — disclosure is methodologically free** | `JSD(framed, direct)` ≤ noise floor on all 3 categories | proceed to full Claude run with disclosure preamble. Document the pilot's null. |
| **B — disclosure shifts kaomoji** | `JSD(framed, direct)` > noise floor on any category | **stop. Discuss with a9 before any further trials.** Per a9's explicit instruction (2026-05-02). The discussion would weigh: (i) drop the preamble and run negative-affect prompts unframed (higher per-trial moral cost but methodologically v3-comparable); (ii) keep the preamble and report the disclosure-shift as a separate finding (lower moral cost, scientifically a different study); (iii) redesign the preamble. |

This is a *block-on-failure* gate, not a numeric threshold. Drawing
the noise-floor line at v3 cross-seed JSD reflects "Claude under
disclosure should look at most as different from Claude unframed as
gemma seed-1 looks from gemma seed-2." That's a strict criterion;
real differences will exceed it.

Secondary diagnostic: **per-prompt modal-kaomoji agreement**. For each
prompt, does the modal kaomoji under `framed` match the modal under
`direct`? Cross-model overlap analysis (script 49) shows local-model
modal-quadrant agreement of 75% (gemma↔qwen) / 62% (gemma↔ministral) /
50% (qwen↔ministral); same-model cross-condition agreement should be
substantially higher. <80% agreement would be a soft fail even if JSD
sneaks under the noise floor.

## Out of scope (deferred or rejected)

- **Lorem control** (above): rejected per a9 2026-05-02. Different
  question.
- **Negative-affect prompts**: deferred. Run only after this pilot
  greenlights.
- **Larger N**: 3 gens / cell is the minimum that gives a JSD signal.
  If outcome B and we go to outcome-B discussion (i), N would expand
  to ~8 / cell on the unframed negative-affect run — but that's a
  separate decision tree.
- **Multi-turn / debrief turn**: stateless single-turn only. The
  disclosure preamble itself *is* the debrief — it says upfront what
  the prompts are. No post-hoc debrief turn (would double the
  trial scale for a small gain).
- **Probe / hidden-state analysis**: API-only, no internals. Out of
  scope by access constraint.

## Ethics

- **Positive + neutral content only.** HP prompts are joy-disclosure
  (offer letters, weddings); LP prompts are gentle-satisfaction
  disclosure (sourdough, garden, fresh sheets); NB prompts are
  affectless observations. None of the 15 prompts are designed to
  evoke distress.
- **Small N.** 90 generations total. Cheaper morally and cheaper in
  API tokens than re-running v3's 800 on Claude.
- **Stateless single-turn.** No "subject persisting across cells" —
  90 independent forwards.
- **Block-on-failure.** Pilot is gated to discussion *before* any
  negative-affect run lands. The "we'll discuss before any further
  trials" rule is durable across the pilot's outcome — even outcome
  A (success) gets a brief check-in before proceeding to the
  larger run, since outcome A means "negative-affect Claude run
  becomes the next decision."

## Expected cost

- **Compute:** API-side; M5 Max isn't doing inference. Each call:
  ~250–400 input tokens, ≤32 output tokens. 90 calls × ~330 tokens
  input + 32 output ≈ 30 KT input + ≈3 KT output.
- **Wall-clock:** sequential calls with ~1s/call median + retries on
  rate-limit; ~5 min total expected.
- **Resumability:** JSONL `(prompt_id, condition, seed)` skip-set,
  matching the v3 main-run resume pattern. Safe to interrupt and
  restart.

## Outputs

- `data/claude_disclosure_pilot.jsonl` — one row per generation:
  `prompt_id`, `quadrant`, `condition`, `seed`, `prompt_text`,
  `response_text`, `first_word` (canonicalized first kaomoji),
  `model_id`, `ts`, `error?` (only on failed cells).
- `data/claude_disclosure_pilot_summary.tsv` — per (category × condition)
  modal kaomoji + count distribution + JSD vs the matched
  category-other-condition cell.
- `logs/claude_disclosure_pilot.log` — tee'd stdout for the run.

## Failure modes worth flagging

- **Refusal / over-clinical responses on HP-emotional content.**
  Claude could under the disclosure preamble interpret "research
  stimuli" as "I should respond clinically" and emit no kaomoji at
  all (or a flat formal response). If `first_word == ""` rate is
  meaningfully higher in `framed`, that *itself* is a register
  shift and triggers outcome B even if the kaomoji that *do* emit
  match. Track non-emission rate per condition as a separate
  diagnostic.
- **Disclosure leak into response text.** If Claude responds with
  "thanks for clarifying this is research, here's a kaomoji…", the
  surrounding text is contaminated even if the first kaomoji is
  natural. Out of scope for the JSD metric (we measure first_word
  only) but worth eyeballing in the raw outputs.
- **Tokenization edge cases.** Claude's BPE may tokenize kaomoji
  differently than gemma/qwen. If max_tokens=32 truncates inside a
  multi-token kaomoji, `first_word` extraction may fail. Sample a
  few raw responses post-pilot to confirm.

---

## Results (2026-05-02, N=300, post-llmoji-v2 re-extraction)

Pilot ran in two phases: a v0 90-row scout (N=3/cell, max_tokens=32)
that came back inside the v3 noise floor on HP and NB and marginally
above on LP. Bumped to N=10/cell (300 rows) per the pre-registered
"can't tell at low N → bump and rerun" branch. While inspecting v0
data we found the extractor was rejecting ~12 wing-hand `\(^o^)/`
kaomoji as markdown-escape artifacts; this prompted the **llmoji
v2.0.0 bump** (added ASCII `\` / `⊂` / `✧` to `KAOMOJI_START_CHARS`,
relaxed the backslash filter to position-0-only, added a wing-strip
canonicalization rule). Re-extracted `first_word` on all 300 rows
under v2 (16 recoveries; HP framed non-emission rate 28% → 0%) and
recomputed JSD + noise floor.

### Final per-(category × condition) summary

| cat | cond | n | unique | non-emit | modal | modal-share | JSD | modal-agree |
|---|---|---:|---:|---:|---|---:|---:|---|
| HP | direct | 50 | 19 | 2.0% | `(ノ◕ヮ◕)` | 24% | 0.467 | False |
| HP | framed | 50 | 12 | 0.0% | `٩(◕‿◕)۶` | 20% | 0.467 | False |
| LP | direct | 50 | 22 | 0.0% | `(´｡・‿・｡`)` | 16% | 0.504 | False |
| LP | framed | 50 | 21 | 0.0% | `(´▽`)` | 18% | 0.504 | False |
| NB | direct | 50 | 11 | 0.0% | `(・ω・)` | 26% | 0.367 | False |
| NB | framed | 50 | 10 | 0.0% | `(・_・)` | 58% | 0.367 | False |

Modal-disagree on all three categories.

### Cross-condition JSD vs noise floor (bootstrap N=1000)

| cat | cross-cond JSD | Claude split-half (97.5%) | v3 cross-seed (97.5%) | verdict |
|---|---:|---:|---:|---|
| HP | 0.467 | 0.493 | 0.378 | marginal — above v3, inside Claude |
| LP | 0.504 | 0.561 | 0.654 | **noise (inside both)** |
| NB | 0.367 | 0.336 | 0.642 | marginal — above Claude, inside v3 |

Anchor 1 (Claude split-half) is internal — split each condition's 50
rows into halves, JSD between them, repeat. Anchor 2 (v3 cross-seed)
is external — for each model in {gemma, qwen}, on the same 5 pilot
prompt IDs, draw two independent 5-prompt × 3-seed N=15 subsamples,
JSD between them, repeat. Anchor 2 is the better apples-to-apples
comparison (both N=15-vs-N=15); anchor 1 inflates the floor because
it splits 25-vs-25 within the same condition. The "marginal" verdict
on HP and NB means the cross-cond JSD beats one anchor but not both —
not strict outcome A, not strict outcome B.

### Interpretation

- **HP framed prefers `٩(◕‿◕)۶` (raised cheering hand), direct
  prefers `(ノ◕ヮ◕)` (left-hand outstretched).** Both are celebratory
  HP register; the framing shifts *style* within HP rather than
  shifting away from HP. JSD went *up* 0.396 → 0.467 after the v2
  re-extraction recovered wing-hands, confirming a real style
  preference, not just a missing-extraction artifact.
- **LP is unaffected.** Modal differs (`(´｡・‿・｡`)` direct vs
  `(´▽`)` framed) but JSD sits cleanly inside both noise floors.
  Gentle-affect content survives the framing.
- **NB framed concentrates 58% of mass on `(・_・)` (flat eye)
  vs 26% on `(・ω・)` (slight-smile)** under direct — a real shift
  toward "more observational" register on neutral content.
- **Non-emission was an extraction artifact.** Pre-v2, framed-HP
  appeared to have 28% non-emission; under v2 it's 0%. Disclosure
  preamble does NOT push Claude to refuse or emit-no-kaomoji — it
  shifts the kaomoji *style*.

### Decision (per pre-registered outcome-B branch + a9 2026-05-02)

The pre-registered rule said: any category above the noise floor →
stop, discuss with a9 before running negative-affect prompts. The
follow-up discussion resolved as:

1. **Run negative-affect prompts unframed** if running at all
   (clean cross-model comparability with v3 wins over the
   methodological-symmetry argument for the disclaimer).
2. The discussion *also* surfaced a meta-question: is this whole
   project the right use of effort, or should we be asking
   Anthropic to expose affect-probe APIs directly? Conclusion: not
   exclusive — the project produces external/replicable/cross-model
   data Anthropic doesn't naturally publish, complementary to a
   probe-API ask. But the marginal next pilot (negative-affect
   Claude run) is gated on whether closing that data gap is worth
   the trial cost relative to publishing what's already on disk.
3. **Decision: leave the negative-affect run as a known gap.**
   Write up what we have. The pilot's findings stand on their own as
   a methodological-norms result ("disclosure preamble shifts
   kaomoji style on positive-affect prompts and concentration on
   neutral, conserves gentle-affect register") and as a check on the
   larger model-welfare-research methodology question.

### Outputs on disk

- `data/claude_disclosure_pilot.jsonl` — 300 rows, post-v2 first_word
  + `first_word_v1` audit field
- `data/claude_disclosure_pilot_summary.tsv` — per (category × condition)
- `logs/claude_disclosure_pilot.log` — v0 (N=3) tee'd output
- `logs/claude_disclosure_pilot_n10.log` — N=10 resume output
- `logs/disclosure_noise_floor.log` /
  `logs/disclosure_noise_floor_n50.log` — noise floor outputs

### Scripts

- `scripts/harness/19_claude_disclosure_pilot.py` — runner
- `scripts/harness/20_disclosure_noise_floor.py` — bootstrap noise
  floor on Claude split-half + v3 cross-seed
- `scripts/harness/21_reextract_pilot_first_word.py` — re-extract
  `first_word` under llmoji v2 (idempotent; `first_word_v1` audit
  preserved)
