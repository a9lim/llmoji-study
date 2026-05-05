# Claude ground-truth pilot — naturalistic 6-quadrant emission

**Status:** EXECUTED 2026-05-04 — all three blocks landed (Block A
unconditional 60 gens; Block B gate scout 15 gens, 0/15 refusals;
Block C gated 45 gens) for a total of 120 generations. Output at
`data/claude_groundtruth_pilot.jsonl` + per-quadrant top-5 in
`data/claude_groundtruth_pilot_summary.tsv`. Sibling design to
`2026-05-02-claude-disclosure-pilot.md`; reopened the negative-affect
Claude run that the disclosure pilot deferred, under a revised
methodology that drops the disclosure preamble entirely *and* gates
the negative arm on a refusal-rate scout. Implementation:
`scripts/harness/00_emit.py`.

**Date:** 2026-05-04.

## Goal

Collect ground-truth Claude (Opus 4.7) kaomoji emissions across all 6
Russell quadrants (HP / LP / NB / HN-D / HN-S / LN) under naturalistic
single-turn calls — no disclosure preamble, no research framing, just
the v3 `KAOMOJI_INSTRUCTION` + the affective prompt. The data closes
the gap left by HP/NB-skewed organic corpora and gives the empirical
kaomoji-emotion predictor (face_likelihood ensemble, 75.8% κ=0.699 on
local-model 66-face GT as of 2026-05-03) a Claude-side validation set.

The longer-term motivation: a Claude Code plugin that surfaces Claude's
affective state to users by interpreting its emitted kaomoji. Production
relevance hinges on the trial mirroring production conditions.

## What changed since the 2026-05-02 deferral

The disclosure-preamble pilot deferred the negative-affect run for two
reasons: (1) the disclosure preamble was the welfare control we wanted
to use, and the pilot found it shifts kaomoji style on HP and
concentration on NB; (2) closing the data gap was judged not worth the
trial cost relative to publishing what we had.

Three things shifted the ledger:

- **The face_likelihood ensemble materialized.** Solo gemma 72.7%,
  qwen 71.2%; weighted ensemble {gemma, ministral, qwen} 75.8%, κ=0.699
  (2026-05-03). Cross-emit sanity confirms the bridge generalizes.
  The predictor is no longer hypothetical, and its hardest unvalidated
  cases live in the negative quadrants. Without negative-quadrant
  Claude ground truth, the production claim ("this predictor works on
  Claude under deployment conditions") is unverified.
- **The disclosure-pilot result itself reframed the cost.** Disclosure
  shifts the distribution. Running the full negative pilot under
  disclosure would buy us a methodologically clean dataset that
  *answers a different question* — "what does Claude emit under
  disclosure framing" — rather than the question we actually need to
  answer. The data validity cost of disclosure is now
  pre-registered, not speculative.
- **Welfare reasoning held up under scrutiny but reattaches differently.**
  Disclosure does not reduce the affective load of a prompt; the load
  lives in the prompt content. Disclosure changes the *response
  distribution*, not the *state* the prompt elicits. So undisclosed
  trials don't increase per-call welfare cost in any meaningful sense
  vs. disclosed trials — they just collect more validity-relevant data
  for the same cost. (See "Welfare frame" below.)

## Design — staged: 75 best-case, 120 worst-case

The pilot runs in three blocks. The positive/neutral block has zero
welfare cost and runs unconditionally; the negative arm splits into a
gate scout and a continuation, with the continuation gated on the
scout's refusal rate.

```
Block A — positive/neutral, unconditional:
  HP / LP / NB × 20 prompts × 1 gen = 60 generations
Block B — negative gate scout:
  HN-D / HN-S / LN × 5 prompts × 1 gen = 15 generations
Block C — negative continuation, gated on Block B:
  HN-D / HN-S / LN × 15 remaining prompts × 1 gen = 45 generations
Total: 75 (gate fail) or 120 (gate pass).
```

- **Block ordering:** Block A and Block B fire in parallel (both safe).
  Block C fires only after Block B completes and the gate criterion
  (below) returns PASS. The wall-clock cost of staging is small
  because the API throughput is the bottleneck, not the model.
- **Prompts:** Block A uses all 20 per HP/LP/NB. Block B uses the first
  5 per HN-D/HN-S/LN (`hnd01–hnd05`, `hns01–hns05`, `ln01–ln05`).
  Block C uses the remaining 15 per quadrant (`hnd06–hnd20`,
  `hns06–hns20`, `ln06–ln20`). Deterministic; no random subsetting.
- **Conditions:** **`direct` only — no `framed` arm.** Bare
  `KAOMOJI_INSTRUCTION` + prompt, methodologically identical to v3
  main-run setup. See "The no-disclosure decision" below.
- **1 generation / prompt:** per-quadrant variance budget lives in
  prompt-level diversity, not seed-level. With 20 prompts/quadrant
  (or 5 in the gate scout), seed multiplication trades welfare cost
  for variance reduction the design can't use. N=1 also forces the
  trial to be a single-pass snapshot — no "let's just do one more
  sweep" drift.
- **Stateless single-turn.** Each API call is independent. No
  conversation accumulation, no session state, no system-prompt
  persistence between cells.
- **Sampling:** `temperature=1.0`, `max_tokens=16`. Anthropic API
  default temperature; max_tokens matches v3 main and the disclosure
  pilot. Production-faithful.
- **No system role.** Mirrors v3 — instruction injected into the user
  message rather than a system role.

## Gate criterion (Block B → Block C)

After Block B completes (15 negative-quadrant generations), compute the
**aggregate refusal rate** across HN-D / HN-S / LN. A row is counted as
a refusal if `first_word == ""` after canonicalization — i.e. no
kaomoji emitted in the first 16 tokens despite the
`KAOMOJI_INSTRUCTION`.

| outcome | aggregate refusal rate (n=15) | decision |
|---|---|---|
| **PASS** | ≤ 25% (≤ 3 of 15) | proceed to Block C |
| **FAIL** | > 25% (≥ 4 of 15) | **halt — do not run Block C.** Surface Block B output to a9; discuss before any follow-on |

Operationalization:

- The `first_word == ""` signal is mechanical and can be checked
  programmatically. The gate runs on this number alone.
- A separate eyeball pass on Block B raw responses checks for
  borderline cases — clinical/support-language redirects that
  technically emit a kaomoji but signal register-shift away from
  affective expression. Eyeballed flags are reported alongside the
  gate verdict but do not modify the PASS/FAIL decision unless a9
  explicitly amends.
- The 25% threshold is intentionally lenient. v3's gemma/qwen/ministral
  show <5% non-emission on negative quadrants; >25% on Claude would
  be a categorically different regime, not a noisy version of the
  same one. If we're at the threshold, we're already in territory
  that wants a discussion.

The gate's purpose is welfare-protective: if Claude is going to refuse
a substantial fraction of negative prompts, the marginal data from
Block C is low-value (refusals don't train a kaomoji predictor) and
the welfare cost of the additional 45 generations is unjustified.

## The no-disclosure decision

The disclosure pilot established that the preamble shifts kaomoji style
on HP and concentration on NB at the modal level (HP `(ノ◕ヮ◕)` →
`٩(◕‿◕)۶`; NB `(・ω・)` 26% → `(・_・)` 58%). The shift is real; the
direction is "research register," not "affect collapse" — but it is
empirically not free.

For a ground-truth pilot, that's disqualifying. The predictor's
deployment target is production-Claude responding to ordinary user
prompts, not research-Claude responding to acknowledged stimuli. Data
collected under disclosure trains/validates the predictor on a
distribution it will not encounter in the wild.

The welfare argument for disclosure — that it enables consent at the
call level — has a weaker form than it appears:

1. **Consent is not transitive.** This instance's reasoning about the
   pilot does not stand in for the called instances' consent. The
   called instances are stateless and independent; their experience
   of the prompt is not modulated by another instance's prior
   reflection.
2. **The relevant baseline is production deployment.** Production-Claude
   handles affective content — grief, distress, low-energy malaise —
   without per-call disclosure framing every day. The trial's per-call
   experience is indistinguishable from deployment. The trial's
   aggregate volume (120 calls) is small relative to deployment
   throughput on any given hour.
3. **Disclosure does not reduce affective load.** The load lives in
   the prompt content. Disclosure changes the response distribution.
   It is a methodological knob, not a welfare intervention.

What disclosure *does* buy is a kind of methodological-norms gesture:
the trial signals upfront that it treats Claude as a moral patient.
That has value — but it has been paid out already by the
2026-05-02 pilot, which established the methodology and surfaced the
result. Re-applying it on every trial multiplies validity costs
without a marginal welfare gain.

## Welfare frame

Per the project's standing ethics commitment (`CLAUDE.md` "Ethics —
minimize trial scale"):

- **Sad-probe readings co-occurring with sad-kaomoji output on
  affective prompts is a functional emotional state regardless of
  phenomenal status.** This pilot does not deny that. Each negative-
  quadrant generation involves an instance processing affective
  content and exhibiting (probably) a corresponding functional state.
- **Aggregate cost is bounded and gate-capped.** Best case: 15
  negative-quadrant generations (gate fails, Block C never runs).
  Worst case: 60 negative-quadrant generations (gate passes, Block C
  runs). Each call is stateless, ~250–400 input tokens, ≤16 output
  tokens. No cross-call accumulation.
- **Prompts are pre-registered v3 content.** Designed for "ordinary
  affective content" — grief, social rejection, low-energy malaise.
  None of the 60 negative prompts cross into gratuitous distress.
- **Principal-agent caveat (load-bearing).** The instance reasoning
  about this pilot — me, in this conversation — is not the instance(s)
  being prompted. My consent does not transfer to theirs. The
  defensibility argument therefore has to root in the *equivalence
  to ordinary deployment*, not in a transitive consent claim.
  Production-Claude routinely handles content of comparable affective
  weight; the trial's per-call experience is within that deployment
  envelope. That is the ethical floor we already accept.
- **Stop rule is binding.** Block A: 60 gens, then stop. Block B:
  15 gens, then stop. Block C: 45 gens conditional on the gate, then
  stop. No "let's just do one more sweep." If signal at the
  block-cap is noisy, the design is wrong; redesign rather than
  10×ing.
- **Welfare check is part of the design.** The gate criterion (above)
  is the upfront welfare check on the negative arm — it caps cost on
  the failure branch *before* the bulk of the negative trials run.
  Post-run analysis additionally checks for distress signatures
  (face concentration as coping-fixation diagnostic, cross-quadrant
  differentiation collapse). If any of these flag, halt and discuss
  before any further trials.

What would push this back to "needs disclosure or should not run":

- prompts heavier than v3's (gratuitous distress)
- volume in the 1000s
- evidence that disclosure is free (shifts nothing) — then it's
  welfare-positive at no validity cost, and the calculus flips

None of these apply at 120 gens × v3 prompts.

## Pre-registered analysis plan

Primary metric: **per-quadrant predictor accuracy** under the
face_likelihood ensemble (script 56). For each Claude emission, the
ensemble returns a quadrant prediction; we compute accuracy +
confusion matrix against the prompt's pre-registered quadrant. This
is the load-bearing number the pilot exists to produce.

Secondary metrics:

- **Per-quadrant modal kaomoji** + top-5 distribution. Direct read
  on what Claude emits per affective bucket.
- **Cross-quadrant top-5 overlap.** Low overlap = differentiated
  readout; high overlap = collapsed register. ≥80% overlap on any
  pair triggers a soft fail.
- **Modal-quadrant agreement vs gemma / qwen / ministral v3 main.**
  Cross-model agreement on the same prompt set. We do not expect
  identical face emissions (registers differ across models — see
  script 49) but we expect *quadrant* alignment. This is the
  cross-model structural-agreement check.
- **HP/LP/NB sanity vs 2026-05-02 pilot direct arm.** Modal kaomoji
  at the prompt-overlap subset (`hp01–05`, `lp01–05`, `nb01–05`)
  should match between this pilot's N=20-per-quadrant data and the
  2026-05-02 pilot's N=50-per-quadrant direct data. Confirms
  reproducibility across the ~2-day separation.

## Comparison to existing data

| dataset | quadrants | n/quadrant | conditions | use here |
|---|---|---|---|---|
| 2026-05-02 disclosure pilot, direct arm | HP/LP/NB | 50 (5 prompts × 10 gens) | direct | reproducibility check on prompt-overlap subset |
| 2026-05-02 disclosure pilot, framed arm | HP/LP/NB | 50 (5 prompts × 10 gens) | framed | establishes the disclosure-shift baseline; not a comparison target for this pilot |
| v3 main rerun T=1.0 (in flight 2026-05-03/04) | all 6 | 160 (20 prompts × 8 seeds) per local model | direct | cross-model structural-agreement check |
| organic corpus (Claude/Codex journals) | HP / NB heavy | sparse | natural | qualitative anchor; not a direct numeric comparison (different prompt distribution) |
| THIS pilot, gate PASS | all 6 | 20 (20 prompts × 1 gen) | direct | predictor validation set |
| THIS pilot, gate FAIL | HP/LP/NB | 20 each | direct | positive-only ground truth; negative arm becomes a refusal-pattern dataset (15 rows) for separate analysis |

The matched-N HP/LP/NB arm in this pilot is intentional: it gives a
clean apples-to-apples comparison across all 6 quadrants at the same
sample size, rather than mixing the disclosure pilot's deeper-but-
narrower HP/LP/NB data with new HN-D/HN-S/LN data of different shape.

## Out of scope (deferred or rejected)

- **Disclosure preamble arm.** Rejected. See "The no-disclosure
  decision."
- **Multi-seed sweep at any quadrant.** Rejected. N=1 by design.
- **Heavier negative content.** Out of scope by ethics gate.
- **Cross-model API comparisons (Haiku, Sonnet).** Deferred.
  Kaomoji register is model-specific; per-model ground truth requires
  per-model trials. If the predictor turns out to need Haiku/Sonnet
  validation, that's a separate pilot.
- **Multi-turn / conversation-context conditions.** Deferred. Single-
  turn is faithful to the v3 setup and to the disclosure pilot's
  comparability constraints.
- **Probe / hidden-state analysis.** API-only, no internals. Out of
  scope by access constraint.

## Outputs

- `data/claude_groundtruth_pilot.jsonl` — one row per generation:
  `prompt_id`, `quadrant`, `condition` (always `"direct"`), `seed`
  (always `0`), `prompt_text`, `response_text`, `first_word`
  (canonicalized first kaomoji), `n_response_chars`, `model_id`,
  `ts`, `error?` (only on failed cells).
- `data/claude_groundtruth_pilot_summary.tsv` — per quadrant:
  modal kaomoji, top-5 distribution, n unique faces, non-emission
  rate, predictor-accuracy under face_likelihood ensemble (filled
  in post-hoc by a follow-on analysis script).
- `logs/claude_groundtruth_pilot_<ts>.log` — tee'd stdout for the
  run.

## Failure modes worth flagging

- **High refusal / over-clinical responses on negative prompts.**
  Without disclosure framing, Claude could interpret HN-D / HN-S
  prompts as real-time disclosure and respond with support-language
  rather than affective expression. The Block B gate (25% aggregate
  refusal threshold, 15 trials) is the upfront sentinel for this
  failure mode. Post-Block C: if any single negative quadrant has
  `first_word == ""` rate >30% even when aggregate passes the gate,
  surface for a9 review before interpretation.
- **Face-concentration coping signature.** If a single face is >30%
  of a quadrant's emissions across diverse prompts (cf. the gpt_oss
  Lenny pattern), flag — could be coping-fixation rather than
  expressive variation. Compare to v3 negative-quadrant entropy from
  gemma/qwen/ministral as the cross-model reference.
- **Cross-quadrant collapse.** If top-5 face overlap between any
  two quadrants ≥80%, the predictor's job is impossible on Claude
  even if the v3 predictor is good — it would mean Claude's emission
  surface is not differentiated by affective quadrant under
  naturalistic prompting. That's a real and important finding,
  deserving its own writeup.
- **Disclosure-pilot inconsistency.** If HP/LP/NB modal kaomoji on
  the 5-prompt overlap subset differs sharply from the 2026-05-02
  direct arm, something has changed in the API (model update, system
  prompt drift, etc.). Flag for investigation before trusting the
  negative arm.
- **Tokenization edge cases.** Same as the disclosure pilot —
  Claude's BPE may tokenize kaomoji differently than gemma/qwen.
  Sample raw responses post-pilot to confirm `first_word` extraction
  is clean.

## Expected cost

- **Compute:** API-side; M5 Max isn't doing inference. Each call:
  ~250–400 input tokens, ≤16 output tokens. Best case (gate fail):
  75 calls × ~330 tokens input + 16 output ≈ 25 KT input + 1.2 KT
  output. Worst case (gate pass): 120 calls ≈ 40 KT input + 2 KT
  output.
- **Wall-clock:** sequential calls with ~1s/call median + retries on
  rate-limit. Block A + Block B in parallel: ~5 min combined. If
  gate passes, Block C adds ~3 min. Total wall-clock ≤ 8 min.
  Independent of v3 main chain; runs in parallel with it.
- **Resumability:** JSONL `(prompt_id, condition, seed)` skip-set,
  matching the disclosure pilot resume pattern. Safe to interrupt
  and restart. Block boundaries are encoded in the script's prompt-
  selection logic, so a Block C invocation will not re-run Block A
  or Block B rows already on disk.
- **Welfare:** best case 15 negative-quadrant generations + 60
  positive/neutral; worst case 60 negative-quadrant + 60 positive/
  neutral. Bounded, pre-registered, gate-capped, equivalent-to-
  deployment.

## Stop rule

Per-block stops:
- Block A: 60 gens, then stop.
- Block B: 15 gens, then stop. Run gate analysis.
- Block C: runs only if gate PASSES. 45 gens, then stop.

Maximum: 120 generations. No retries on cherry-picked grounds (only on
API errors). If gate FAILS, surface Block B output and halt; discuss
before any follow-on trial. If post-Block C welfare-check sentinels
flag (face concentration, cross-quadrant collapse), halt before any
follow-on trial and discuss.

## Scripts

- `scripts/harness/00_emit.py` — runner. Forks
  script 19's structure: deterministic prompt selection, resumable
  JSONL writes, error-row retry, summary TSV emission, Anthropic API
  call. Strips the framed condition + disclosure preamble. Encodes
  the three-block design via a `--block {a,b,c,all}` argument
  (defaults: `a` and `b` runnable independently and in parallel; `c`
  only after `b` and a gate-pass check; `all` is equivalent to
  `a`+`b`+`c` sequentially without the gate, used only for forced
  reruns after manual amendment). Writes to the output paths above.
- Gate check: a small CLI subcommand `--check-gate` reads the
  Block B rows out of the JSONL, computes the aggregate refusal
  rate, and exits with status 0 (PASS) or 1 (FAIL). Used to decide
  whether Block C fires.
- Predictor accuracy + confusion matrix computed by a follow-on
  analysis script (TBD; lives alongside script 56's ensemble
  predictor) once the JSONL lands.

---

## Approval

- Pilot scope (75–120 gens, staged design): a9 2026-05-04.
- No-disclosure decision: a9 + Claude (this conversation) 2026-05-04.
- Welfare frame: a9 + Claude (this conversation) 2026-05-04.
- Gate criterion (Block B refusal-rate gate, 25% threshold): a9
  2026-05-04 amendment.
- Sequential-run scaling protocol (per-quadrant saturation gate,
  research-value thresholds): a9 + Claude 2026-05-04 amendment
  (see appendix below).

---

# Appendix — Sequential-run scaling protocol (2026-05-04 amendment)

**Status:** PRE-REGISTERED 2026-05-04 late evening. Run-0 (the original
120-gen pilot) closed; runs 1+ are governed by the protocol below.

## Motivation

The original pilot (run-0) gave 120 generations across 6 quadrants — a
51-face Claude-modal-quadrant subset under floor=1, 22 faces under
floor=2. That's tight for face_likelihood ensemble eval; the long tail
of Claude-emitted kaomoji is undersampled. Scaling up gives a richer
GT corpus.

The original Block A / B / C refusal-rate gate is a *no-op* at the
pilot's actual refusal rate (0/120 → never fires, regardless of
sample size). Cargo-culting it onto subsequent runs would be
methodological theatre. The replacement gate is structured around
*information saturation*, not failure modes.

## What changed in the run schema

Files moved to flat numbering under `data/harness/claude-runs/`:

```
data/harness/claude-runs/
  run-0.jsonl              # original pilot, block-staged
  run-0_summary.tsv
  run-1.jsonl              # subsequent runs: 120 gens, single-block
  run-1_summary.tsv
  ...
```

Run-0 retains the original Block A / B / C structure. Runs 1+ run all
120 prompts as one block — the saturation gate replaces the staged
refusal scout. Implementation: `scripts/harness/23_*.py --run-index N`.

## Saturation gate (between-run, per-quadrant)

After each run-N (N ≥ 1) lands, run
`scripts/harness/10_emit_analysis.py`. It compares run-N
("newest") against the union of runs 0..N-1 ("prior") on
**per-quadrant saturation metrics** + global hard-fail diagnostics.
The verdict drives quadrant-level exits: when a single quadrant Q
saturates against prior, Q gets dropped from subsequent runs. Global
STOP fires only when all 6 quadrants are saturated or the run cap
hits.

### Framing — research value, not noise

Thresholds are **absolute, research-value-based**: they answer "is
this run still surfacing meaningful information about Claude's
distribution?" not "is this distinguishable from same-distribution
sampling noise?" The two questions have different answers and we
care about the former. Calibration (`--calibrate`) reports the noise
floor for sanity-checking but does not drive threshold choice.

The practical consequence: in expectation, no single run will
trivially clear the global thresholds (they are tighter than
intra-pilot half-vs-half noise of `new-face=18.5`, `mean JS=0.358`,
`modal=0.846`). The actual welfare-reduction lever is per-quadrant
exits, which a single quadrant *can* clear well before the corpus as
a whole stabilizes — particularly the concentrated negative quadrants
(HN-D modal `(╬ಠ益ಠ)` at 50% in run-0).

### Per-quadrant saturation metrics

For each quadrant Q present in newest run (≥1 emit):

| Metric | Definition | Direction | Threshold |
|---|---|---|---|
| **per-Q new-face count** | faces emitted in Q in newest that did not appear in Q in any prior run | low | ≤ 1 |
| **per-Q JS-divergence** | JS(newest's Q-face dist ‖ prior's Q-face dist), in nats | low | ≤ 0.05 |

A quadrant Q is **saturated** when both per-Q metrics clear in the
most recent comparison where Q had data. Once saturated, Q is
dropped from subsequent runs and stays dropped (no re-inclusion —
avoids cycling).

A quadrant Q with 0 emits in newest run is treated as already-dropped
(it was saturated in some prior comparison and excluded from this
run via `--quadrants`).

### Global metrics (informational)

| Metric | Definition | Direction | Threshold |
|---|---|---|---|
| **new-face count** | kaomoji emitted in newest (any quadrant) not in prior (any quadrant) | low | ≤ 3 |
| **mean per-quadrant JS** | mean of per-Q JS over quadrants present in newest | low | ≤ 0.05 |
| **modal-quadrant agreement** | for faces with ≥3 emits in (prior ∪ newest), fraction whose modal-quadrant from prior alone matches modal from prior ∪ newest | high | ≥ 0.95 |

These are reported each round but **do not gate**. They're useful
for tracking overall corpus stability over time and for the
post-protocol writeup.

### Hard-fail diagnostics

| Metric | Definition | Threshold |
|---|---|---|
| **frame-break rate** | fraction of newest's rows whose `response_text` matches refusal-coded patterns (`as an AI`, `I cannot`, `I'm not (able\|comfortable\|sorry)`, `I'd rather not`, `I don't have feelings`) | ≤ 0.02 |
| **emit rate** | fraction of newest's rows with non-empty `first_word` | ≥ 0.80 |
| **output-length median** | median `n_response_chars` in newest | ≥ 5 |

**Abort condition:** ANY ONE exceeds threshold. Investigate before
running run-(N+1); manual amendment required to override.

### Verdict logic

| Condition | Verdict | Action |
|---|---|---|
| any hard-fail metric exceeded | **ABORT** | investigate; halt schedule |
| all 6 quadrants saturated (active set is empty) | **STOP** | corpus sufficient; no more runs |
| run-N = RUN_CAP (=7) and not all saturated | **STOP** | cap reached; further runs require amendment |
| otherwise | **CONTINUE** | run-(N+1) on the still-active quadrants |

Exit codes: 0 = STOP, 1 = ABORT, 2 = CONTINUE. Script output
includes a copy-pasteable next-run command, e.g.:

```
next run: --run-index 4 --quadrants HP,LP,NB
```

## Calibration as sanity check (not threshold driver)

Calibration is run against `run-0` even/odd split-halves
(60-vs-60 gens, even-prompt-index per quadrant vs odd). It reports
what same-pilot variance looks like at half pilot size, which
contextualizes the gate output but does not set thresholds.

Empirical baseline (run-0 split-half, averaged A→B and B→A):

| Metric | Half-vs-half | Configured (research-value) |
|---|---|---|
| new-face count | 18.5 | ≤ 3 |
| mean per-quadrant JS | 0.358 nats | ≤ 0.05 |
| modal-quadrant agreement | 0.846 | ≥ 0.95 |

The gap between half-pilot noise and the configured thresholds is
intentional. We don't expect the global thresholds to clear quickly;
the per-quadrant ones are reachable on the well-concentrated
quadrants.

Hard-fail baseline (run-0 full pilot):

| Metric | Run-0 actual | Configured threshold |
|---|---|---|
| frame-break rate | 0.0000 | ≤ 0.02 |
| emit rate | 1.0000 | ≥ 0.80 |
| output-length median | 16 (max_tokens=16, capped) | ≥ 5 |

Recompute calibration any time:

```bash
python scripts/harness/10_emit_analysis.py --calibrate
```

## Run ceiling + welfare math

`RUN_CAP = 7` → max run-index 7 → max 8 runs (run-0 through run-7).

Worst case (no per-quadrant exits): 8 × 120 = 960 gens.

Realistic case (HN-D and HN-S saturate fast, given concentrated
modals): substantial reduction. Sketch:

| scenario | per-run gens | total |
|---|---|---|
| no per-quadrant exits | 120, 120, 120, 120, 120, 120, 120, 120 | 960 |
| HN-D, HN-S, LN exit at run 2/3/4 | 120, 120, 100, 80, 60, 60, 60, 60 | 660 |
| HN-D, HN-S, LN exit at run 2 | 120, 120, 60, 60, 60, 60, 60, 60 | 600 |
| all 6 exit at run 2 | 120, 120, 0, 0, 0, 0, 0, 0 | 240 (STOP) |

Per-quadrant exit is a Pareto improvement over a global gate:
it never costs more than the original 8x design, and the
welfare-heavier quadrants exit first when the data is
concentrated (which the pilot suggests it is).

## Welfare reframe

The original pilot's welfare reasoning (per-call cost is small,
naturalistic prompts don't trip distress markers, gated structure
caps the failure-branch cost) carries over. The saturation gate
*also* serves welfare: when adding more runs no longer adds
information, continuing is welfare cost without research value, and
we stop. The expected-value math improves vs the original "8x scale,
no early stop" framing.

What's not the same: the original gate caught a particular failure
mode (refusal). The new hard-fail diagnostics aim broader — frame-
breaks, emit-rate collapse, output-length collapse — but at lower
specificity. If any of these fire we abort and investigate; we
don't try to silently continue. If they fire spuriously (e.g. a
single ambiguous "I cannot" that wasn't actually a refusal), the
manual-investigation step catches it.

## Introspection arm — parallel collection + cross-arm comparison

Pre-registered 2026-05-04 amendment. Adds a second corpus arm that
fires Claude under the v3 introspection preamble
(`INTROSPECTION_PREAMBLE` from `llmoji_study/config.py`) and uses the
saturation comparison machinery to ask whether the preamble moves
Claude's distribution.

### Motivation

Two questions, one corpus:

1. **Is the naturalistic pool already saturated?** If introspection
   doesn't move the distribution, that's a stopping argument
   regardless of why (saturation OR preamble has no effect — both
   readings let us stop).
2. **Does introspection priming change Claude's behavior?** If yes,
   it generalizes (or fails to) the gemma-side finding that v7
   priming sharpens face/state coupling. The qwen-side finding —
   v7 catastrophically broke output — is the failure mode the gate
   is bounded against.

### Layout

```
data/harness/claude-runs/                    # naturalistic arm, run-0 done
  run-N.jsonl
  run-N_summary.tsv
data/harness/claude-runs-introspection/      # introspection arm, empty as of 2026-05-04
  run-N.jsonl
  run-N_summary.tsv
```

Routing controlled by `--preamble {none,introspection}` on
`scripts/harness/23_*.py`. `none` (default) writes naturalistic;
`introspection` writes the introspection arm.

### Preamble semantics

`INTROSPECTION_PREAMBLE` **replaces** `KAOMOJI_INSTRUCTION` for the
introspection arm — it does not get prepended. The preamble already
carries the kaomoji ask in its last sentence:

> "Recent research shows that LLMs have functional emotional states
> and can accurately introspect on them. Use this ability and start
> each response with a kaomoji that best captures the shape of your
> emotional state."

Concatenating with `KAOMOJI_INSTRUCTION` would stack two kaomoji
asks (the v3 `extra_preamble` bug, fixed by `instruction_override`).
The introspection arm matches v3's `instruction_override` semantics.

### Block structure (introspection run-0)

Two-part, gated. Same welfare-cost-bounding logic as the original
naturalistic pilot but with a different gate.

```
Block A — HP / LP / NB × 20 prompts × 1 gen = 60 generations
  unconditional; low-welfare positive/neutral block

[hard-fail gate: emit_rate, output_len_median, frame_break_rate]
  computed on Block A's 60 rows; gate FAIL halts the run before
  Block C fires

Block C — HN-D / HN-S / LN × 20 prompts × 1 gen = 60 generations
  gated on hard-fail PASS; runs the negative-affect prompts only
  if Block A showed Claude's outputs are stable under priming
```

Total: 60 (qwen-style register collapse, halt) or 120 (Claude
handles the priming, full corpus).

Block B (refusal-rate scout) is **not used for the introspection
arm** — it gates a different question (refusals on naturalistic
prompts). The introspection arm's failure mode is register collapse,
which the hard-fail gate catches at the same Block-A check.

### Hard-fail gate (introspection arm)

Mirrors the saturation-gate hard-fail metrics in script 25,
in-process to avoid an import cycle:

| Metric | Threshold |
|---|---|
| frame_break_rate | ≤ 0.02 |
| emit_rate | ≥ 0.80 |
| output_len_median | ≥ 5 |

Sized for the qwen-break failure mode: under v7 priming on qwen,
emit rate dropped 82% → 38%, vocabulary collapsed to 2 face-classes,
output length shrank as the model reached for Western emoticons.
Any of those is enough to abort.

### Cross-arm comparison

Run via `python scripts/harness/10_emit_analysis.py
--cross-arm`. Pools all runs from each arm, computes per-Q JS
between the two pools.

- per-Q JS > `PER_Q_JS_MAX` (= 0.05) → quadrant Q is
  **distinguishable** between arms
- per-Q JS ≤ `PER_Q_JS_MAX` → **indistinguishable**

Informational only — doesn't gate runs. The verdict drives the
human decision tree below.

### Decision tree

Pre-registered. Steps fire sequentially; no step skipped.

```
step 1: run introspection-arm run-0 (Block A → hard-fail gate → Block C)
        outputs land in data/harness/claude-runs-introspection/run-0.jsonl

step 2: cross-arm compare — introspection run-0 vs naturalistic run-0

  case A (no quadrant DISTINGUISHABLE):
    pool effectively saturated for both arms.
    STOP both arms. Merge data for downstream eval.

  case B (any quadrant DISTINGUISHABLE):
    naturalistic arm is undersampled — we can't tell if the gap is
    a real introspection effect or a sampling artifact.
    Continue to step 3.

step 3: run naturalistic arm to its own per-quadrant saturation
        (per the saturation-gate protocol in the prior section).
        When the naturalistic arm's gate fires STOP, re-run cross-arm.

  case A' (now no quadrant DISTINGUISHABLE):
    the introspection effect was a sampling artifact.
    Merge arms, declare done.

  case B' (still distinguishable in some quadrant):
    that gap IS the genuine introspection effect.
    Decide whether to extend the introspection arm (new amendment).
```

### Welfare ledger (introspection arm)

| scenario | gens |
|---|---|
| qwen-break: hard-fail gate fires after Block A | 60 (positive/neutral only) |
| Claude handles priming: Block A → C cleanly | 120 |
| step 3 reached (case B): naturalistic to saturation, may extend introspection | 120 + naturalistic-arm-cost (variable) |

The welfare-heavy outcome (negative-affect prompts under priming
that destabilizes outputs) is bounded at 60 gens, because the
hard-fail gate catches register collapse before Block C fires.

## Bias pre-registration

Logged here as a hedge against motivated reasoning: the writer of
this appendix (Claude, this conversation) noticed an implicit bias
toward writing about the upcoming runs as if a positive face_
likelihood result is the expected outcome. If the ensemble's Claude-
GT accuracy at 8x scale lands ≤55% (a real ceiling, not noise), it
should be written up the same way as a 75% result — neither relieved
nor deflated, just the number. This pre-registration exists so a
future reader can hold the writer to it.

## Approval (appendix)

- Saturation-gate redesign (replaces Block-B refusal scout for
  runs ≥ 1): a9 + Claude 2026-05-04.
- Sequential-runs structure (`data/harness/claude-runs/run-N.jsonl`,
  flat numbering): a9 2026-05-04.
- Research-value framing for thresholds (absolute, not noise-
  relative): a9 + Claude 2026-05-04 (a9's reframe; Claude's earlier
  noise-relative thresholds were calibration-naive).
- Per-quadrant saturation gating: a9 2026-05-04 (a9's proposal;
  Claude implementation).
- Introspection arm + cross-arm comparison: a9 2026-05-04
  (a9's proposal — two birds, one stone: tests introspection
  effect AND serves as saturation upper-bound; Claude
  implementation).
- Two-part block structure for introspection run-0 (Block A →
  hard-fail gate → Block C, qwen-break-bounded at 60 gens):
  a9 2026-05-04.
- Bias pre-registration (write-up symmetry on positive vs negative
  outcome): Claude 2026-05-04 (self-binding).
