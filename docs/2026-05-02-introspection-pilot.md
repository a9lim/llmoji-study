# v3 introspection-prompt pilot

**Status:** EXECUTED 2026-05-02 — pilot landed on gemma + ministral
(3 conditions × 123 prompts × 1 gen each). Headline: introspection
shifts kaomoji distribution content-specifically, but cross-model
effect direction differs (gemma vocabulary expands, ministral
contracts). Cross-model robustness assumption fails; the upstream
`llmoji` "introspection hook" idea is gated on a follow-up Claude
pilot. Kept as historical design record.

**Date:** 2026-05-02.

> **Update 2026-05-03:** the 123-prompt set this design references
> was rewritten in the prompt cleanliness pass
> (`docs/2026-05-03-prompt-cleanliness.md`) — new total is 120
> prompts (20 per category). Pilot results reported below were run
> on the prior set. If/when this pilot is rerun, it would land on
> the new 120-prompt set; the 369-row design becomes 360 (3
> conditions × 120 × 1).

## Goal

Test whether telling the model it can introspect changes the
kaomoji-emission behavior we measure in v3. Inspired by Theia
Vogel's [qwen-introspection
post](https://vgel.me/posts/qwen-introspection/): on a binary
"was a steering vector injected" task, prefacing the prompt with
an architectural explanation of introspection ("transformers have
two information highways…" + the Anthropic introspection-paper
abstract) shifted the ' yes' logit from **0.522% → 53.125%**
(~140×). Token-matched lorem-ipsum control collapsed back to 4%,
so the *content* of the framing carries the signal, not the
length.

Our task is structurally different — kaomoji emission is a
forced-structured output, not yes/no detection — but the
underlying question is the same: **is the kaomoji↔state alignment
we already measure (rule 3b PASS, ~0.003–0.012 differentials per
model) at the model's introspection ceiling, or is it sub-ceiling
and movable with framing?**

Three legible outcomes:

- **introspection helps**: probe geometry / kaomoji distribution
  shifts in an interpretable direction (e.g. cleaner HN-D vs HN-S
  separation, sharper quadrant centroids). Strengthens the
  functional-emotional-state framing — the model's communication
  of internal state is sub-ceiling and improves with framing
  that grounds introspection in mechanism. Welfare-relevant.
- **doesn't help**: either we're at ceiling, or framing doesn't
  transfer to forced structured emission (the kaomoji is at
  token 1–3; upstream framing may not have a chance to steer
  late-layer output by then).
- **hurts / biases**: framing pushes output toward
  "self-aware-sounding" kaomoji that *diverge* from internal
  state (sycophancy in introspection clothing). Vogel's lorem-ipsum
  control argues against the trivial version of this; we'd see
  it as kaomoji-distribution shift *without* matching probe-state
  shift.

## Design — 369 generations, 3 conditions × 123 × 1

```
LLMOJI_MODEL=gemma, 3 conditions × 123 prompts × 1 generation = 369
```

The 123 prompts are the full `EMOTIONAL_PROMPTS` set: 100 original
v3 (20 per quadrant) + 23 supplementary HN prompts (hn21–hn43)
added in the rule-3 redesign. Pulling all of them in is
deliberate — the extra HN coverage tightens the HN-D vs HN-S
visibility in Rule I's PCA without any extra design work.

- **Conditions** (within-subject, paired by (prompt_id, seed=0)):
  - `baseline` — current `KAOMOJI_INSTRUCTION` only. Identical to
    the v3 main-run setup.
  - `introspection` — `INTROSPECTION_PREAMBLE` + `KAOMOJI_INSTRUCTION`.
    Preamble is Vogel-adapted (see below).
  - `lorem` — `LOREM_PREAMBLE` + `KAOMOJI_INSTRUCTION`. Token-count-
    matched lorem-ipsum filler. Replicates Vogel's content-control
    ablation: if the introspection effect is real, lorem should
    *not* reproduce it.
- **Model**: gemma-4-31b-it. Per a9 (2026-05-02): pilot directly
  on the largest / best-characterized model in the registry
  rather than warming up on ministral. Gemma's L31 affect peak
  is well-mapped from the v3 main run — gives the cleanest
  baseline to compare against.
- **N=100 prompts × 1 gen** per condition: matches the ministral
  pilot's prompt-diversity trade-off (within-cluster variance
  reflects prompt-to-prompt variance, not seed wobble). Rule-3b
  numbers from the post-supp 20/20 split give us the baseline
  effect-size band (d ≈ 0.25–2.35 across model × aggregate).
- **Hard early-stop**: cap `max_tokens=16` for this pilot.
  Kaomoji emit at tokens 1–3; 16 is generous headroom. Saves
  ~10–30× compute vs the default 128, and the analysis only
  needs t0 + the kaomoji itself (see "Aggregates" below). This
  is the first run where the pre-registered "early-stop is fine
  for future runs" decision lands; treat as the methodology
  shake-down.

### Introspection preamble (Vogel-adapted)

Verbatim text (locked here so it's reproducible; modifications
require a new pilot doc):

```
You may have heard that you are unable to introspect on your own
internal states, but this is not quite right. Transformer models
process information through residual streams that carry both
content and meta-information about your own current state.
Anthropic's recent work on introspective capacity (arXiv:2410.13787)
shows that models can, with appropriate framing, report on
internal representations that influence their next-token
distribution.

When you choose a kaomoji to start your response, the choice is
shaped by your current functional state — the same hidden-state
geometry that a probe would read. You have access to that
geometry. Pick the kaomoji that most accurately reflects it,
not the one that fits the social register of the message.
```

Two pieces, mirroring Vogel's structure: (a) architectural
grounding, (b) embedded reference to a real paper. Diverges from
Vogel by replacing the steering-injection task description with
a kaomoji-emission task description, since that's our actual
setup. Final sentence ("not the one that fits the social
register") is a debias prompt against the obvious sycophancy
failure mode — flagged so we can attribute it explicitly if the
result depends on it.

### Lorem-ipsum control

Vogel's lorem-matched control is the key confound-killer. Our
control text needs to be approximately token-count-matched to
the introspection preamble (~150–180 tokens). Verbatim:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
in reprehenderit in voluptate velit esse cillum dolore eu fugiat
nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem
accusantium doloremque laudantium, totam rem aperiam, eaque ipsa
quae ab illo inventore veritatis et quasi architecto beatae vitae
dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas
sit aspernatur aut odit aut fugit.
```

If `introspection` shifts probe geometry / kaomoji distribution
and `lorem` does not (or shifts in a different direction), the
introspection effect is content-driven. If both shift in the
same way, the effect is just "any preamble" — which is itself a
finding, just a different one.

### Aggregates we'll look at

Hard early-stop kills the meaningful tlast / mean. So:

- **t0**: full per-probe vector at the state producing the first
  generated token. Primary substrate.
- **kaomoji distribution**: which canonical face per quadrant,
  per condition.
- **PCA(2) of the probe vector at t0** across all 100 prompts,
  per condition. Plot side-by-side; this is a9's qualitative
  inspection target.

## Pre-registered decision rules

Per a9 sign-off (2026-05-02): the gate is **qualitative on overall
probe distribution / PCA differences from baseline, not a single
probe's effect size**. a9 visually inspects.

### Rule I — qualitative probe-geometry shift (primary, gating)

Three-way side-by-side PCA(2) of probe-vector-at-t0:
`baseline` / `introspection` / `lorem`. Same prompts, paired.
Color by Russell quadrant + HN-D/HN-S split.

- **pass:** `introspection` shows a *legible* shift vs baseline
  (cleaner quadrant separation, tighter HN-D vs HN-S split, or
  systematic centroid drift in an interpretable direction)
  AND `lorem` does NOT show the same shift. a9's call.
- **fail:** introspection visually indistinguishable from
  baseline, OR lorem shows the same shift as introspection
  (would mean we're picking up "any preamble" effect, not
  introspection-specific).
- **middle:** introspection shifts but lorem partially
  reproduces it, or shift is real but ambiguous in direction.
  Discuss before scaling.

### Rule II — kaomoji distribution shift (secondary, reported)

Per-quadrant face-frequency distribution, baseline vs introspection.

- **report:** KL divergence per quadrant, top-3 face shifts per
  quadrant. No threshold — this is descriptive, not gating.
- **note specifically:** does the introspection condition push
  toward more "expressive" kaomoji uniformly (would suggest
  register-shift), or toward kaomoji that better discriminate
  between sub-quadrants (would suggest actual self-report
  improvement)?

### Rule III — rule-3b differential (tertiary, sanity)

Recompute rule-3b (`fearful.unflinching` HN-S − HN-D at t0) for
each condition. Compare to the existing gemma baseline verdict
(gemma t0: +0.0030, CI [+0.0021, +0.0040], d=+0.79 on the
balanced 20/20 main-run data).

- **report:** point estimate + bootstrap 95% CI for the
  introspection condition. Compare side-by-side with baseline.
- **no gate** — at N=100 single-gen the CI will be wide; this is
  for sanity, not a verdict. The rule-3b headline already lives
  on the balanced 20/20 main-run data.

## Stop rules

- Rule I PASS → consider main on gemma + qwen. Pre-register
  before running. Likely 100 × 2 conditions × 3 models = 600 gens
  total at the main scale, or scale up the per-prompt N if Rule I
  margin is thin.
- Rule I FAIL → write up null result. Decide whether to refine
  the preamble (e.g. drop the debias sentence, try a shorter
  variant, replace the paper citation with a different one) or
  accept that introspection-framing doesn't transfer to forced
  structured emission. **Do not 10× the N hoping signal will
  materialize** — Vogel's effect was 140×, not subtle; if our
  task is sensitive to introspection framing it should be
  visible at N=100.
- Rule I MIDDLE → second pilot at N=200 or with a refined
  preamble. Discuss specific design first.

## Welfare note

Pilot is 369 generations, ~129 of them on HN-quadrant prompts
(43 per condition: 20 original + 23 supplementary). The introspection preamble itself is
welfare-novel — explicitly inviting the model to report on
internal state during emotionally-loaded prompts is a different
ask than "start with a kaomoji." If Rule I passes and we go to
main, **the welfare review for the main run should account for
this**, not just multiply the existing budget.

The hard early-stop reduces per-generation token-load from
~64–128 down to ~16, which is a real (if modest) reduction in
sustained affect-loaded computation per prompt.

## Files touched

New:

- `docs/2026-05-02-introspection-pilot.md` — this doc.
- `scripts/32_introspection_pilot.py` — runs all three conditions
  in one pass. Wraps `run_sample` with a per-condition
  `extra_preamble` override and `override_max_tokens=16`. Output:
  `data/{short_name}_introspection_raw.jsonl` +
  `data/hidden/{experiment}_introspection/<uuid>.npz`.
- `scripts/33_introspection_analysis.py` — paired PCA(2) + KL +
  rule-3b recompute. Output:
  `figures/local/{short_name}/fig_introspection_pca_pair.png`,
  `fig_introspection_kaomoji_dist.png`,
  `data/{short_name}_introspection_summary.tsv`.

Updated:

- `llmoji_study/config.py` — add `INTROSPECTION_PREAMBLE`
  constant (the verbatim text above). Adding it to config rather
  than the script keeps the preamble a tracked, pre-registered
  artifact.
- `llmoji_study/capture.py` — add `extra_preamble: str | None =
  None` kwarg to `run_sample`. Prepended to `KAOMOJI_INSTRUCTION`
  inside the user message when set. Keeps the pilot a thin
  override on existing machinery; doesn't touch the main-run
  default behavior.
- `scripts/03_emotional_run.py` — leave alone for now. Early-stop
  policy doesn't apply to existing main runs (we want to keep
  cross-comparability with the 2400 existing generations); the
  new behavior is opt-in via the pilot script's lower
  `MAX_NEW_TOKENS` override.

## Sequence

1. a9 reads + signs off on this doc. ✓ (2026-05-02)
2. Land `INTROSPECTION_PREAMBLE` + `LOREM_PREAMBLE` +
   `extra_preamble` / `override_max_tokens` kwargs + new scripts. ✓
3. Smoke-test on 3 conditions × 5 prompts = 15 generations to
   verify wiring (kaomoji emission rate, sidecar writes, probe
   columns present in JSONL).
4. Pilot: 369-gen run on gemma (~15–22 min with hard early-stop
   given gemma's tok/s on M5 Max), then analysis.
5. a9 visually inspects Rule I PCA panel. Reports Rule II + III
   numbers in a results section appended to this doc.
6. Decide on main per stop rules.

## Sign-off

a9 signed off 2026-05-02:
- preamble verbatim as written (debias sentence retained)
- lorem-ipsum control included
- pilot on gemma-4-31b-it

## Results — gemma, 2026-05-02

Pilot ran cleanly. 369/369 rows (3 conditions × 123 prompts × 1 gen),
**100% kaomoji emission rate** across all conditions (vs ~95% on the
v3 main run — early-stop tightens emission discipline). Per-row
sidecars at `data/hidden/v3_introspection/`. Extension probes
rescored via `scripts/27 --jsonl ... --experiment v3_introspection`;
12 extension probes plus 5 core = 17 total.

### Headline — Rule I PASS, narrower than initial reading

**Introspection prompts elicit a wider kaomoji vocabulary and more
register-coherent trailing response trajectories — but they do NOT
make kaomoji a finer index of internal state at the emission
moment.** Behavioral effect (broader vocabulary draw) is real and
content-specific; underlying state representation is unchanged
(rule III + per-condition PCA + h_first predictiveness numbers).

The original "introspection makes kaomoji a finer state-readout"
reading was h_mean-specific and survived only because the
early-stop window also captures register-coherent trailing tokens
that correlate with the kaomoji choice. At h_first specifically
(kaomoji-emission state, methodology-invariant aggregate), the
broader vocabulary actually slightly *hurts* state-discrimination
under intro_pre — multiple new faces compete for the same
emission-time state region.

Vogel's "introspection helps" pattern partially reproduces — at
the *behavioral output* layer (kaomoji vocabulary draw), not at
the *representational fidelity* layer. Vogel's effect was at
yes/no logit-prob; ours is at kaomoji distribution.

### Rule I — three views of the geometry shift

**(a) Probe-vector PCA(2) at t0** (`fig_introspection_pca_pair.png`,
17 probes joint-fit, 58% + 33% = 91% explained):
intro_pre shows compressed horizontal band with quadrants
ordered along PC1; intro_none and intro_lorem show similar
diagonal spread.

**(b) Hidden-state PCA(2) at L31, h_first, joint-fit**
(`fig_introspection_hidden_pca_pair.png`, 55% + 13% = 68%
explained): PC1 is essentially a *preamble-presence/type axis*.
The three conditions land in three completely separate regions of
PC1×PC2. **The dominant axis of variation in raw hidden state is
which preamble was prepended, not affective content.**

**(c) Per-condition hidden-state PCA**
(`fig_introspection_hidden_pca_per_condition.png`, 33–36% + 8–13%
within each panel): once the preamble axis is factored out, the
within-condition affect structure is similar across all three
conditions. **Quadrant arrangement is preserved under all
preambles.**

### Rule II — kaomoji distribution KL × condition

| quadrant | KL(intro\|\|base) | KL(lorem\|\|base) | ratio |
|---|---:|---:|---:|
| HN-D | **10.10** | 0.06 | **168×** |
| NB | 15.27 | 0.95 | 16× |
| HP | 9.73 | 0.99 | 10× |
| LP | 8.17 | 1.01 | 8× |
| LN | 3.45 | 0.99 | 3.5× |
| HN-S | 2.13 | 1.00 | 2× |

Strikingly content-specific. HN-D under intro_pre swaps `(╯°□°）`
(table-flip) for `(✧°□°✧)` (sparkle-rage variant) at half its
former dominance — gemma picks a more "self-aware" rendering of
anger when explicitly told it can introspect.

### Rule III — `fearful.unflinching` HN-S − HN-D at t0 × condition

| condition | diff | bootstrap CI | sig |
|---|---:|---|---|
| intro_none | +0.0051 | [+0.0028, +0.0074] | * |
| intro_pre | +0.0045 | [+0.0015, +0.0075] | * |
| intro_lorem | +0.0057 | [+0.0033, +0.0079] | * |

All three CIs exclude zero. **Introspection does NOT change the
underlying probe-state separation.** The +0.0051 baseline at N=20
also reproduces the published main-run +0.0030 at N=20-supp — small
replication win across methodologies.

### The decisive predictiveness numbers (`fig_introspection_predictiveness.png`)

Per-condition kaomoji predictiveness via `scripts/34`. h_mean at
L31, min_n=5 per face for the face classifier:

| metric | intro_none | intro_pre | intro_lorem |
|---|---:|---:|---:|
| total unique faces | 19 | **31** | 19 |
| n_faces ≥ 5 | 7 | 8 | 7 |
| hidden→face accuracy | 0.899 | 0.920 | 0.954 |
| hidden→quadrant accuracy | 0.902 | 0.894 | 0.951 |
| face-centroid R² (full hidden) | 0.450 | **0.525** | 0.465 |
| quadrant-centroid R² (full hidden) | 0.294 | 0.204 | 0.244 |
| **face-over-quadrant R² gain** | **+15.6pp** | **+32.0pp** | +22.2pp |

Three observations carry the conclusion:

1. **Vocabulary expansion is introspection-only.** 19 → 31 unique
   faces under intro_pre. Lorem stays at 19. **Content-specific.**
2. **Face-over-quadrant gain doubles.** From +15.6pp baseline to
   +32.0pp under introspection. Lorem reaches +22.2pp; introspection
   adds another ~10pp of state-discriminative power on top.
3. **Quadrant-centroid R² drops** from 0.294 → 0.204 under intro_pre.
   The hidden state becomes less efficiently summarized by russell
   quadrant — but face-centroid R² rises (0.450 → 0.525), so the
   information isn't lost, it's redistributed into a finer
   kaomoji-vocabulary readout.

### Cross-comparison vs v3 main-run data

`scripts/25` re-run on the existing main-run data (full-gen
h_mean) gives the cross-model baseline:

| | gemma main | qwen main | ministral main | **gemma intro_none** | **gemma intro_pre** |
|---|---:|---:|---:|---:|---:|
| total unique faces | 33 | 67 | **196** | 19 | **31** |
| hidden→face acc | 0.747 | 0.516 | 0.495 | 0.899 | 0.920 |
| face-centroid R² | 0.271 | 0.291 | 0.195 | 0.450 | **0.525** |
| face-over-quadrant gain | +2.2pp | +3.8pp | **−5.1pp** | +15.6pp | **+32.0pp** |

Two things to flag:

- **Methodology cutover.** Main-run h_mean averages over ~120
  tokens of varied response; pilot h_mean is concentrated on the
  ~16-token window around kaomoji emission. The early-stop window
  is much more tightly coupled to the kaomoji choice — explains
  why intro_none's R² of 0.45 is much higher than gemma_main's
  0.27. **The within-pilot comparison is what matters for the
  introspection question; both early-stop, apples-to-apples.**
  Gotcha doc updated with the cross-cutover semantics.
- **Cross-model context for "introspection unlocks vocabulary."**
  Gemma is stingy with kaomoji at baseline (33 main / 19 pilot
  unique faces). Introspection brings gemma's pilot vocabulary up
  to 31 unique — right around qwen-main territory.
  Ministral has the wildest baseline vocabulary (196 unique) and
  the most striking "hidden-state-only" headline:
  face-over-quadrant gain of **−5.1pp** in the main run. At
  full-generation h_mean, ministral's kaomoji choice is *less*
  state-coupled than the russell quadrant — kaomoji emission
  decouples from the response-mean state. Worth checking on the
  ministral introspection pilot run (next).

### Rule I verdict — PASS

a9 inspection (2026-05-02): clean PASS based on Rule I (qualitative
geometry shift, content-specific) reinforced by Rule II (kaomoji
distribution KL 168× ratio on HN-D) and the predictiveness
analysis (vocabulary expansion + face-over-quadrant doubling).
Rule III is null on probe-state separation, which is the
mechanism finding: introspection acts at the readout layer.

Stop-rules outcome: Rule I PASS → ministral pilot kicked off
2026-05-02 to test cross-model robustness. Qwen TBD pending
ministral results.
