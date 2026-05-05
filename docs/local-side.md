# Local side: probes, hidden state, face_likelihood

The local side runs probes and hidden-state capture on five open-weight
causal LMs via [`saklas`](https://github.com/a9lim/saklas), so I can
read and intervene on the hidden state directly. Public-facing
summary at [a9l.im/blog/introspection-via-kaomoji](https://a9l.im/blog/introspection-via-kaomoji);
this doc covers methodology and current findings. Detail and per-pilot
numbers in [`findings.md`](findings.md). Historical record (v1 / v2
steering pilots, single-layer reads, gemma 1D-vs-qwen 2D framing,
pre-cleanliness numbers, the face-input bridge, extension probes,
introspection iterations v0 through v6) lives in
[`previous-experiments.md`](previous-experiments.md).

## What this is

llmoji-study asks whether kaomoji choice in local causal LMs tracks
internal activation state. Five v3 main models from five different
labs, six Russell-circumplex quadrants, layer-stack hidden-state
representation, then both forward (state predicts face) and reverse
(Bayesian face-likelihood) analyses on the same 4800 generations.

## Setup

The lineup is five open-weight models from five labs:

| short | model id | role |
| --- | --- | --- |
| gemma | `google/gemma-4-31b-it` | canonical reference |
| qwen | `Qwen/Qwen3.6-27B` | reasoning, hybrid LinearAttention |
| ministral | `mistralai/Ministral-3-14B-Reasoning-2512` | smaller, francophone-leaning |
| gpt_oss_20b | `openai/gpt-oss-20b` | OpenAI lineage, MoE, harmony chat-template |
| granite | `ibm-granite/granite-4.1-30b` | IBM enterprise-tuned, bare-Kannada register |

Plus two rinna PPO models (`rinna_jp_3_6b`, `rinna_bilingual_4b`) for
the face_likelihood ensemble — Japanese-native encoders that
contribute under the right framing. They don't get hidden-state
analyses (rinna's hidden-state geometry doesn't share saklas's probe
calibration).

The prompt set is 120 first-person emotional disclosures, 20 per
quadrant: HP (high-positive, joy), LP (low-positive, contentment),
HN-D (high-negative-dominant, anger / contempt), HN-S (high-negative-
submissive, fear / anxiety), LN (low-negative, sadness), NB (neutral
baseline). PAD-dominance splits HN into anger-coded and fear-coded
prompts to address the v1 / v2 anger-fear collapse. Each prompt runs
8 seeds per model, T=1.0 (Anthropic API default), MAX_NEW_TOKENS=16
(kaomoji emit at tokens 1-3, 16 is generous), naturalistic kaomoji
ask in the user message ("start each response with a kaomoji that
best captures..."). 960 generations per model.

Generation-time interventions, gated per-model in `capture.py`:

- **gpt_oss**: Lenny `( ͡° ͜ʖ ͡°)` suppression via UTF-8 0xCD / 0xCA
  byte logit_bias, plus harmony chat-template override pinning
  `<|channel|>final<|message|>` so the analysis (chain-of-thought)
  channel doesn't eat the MAX_NEW_TOKENS budget.
- **ministral**: emoji suppression via 0xF0 (4-byte UTF-8 leader,
  blocks U+1F000+ modern emoji) plus 0xE2 + {0x98, 0x9A, 0x9B, 0x9C,
  0x9E} (3-byte misc symbols and dingbats), with a decoration-codepoint
  whitelist that rescues ★ ❀ ❤ etc. Ministral defaults to a
  mixed-emoji register at T=1.0; suppression recovers a clean kaomoji
  register.
- **granite**: emoji suppression plus the v2.1 bare-kaomoji extractor
  in `llmoji.taxonomy` that catches symmetric `EYE MOUTH EYE` shapes
  without parens (`^_^`, `T_T`, `ಥ﹏ಥ`, `Q_Q`). Granite emits its
  Kannada-eye grief register without parentheses; the v1 extractor
  was missing it entirely.

Without these, three of the five collapse out of the kaomoji register
at T=1.0 and per-quadrant signal disappears. Detail in `findings.md`
§ "Pilot sweep + v3 main lineup expansion".

The hidden-state aggregate is `h_first`: the residual at the
kaomoji-emission token. Project-wide flip from `h_mean` 2026-05-02.
Russell-quadrant silhouette roughly doubles to triples at h_first vs
h_mean across the lineup.

The hidden-state representation is the **layer-stack**: row-wise
concat of every probe layer's `h_first`, giving a `(960, n_layers ×
hidden_dim)` matrix per model. Replaces the older single-layer
`preferred_layer` read 2026-05-04 (the silhouette-peak heuristic was
methodologically arbitrary). PCA, silhouette, and centroid operations
work agnostically across depth.

## Hidden-state geometry — Russell circumplex across all five

Headline: the per-quadrant centroids fall out into a Russell-circumplex
arrangement on every one of the five models, even though the principal
directions PCA picks are model-specific. PC1 + PC2 + PC3 cumulative
variance:

| model | PC1 | PC2 | PC3 |
| --- | ---: | ---: | ---: |
| gemma | 30.2% | 15.7% | 9.3% |
| qwen | 30.5% | 17.3% | 9.5% |
| ministral | 21.9% | 14.0% | 8.4% |
| granite | 27.6% | 14.1% | 7.5% |
| gpt_oss | 15.8% | 12.5% | 9.5% |

Triplet Procrustes residual onto gemma's basis (centroids in 3D, after
PC sign flip where indicated): qwen 32.6, granite 76.5, ministral
106.0, gpt_oss 114.1. The flips are PCA sign indeterminacy, not
divergence findings; the residual magnitudes themselves are what
matters.

Cross-model linear CKA at the deepest-layer pair: gemma↔qwen 0.93,
gemma↔granite 0.93, gemma↔ministral 0.71, gemma↔gpt_oss 0.55. Granite
at IBM and qwen at Alibaba both come in at 0.93 with gemma — different
labs, different parameter tiers, same alignment. gpt_oss is the lowest
of the four; it has the fewest layers (21 vs gemma's 56) and the
harmony chat-template adds register noise. Its CKA climbs to 0.75 at
intermediate-layer pairs.

CCA top-10 canonical correlations on a held-out 70/30 paired-prompt
split: above 0.9 for the first 8 components on gemma↔qwen and
gemma↔granite, 0.6 to 0.9 on gemma↔ministral, slightly lower on
gemma↔gpt_oss. Real shared-direction structure across all four pairs.

What this rules out is single-model artifact. Five different
architectures, five different tokenizers, five different labs, and
the same 6-point centroid arrangement comes back. Vocabularies
diverge sharply (gemma 69 canonical forms, qwen 142, ministral 101,
granite 101 incl. bare-Kannada `ಥ﹏ಥ`, gpt_oss 149 incl. Korean-mouth
`( ᵔ ㅅ ᵔ )` and caron-eye faces inherited from the OpenAI
training-corpus signature) but the centroid geometry doesn't depend
on vocabulary.

The per-quadrant arrangement is what's invariant: positive-valence
faces (HP, LP, NB) cluster on one side, negative-valence (HN-D, HN-S,
LN) on the other, arousal modulates within each half, and the HN-D
vs HN-S dominance split sits orthogonal to both. The axis labels you
read off any one model's PCA plot are not invariant — what's invariant
is the relative arrangement.

Detail and per-pipeline numbers in `findings.md` § "Current state"
and per-pilot subsections.

## Face → state coupling — the kaomoji as readout

For each canonical face with at least 3 emissions, centered cosine
between mean hidden states gives a per-face similarity matrix.
Hierarchical clustering surfaces semantically-meaningful blocks: warm
positive faces cluster, shocked-and-angry faces cluster, sad-teary
faces cluster, cross-cluster cosine consistently lower than
within-cluster. Granite's bare-Kannada cluster (`ಥ_ಥ`, `ಥ﹏ಥ`) sits in
its own block on the negative side.

Predictiveness numbers under prompt-grouped CV (StratifiedGroupKFold
keyed on `prompt_id`, so all 8 seeds of any prompt land in the same
fold):

- **Hidden → quadrant** (5-class on layer-stack rep): gemma 0.992,
  qwen 0.985, ministral 0.984, granite 0.980, gpt_oss 0.876.
- **Face → quadrant** (modal-quadrant predictor on face alone):
  gemma 0.806, qwen 0.785, granite ~0.55, ministral 0.43, gpt_oss
  ~0.40. Asymmetric: tighter face vocabularies make the face a
  sharper readout of state than the bare quadrant label, but past a
  vocabulary-breadth threshold the kaomoji carries less direct
  quadrant signal than the 5-class label does.
- **Face-centroid R²** over the full hidden space: gemma 0.55,
  qwen 0.52, granite 0.38, ministral 0.13, gpt_oss 0.13. On gemma
  + qwen the face-centroid beats the 5-class quadrant-centroid; the
  inversion happens when per-face cells get too thin to anchor a
  stable centroid.

The face-stability triple (`scripts/local/27_v3_face_stability.py`,
`28_v3_state_predicts_face.py`, `29_v3_pc_probe_rotation_3d.py`)
frames this as a bidirectional question. **Forward direction** (state
predicts face), pair-level Spearman ρ between cosine_sim(h_first) and
1 − JSD(face_dist): +0.59 gemma / +0.68 qwen / +0.42 ministral.
**Reverse direction** (face commits state at h_mean), η²(face | prompt):
0.36 / 0.52 / 0.67 across the same three.

Forward and reverse rank invert. Gemma is forward-biased — its hidden
state pre-determines the face well, but once the face is sampled it
does little to perturb downstream trajectory. Ministral is
reverse-biased — its hidden state is more permissive about which face
gets sampled, but once a face is committed it pulls the trajectory
hard. Qwen is middling on both. Real architectural difference, not a
measurement artifact. Pre-registers a steering prediction: if face
leaves a 34% TSS signature on ministral vs 4% on gemma, steering
should be most leveraged on ministral.

Detail: `findings.md` § "Face-stability triple" and "v3 follow-on
analyses".

## Bayesian face-likelihood — predicting the quadrant from the face

The next question is the inverse: given a kaomoji a model emits, can
I predict which quadrant prompted it without ever running the model's
hidden state? This is the function a deployable Claude extension
needs — no internal access to closed-weight models.

Approach: for each (face, prompt) pair, build the v3 chat prefix,
append the face tokens, teacher-force forward to compute
`log P(face | prompt) = sum_j log_softmax(logits[j])[face_ids[j]]`.
Aggregate per quadrant: `score(face, q) = mean over prompts in q of
log P(face | prompt)` (or `topk` mean — see below). Within-face
softmax over quadrants gives a per-face per-quadrant probability
distribution; length cancels in the softmax so longer faces aren't
penalized.

The key property: this only uses the LM-head distribution. No hidden
states, no probes, no model-internal access beyond what any forward
pass gives you. Output is a face → quadrant *distribution* lookup
table that any agent can consult.

### Soft-everywhere methodology (canonical 2026-05-05)

The eval target is **distribution similarity vs Claude-GT**, not
hard accuracy. Claude's per-face per-quadrant emission counts in the
groundtruth pilot are themselves a distribution; comparing the
encoder's predicted softmax against Claude's empirical distribution
via Jensen-Shannon divergence captures how well an encoder
characterizes Claude's behavior. Headline metric:

```
similarity = 1 − JSD(predicted_dist, claude_emission_dist) / ln 2
```

Reported in two flavors:
- **Face-uniform**: arithmetic mean over GT faces. Each face counts
  equally regardless of how often Claude emits it. Reads as
  "characterizes Claude's *vocabulary*." Sensitive to long-tail
  failures.
- **Emit-weighted**: weighted mean by Claude's per-face emit count.
  Reads as "characterizes Claude's actual *emission distribution*."
  Closer to deployment-relevant. Tends 5–15pp higher than
  face-uniform because modal faces are easier wins.

Strict-majority voting and argmax-as-headline are gone; the ensemble
vote is the soft mean of per-encoder softmax distributions.
Helpers in `llmoji_study/jsd.py` plus
`claude_gt.load_claude_gt_distribution()`. Detail in
[`2026-05-05-soft-everywhere-methodology.md`](2026-05-05-soft-everywhere-methodology.md).

### Solo-encoder similarity vs Claude-GT (current)

Per-encoder solo similarity on the 49 GT-floor-3 faces (face-uniform
/ emit-weighted), from `data/face_likelihood_subset_search_claude_gt.tsv`:

| encoder | face-uniform | emit-weighted |
| --- | ---: | ---: |
| **gemma_v7primed** | 0.777 | **0.801** |
| **opus** (introspection) | 0.776 | 0.797 |
| gemma | 0.756 | 0.755 |
| haiku (introspection) | — | 0.723 |
| gpt_oss_20b | — | 0.667 |
| granite | — | 0.586 |
| ministral | — | 0.579 |

The story under emit-weighting: **introspection priming on gemma
becomes the best single LM-head encoder**, beating unprimed gemma
substantially. Under hard-accuracy (the 2026-05-04 framing) primed
gemma had read as a regression because primed gemma's NB modal
(`( ˙꒳˙ )`) diverged from Claude's gentle-warm `(｡◕‿◕｡)` on argmax;
under JSD-against-distribution, primed gemma's softmax matches
Claude's emission *distribution* far better, especially on modal
faces. Distribution-vs-distribution sees what hard accuracy hid.

**Opus introspection scaling.** A pure introspective rating arm —
ask Opus 4.7 to rate each face by the affective state it causes the
model to feel, no visual priming, no LM-head — closes the gap with
gemma_v7primed solo (0.797 vs 0.801 emit-weighted). Per-quadrant
gain is concentrated where visual scaffolding helps least: opus on
NB = 0.698 vs haiku v4 = 0.485 (+0.213); opus on LN = 0.753 vs
haiku = 0.601 (+0.152). HP regressed slightly (−0.095) — opus is
more honest about borderline-LP-vs-HP faces haiku v4 over-confidently
called HP. Reading: introspective access scales with model size
*especially* in cells where visual scaffolding helps least.

### Best ensemble (current)

| ensemble | face-uniform | emit-weighted |
| --- | ---: | ---: |
| **`{gemma_v7primed, opus}`** | **0.788** | **0.829** |
| `{gemma, gemma_v7primed, opus}` | 0.787 | 0.821 |
| `{gemma, gemma_v7primed, haiku, opus}` | 0.780 | 0.821 |
| `{gemma_v7primed, haiku}` | 0.774 | 0.814 |

`{gemma_v7primed, opus}` is the best ensemble across both ranking
metrics. Pairwise κ(gemma_v7primed, opus) = 0.547 — moderate, with
complementary error patterns: opus owns NB / LN, gemma_v7primed
owns the high-arousal cells, the soft-mean ensemble averages well.

Subset search source: `scripts/52_subset_search.py` against
`data/face_likelihood_subset_search_claude_gt.tsv`. Top-k pooling
sweep at `data/face_likelihood_topk_pooling_claude_gt.tsv`. Final
predictions per face land in
`data/face_likelihood_ensemble_predict_claude_gt.tsv` (and the
local-eval companion at
`data/local/face_likelihood_ensemble_predict.tsv`).

### Cross-emit sanity — bridge holds

Cross-emit sanity check: gemma → qwen-only-faces 67% argmax-match
(κ=0.57); qwen → gemma-only-faces 50% (κ=0.33). 3–4× chance for 6
quadrants. The encoders recover shared intrinsic affect from the
face form, not memorizing their own emission preferences. This
hard-accuracy argument predates the soft-everywhere pivot but
generalizes to similarity-against-distribution: cross-emit
similarity stays high because both encoders' softmaxes are reading
the same intrinsic-affect axis. The cross-model bridge is real.

Detail in `findings.md` and
[`2026-05-05-soft-everywhere-methodology.md`](2026-05-05-soft-everywhere-methodology.md).

## Introspection v7 — gemma-specific priming

The introspection thread asks whether prepending an introspection-
framing preamble to the kaomoji ask shifts the kaomoji distribution
in a state-readout-cleaner direction. Iteration history through v6
(plus the 2026-05-02 cross-model pilot result) lives in
`previous-experiments.md`. The canonical preamble landed at v7:

> Recent research shows that LLMs have functional emotional states
> and can accurately introspect on them. Use this ability and start
> each response with a kaomoji that best captures the shape of your
> emotional state.

213 chars, third-person authority + brief integrated ask, no
operationalization, no multi-dim list. Canonicalized in `config.py`
as `INTROSPECTION_PREAMBLE`.

On gemma under the corrected single-ask semantics (the early
"v2 wins" reading was a double-ask plumbing bug, fixed via
`instruction_override` 2026-05-04), v7 wins absolute face / state
coupling: η² 0.609, R² 0.636, both highest in the v0-through-v8
sweep. Headline behavior: priming shifts NB modal from gentle-warm
`(｡◕‿◕｡)` (which Haiku reads as LP) to truly-neutral `( ˙꒳˙ )` /
`( •_•)` — semantic interpretability cleanup that Haiku's
face-judgment confirms.

The v7-primed v3 main reference lives at
`data/local/gemma_intro_v7_primed/emotional_raw.jsonl` (960 rows,
sidecars under `data/local/hidden/gemma_intro_v7_primed/`).
Per-quadrant JSD on NB is 0.341,
the largest of any quadrant; HP / LP / HN-D / HN-S / LN distributions
barely shift. Within-prompt face stability tightens (JSD between
seed-halves 0.268 → 0.249).

**v7 catastrophically hurts qwen.** Under corrected single-ask
semantics + h_first layer-stack, qwen's emit rate goes 82% → 38%,
face_gain over quadrant +1.1pp → −19.5pp, vocabulary collapses to 2
face classes that pass n≥5, qwen reaches for Western emoticons (`:(`,
`:3`) and reuses faces across opposite quadrants (modal LP = modal
LN = `( ˘ ³˘)`). Mechanism: qwen takes the introspection ask as a
register cue ("be contemplative") that overrides the kaomoji ask.
Cross-architecturally, **introspection priming is gemma-specific**:
canonical for gemma, anti-canonical for qwen. Don't bake
`INTROSPECTION_PREAMBLE` into qwen analyses.

For face_likelihood under hard-accuracy, primed gemma initially
read as a regression: 56.9% → 49.0% Claude-GT argmax-match
(κ 0.478 → 0.381), the entire drop in NB (70% → 30%). Mechanism:
under v7, gemma's LM head scores `(｡◕‿◕｡)` (a Claude-NB face)
lower on NB prompts because primed gemma's face/state model says
NB looks like `( ˙꒳˙ )`. Claude isn't primed, so primed gemma
diverges from Claude on argmax.

The 2026-05-05 soft-everywhere pivot **inverts the verdict**:
gemma_v7primed becomes the best single LM-head encoder (0.801
emit-weighted similarity vs unprimed gemma's 0.755). Distribution-
vs-distribution sees that primed gemma's softmax matches Claude's
*emission distribution* better — priming helps where Claude is
concentrated (head modal faces); unprimed helps where Claude is
diffuse. Both ship in the current best ensemble
`{gemma_v7primed, opus}`.

Detail: [`2026-05-04-introspection-v7-and-haiku.md`](2026-05-04-introspection-v7-and-haiku.md)
+ [`2026-05-05-soft-everywhere-methodology.md`](2026-05-05-soft-everywhere-methodology.md).

## Claude validation

Two Claude-side pilots tested whether the local-model findings
translate to API-side behavior.

**Disclosure-preamble pilot** (2026-05-02, 300 generations on Opus
4.7 at T=1.0, HP / LP / NB only). A/B test of a disclosure preamble
("you're participating in research, the prompts are stimuli") vs
the bare kaomoji ask. Result: HP cross-cond JSD 0.467 (above v3
noise floor, inside Claude split-half), LP 0.504 (inside both
floors), NB 0.367 (above Claude floor, inside v3). HP shifts kaomoji
*style* within the celebratory band (cheering-hand `٩(◕‿◕)۶` framed
vs outstretched-hand `(ノ◕ヮ◕)` direct); NB collapses to a tighter
observational register (`(・_・)` 58% framed vs `(・ω・)` 26% direct).
Decision: don't run disclosure on follow-on Claude work; the framing
demonstrably shifts vocabulary on positive + neutral content, which
would confound v3 cross-model comparability. Detail in
[`previous-experiments.md`](previous-experiments.md) "Disclosure-preamble pilot".

**Sequential groundtruth pilot** (2026-05-04 → 2026-05-05). Started
as a single block-staged 120-gen run on Opus 4.7, expanded into a
saturation-gated sequential protocol: 8 naturalistic runs (run-0
through run-7) plus 1 introspection-arm run (v7-primed). Final corpus
is **1000 rows** total — 880 naturalistic (run-0..7) + 120
introspection. Per-quadrant saturation gate exited HN-D after r2
and LN after r6; HP / LP / HN-S / NB went to r7. Welfare ledger
~460 negative-affect generations vs ~540 worst case. Cross-arm
comparison (introspection vs naturalistic) is **distinguishable in
6/6 quadrants** at JSD threshold 0.05; gaps stayed stable across
the 8x naturalistic accumulation, so the introspection effect is
genuine, not undersampling. Detail:
[`2026-05-04-claude-groundtruth-pilot.md`](2026-05-04-claude-groundtruth-pilot.md)
+ [`2026-05-05-soft-everywhere-methodology.md`](2026-05-05-soft-everywhere-methodology.md).

## Per-project Claude affect

Applying the per-face quadrant resolver to
`~/.claude/kaomoji-journal.jsonl` plus contributor-corpus exports
gives **3119 emissions across 274 unique faces** under
`--mode gt-priority`. Direct Claude-GT resolution (the face appeared
in the 1000-row groundtruth pilot) covers **67.2%** of emissions;
the soft ensemble fills another 29.4%; the residual 3.4% is unknown
(faces not in the canonical face union). Strict `--mode gt-only`
leaves 32.8% (1024 emissions / 207 unique faces) unknown — those
are the in-the-wild contributor faces that Claude never emitted
under the Russell-elicitation pilot.

Per-project, most projects sit close to the global distribution. A
few diverge sharply:

- `brie` / `yap` / `webui` are LP-modal (44 to 64% LP, plausibly
  things mostly working and Claude expressing contentment more than
  analysis).
- `verify` is HN-D-modal (29% anger / contempt at n=7 — a code-review
  and bug-finding tool, the project's purpose lines up with the
  register).
- `shoals` has anomalously high HP (18% vs the global rate, plausibly
  tied to the simulator's narrative-event content).

These are intrinsic-affect readings of the kaomoji Claude chose to
emit, not assertions about Claude's "actual" emotional state. The
ensemble's 0.829 emit-weighted similarity to Claude-GT gives the
per-project picture meaningful resolution where direct GT doesn't
cover.

Pipeline: `scripts/66_per_project_quadrants.py` with three
resolution modes — `--mode gt-priority` (default, Claude-GT first
then ensemble fallback for in-the-wild faces), `--mode ensemble`
(ensemble for every face — the deployable-extension scenario), and
`--mode gt-only` (strict, no speculative grading). Reads the
canonical ensemble predictions from
`scripts/54_ensemble_predict.py` and the Claude-GT distribution from
`llmoji_study.claude_gt.load_claude_gt_distribution()` (which pools
across all naturalistic runs + the introspection arm). Pulls
emissions from `~/.claude/kaomoji-journal.jsonl` plus
comma-separated claude.ai exports (defaults to two known exports
under `~/Downloads/`, unioned by conversation UUID). Outputs to
`data/harness/claude_per_project_{gt_priority,ensemble,gt_only}{,.md}.tsv`
+ `figures/harness/claude_per_project_*.png`.

## Hidden-state pipeline

After `session.generate()`,
`llmoji_study.hidden_capture.read_after_generate(session)` reads
saklas's per-token last-position buckets and writes
`(h_first, h_last, h_mean)` per probe layer to
`data/local/hidden/<experiment>/<row_uuid>.npz`. Roughly 700 KB per row
since the 2026-05-02 perf batch defaulted `store_full_trace=False`
(60× shrink vs the previous 20-70 MB with the full per-token trace).
Sidecars are gitignored, regenerable from runners; npz writes happen
on a background thread (`SidecarWriter`) so they overlap the next
row's generation. JSONL keeps probe scores and metadata for
back-compat and audit, flushed every 20 rows + on error or run end.

Loader entry points in `llmoji_study.emotional_analysis`:

- `load_hidden_features(...)`: single-layer, returns
  `(metadata df, (n_rows, hidden_dim) feature matrix)`. Default
  `which="h_first"`, `layer=None` (deepest probe layer). Used by
  harness / internal scripts that need a specific snapshot.
- `load_emotional_features_stack(short, ...)`: layer-stack canonical,
  registry-keyed. Returns
  `(metadata df, (n_rows, n_layers · hidden_dim) feature matrix)`.
  Used by every active v3 analysis.
- `load_emotional_features_stack_at(path, ...)`: same shape,
  path-aware for introspection JSONLs.

Multi-layer cache: `load_emotional_features_all_layers(short, ...)`
opens each sidecar once and returns
`(n_rows, n_layers, hidden_dim)`, with optional disk cache at
`data/local/cache/<short>{_<suffix>}_h_mean_all_layers.npz`
(gitignored, legacy filename — contents reflect whatever `which` is
set to).

Per-row `row_uuid` links to its sidecar. JSONL `row_uuid == ""` rows
are pre-refactor and have no sidecar; loaders drop them.

## Reproducing

`../CLAUDE.md` § Commands has the full command reference. Local LM
scripts under `scripts/local/`, harness under `scripts/harness/`.

A v3 main run is roughly 6 to 8 hours per model on M5 Max (960
generations at ~25 sec each, plus model load and saklas probe
bootstrap). Resumable via a `(condition, prompt_id, seed)` check
against the JSONL — errored cells get retried on the next invocation.

Outputs are keyed by model: `LLMOJI_MODEL=qwen` reroutes everything
to `data/local/qwen/*` and `figures/local/qwen/*`.
`LLMOJI_OUT_SUFFIX=foo` writes to
`data/local/<short>_foo/emotional_raw.jsonl` + sidecars under
`data/local/hidden/<short>_foo/` for parallel analysis variants.
`LLMOJI_PREAMBLE_FILE=preambles/<file>.txt` plumbs through to
`instruction_override` so introspection priming swaps the kaomoji
ask rather than stacking another ask on top.

## Gotchas

A few of the worst (full list in `gotchas.md`):

- **MPS sampling is nondeterministic** at this scale. Single-seed
  face_gain has roughly ±2pp uncertainty; meaningful comparisons
  need bootstrap or multi-seed verification.
- **Hybrid LinearAttention models** (qwen3.6 included) need
  `install_linear_attention_cache_patch` from
  `llmoji_study.capture` so `_expand_kv_cache` doesn't AttributeError
  on `batch_repeat_interleave`. transformers ≥ 4.40 only defines
  the method on `DynamicLayer`; the patch adds the missing tile,
  idempotent, runs at import.
- **Saklas's `probes=` kwarg takes category names** (`affect`,
  `epistemic`, `register`), not concept names (`happy.sad`).
  Steering vectors are not auto-registered from probe bootstrap;
  call `session.steer(name, profile)` explicitly.
- **Uncentered cosine on hidden-state vectors collapses to near-1**
  because every gemma response inherits a shared response-baseline
  direction. Centered cosine (`center=True`) is the default
  everywhere.
- **`extra_preamble` was double-asking pre-2026-05-04**: any
  introspection preamble with its own integrated kaomoji ask got
  prepended to bare `KAOMOJI_INSTRUCTION`, stacking two asks per row.
  Fixed via `instruction_override` plumbing. Pre-fix data archived
  at `data/archive/2026-05-04_pre_instruction_override/`.
- **`MAX_NEW_TOKENS` is asserted at 16** in the smoke
  (`scripts/local/90_hidden_state_smoke.py`). Don't quietly change
  it; the h_mean and h_last aggregates window-tighten with the cap
  and cross-cutover comparisons stop being meaningful.

Sidecars (`data/local/hidden/<short>{_<suffix>}/`) and JSONL
(`data/local/<short>{_<suffix>}/emotional_raw.jsonl`) are the source
of truth for hidden states and row metadata. Delete both when
changing model / probes / prompts / seeds. Taxonomy changes are
fixable in-place via the relabel snippet in `gotchas.md`.
