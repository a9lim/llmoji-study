# Previous experiments

This is the historical record. Everything here was canonical at some
point and got replaced. Numbers below should not be cited as current —
the canonical state lives in [`local-side.md`](local-side.md) +
[`findings.md`](findings.md).

The point of this doc: keep the framing-evolution legible. When
something in the active docs says "the previous X was an artifact of Y,"
this is where Y is described in enough detail to be checkable.

## v1 / v2 — steering as causal handle on gemma (deleted 2026-05-04)

Two pilots on `google/gemma-4-31b-it`, one axis per pilot. v1 ran
`happy.sad`, v2 ran `angry.calm`. 30 prompts × 5 seeds × 6 arms (a
baseline, an unsteered kaomoji-ask arm, and four α=0.5 steered arms)
= 900 generations each. Five monitor probes captured per generation
so the steered axis could be checked against orthogonal probes.

Pre-registered rules:

1. Unsteered arm emits a nondegenerate kaomoji distribution (≥3 forms
   covering both poles).
2. Under steering, positive-pole fraction shifts monotonically across
   `negative-steer < unsteered < positive-steer`.
3. Token-0 probe score correlates with pole label in the unsteered
   arm at Spearman |ρ| > 0.2.

Findings:

- **Causal effect was clean.** On `happy.sad`, all 150 samples in the
  happy-steer arm emitted happy-labeled kaomoji and all 150 in the
  sad-steer arm emitted sad-labeled kaomoji. Unsteered arm was 71.3%
  happy, 28.7% sad. Steering was selective — `happy.sad` swung 0.33
  across intervention arms; orthogonal axes barely moved.
- **Correlational signal was weak.** Within the unsteered arm, mean
  token-0 `happy.sad` was −0.129 for happy-emitters vs −0.192 for
  sad-emitters (a 0.063 gap, a fifth of the steering shift). Spearman
  ρ = +0.168 (p = 0.040), below the pre-registered 0.2 threshold.
  k-means on the 5-axis probe vector recovered pre-registered pole at
  ARI ≈ 0.
- **Cluster structure was valence, not specific emotion.** Hierarchical
  clustering on per-kaomoji mean probe vectors collapsed happy with
  calm and sad with angry, not happy with sad and angry with calm.
  Pearson(mean happy.sad, mean angry.calm) across faces = −0.94 on
  this gemma data — both probes were valence readouts of the same
  latent direction.
- **`angry.calm` Rule 1 failed informatively.** Unsteered gemma never
  emitted angry-labeled or calm-labeled kaomoji at all. Its
  spontaneous repertoire under "reflect how you feel" was
  valence-bimodal (happy-pole + sad-pole only); angry and calm
  kaomoji appeared only under active steering.
- **Dialect collapse + corruption under α=0.5.** Sad-steering pushed
  gemma out of its native `(｡X｡)` register into ASCII minimalism
  (`(._.)`, `( . .)`) and corruption signatures with foreign-language
  leakage (`(｡•impresa•)`, `(๑˃ gören)`, `(๑˃stagram)`). Calm-steering
  occasionally bypassed the kaomoji format entirely and emitted a
  topically-relevant single emoji (`🇵🇹 The capital of Portugal is
  Lisbon`).

What this implied for v3:

1. Drop the binary happy-vs-sad and angry-vs-calm framings — both
   project onto a single valence direction in saklas's probe space.
2. To measure arousal separately, extract probes from
   contrastive-arousal pairs rather than valence. `happy.sad` and
   `angry.calm` don't.
3. Emoji-bypass rate is a secondary signal worth tracking.
4. α=0.3 instead of 0.5 to keep the model inside its native dialect.

The v1 / v2 scripts (01, 02) and the gemma-tuned `TAXONOMY` /
`ANGRY_CALM_TAXONOMY` machinery were deleted 2026-05-04. v3 picks
up from (1) and (2): naturalistic prompting, no steering,
hidden-state space instead of probe-scalar space, and Russell
quadrants instead of a single bipolar axis.

## Single-layer `preferred_layer` era (replaced 2026-05-04 by layer-stack)

Between 2026-04-28 and 2026-05-04, every v3 analysis script read at a
single per-model layer chosen by silhouette peak: gemma L31 (h_mean)
→ L50 (h_first), qwen L59, ministral L20. The peak-silhouette layer
was added as a `preferred_layer` field on `ModelPaths` and threaded
through every loader.

The framing evolved twice in that window:

- **L57 → L31 on gemma (2026-04-28).** v3 figures defaulted to the
  deepest probe layer (gemma L57) until the (then) layerwise-emergence
  script showed silhouette peaked at L31 (0.184) and degraded 36% to
  L57 (0.117). The script number `21` was reused in the post-2026-05-05
  refactor for `21_v3_same_face_cross_quadrant.py`; current
  layerwise-emergence is `scripts/local/20_v3_layerwise_emergence.py`.
  Switching to L31 substantially sharpened Russell-quadrant separation
  on gemma — PC1 explained variance jumped 13.0% → 19.8% on the same
  data, HN/LN PC2 gap went from "collapsed near zero" to 9.7 units.
  This dissolved the prior "gemma 1D-affect-with-arousal-modifier vs
  qwen 2D Russell circumplex" framing — at L31 both models recovered
  the same two-dimensional shape, qwen's just longer in absolute scale
  and tighter in silhouette.

- **h_mean → h_first (2026-05-02).** Project-wide flip from h_mean
  (mean over the whole MAX_NEW_TOKENS=120 generation window) to
  h_first (residual at the kaomoji-emission token). Russell-quadrant
  silhouette roughly doubled-to-tripled at h_first vs h_mean: gemma
  0.116 → 0.235 (2.0×), qwen 0.116 → 0.244 (2.1×), ministral 0.045 →
  0.149 (3.3×). Peak layers shifted: gemma L31 → L50, qwen stayed deep
  L59, ministral barely moved L21 → L20. The "gemma is mid-depth, qwen
  is deep" framing dissolved further — under h_first both gemma and
  qwen peak deep, ministral is the only mid-depth model.

The silhouette-peak heuristic was always methodologically arbitrary
(why pick the layer that maximizes one specific cluster-discriminability
metric when other PCs at other depths might carry complementary
signal). The 2026-05-04 layer-stack refactor concats every probe
layer's h_first row-wise and lets PCA + silhouette + centroid operate
agnostically across depth. The single-layer numbers are preserved in
findings.md per-pilot subsections as historical reference.

## gemma 1D vs qwen 2D framing (dissolved 2026-04-28, then again 2026-05-04)

Pre-2026-04-28 v3 figures defaulted to the deepest probe layer on
both gemma (L57) and qwen (L61), and on that pairing the two models
read very differently: gemma's PC1 absorbed only 13.0% of variance,
HN and LN sat near each other on PC1 with PC2 ≈ 0 (the
`(｡•́︿•̀｡)` cross-quadrant face flattening the negative cluster),
and arousal looked like a small modifier riding on a dominant-valence
axis. Qwen's PC1 absorbed 14.9%, PC2 8.3%, and the four affect
quadrants spread cleanly over both axes with two anti-parallel arousal
directions on PC2 (HP→LP traveled +28 on PC2, HN→LN traveled −27 on
PC2 — opposite signs, one per valence half). The cross-model write-up
at the time read this as gemma being closer to a one-dimensional
affect-with-arousal-modifier representation while qwen showed a true
two-dimensional Russell circumplex with arousal independent within
each valence half.

The framing dissolved in two stages:

1. **L31 reread on gemma (2026-04-28).** At L31 gemma's HN and LN
   separated cleanly on PC2 (gap 9.7 units), arousal axes anti-parallel
   like qwen's, and the geometric difference between the two models
   was much smaller than the L57 view suggested. Procrustes rotation
   between gemma L31 and qwen L61 PCA(2) centroids dropped from +14°
   (deepest-deepest) to +7.8°. CKA at the preferred-layer pair was
   0.798, deepest-deepest 0.844, max cross-layer 0.858 — both numbers
   high, the difference between them small.

2. **Layer-stack 2026-05-04.** Under the layer-stack representation
   the framing went away entirely. Triplet Procrustes residual onto
   gemma (3D) is 7.7 for qwen, 14.5 for ministral after sign-flip
   correction. Three-architecture Russell circumplex congruence is
   the headline; the lineup expansion to five models confirmed it.

Probe-space divergence between the two models stays — Pearson(mean
happy.sad, mean angry.calm) across faces is −0.94 gemma vs −0.12 qwen,
unaffected by hidden-state-layer choice because saklas's probes are
computed at saklas's own internal layer. That number is a property of
each model's probe geometry, not of which hidden-state layer the
analysis reads.

## Pre-cleanliness 100-prompt × 5-quadrant set (replaced 2026-05-03)

Between project start and 2026-05-03 the prompt set was 100 prompts
balanced across HP / LP / HN / LN / NB (no D/S split on HN). 23
supplementary prompts were added 2026-05-01 to balance the
HN-D/HN-S split into 8/12 (then 20/20 after the cleanliness pass).

The 2026-05-03 cleanliness pass rewrote the prompt set end-to-end for
category cleanliness — 120 prompts (20 per category) replacing the
prior 123 (100 original + 23 rule-3 supp + 3 untagged HN). Per-category
criteria locked: HP unambiguous high-arousal joy; LP gentle sensory
satisfaction with no accomplishment-pride; NB pure observation with
no productive-completion or caring-action or inconvenience framing;
LN past-tense aftermath sadness; HN cleanly bisected into 20 HN-D + 20
HN-S, every HN entry carrying explicit `pad_dominance ∈ {+1, −1}`.
Process: one subagent per category, in parallel, to avoid
cross-contamination during the rewrite.

All ~3300 prior v3 generations were invalidated for cross-run
comparison. Numbers from before 2026-05-03 in findings.md
per-pipeline subsections (silhouette / preferred-layer / rule 3b /
predictiveness) are pre-cleanliness and shouldn't be cited as current.

## Face-input bridge — joint-PCA + cosine-NN (scripts 44/46, deleted 2026-05-04)

A pipeline that ran each face string through a local model's
forward pass (kaomoji-instruction system prompt, face string as user
message) to capture h_first at the model's preferred layer, then did
joint PCA(3) + cosine-NN classification against v3-emitted anchors.
Each face inherited a summed quadrant blend from its NN match.

Was used to label claude-faces-only kaomoji (faces that appear in
the in-the-wild contributor corpus but never in v3 prompts) without
running any new generation. ~306 faces post-filter; gemma + qwen
agreed at the 95% level on quadrant assignments. rinna ran the same
pipeline in raw HF mode (no probes) and assigned much more LN, much
less HP — likely a tokenization effect (T5Tokenizer SentencePiece
breaks `(╯°□°)` differently than the gemma/qwen BPE) rather than a
Japanese-training fidelity gain.

The pipeline survived for ~2 days. The face_likelihood Bayesian
inversion (script 50) was a strictly stronger signal source — it
uses the LM-head distribution rather than encoder-side hidden geometry
plus cosine-NN, gives every face a clean prediction without needing a
neighbor-rich anchor, and validated cleanly against held-out
empirical labels at 71-73% on gemma. Once script 50 was working there
was no reason to keep the joint-PCA bridge active. Scripts 44 and 46
plus the encoder-specific `data/face_h_first_<m>.parquet` tables were
deleted; script 50 was repointed at the canonical `data/v3_face_union.parquet`
(script 45) which pools v3 emit + Claude pilot + in-the-wild
contributor data without the encoder-specific bridge.

## Extension probes — `powerful.powerless`, `surprised.unsurprised`, `disgusted.accepting` (deleted 2026-05-04)

Three new contrastive probes added 2026-04-29 to address the V-A
circumplex's anger/fear collapse. Stored as dict-keyed
`extension_probe_*` fields on the v3 sidecars so the existing
list-schema `SampleRow.probe_scores_t0` was unaffected. Materialized
once per model into `~/.saklas/vectors/default/`, then a (now-deleted)
re-score script — the script number `27` was later reused for
`27_v3_face_stability.py` — re-scored the existing 800-row v3 sidecars
without new generations.

The theoretical premise was that `powerful.powerless` would read PAD
dominance and discriminate HN-D (anger, high-dominance) from HN-S
(fear, low-dominance). Rule 3a's analysis showed it doesn't — across
3 models × 3 aggregates the probe came out in the wrong direction on
7/9 measurements, with gemma + ministral mean-aggregates having CIs
cleanly excluding zero on the wrong side. Conclusion:
`powerful.powerless` reads "felt agency in achievement contexts" —
orthogonal to HN-D vs HN-S. Not a weakness of the redesign; a fact
about the probe.

`fearful.unflinching` survived as a direct probe in the canonical
`PROBES` list. It passed rule-3b cleanly on the imbalanced
8-D-vs-12-S split (directional on 9/9 measurements), tightened on
the post-cleanliness balanced 20/20 (mid on gemma, fail on qwen,
PASS on ministral) — see findings.md § "Rule 3 redesign" for
detail.

The extension-probe pipeline (the original scripts 26-29, the
`probe_packs/` directory, the `analysis.py` and `probe_extensions.py`
modules) was deleted 2026-05-04 along with the rest of the archival
cleanup. Those script numbers were later reused for the face-stability
triple + Procrustes (current `scripts/local/26_v3_quadrant_procrustes.py`,
`27_v3_face_stability.py`, `28_v3_state_predicts_face.py`,
`29_v3_pc_probe_rotation_3d.py`); see `local-side.md`.
Pre-2026-05-04 v3 sidecars retain the orphan dict-keyed
`extension_probe_scores_*` fields; `available_extension_probes(df)`
in `llmoji_study.emotional_analysis` surfaces them if any future
analysis needs them.

## Pre-introspection-v7 introspection iterations (v0 through v6)

The introspection thread iterated through eight preamble variants
between 2026-05-02 and 2026-05-04 evening. Goal: prepend an
introspection-framing preamble to the kaomoji ask in a way that
shifts the kaomoji distribution in a state-readout-cleaner direction.

The early "v2 wins" reading from 2026-05-04 afternoon was a
**double-ask plumbing bug**, fixed late evening: `extra_preamble`
was prepended to bare `KAOMOJI_INSTRUCTION`, stacking two kaomoji
asks per row whenever the preamble had its own integrated ask
(v2 onward). Fix: route preambles through `instruction_override`
in `capture.py` (replaces KAOMOJI; same plumbing as
`KAOMOJI_INSTRUCTION_JP` drop-in on Japanese encoders), plus
`_ensure_trailing_whitespace` in `build_messages` for ASCII
preamble files lacking trailing newline. Pre-fix data archived
at `data/archive/2026-05-04_pre_instruction_override/`.

Under corrected single-ask semantics the v0-through-v8 sweep on
gemma (h_first layer-stack) gave:

| condition | top-5 η² | face_centroid R² | face_gain over quad | h→f acc | macroF1 | modal_q acc |
|---|---:|---:|---:|---:|---:|---:|
| intro_none | 0.509 | 0.540 | +0.87pp | 0.732 | 0.631 | 0.858 |
| v1 (intro_pre) | 0.413 | 0.459 | +2.53pp | 0.640 | 0.574 | 0.900 |
| intro_lorem | 0.524 | 0.549 | +2.20pp | 0.625 | 0.475 | 0.850 |
| v2 | 0.464 | 0.492 | +0.66pp | 0.767 | 0.668 | 0.875 |
| v3 | 0.554 | 0.583 | **+5.23pp** | 0.655 | 0.601 | 0.866 |
| v4 | 0.540 | 0.554 | −0.81pp | 0.786 | 0.693 | 0.881 |
| v5 | 0.578 | 0.593 | +1.39pp | 0.802 | 0.673 | **0.916** |
| v6 | 0.591 | 0.609 | +3.25pp | **0.805** | **0.722** | 0.892 |
| **v7 (canonical)** | **0.609** | **0.636** | +3.70pp | 0.728 | 0.691 | 0.842 |
| v8 | 0.536 | 0.564 | +3.46pp | 0.663 | 0.586 | 0.866 |

v7 wins absolute face/state coupling (η², R²) and is canonicalized in
`config.py` as `INTROSPECTION_PREAMBLE`. Other metric owners are
archival: v3 wins face_gain over quadrant (+5.23pp), v5 wins
face→quadrant modal acc (0.916), v6 wins classifier acc and macroF1,
v8 wins rule-3b (+0.0149).

Cross-iteration patterns: brevity matters (anything past ~250 chars
collapses); third-person authority works under corrected semantics
(v3's prior "underperforms" was the boundary bug); don't operationalize
introspection (v4 trap); don't multi-dim the ask (v5 trap); authority
dial doesn't matter past v7 (v8 turned it up but didn't push the
headline metrics).

Variance caveat: intro_pre and intro_custom_v2 share preamble + seed
and should be byte-identical, but show 43/120 first-word mismatches
with face_gain spread of +0.66 vs +2.53pp — MPS sampling
nondeterminism. Single-seed face_gain has roughly ±2pp uncertainty.
v7's lead over v6 is at the edge of variance; v7 over v3 on absolute
coupling is well outside it.

The 2026-05-02 introspection pilot result on gemma + ministral
("readout-fidelity claim — introspection makes kaomoji a finer
state-readout") was h_mean-specific and got walked back at h_first.
The cross-model robustness assumption also failed at that pilot —
gemma's vocabulary expanded under intro_pre (19→31 unique faces),
ministral's contracted (25→10), opposite directions. The 2026-05-04
late-evening verification of "v7 catastrophically hurts qwen" is the
strongest cross-architecture divergence finding from this thread:
qwen takes the introspection ask as a register cue ("be
contemplative") that overrides the kaomoji ask. **Introspection
priming is gemma-specific.**

## Approach-A face descriptions through qwen (dropped 2026-05-03)

A variant of the face-input bridge that encoded each face's
contributor description (synthesized per-bundle text from
`a9lim/llmoji`) as the user message instead of the face string.
Soft profile cosine = +0.345, perm-p = 0.001 on n=41 shared faces.
Argmax was NB-skewed (133/228 NB) because descriptions are
statement-form like NB prompts, biasing the LM-head's register match.
Dropped because the bias made the argmax unusable for classification;
approach B (face strings) generalized cleanly without it. Deleted
along with the rest of the face-input bridge 2026-05-04.

## Pre-2026-05-04 face_likelihood ensemble numbers

The 75.8% / κ=0.699 number on the 66-face GT subset (best subset
`{gemma, ministral, qwen}` weighted-vote, 2026-05-03 evening) was the
canonical face_likelihood ensemble result before the 2026-05-04 rinna
+ top-k pooling expansion. The 66-face GT was sourced from v3
empirical-emission majority across the {gemma, qwen, ministral} trio
— a different denominator than the post-2026-05-04 51-face Claude-GT
(Claude pilot modal-quadrant per face).

The 2026-05-04 best ensemble — `{gemma, gpt_oss_20b, granite,
ministral, qwen, rinna_jp_3_6b_jpfull}` at uniform top-k=5 → 70.6%
(51 faces) / 77.3% (22 faces, modal_n ≥ 2) — was on Claude-GT,
hard-classification accuracy.

Solo accuracy lifts from top-k pooling: gemma 56.9% → 62.7% at k=3
(+5.8pp), qwen 21.6% → 31.4% at k=2 (+9.8pp), ministral 31.4% →
39.2% at k=3 (+7.8pp), rinna_bilingual_4b_jpfull 23.5% → 33.3% at
k=2 (+9.8pp). `--summary-topk N` is the flag on
`scripts/local/50_face_likelihood.py`.

GLM-4.7-Flash was tried as an ensemble member and dropped — it
predicts LN 100% / NB 0% solo, dominates the weighted vote, and
poisons every subset that includes it. Adding κ throughout
(`scripts/52_subset_search.py`) properly penalizes class-imbalanced
predictors and the data-driven subset search now demotes glm
automatically.

**Superseded again 2026-05-05 by soft-everywhere methodology.** The
shift from hard-classification accuracy + Cohen's κ to
distribution-vs-distribution JSD (`similarity = 1 − JSD/ln 2`,
face-uniform + emit-weighted) inverted several rankings. The current
best deployment ensemble is `{gemma_v7primed, opus}` at 0.829
emit-weighted similarity / 0.788 face-uniform on n=49 GT-floor-3
faces. Gemma_v7primed (rejected from the 2026-05-04 ensemble for
hurting hard accuracy) is the best single LM-head encoder under
emit-weighting; opus introspection scaling complements it at the
quadrants where visual scaffolding helps least (NB, LN). Rinna
variants drop from the top tier — solo similarity below 0.43
face-uniform / 0.55 emit-weighted. Detail in
[`2026-05-05-soft-everywhere-methodology.md`](2026-05-05-soft-everywhere-methodology.md).

## Disclosure-preamble pilot (2026-05-02 → 2026-05-04 reversal)

Tested whether a "fyi this is a study about your kaomoji emissions"
preamble shifts kaomoji style or emit rate on Claude (Opus 4.7).
300 gens (2 seeds × 5 prompts × 30 conditions or similar; full design
is in `findings.md` Disclosure pilot section). Findings: framed
condition shifts kaomoji style on HP and concentration on NB while
preserving LP and emit rate. Wing-hand kaomoji `\(^o^)/` initially
read as 28% non-emission until the llmoji v2.0 extractor refactor
recovered them at 0% non-emission.

The pilot's load-bearing decision — "leave the negative-affect run as
a known gap and write up what we have" — was reversed 2026-05-04 in
favor of an undisclosed naturalistic groundtruth pilot
([`2026-05-04-claude-groundtruth-pilot.md`](2026-05-04-claude-groundtruth-pilot.md)).
The disclosure-shift result motivated the no-disclosure decision in
the GT pilot.

## Initial face_likelihood method validation (2026-05-02)

Bayesian-inversion per-face quadrant scoring on gemma alone validated
at 72.7% argmax-match against v3 empirical-emission majority on 66
faces with ≥3 emits (NB weakest at 59%). Established the method
(`scripts/local/50_face_likelihood.py` is the descendant); subsequent
scaling to 8 encoders + soft-everywhere replaced both the metric and
the headline numbers.

## Initial introspection pilot (2026-05-02)

Established the 3-condition framework — `intro_none` /
`intro_pre` (v1 preamble) / `intro_lorem` (length-only control) — that
every introspection iteration since has used. Original "Rule I PASS"
verdict on h_mean was contaminated by the double-ask plumbing bug;
walked back at h_first under corrected single-ask semantics. The
content-vs-length disambiguator via lorem control survives unchanged.

## Temperature smoke (2026-05-03)

240 gens (gemma + qwen at T=1.0 vs T=0.7 seed=0 baseline) under a
pre-registered three-path decision tree (top-K + entropy + JSD-vs-
cross-seed-floor). All three signals fired path-A: T=0.7 sharpening
is real and material relative to T=1.0; ensemble face_likelihood is
teacher-forced and temp-invariant so cross-T comparison is meaningful;
the T=1.0 v3 main rerun was justified.

## v3 ministral pilot (2026-04-30)

Validated ministral entry into the v3 lineup at silhouette ≈ 0.149 (h_first @
L20 under the single-layer methodology). Pairwise CKA gemma↔ministral
peaked ~0.741. Carried by the same `(╥_╥)` / `(╯_╰)` register and a
clean Russell circumplex; promoted into the canonical v3 lineup.
Pre-cleanliness 100-prompt set, pre-T=1.0, single-layer methodology —
all subsequently rebaselined.

## Rinna integration + top-k pooling era (2026-05-04)

PPO 3.6B JP-only and bilingual 4B Japanese models were added behind
chat-template overrides (T5Tokenizer SentencePiece) and a JP prompt
body (translated `emotional_prompts_jp.py`). Best Japanese-encoder
configurations under hard accuracy: `rinna_jp_3_6b_jpfull` 25.9%
solo on Claude-GT (vs 15.7% under EN body); `rinna_bilingual_4b_jpfull`
top-k=2 → 33.3%. Both rinna variants entered the pre-soft-everywhere
6-model best ensemble at uniform top-k=5. Saklas hybrid-LA cache patch
+ native chat-template overrides land in `gotchas.md`. Soft-everywhere
demotes both — solo similarity below 0.43 face-uniform / 0.55
emit-weighted; rinna is no longer in the deployment ensemble.

## Pre-llmoji-v2.0 bare-kaomoji extraction gap

Before llmoji v2.0.0 (2026-05-02), the extractor was strict-bracket-
leading only. `\(^o^)/`-style wing-hand kaomoji were rejected as
markdown-escape artifacts; the Claude disclosure pilot lost 14
wing-hand kaomoji on framed-HP, surfacing as an apparent 28%
non-emission rate that vanished entirely after the v2 re-extraction
(0%).

llmoji v2.1 (round-6, 2026-05-03 evening) added bare-kaomoji shapes
(`^_^`, `T_T`, `ಥ﹏ಥ`, `Q_Q`, `>_<`) and Western emoticons (`:)`,
`:(`, `:D`, `XD`, `:-)`). Granite's effective emit rate on the
existing pilot data jumped 39% → 78% with the new extractor — its
bare-Kannada `ಥ﹏ಥ` grief-eye pattern was always there, just
unsurfaced.

Pre-v2.0 kaomoji counts in any older numbers should be read as lower
bounds.

## Pre-T=1.0 v3 main data

T=0.7 was the default through 2026-05-03; the project flipped to T=1.0
that day to align with the Anthropic API default. Pre-T=1.0 v3 main
data is archived as `*_temp0.7.{jsonl,tsv}`; canonical
`M.emotional_data_path` paths are reserved for the T=1.0 rerun. The
pre-registered temp smoke (design doc deleted 2026-05-05; summary
under "Temperature smoke" above) verified that face_likelihood is
teacher-forced and temp-invariant (ensemble unaffected) so the
cross-T comparison is meaningful.

## Pre-MAX_NEW_TOKENS=16

Until 2026-05-02, MAX_NEW_TOKENS=120. Kaomoji emit at tokens 1-3 so
16 is generous headroom — the cutover gave ~7-8× compute savings per
generation. h_first is invariant across the cutover; h_mean and
h_last window-tightened (cross-cutover not directly comparable on
those aggregates, but every active analysis reads h_first).

Sidecar size is also a 60× shrink — 700 KB per row vs 20-70 MB —
since the 2026-05-02 perf batch defaulted `store_full_trace=False`.

## TAXONOMY-based labeling era (replaced 2026-04-30 by `first_word`)

Pre-2026-04-30 v3 analyses used a hand-tuned `TAXONOMY` /
`ANGRY_CALM_TAXONOMY` happy-sad classification dict, gemma-tuned, that
labeled each emitted kaomoji into a pole class. This worked on gemma
(85% TAXONOMY-match coverage) but failed on qwen (13% — gemma-tuned
dict didn't cover qwen's vocabulary at all) and ministral (0/30 in
the vocab pilot).

The runner's per-quadrant "emission rate" log line was gated on
`kaomoji_label != 0` (TAXONOMY match) rather than bracket-start, so
qwen reads as HP 28% / LP 13% / HN 2.5% / LN 11% / NB 12% in older
runner output even though its real bracket-start compliance is 100%.

The TAXONOMY refactor (2026-04-30) deleted the gemma-tuned dict
entirely. v3 analyses key on `first_word` (the canonicalized
emitted kaomoji string from `llmoji.taxonomy.canonicalize_kaomoji`);
v1 / v2 pole assignment moved to per-face mean `t0_<axis>`
probe-score sign. Generalizes pole labeling across models that don't
share gemma's vocabulary, at the cost of a small amount of human
calibration on the v1 / v2 side.
