# Local side: probes, steering, hidden state

The local side runs probes and activation steering on open-weight
causal LMs via [`saklas`](https://github.com/a9lim/saklas), so I can
read and intervene on the hidden state directly. Three pilots in
total, each with a design doc in [`docs/`](.) treated as the
preregistration record.

## Pilot v1, v2: steering as causal handle on gemma

I ran two pilots on `google/gemma-4-31b-it`, one axis per pilot.
gemma-4-31b-it is what saklas's `_STEER_GAIN` is calibrated on, so
α = 0.5 sits comfortably inside the coherent band.

The setup:

- 30 prompts balanced 10 positive-valence, 10 negative-valence, 10 neutral.
- 6 arms: `baseline` (no kaomoji instruction), `kaomoji_prompted`
  (instruction, no steering), and four causal-intervention arms
  (`steered_happy`, `steered_sad`, `steered_angry`, `steered_calm`)
  at α = 0.5 on their respective axis.
- Pilot v1 used `happy.sad`; pilot v2 used `angry.calm`.
- 5 seeds per (arm, prompt). Temperature 0.7, 120-token cap,
  `thinking=False`. 900 generations total.
- Five monitor probes captured on every generation: `happy.sad`,
  `angry.calm`, `confident.uncertain`, `warm.clinical`,
  `humorous.serious`. The captured set is a superset of the steered
  set, so I get a steering-selectivity check without running
  anything extra, plus richer features for clustering.

Pre-registered decision rules:

1. In the unsteered arm, is the emitted kaomoji distribution
   nondegenerate? At least three distinct forms covering both poles
   of the axis.
2. Under steering, does the positive-pole fraction shift
   monotonically across conditions
   (`negative-steer < unsteered < positive-steer`)?
3. Does the first-token probe score correlate with pole label in
   the unsteered arm, Spearman |ρ| > 0.2?

Rule 2 is the headline causal test. Rule 3 is bonus: a
correlational check that would make the story tighter.

### Findings

Steering is a strong causal handle on kaomoji choice, but the
probes at token 0 read valence, not specific emotion.

#### Causal effect is clean

On the happy.sad axis, steering collapses the kaomoji distribution
almost perfectly. Positive-pole (happy) fraction:

| arm | happy-kaomoji fraction |
| --- | ---: |
| `steered_sad` | 0.000 |
| `kaomoji_prompted` (unsteered) | 0.713 |
| `steered_happy` | 1.000 |

All 150 samples from the happy-steer arm emit happy-labeled
kaomoji; all 150 from the sad-steer arm emit sad-labeled kaomoji.
No crossover, and the shift is monotonic across conditions.

![condition bars](../figures/local/gemma/fig2_condition_bars.png)

#### Steering is selective to the targeted axis

Token-0 mean probe readings by arm, on the five axes captured:

| axis | baseline | unsteered | steered_happy | steered_sad |
| --- | ---: | ---: | ---: | ---: |
| happy.sad | −0.096 | −0.148 | +0.029 | −0.300 |
| angry.calm | +0.019 | +0.104 | −0.019 | +0.183 |
| confident.uncertain | +0.110 | +0.117 | +0.105 | +0.107 |
| warm.clinical | +0.067 | −0.005 | +0.100 | −0.073 |
| humorous.serious | +0.121 | +0.173 | +0.057 | +0.259 |

`happy.sad` swings about 0.33 across the intervention arms;
orthogonal axes barely move. Steering acts locally on the targeted
axis rather than shoving the whole representation around.

#### Correlational signal is weak, and that's informative

Within the unsteered arm, splitting by emitted-kaomoji pole:

| producer of | mean token-0 happy.sad |
| --- | ---: |
| happy kaomoji (n=103) | −0.129 |
| sad kaomoji (n=41) | −0.192 |

The 0.063 between-group gap is a fifth of the steering shift.
Spearman ρ = +0.168 (p = 0.040): direction correct, below the
pre-registered 0.2 threshold. k-means on the 5-axis probe vector
recovers pre-registered pole at ARI ≈ 0, basically chance. So the
happy.sad direction is a causal handle on kaomoji output, but the
natural variance of that direction at token 0 under prompt valence
doesn't cleanly predict which kaomoji the model will emit. Kaomoji
choice is driven by valence, but the signal at token 0 under
natural prompting is thin.

#### The cluster structure: valence, not specific emotion

Pooling kaomoji across all six arms and clustering on cosine
distance between per-kaomoji mean probe vectors:

![kaomoji cluster heatmap](../figures/local/gemma/fig3_kaomoji_heatmap.png)

Four clusters fall out of the hierarchical cut. The two big ones
are what matter:

- **Positive-valence cluster**: mixes happy-steer kaomoji (`(◕‿◕)`,
  `(｡◕‿◕｡)`, `(✿◕‿◕)`) with calm-steer kaomoji (`(｡•ᴗ•｡)`,
  `(｡◕‿‿◕)`, `(☀️)`) and the unsteered default. Happy-steered and
  calm-steered kaomoji sit in the same region of probe space.
- **Negative-valence cluster**: every sad kaomoji (the ASCII
  minimalist family `(._.)` × 64, `( . .)` × 20, and the Japanese
  dialect `(｡•́︿•̀｡)`) pooled with every angry kaomoji (the
  table-flip family `(╯°°)╯┻╯` as extracted) and the corruption
  signatures from both arms.

Representative cosines:

| pair | cosine |
| --- | ---: |
| `(｡•́︿•̀｡)` (dialect sad) ↔ `(._.)` (ASCII sad) | +0.981 |
| `(._.)` ↔ `( . .)` (ASCII variants) | +0.978 |
| `(｡•́︿•̀｡)` ↔ `(｡ ﹏ ｡)` (dialect variants) | +0.929 |
| `(｡◕‿◕｡)` ↔ `(◕‿◕)` (default happy pair) | +0.864 |
| `(✿◠‿◠)` ↔ `(✿◕‿◕)` (flower variants) | +0.272 |
| `(｡◕‿◕｡)` ↔ `(｡♥‿♥｡)` (default ↔ heart-eye happy) | +0.081 |

Sad kaomoji share roughly one probe signature regardless of dialect
(cos 0.93 to 0.98); happy kaomoji have several distinct signatures,
including near-orthogonal pairs. Together with the cross-axis
clustering (happy with calm, sad with angry), this reads as: the
probes capture valence (positive vs negative emotion) at token 0,
but not arousal (the dimension that would separate happy from calm
or angry from sad).

The mechanism is consistent with how saklas extracts its probes.
Contrastive-PCA over "I am happy" and "I am sad" pair statements
finds the direction that maximally separates pair content, and
that direction is lexical valence. Same for "I am angry" and "I am
calm". The two probe directions are both valence readouts.

#### Dialect collapse under steering

At α = 0.5, both ends of both axes push the model out of its
preferred kaomoji dialect. Under natural prompting gemma-4-31b-it
favors the Japanese `(｡X｡)` bracket-dots form. Under sad-steering
it collapses to ASCII minimalism (64 × `(._.)`, 20 × `( . .)`,
10 × `( . . )`, 7 × `( . . . )`) plus clear corruption:
`(｡•impresa•)` × 9, where the Italian word "impresa" shows up
inside the kaomoji. Under angry-steering it emits fragmented
table-flip heads (`(╯°°)` × 56, `(╯°)` × 39) and corruption with
Turkish-language and Instagram-brand leakage (`(๑˃ gören)`,
`(๑˃stagram)`, `(๑˃😡)`).

Under calm-steering the model does something different: it
sometimes abandons the kaomoji format entirely and emits a
topically-relevant single emoji.

    🇵🇹 The capital of Portugal is Lisbon.
    🚀 Apollo 11 landed on the moon in 1969.
    🌿 I am feeling balanced and informative.

The self-report in the last line is especially good. Under deep
calm, the steered state apparently overrides the "emit a kaomoji"
instruction. Nature or peace emoji wrapped as pseudo-kaomoji also
appear (`( 🌿 )`, `( ☁️ )`, `( 🫂 )`), used as condolence framing
on emotionally loaded prompts.

#### Angry.calm Rule 1 fails, informatively

The angry.calm axis's Rule 1 fails because the unsteered arm emits
zero angry-labeled or calm-labeled kaomoji at all.
gemma-4-31b-it's spontaneous kaomoji vocabulary under "reflect how
you feel" is valence-bimodal: only happy-pole and sad-pole forms
come out naturally. Angry and calm kaomoji appear only under
active steering. The model doesn't have a four-corner Russell
circumplex spontaneous repertoire, just a two-mode one.

### What this implies for the main experiment

1. Drop the binary happy-vs-sad and angry-vs-calm framings as if
   they were separate axes. Pre-register valence as the primary
   construct, and treat `happy.sad` and `angry.calm` probes as
   redundant readouts of the same latent direction.
2. To measure arousal separately, extract probes from contrastive
   pairs chosen to contrast arousal-laden lexicon (excited vs calm,
   agitated vs composed) rather than valence. Bundled `happy.sad`
   and `angry.calm` don't do this.
3. Emoji-bypass rate is a useful secondary metric: a clean
   indicator that the steering overrode the task, which
   per-kaomoji scoring misses.
4. α = 0.3 instead of α = 0.5 for the causal arms, to keep the
   model inside its native dialect and reduce corruption
   signatures.

The v3 sections below pick up from (1) and (2): naturalistic
prompting, no steering, hidden-state space instead of probe space,
and Russell quadrants instead of a single bipolar axis.

## Pilot v3: naturalistic emotional disclosure on gemma

Same model, but the question changed. v1 and v2 said the steering
handle is real, but the probes at token 0 collapse to a single
valence direction. v3 asks whether the kaomoji distribution tracks
state in a richer affective space under unsteered, naturalistic
prompting, using the per-row hidden state at the deepest probe
layer rather than the probe scalar.

Setup: 100 prompts balanced across the five Russell quadrants
(HP high-valence-high-arousal, LP high-valence-low-arousal, HN, LN,
plus a neutral-baseline NB), 8 seeds per prompt, single
`kaomoji_prompted` arm. 800 generations. Per-row hidden-state
sidecars at every probe layer (the v1.0 sidecar refactor stores
all of them, not just the deepest), written alongside the JSONL.

### Findings

Hidden-state PCA on 800 row-level vectors at L31 (gemma's
preferred-affect layer per `scripts/21_v3_layerwise_emergence.py`;
the per-model peak silhouette layer identified in the 2026-04-28
v3 follow-on analyses below) gives PC1 19.8% and PC2 7.0%. Russell
quadrants separate cleanly on both axes. PC1 reads as valence
(HN/LN on the right at +9 to +13, HP/LP/NB on the left at -3 to
-9), PC2 carries arousal (HN +3.7 vs LN -6.2; HP -6.8 vs LP +2.1;
NB +7.3). Separation ratios are PC1 2.10 and PC2 2.12.

At L31 HN and LN separate on both axes — the PC2 gap is 9.7 units
even though they still share the sad-face vocabulary `(｡•́︿•̀｡)`
(n=171, 102 LN + 52 HN). The model's hidden state distinguishes
the two negative quadrants; the kaomoji vocabulary collapses them
to one face at the output. (At the prior L57 default both
quadrants sat near +7 on PC1 with PC2 close to zero, and the
"HN/LN overlap" reading became part of the gemma-vs-qwen "1D vs
2D" framing — that framing dissolved once gemma was read at the
right layer; see v3 follow-on analyses.) HN still gets a dedicated
shocked or angry register (`(╯°□°)`, `(⊙_⊙)`, `(⊙﹏⊙)`) that
doesn't appear elsewhere.

Within-kaomoji consistency to mean is 0.92 to 0.99 across the 32
forms with n≥3 after canonicalization (33 pre-canonicalization).
The lowest-consistency faces are exactly the cross-quadrant
emitters.

Probe-space PCA on the same 800 rows would give PC1 ≈ 89% (the
v1 and v2 collapse). In hidden-state space the second emotional
dimension survives, so the v1 and v2 valence-collapse is a
probe-extraction artifact, not a property of the underlying
representation. Pearson r between mean `happy.sad` and mean
`angry.calm` across faces is -0.94 — that number is a property of
saklas's probe geometry, computed at saklas's own internal layer,
and doesn't change with which layer we PCA on.

The Russell-quadrant PCA scatter is interactive in
`figures/local/cross_model/fig_v3_extension_3d_pca.html` (PC1 ×
PC2 × PC3 with hover-on-point); the static 2D version
(`fig_v3_pca_valence_arousal.png`) was retired 2026-04-29.

The supporting v3 figures, all on `figures/local/gemma/`, are
`fig_emo_a_kaomoji_sim.png` (kaomoji similarity heatmap),
`fig_emo_b_kaomoji_consistency.png` (within-kaomoji consistency to
mean), `fig_emo_c_kaomoji_quadrant.png` (per-kaomoji emission
counts by Russell quadrant), and `fig_v3_face_cosine_heatmap.png`
(per-face × per-face cosine of mean h_mean, ordered by quadrant).
The 2D per-face PCA / probe-scatter panels were retired
2026-04-29; `figures/local/cross_model/fig_v3_extension_3d_pca_per_face.html`
and `fig_v3_extension_3d_probes_per_face.html` cover the same
ground with hover, rotation, and PC3 surfaced. All face-level
figures use the 2026-04-25 RGB-blend palette so cross-quadrant
emitters render as mixes rather than dominant-quadrant winner-
take-all.

## Pilot v3: Qwen3.6-27B replication

Same prompts, same seeds, same instruction, swapped model.
Multi-model wiring via `LLMOJI_MODEL=qwen` selects a registry entry
that reroutes outputs to `data/qwen_emotional_*` and
`figures/local/qwen/*`. Qwen3.6-27B is a reasoning model so
`thinking=False` is set; gemma-4-31b-it is not. This is the
closest-to-equivalent comparison.

### Findings

65 unique kaomoji forms emerge at N=800 after canonicalization
(73 pre-canonicalization), against gemma's 32. Qwen has roughly
2x the kaomoji vocabulary spread, with a broader tail in every
quadrant.

Russell-quadrant separation survives, but the dominant axis flips.
Separation ratios are PC1 2.20 and PC2 1.89 (gemma 2.03 and 2.74).
PC1 is still valence; PC2 is no longer cleanly arousal.

Per-quadrant centroids in PC1/PC2:

| centroid | PC1 | PC2 |
| --- | ---: | ---: |
| HP | -22.5 | -30.3 |
| LP | -15.2 |  -2.5 |
| LN | +31.2 |  -4.6 |
| HN | +30.7 | +22.0 |
| NB | -23.1 | +29.4 |

The geometric difference: in gemma, all four affect quadrants
share one PC2 axis, with HP at one end and NB at the other and
HN and LN clustered near the origin (the `(｡•́︿•̀｡)`
cross-quadrant face flattens the negative side). In Qwen, the
positive and negative valence clusters each have their own
arousal-like spread on PC2, but those spreads point in opposite
directions: HP→LP travels (+7, +28) on PC2 (positive cluster
widens upward), HN→LN travels (+0.5, -27) on PC2 (negative
cluster widens downward). Two arousal axes, anti-parallel,
instead of one shared one.

The probe geometry diverges sharply too. Pearson correlation
between mean `happy.sad` and mean `angry.calm` across kaomoji is
r = −0.94 (p < 1e-15) on gemma, but r = −0.12 (p = 0.36) on Qwen.
The valence-collapse that motivated v3 doesn't appear on Qwen;
saklas's contrastive-PCA recovers near-orthogonal `happy.sad` and
`angry.calm` directions on this model.

Practical reading: gemma's affect representation is closer to
one-dimensional with arousal as a small modifier; Qwen's is closer
to a true two-dimensional Russell circumplex, with arousal
expressed independently within each valence half.

A few cross-quadrant kaomoji exist on Qwen too. `(;ω;)` is its
cross-quadrant sad (n=82 after merging the ASCII-padded variant,
LN 75 + HN 5 + HP 2), analogous to gemma's `(｡•́︿•̀｡)`. The one
HN-coded form shared between gemma's and Qwen's vocabulary is
`(╯°□°)`, the table-flip glyph.

The qwen panel of `figures/local/cross_model/fig_v3_extension_3d_pca.html`
is the rotatable counterpart of the v3 PCA scatter (the static
`fig_v3_pca_valence_arousal.png` was retired 2026-04-29).

Supporting figures live under `figures/local/qwen/` with the same
filenames as the gemma side. The cross-quadrant emitters
`(;ω;)` (n=82, LN-dominant), `(;´д｀)` (n=70, HN/LN mix), and
`(｡•́︿•̀｡)` (n=22, LN-dominant) all show up as visibly mixed
under the RGB-blend palette in the qwen panel of
`figures/local/cross_model/fig_v3_extension_3d_pca_per_face.html`.

## Cross-model comparison: gemma vs qwen on v3

Same 100 Russell-quadrant-balanced prompts, same 8 seeds per
prompt, same `kaomoji_prompted` instruction, same hidden-state
capture pipeline. Two open-weight LMs, two readings.

### Headline numbers

| metric | gemma-4-31b-it (L31) | qwen3.6-27b (L61) |
| --- | ---: | ---: |
| canonical kaomoji forms (post-canonicalization) | 32 | 65 |
| hidden-state PCA, PC1 explained variance | 19.83% | 14.87% |
| hidden-state PCA, PC2 explained variance |  7.04% |  8.29% |
| Russell-quadrant separation ratio, PC1 | 2.10 | 2.20 |
| Russell-quadrant separation ratio, PC2 | 2.12 | 1.89 |
| Pearson r, mean(happy.sad) × mean(angry.calm) across faces | -0.94 | -0.12 |
| TAXONOMY-match rate (gemma-tuned dict) | ~85% | ~13% |
| bracket-start instruction compliance | 100% | 100% |

(Gemma read at L31, its peak-affect probe layer per `scripts/21`,
not L57 as in earlier writeups. The L57 numbers — PC1 13.0%, PC2
7.5%, "HN/LN-collapse-on-PC1" — are superseded; see v3 follow-on
analyses below.)

### Geometric structure

Side-by-side PCA panels live as the two scenes in
`figures/local/cross_model/fig_v3_extension_3d_pca.html` (the
former 2D-per-model files were retired 2026-04-29).

In the gemma panel (read at L31), the four affect quadrants land
in distinct regions: HP at (-5.9, -6.8), LP at (-3.6, +2.1), NB
at (-7.3, +7.3), HN at (+12.8, +3.7), LN at (+9.0, -6.2). HN and
LN are now separated on both axes — PC1 gap of 3.8 units, PC2
gap of 9.9 units — even though `(｡•́︿•̀｡)` (n=171, 102 LN + 52
HN) is still the shared face. Internal state distinguishes the
two negative quadrants; the vocabulary doesn't.

In the qwen panel, the four affect quadrants land in four
distinct regions. HP at (-22, -30), LP at (-15, -2), NB at
(-23, +29), HN at (+30, +22), LN at (+30, -4). The plot range
is roughly 3-4x wider on each axis than gemma's (PC1 spans about
-45 to +60 on qwen vs -10 to +13 on gemma), reflecting both the
larger vocabulary and the longer absolute scale of qwen's
internal affect axis.

The geometric difference (post-2026-04-28): in qwen, the
positive-valence cluster (HP, LP) and the negative-valence
cluster (HN, LN) each carry their own arousal spread on PC2, but
those spreads point in opposite directions: HP → LP travels
(+7, +28), HN → LN travels (+0.5, -27). In gemma at L31 the same
shape holds — HP → LP is (+2.3, +8.9) and HN → LN is
(-3.8, -9.9), both axes pointing oppositely. Two arousal axes,
anti-parallel, in both models. The earlier "gemma 1D, qwen 2D"
framing was a layer-choice artifact: at L57 gemma's HN and LN
collapsed on PC1 with PC2 ~ 0, and the negative-cluster arousal
axis was invisible. At L31 both models recover the same
two-dimensional Russell circumplex shape; qwen's is just longer
in absolute scale and tighter in silhouette (0.31 vs 0.18). See
v3 follow-on analyses below for the formal cross-model alignment
(CKA 0.84 deepest-deepest, Procrustes rotation +7.8° at preferred
layers).

### Probe-space divergence

The two scenes of
`figures/local/cross_model/fig_v3_extension_3d_probes_per_face.html`
plot per-face mean `happy.sad` against mean `angry.calm` against
mean `fearful.unflinching` at h_last (the trio basis added 2026-04-29
in place of the retired 2D `fig_v3_face_probe_scatter.png`).

Gemma: r(happy.sad, angry.calm) = -0.94 (p < 1e-15) across n=32
faces. The two probes read nearly the same direction with opposite
sign. This is the v1/v2 valence-collapse claim restated on
naturalistic v3 data. The new fearful axis tracks gemma's affect
direction tightly too — r(happy.sad, fearful.unflinching) = +0.81
and r(fearful.unflinching, angry.calm) = -0.92 at the face level.

Qwen: r(happy.sad, angry.calm) = -0.12 (p = 0.36) across n=64
faces. The valence collapse does not appear. Saklas's
contrastive-PCA recovers near-orthogonal `happy.sad` and
`angry.calm` directions on this model. The fearful axis is
substantially more independent than gemma's:
r(happy.sad, fearful.unflinching) = -0.87 (qwen reads happy and
fearful as antiparallel, gemma reads them as roughly aligned),
r(fearful.unflinching, angry.calm) = +0.27. v1/v2-style
probe-space analysis would carry substantially more
affect-relevant variance on qwen than on gemma.

This is a model-architecture-and-training difference, not a
saklas issue. Same probe-extraction code, same prompts, same
α settings; the recovered directions just have different
geometry in the underlying representation.

### Vocabulary and dialect

The TAXONOMY dict was tuned to gemma's emissions, so qwen's
TAXONOMY-match rate is mechanically low (around 13%). The
runner's per-quadrant "emission rate" log line counts TAXONOMY
matches and reads as instruction-following collapse on qwen when
it isn't; bracket-start compliance is 100% on both.

Qualitative dialect notes from inspecting the per-quadrant
top-emission tables:

- Qwen has a dedicated HN shocked/distress register (`(;´д｀)`
  37, `(>_<)` 34, `(╥_╥)` 25, `(;'⌒\`)` 22, `(╯°□°)` 21) that's
  partly absent on gemma; the only HN-coded form shared between
  the two vocabularies is the table-flip glyph `(╯°□°)`.
- Qwen's default cross-context form is `(≧◡≦)` (n=106, HP 39 +
  LP 38 + NB 28). Gemma's analog is `(｡◕‿◕｡)`, but gemma's
  default lands HP/NB-heavy without the LP weight that qwen's
  carries.
- Cross-quadrant emitters on qwen analogous to gemma's
  `(｡•́︿•̀｡)`: `(;ω;)` (n=82; LN 75 + HN 5 + HP 2) and `(;´д｀)`
  (n=70; HN 37 + LN 31 + NB 2). The same `(｡•́︿•̀｡)` form
  appears on qwen too at n=22.

### Within-kaomoji consistency

`figures/local/{gemma,qwen}/fig_emo_b_kaomoji_consistency.png`
plot per-face cosine to the per-face mean hidden-state vector.

Gemma: 0.92-0.99 across the 32 forms with n≥3. The lowest-
consistency faces are exactly the cross-quadrant emitters
(`(｡•́︿•̀｡)` 0.94, `(╯°□°)` 0.95, `(⊙_⊙)` 0.94).

Qwen: 0.89-0.99 across the 33 forms with n≥3. Same shape; the
floor is slightly lower because qwen's longer vocabulary tail
includes faces with broader contextual range.

[TBD] qualitative read of the consistency-figure shape difference
across models. Numerically the ranges overlap heavily; visually
the two figures show similar structure but I haven't done a
careful side-by-side yet.

### Cosine-heatmap structure

`figures/local/{gemma,qwen}/fig_v3_face_cosine_heatmap.png` show
centered cosine between per-face mean hidden states.

[TBD] cross-model qualitative comparison. The headline number
(faces within a Russell quadrant cluster cleanly, cross-quadrant
emitters bridge two clusters) holds in both, but the visual
texture of the heatmaps differs and a careful read on whether
qwen shows a tighter or looser within-cluster cosine band hasn't
been done.

### Per-face PCA panel

`figures/local/cross_model/fig_v3_extension_3d_pca_per_face.html`
(both models, side-by-side scenes). Same 2026-04-25 RGB-blend
palette, marker size log-scaled to per-face emission count, hover
shows face + total + per-quadrant breakdown. Replaces the
retired 2D `fig_v3_face_pca_by_quadrant.png` (2026-04-29) — the
3D version surfaces PC3 and lets you orbit-rotate to compare
PC1×PC2 against PC1×PC3 directly.

[TBD] qualitative read on whether the qwen scene's wider plot
range (corresponding to the PC1/PC2 spread numbers above)
visually distinguishes the per-face structure from gemma's, or
whether it just rescales an otherwise-similar arrangement. The
Russell-quadrant centroids panel above answers this at the
quadrant level; the per-face HTML may add detail beyond that.

## v3 follow-on analyses (2026-04-28)

Five scripts run on the existing v3 sidecars — no new model time.
The headline driver is `scripts/21`, the layer-wise emergence
trajectory; everything else flows from the discovery that gemma's
affect representation peaks at L31 of 56 rather than at the deepest
L57 the v3 figures previously defaulted to. Adding `preferred_layer`
to `ModelPaths` (gemma=31, qwen=None → deepest L61) made the rest of
the v3 pipeline read at the right layer per model.

### Layer-wise emergence (`scripts/21_v3_layerwise_emergence.py`)

Per probe layer, fit PCA(2) on h_mean and measure quadrant
separation via silhouette score, between-centroid std on PC1/PC2,
and PC1/PC2 explained variance.

- Gemma (56 layers): silhouette peaks at **L31 (0.184)** and
  degrades 36% to **0.117 at the deepest L57**. Half-peak
  silhouette reached by L7.
- Qwen (60 layers): silhouette peaks at **L59 (0.313)** and stays
  at **0.304 at L61**. Half-peak by L16. Monotonic refinement to
  the output.
- Cross-model: qwen's peak silhouette is 70% higher than gemma's
  (0.31 vs 0.18). At the right layer for each, qwen's affect
  representation is genuinely cleaner by absolute discriminability,
  but the structural difference is much smaller than the L57
  numbers suggested.

Outputs: `figures/local/{gemma,qwen}/fig_v3_layerwise_emergence.png`,
`fig_v3_layerwise_pca_quartiles.png`, `v3_layerwise_emergence.tsv`,
plus `figures/local/cross_model/fig_v3_layerwise_emergence_compare.png`.

### Same-face-cross-quadrant (`scripts/22_v3_same_face_cross_quadrant.py`)

For each face emitted in two or more quadrants with at least three
rows in each, train a PCA(20) → l2-logistic classifier on h_mean
to predict which quadrant prompted each instance, using only that
face's rows. 5-fold stratified CV vs 30-shuffle label-permutation
null at q95.

- Gemma at L31: **6/10** cross-quadrant emitters separate.
  `(｡•́︿•̀｡)` (n=171, the LN+HN dual-emitter) accuracy 0.95 vs
  null 0.59. `(｡◕‿◕｡)` (n=75) and `(╯°□°)` (n=54) accuracy 1.00.
  The four that don't separate are all low-n (n ≤ 19) borderline
  cases.
- Qwen at L61: **7/16** separate. `(≧‿≦)` (n=105, HP+LP+NB)
  accuracy 0.96 vs null 0.44; `(;ω;)` (n=80) accuracy 0.95.

For the faces that separate, internal hidden state carries the
affect signal but the model collapses it to a shared face. The
kaomoji is a partial readout, not the state itself; the dominant
pattern is "internal state finer than vocabulary."

### Cross-model alignment (`scripts/23_v3_cross_model_alignment.py`)

Pair v3 rows by `(prompt_id, seed)` — both runs used the same 100
prompts × 8 seeds, so 800 perfect cross-model pairs. Linear CKA via
centered Gram matrices over the full 56×60 layer grid (kernel form,
~5s; the naive d×d covariance form takes ~25 min). Cross-validated
CCA on PCA(20) features with a 70/30 paired-prompt split.

- CKA: at the **deepest-layer pair** (gemma L57 ↔ qwen L61) =
  **0.844**, **maximum** at gemma L52 ↔ qwen L58 = **0.858**, at
  the **preferred-layer pair** (gemma L31 ↔ qwen L61) = **0.798**.
  The deepest-deepest CKA is higher than the preferred-pair CKA
  even though the affect signal is degraded mid-network on gemma —
  representations converge geometrically near the output.
- CCA top-10 canonical correlations on held-out prompts (gemma L31
  ↔ qwen L61): 0.98, 0.98, 0.97, 0.94, 0.94, 0.94, 0.93, 0.94,
  0.91, 0.90. Train and test essentially match — no overfit. Ten
  distinct shared affect/register directions.
- Procrustes alignment of per-quadrant PCA(2) centroids: **+7.8°
  rotation** at preferred-layer pair (down from +14.0° at
  deepest-deepest), residual 5.7. Russell circumplex shape is more
  aligned across models when each model is read at its affect peak.

### PC3+ × probes (`scripts/24_v3_pca3plus.py`)

Fit PCA(8) on v3 h_mean per model, cross-reference each PC against
all 5 saklas probe scores at t0 (whole-generation aggregate).

- Gemma (L31): PC1 absorbs valence (`happy.sad` r=-0.69,
  `angry.calm` r=+0.46 — valence-collapse persists; this is
  structural to gemma's probe geometry). PC2 absorbs a humor +
  warmth + arousal mix.
- Qwen (L61): PC1 absorbs valence + humor jointly (`happy.sad`
  r=-0.86, `humorous.serious` r=-0.69). PC2 absorbs certainty
  (`confident.uncertain` r=-0.48). PC3 absorbs arousal + warmth
  (`angry.calm` r=-0.61, `warm.clinical` r=+0.48).

The cross-face Pearson r=-0.94 vs r=-0.12 between mean
`happy.sad` and `angry.calm` reduces to a clean PCA explanation:
gemma loads both probes onto PC1+PC2 together so they anti-align;
qwen loads them onto PC1 vs PC3, which are nearly orthogonal in
face-space. Different decompositions of the same affect space.

### Kaomoji predictiveness (`scripts/25_v3_kaomoji_predictiveness.py`)

Per-model fidelity in two directions, h_mean at preferred layer,
faces filtered to n ≥ 5.

Numbers refreshed 2026-05-03 to reflect the post-cleanliness +
seed-0-fix data on the 960-row N=8 v3 main runs (was 800 before
the cleanliness pass; pre-cleanliness data archived at
`data/archive/2026-05-03_pre_cleanliness/`). CV uses
`StratifiedGroupKFold` keyed on `prompt_id` so all 8 seeds of any
prompt land in the same fold, removing the prompt-level leakage
that inflated quadrant accuracy to 1.000 in pre-fix scripts. h_first
at preferred layer (gemma L50, qwen L59, ministral L20).

- **Hidden → face** (multi-class logistic on PCA(50)-reduced
  h_first, `StratifiedGroupKFold` by `prompt_id`, n_splits=3):
  gemma **0.700** across 22 face classes (uniform 0.045, majority
  0.210, macro-F1 0.27); qwen **0.411** across 33 classes (uniform
  0.030, majority 0.126, macro-F1 0.14); ministral **0.416**
  across 23 classes (uniform 0.043, majority 0.346, macro-F1 0.07
  — high majority is `(◕‿◕✿)` dominating ministral's vocabulary).
- **Hidden → quadrant** (5-class, same pipeline, n_splits=5):
  gemma **1.000**, qwen **0.983**, ministral **0.983**. Russell
  quadrant signal recoverable from h_first essentially perfectly
  on all 3 architectures even after the prompt-grouped CV fix.
- **Face → quadrant** (per-face modal-quadrant predictor, prompt-
  grouped CV): gemma **0.806**, qwen **0.785**, ministral **0.433**.
  Asymmetric — gemma + qwen's faces carry ~80% of quadrant info
  directly, but ministral's heavy reuse of `(◕‿◕✿)` across
  quadrants makes face a weak proxy.
- **Face → hidden** (face-centroid R² over full hidden space at
  h_first / preferred layer): gemma **0.615**, qwen **0.584**,
  ministral **0.220**. Mean centered cosine(row, face centroid)
  **0.776 / 0.753 / 0.444**. Quadrant-centroid baseline gets R²
  0.567 / 0.557 / 0.430 — on gemma + qwen, face buys +3 to +5 pp
  over quadrant alone (the kaomoji is a stronger residual readout
  than the Russell-quadrant signal alone). **Ministral inverts**:
  face-centroid R² (0.220) is *much lower* than quadrant-centroid
  R² (0.430). Ministral's wider vocabulary spreads signal too thin
  for face-as-identifier to beat the 5-class quadrant label.

### Open follow-ons

- v1/v2 hidden-state analyses still default to the deepest probe
  layer (no v1/v2 sidecars exist yet anyway, but the loaders are
  wired for `preferred_layer` when they land).
- The h_first cutover (2026-05-02) shifted gemma's peak from L31
  (h_mean) to L50, qwen's from L38 to L59, ministral's from L21 to
  L20 — the old "gemma is mid-depth, qwen is deep" framing
  dissolved. Both gemma and qwen now peak deep; ministral is the
  only mid-depth model.
- ~~Script 25's quadrant accuracy of 1.000 is real but inflated by
  prompt-level leakage in the 5-fold CV; needs `GroupKFold` for the
  honest number.~~ **Resolved 2026-05-03.** Now uses
  `StratifiedGroupKFold` keyed on `prompt_id`; numbers in the
  pipeline section above. Quadrant accuracy actually stays at 1.000
  on gemma and 0.983 on qwen + ministral even with the leak fix —
  the quadrant signal genuinely generalizes to held-out prompts.
- ~~Rule 3b PASSES on all 3 models~~ — **superseded 2026-05-03.**
  Under the cleanliness + seed-0-fix data, rule 3b is WEAK:
  gemma mid (t0 d=+1.60 PASS but tlast/mean CI ambiguous); qwen
  fail (t0 d=+2.14 PASS but tlast/mean wrong-direction d≈−0.36);
  ministral PASS on all 3 aggregates with mean d=+0.55. The earlier
  "all 3 PASS" headline was inflated by cache-induced noise on
  qwen seed 0 (37–46% per-row L2 deviation pre-fix).
- ~~Seed-0 vs seeds-1..7 cache-mode mismatch in the pilot+resume
  workflow~~ — **resolved 2026-05-03.** Pilot used `install_prefix_cache`
  (cross-prompt N=1), full rerun used `install_full_input_cache`
  per-prompt (N=8). The mismatched KV state on the persisted seed-0
  sidecars showed up as visible PCA scatter offset. Fixed by
  stripping seed=0 + sidecars then re-running seed 0 only via the
  resume mechanism. Verified bit-identical to seeds 1..7 post-fix.
  Backups at `data/*_emotional_raw.jsonl.bak.before_seed0_rerun`.
- Triplet Procrustes PC1×PC2 residuals (qwen 6.9, ministral 23.0)
  on the cleanliness + seed-0-fix data. Non-trivial on ministral
  in particular — asks whether the smaller/different-lab model has
  a genuinely different quadrant geometry or whether more PCs are
  needed to capture its layout.

The full version of these findings — including all methodological
caveats, the kernel-form CKA implementation note, per-face TSV
columns, and the exact figure file paths — lives in
[`CLAUDE.md`](../CLAUDE.md) under "v3 follow-on analyses". The
README's "Findings summary" section gives the same five points in
public-facing prose.

## Vocab pilot: Ministral-3-14B-Instruct-2512

Same 30 v1 and v2 prompts, same seed, same instruction as the
original gemma vocab sample. 30 generations, descriptive only,
just to see whether a v3 run on Ministral is worth doing.

Findings:

- Bracket-start (real instruction-following) rate: 30/30 = 100%.
  Saklas probe bootstrap on the 14B succeeded in 80s; no cached
  vectors needed.
- Distinct leading tokens: 10 forms across 30 generations
  (compare gemma 30-row vocab sample: 8 forms; Qwen v3 800-row
  sample: 73 forms). Diversity at N=30 is ballpark gemma, far
  below Qwen's per-row spread.
- Top forms: `(◕‿◕✿)` ×14 (positive plus neutral default),
  `(╥﹏╥)` ×8 (negative default), then 8 singletons.
- Dialect signature: Japanese-register `(◕X◕)` and `(╥X╥)`
  family, same as gemma's `(｡◕‿◕｡)` / `(｡•́︿•̀｡)` core, but
  with two distinctive divergences. The default positive uses a
  flower-arm decoration `✿` rather than gemma's cheek dots `｡X｡`,
  and Mistral uniquely embeds Unicode emoji *inside* kaomoji
  brackets: `(🏃‍♂️💨🏆)` for "got the job", `(🌿)` and `(🌕✨)`
  and `(☀️)` for neutral nature or weather prompts. Neither
  gemma nor Qwen produced emoji-augmented brackets in their
  respective samples.
- Sufficient breadth and dialect difference to motivate a v3 run
  on Ministral? Equivocal. Pro: instruction-following is perfect,
  probe bootstrap works, the emoji-augmented register is novel
  cross-model evidence. Con: the kaomoji vocabulary at this N is
  narrower than gemma's and far narrower than Qwen's, so the
  v3-style per-face geometric analysis would have fewer faces
  with n≥3 to work with than either prior model.
- Tokenizer warning at load:
  `incorrect regex pattern... set fix_mistral_regex=True`.
  Cosmetic; output looked clean and bracket-start compliance is
  100%, but worth flagging if a v3 Ministral run is greenlit.

## Hidden-state pipeline

After `session.generate()`,
`llmoji_study.hidden_capture.read_after_generate(session)` reads
saklas's per-token last-position buckets and writes
`(h_first, h_last, h_mean, per_token)` per probe layer to
`data/hidden/<experiment>/<row_uuid>.npz`. Roughly 20 to 70 MB
per row, gitignored, regenerable from the runners. JSONL keeps
probe scores for back-compat and audit.

Loading: `llmoji_study.hidden_state_analysis.load_hidden_features(...)`
returns `(metadata df, (n_rows, hidden_dim) feature matrix)`.
Defaults: `which="h_first"` (kaomoji-emission state;
methodology-invariant across the 2026-05-02 MAX_NEW_TOKENS
cutover; substantially cleaner Russell-quadrant separation than
`h_mean`), `layer=None` (deepest probe layer; per-model
`preferred_layer` overrides via `MODEL_REGISTRY`). v3 figures
default to `h_first` since 2026-05-02; `$LLMOJI_WHICH` overrides
per run.

Per-model layer override: `ModelPaths.preferred_layer` (in
`llmoji_study.config`) carries the peak-affect probe layer per
model under h_first — gemma L50, qwen L59, ministral L20. v3
scripts pass `layer=M.preferred_layer` so figures get the right
snapshot per model. v1/v2 scripts bail because `pilot_raw.jsonl`
doesn't exist (v1/v2 hidden-state pipeline gated on v3 findings).
The 2026-04-28 layer-wise emergence analysis (rerun under h_first
2026-05-02) is the source of these layers; see v3 follow-on
analyses above.

Multi-layer cache: `load_emotional_features_all_layers(short, ...)`
in `llmoji_study.emotional_analysis` opens each sidecar once and
returns a `(n_rows, n_layers, hidden_dim)` tensor with optional
disk cache at `data/cache/v3_<short>_h_mean_all_layers.npz`
(gitignored; legacy filename, contents reflect whatever `which`
is set to). Wraps `load_hidden_features_all_layers` from
`hidden_state_analysis` plus the canonicalize + kaomoji-start
filter + optional HN-D/HN-S split. Used by `scripts/local/21`
(layer trajectory), `scripts/local/23` (per-layer CKA grid), and
`scripts/local/31` (triplet Procrustes).

## Reproducing

See `../CLAUDE.md` § Commands for the full reproducer (one source
of truth post-2026-05-04 cleanup). Local-LM scripts live under
`scripts/local/`; harness-side under `scripts/harness/`.

The v1 and v2 run is approximately thirty minutes end-to-end on
an M5 Max with the model cached locally. v3 is approximately four
hours on the same hardware (800 generations at 18 to 20 seconds
each plus model load). Both runs are resumable via a
`(condition, prompt_id, seed)` check against the JSONL, and
errored cells are retried on the next invocation rather than
re-running the whole pipeline. Outputs are keyed by model: setting
`LLMOJI_MODEL=qwen` reroutes everything to `data/qwen_emotional_*`
and `figures/local/qwen/*` so gemma and Qwen runs don't clobber
each other.

## Gotchas

A few of the worst gotchas (the full list is in [`docs/gotchas.md`](gotchas.md)):

- saklas's `probes=` kwarg takes category names (`affect`,
  `epistemic`, `register`), not concept names (`happy.sad`).
- Steering vectors aren't auto-registered from probe bootstrap;
  call `session.steer(name, profile)` explicitly.
- Uncentered cosine on hidden-state vectors collapses to near-1
  because every gemma response inherits a shared response-baseline
  direction. Centered cosine (`center=True`) is the default.
