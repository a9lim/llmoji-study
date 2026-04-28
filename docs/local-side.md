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

![condition bars](../figures/fig2_condition_bars.png)

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

![kaomoji cluster heatmap](../figures/fig3_kaomoji_heatmap.png)

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
sidecars at the deepest probe layer, written alongside the JSONL.

### Findings

Hidden-state PCA on 800 row-level vectors gives PC1 13.0% and PC2
7.5%. Russell quadrants separate cleanly. PC1 reads as valence
(HN and LN on the right at +7, HP and LP and NB on the left at -2
to -5), PC2 reads as activation (NB and LP at +4 to +6, HP at -6).
Separation ratios are PC1 2.02 and PC2 2.73.

HP and LP discriminate cleanly. HN and LN overlap on PC1, because
they share the sad-face vocabulary `(｡•́︿•̀｡)` (n=171, 102 LN +
52 HN); a single cross-quadrant face flattens the negative-side
arousal information. HN gets a dedicated shocked or angry register
(`(╯°□°)`, `(⊙_⊙)`, `(⊙﹏⊙)`) that doesn't appear elsewhere.

Within-kaomoji consistency to mean is 0.92 to 0.99 across the 32
forms with n≥3 after canonicalization (33 pre-canonicalization).
The lowest-consistency faces are exactly the cross-quadrant
emitters.

Probe-space PCA on the same 800 rows would give PC1 ≈ 89% (the
v1 and v2 collapse). In hidden-state space the second emotional
dimension survives, so the v1 and v2 valence-collapse is a
probe-extraction artifact, not a property of the underlying
representation.

![v3 PCA](../figures/fig_v3_pca_valence_arousal.png)

## Pilot v3: Qwen3.6-27B replication

Same prompts, same seeds, same instruction, swapped model.
Multi-model wiring via `LLMOJI_MODEL=qwen` selects a registry entry
that reroutes outputs to `data/qwen_emotional_*` and
`figures/qwen/*`. Qwen3.6-27B is a reasoning model so
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

![v3 Qwen PCA](../figures/qwen/fig_v3_pca_valence_arousal.png)

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
Defaults: `which="h_mean"` (whole-generation aggregate; smoother
and more probative than `h_last`), `layer=None` (deepest probe
layer). All v3 figures use `h_mean`.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji   # during dev; replace with `pip install llmoji>=1.0,<2` once published
pip install -e .

# Smoke test the hidden-state pipeline (~5 min)
python scripts/99_hidden_state_smoke.py

# v1/v2 (gemma steering, 900 generations)
python scripts/00_vocab_sample.py
python scripts/01_pilot_run.py
python scripts/02_pilot_analysis.py

# v3 (naturalistic, 800 generations); gemma default
python scripts/03_emotional_run.py
python scripts/04_emotional_analysis.py             # Fig A/B/C + summary TSV
python scripts/13_emotional_pca_valence_arousal.py  # Russell-quadrant PCA
python scripts/17_v3_face_scatters.py               # per-face PCA, cosine, probe scatter

# v3 on a non-gemma model (registry: gemma | qwen | ministral)
LLMOJI_MODEL=qwen python scripts/03_emotional_run.py
LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py
LLMOJI_MODEL=qwen python scripts/13_emotional_pca_valence_arousal.py
LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
# outputs land at data/{short_name}_emotional_*, figures/{short_name}/*

# Cross-pilot + v3-extension analyses
python scripts/10_cross_pilot_clustering.py
python scripts/11_emotional_probe_correlations.py
python scripts/12_emotional_prompt_matrix.py
```

The v1 and v2 run is approximately thirty minutes end-to-end on
an M5 Max with the model cached locally. v3 is approximately four
hours on the same hardware (800 generations at 18 to 20 seconds
each plus model load). Both runs are resumable via a
`(condition, prompt_id, seed)` check against the JSONL, and
errored cells are retried on the next invocation rather than
re-running the whole pipeline. Outputs are keyed by model: setting
`LLMOJI_MODEL=qwen` reroutes everything to `data/qwen_emotional_*`
and `figures/qwen/*` so gemma and Qwen runs don't clobber each
other.

## Gotchas

A few of the worst gotchas (the full list is in [`CLAUDE.md`](../CLAUDE.md)):

- saklas's `probes=` kwarg takes category names (`affect`,
  `epistemic`, `register`), not concept names (`happy.sad`).
- Steering vectors aren't auto-registered from probe bootstrap;
  call `session.steer(name, profile)` explicitly.
- The kaomoji taxonomy is strongly model-dialect-specific; always
  run `00_vocab_sample.py` before locking a taxonomy for a new
  model.
- The v3 runner's per-quadrant "emission rate" log line counts
  TAXONOMY matches, not instruction-following compliance, which
  reads as collapse on any non-gemma model when it isn't.
- Uncentered cosine on hidden-state vectors collapses to near-1
  because every gemma response inherits a shared response-baseline
  direction. Centered cosine (`center=True`) is the default.
