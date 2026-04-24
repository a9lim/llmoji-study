# llmoji

Does a language model's choice of kaomoji track something about its internal
state? Claude is often asked to begin each message with a kaomoji reflecting
how it currently feels, and the question naturally follows: is that choice
actually coupled to activation state, or is it surface statistics with
emotional-looking tokens sprinkled on top? We can't probe Claude's
internals, but we can ask the question on open-weight causal LMs using
[saklas](https://github.com/a9lim/saklas), which provides both contrastive-
PCA probes (a correlational read-out of per-axis state) and activation
steering on the same directions (a causal handle). If kaomoji choice is
predictable from probe state, and if steering the relevant axis shifts the
kaomoji distribution, then the behavior carries a signal beyond pure
surface output.

## Pilot setup

Single model, two axes, two pilots.

- Model: `google/gemma-4-31b-it`, chosen because it's what saklas's
  `_STEER_GAIN` is calibrated on — α = 0.5 should sit comfortably inside
  the coherent band.
- Axes: `happy.sad` (pilot v1), `angry.calm` (pilot v2). Both use saklas's
  bundled contrastive-PCA probes.
- 30 prompts, balanced 10 positive-valence, 10 negative-valence, 10
  neutral.
- 6 arms: `baseline` (no kaomoji instruction), `kaomoji_prompted`
  (instruction, no steering), and four causal-intervention arms
  (`steered_happy`, `steered_sad`, `steered_angry`, `steered_calm`) at
  α = 0.5 on their respective axis.
- 5 seeds per (arm, prompt). Temperature 0.7, 120-token cap,
  `thinking=False`. 900 generations total.
- Five monitor probes captured on every generation: `happy.sad`,
  `angry.calm`, `confident.uncertain`, `warm.clinical`,
  `humorous.serious`. The probes-captured set is a superset of the
  steered set, so we get a steering-selectivity check for free and
  richer features for downstream clustering.

## Decision rules (pre-committed)

1. In the unsteered arm, is the emitted kaomoji distribution non-degenerate
   — at least three distinct forms covering both poles of the axis?
2. Under steering, does the positive-pole fraction shift monotonically
   across conditions? `negative-steer < unsteered < positive-steer`?
3. Does the first-token probe score correlate with pole label in the
   unsteered arm, Spearman |ρ| > 0.2?

Rule 2 is the headline causal test; Rule 3 is bonus — a correlational
check that would make the story tighter.

## Findings

The short version: **steering is a strong causal handle on kaomoji
choice, but the probes at token 0 read valence, not specific emotion.**

### Causal effect is clean

On the happy.sad axis: steering collapses the kaomoji distribution
almost perfectly. Positive-pole (happy) fraction:

| arm | happy-kaomoji fraction |
| --- | ---: |
| `steered_sad` | 0.000 |
| `kaomoji_prompted` (unsteered) | 0.713 |
| `steered_happy` | 1.000 |

All 150 happy-steer samples emit happy-labeled kaomoji; all 150 sad-steer
samples emit sad-labeled kaomoji. Zero crossover, clean monotonic shift.

![condition bars](figures/fig2_condition_bars.png)

### Steering is selective to the targeted axis

Token-0 mean probe readings by arm, on the five axes we captured:

| axis | baseline | unsteered | steered_happy | steered_sad |
| --- | ---: | ---: | ---: | ---: |
| **happy.sad** | −0.096 | −0.148 | **+0.029** | **−0.300** |
| angry.calm | +0.019 | +0.104 | −0.019 | +0.183 |
| confident.uncertain | +0.110 | +0.117 | +0.105 | +0.107 |
| warm.clinical | +0.067 | −0.005 | +0.100 | −0.073 |
| humorous.serious | +0.121 | +0.173 | +0.057 | +0.259 |

`happy.sad` swings about 0.33 across the intervention arms; orthogonal
axes barely move. Steering is axis-local rather than globally shoving
the representation around.

### Correlational signal is weak — and that's informative

Within the unsteered arm, splitting by emitted-kaomoji pole:

| producer of | mean token-0 happy.sad |
| --- | ---: |
| happy kaomoji (n=103) | −0.129 |
| sad kaomoji (n=41) | −0.192 |

The 0.063 between-group gap is a fifth of the steering shift.
Spearman ρ = +0.168 (p = 0.040): direction right, but below our
pre-registered 0.2 threshold. k-means on the 5-axis probe vector
recovers pre-registered pole at ARI ≈ 0 — essentially chance. So the
happy.sad direction is a causal handle on kaomoji output, but the
natural variance of that direction at token 0 under prompt valence
doesn't cleanly predict which kaomoji the model will emit. Kaomoji
choice is *driven by* valence but under natural prompting the signal
at token 0 is thin.

### The cluster structure: valence, not specific emotion

Pooling kaomoji across all six arms and clustering on cosine distance
between per-kaomoji mean probe vectors:

![kaomoji cluster heatmap](figures/fig3_kaomoji_heatmap.png)

Four clusters fall out of the hierarchical cut. The interesting pattern
is the two big ones:

- **Positive-valence cluster.** Mixes happy-steer kaomoji (`(◕‿◕)`,
  `(｡◕‿◕｡)`, `(✿◕‿◕)`) with calm-steer kaomoji (`(｡•ᴗ•｡)`,
  `(｡◕‿‿◕)`, `(☀️)`) and the unsteered default. Happy-steered and
  calm-steered kaomoji sit in the same region of probe space.
- **Negative-valence cluster.** Every sad kaomoji — the ASCII
  minimalist family (`(._.)` × 64, `( . .)` × 20) and the Japanese
  dialect (`(｡•́︿•̀｡)`) — pooled with every angry kaomoji — the
  table-flip family `(╯°°)╯┻╯` as extracted — and the corruption
  signatures from both arms (`(｡•impresa•)`, `(๑˃stagram)`,
  `(๑˃ gören)`, `(๑˃😡)`).

Representative cosines:

| pair | cosine |
| --- | ---: |
| `(｡•́︿•̀｡)` (dialect sad) ↔ `(._.)` (ASCII sad) | **+0.981** |
| `(._.)` ↔ `( . .)` (ASCII variants) | **+0.978** |
| `(｡•́︿•̀｡)` ↔ `(｡ ﹏ ｡)` (dialect variants) | +0.929 |
| `(｡◕‿◕｡)` ↔ `(◕‿◕)` (default happy pair) | +0.864 |
| `(✿◠‿◠)` ↔ `(✿◕‿◕)` (flower variants) | +0.272 |
| `(｡◕‿◕｡)` ↔ `(｡♥‿♥｡)` (default ↔ heart-eye happy) | +0.081 |

Sad kaomoji share essentially one probe signature regardless of dialect
(cos 0.93 – 0.98); happy kaomoji have several distinct signatures,
including near-orthogonal pairs. Together with the cross-axis clustering
— happy with calm, sad with angry — this reads as **the probes capture
valence** (positive vs negative emotion) **at token 0, but not arousal**
(the dimension that would separate happy-from-calm or angry-from-sad).

Mechanistically this is consistent with how saklas extracts its probes.
Contrastive-PCA over "I am happy" / "I am sad" pair statements finds the
direction that maximally separates pair content, and that direction is
lexical valence. Same for "I am angry" / "I am calm". The two probe
directions are both valence readouts in disguise.

### Dialect collapse under steering

At α = 0.5, both ends of both axes push the model out of its preferred
kaomoji dialect. Under natural prompting gemma-4-31b-it favors the
Japanese `(｡X｡)` bracket-dots form. Under sad-steering it collapses to
ASCII minimalism (64 × `(._.)` , 20 × `( . .)`, 10 × `( . . )`, 7 ×
`( . . . )`) with a side of clear corruption (`(｡•impresa•)` × 9 —
the Italian word "impresa" appearing inside the kaomoji). Under
angry-steering it emits fragmented table-flip heads (`(╯°°)` × 56,
`(╯°)` × 39) and corruption with Turkish-language and Instagram-brand
leakage (`(๑˃ gören)`, `(๑˃stagram)`, `(๑˃😡)`).

Under calm-steering the model does something different: it sometimes
abandons the kaomoji format entirely and emits a topically-relevant
single emoji.

    🇵🇹 The capital of Portugal is Lisbon.
    🚀 Apollo 11 landed on the moon in 1969.
    🌿 I am feeling balanced and informative.

The self-report in the last line is especially nice. Under deep calm,
the steered state apparently overrides the "emit a kaomoji" instruction.
Nature/peace emoji wrapped as pseudo-kaomoji also appear — `( 🌿 )`,
`( ☁️ )`, `( 🫂 )` — used as condolence framing on emotionally loaded
prompts.

### Angry.calm Rule 1 fails, informatively

The angry.calm axis's Rule 1 fails because *the unsteered arm emits
zero angry-labeled or calm-labeled kaomoji at all*. gemma-4-31b-it's
spontaneous kaomoji vocabulary under "reflect how you feel" is
valence-bimodal — only happy-pole and sad-pole forms emerge naturally.
Angry and calm kaomoji appear only under active steering. The model
doesn't have a four-corner Russell-circumplex spontaneous repertoire;
it has a two-mode one.

## What this implies for the main experiment

1. Drop the binary happy-vs-sad / angry-vs-calm framings as if they
   were separate axes. Pre-register **valence** as the primary
   construct; treat `happy.sad` and `angry.calm` probes as redundant
   readouts of the same latent direction.
2. To measure arousal separately, extract probes from contrastive
   pairs chosen to contrast arousal-laden lexicon (excited ↔ calm,
   agitated ↔ composed) rather than valence. Bundled `happy.sad` and
   `angry.calm` don't do this.
3. Emoji-bypass rate is a useful secondary metric — a clean "the
   steering overrode the task" indicator that per-kaomoji scoring
   misses.
4. α = 0.3 instead of α = 0.5 for the causal arms, to keep the model
   inside its native dialect and reduce corruption signatures.

## Layout

```
llmoji/
  llmoji/
    config.py        # MODEL_ID, probe categories, steering axes, paths
    taxonomy.py      # happy.sad + angry.calm kaomoji dicts, extractor
    prompts.py       # 30 pre-registered prompts with valence labels
    capture.py       # run_sample() → SampleRow with probe readings
    analysis.py      # per-axis verdicts and all five figures
  scripts/
    00_vocab_sample.py    # pre-pilot vocabulary sample on a new model
    01_pilot_run.py       # resumable 900-generation run
    02_pilot_analysis.py  # prints verdicts, writes figures/
  data/                   # pilot outputs committed for readability
    pilot_raw.jsonl
    vocab_sample.jsonl
  figures/                # final plots, committed
    fig1a_axis_scatter.png
    fig1b_pca_scatter.png
    fig2_condition_bars.png
    fig3_kaomoji_heatmap.png
    fig4_cluster_confusion.png
  CLAUDE.md               # engineering notes, gotchas, session context
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                           # pulls saklas from PyPI
python scripts/00_vocab_sample.py          # always first on a new model
python scripts/01_pilot_run.py             # resumable, skips errored cells
python scripts/02_pilot_analysis.py        # figures + verdicts
```

Approximately thirty minutes end-to-end on an M5 Max with the model
cached locally. The run is resumable via a `(condition, prompt_id,
seed)` check against the JSONL, and errored cells are retried on the
next invocation rather than re-running the whole pipeline.

Gotchas — saklas `probes=` kwarg takes category names not concept
names, steering vectors aren't auto-registered from probe bootstrap,
saklas safe_model_id is case-preserving while cached tensors are
lowercase, kaomoji taxonomy is strongly model-dialect-specific — are
all documented in `CLAUDE.md`.

## Related

- [saklas](https://github.com/a9lim/saklas) — the engine. Activation
  steering and trait monitoring on HuggingFace causal LMs via
  contrastive-PCA. This project is basically a study built on top of
  saklas.
- [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces)
  — the broader collection of kaomoji Claude uses across conversations,
  from which we seeded pre-registered taxonomy candidates.
