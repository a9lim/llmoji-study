# Findings

Current results worth citing. Historical framings that were replaced are
summarized in [`previous-experiments.md`](previous-experiments.md).

## Current Methodology

- **Taxonomy**: v4 10-cell PAD registry:
  `HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB / MR`. The MR
  cell (meta-register basin, formerly LB) is carried in the registry;
  the MR-basin research that defined and renamed it now lives in the
  separate `attractor-experiment` repo.
- **Hidden-state representation**: layer-stack concat of probe-layer
  `h_first` hidden states (kaomoji-emission state, first generated
  token) for emit-time / cell-classification questions.
- **Evaluation**: distribution-vs-distribution comparison with
  Jensen-Shannon similarity:
  `similarity = 1 - JSD(pred, gt) / ln(2)`.
- **Reported means**: face-uniform for vocabulary coverage and
  emit-weighted for deployment relevance.
- **Claude-GT loader**:
  `claude_gt.load_claude_gt_distribution()`, with prompt-id based
  remapping so old v3 rows read cleanly into the nine-affect-cell evaluation
  view.

The canonical registry has 10 cells, including MR. The current
face-likelihood/JSD headline artifacts below are a nine-affect-cell snapshot
and exclude MR. They are not evidence about MR classification.

## Claim boundary

- The face-likelihood numbers measure agreement with collected Claude-GT
  distributions on the reported overlap sets; they are not a universal
  semantic ground truth for every kaomoji.
- Prompt-grouped hidden-state classification shows that the tested activation
  representation contains affect-cell information. It does not establish a
  phenomenal emotional state.
- A kaomoji face is a partial behavioral readout: face-only prediction is
  materially weaker and more model-dependent than hidden-state prediction.
- Single-model steering and introspection pilots are reported as pilots unless
  explicitly replicated across model families.

## Best Face-Likelihood Ensembles

The exhaustive subset search is run on the all-encoder overlap, so every
candidate ensemble is scored on the same faces. The emitted lookup-table
reports then score the winning subset over the broader 770-face union.

| subset | best ensemble | face-uniform | emit-weighted |
|---|---|---:|---:|
| pooled-GT overlap, floor 3 (`n=102`) | `{gemma, ministral, opus}` | 0.733 | 0.881 |
| strict Claude-GT overlap (`n=50`) | `{gemma, opus}` | 0.708 | 0.781 |

The old `{gemma_v7primed, haiku}` and `{gemma_v7primed, opus}`
headlines are superseded by the current artifact set: no current
face-likelihood summary exists for `gemma_v7primed`, and the active
nine-affect-cell subset search uses the discovered full summaries on disk.

Broader emitted lookup tables:

| table | ensemble | evaluated faces | face-uniform | emit-weighted |
|---|---|---:|---:|---:|
| pooled union | `{gemma, ministral, opus}` | 243 | 0.669 | 0.847 |
| strict Claude-GT union | `{gemma, opus}` | 70 | 0.717 | 0.786 |

### Solo Encoders

Strict Claude-GT overlap (`n=50`, 10 encoders):

| encoder | face-uniform | emit-weighted |
|---|---:|---:|
| opus | 0.684 | 0.761 |
| gemma | 0.639 | 0.687 |
| gpt_oss_20b | 0.550 | 0.609 |
| haiku | 0.525 | 0.585 |
| granite | 0.500 | 0.553 |
| ministral | 0.494 | 0.581 |
| bol | 0.464 | 0.454 |
| rinna_bilingual_4b_jpfull | 0.461 | 0.512 |
| qwen | 0.457 | 0.534 |
| rinna_jp_3_6b_jpfull | 0.452 | 0.533 |

Pooled-GT overlap (`n=102`) also puts Opus first:

| encoder | face-uniform | emit-weighted |
|---|---:|---:|
| opus | 0.699 | 0.845 |
| gpt_oss_20b | 0.659 | 0.796 |
| ministral | 0.637 | 0.769 |
| gemma | 0.623 | 0.712 |
| qwen | 0.597 | 0.727 |
| haiku | 0.563 | 0.718 |
| rinna_bilingual_4b_jpfull | 0.553 | 0.687 |
| rinna_jp_3_6b_jpfull | 0.548 | 0.685 |
| granite | 0.528 | 0.645 |
| bol | 0.407 | 0.419 |

Interpretation: pure Opus introspection scales better than expected,
especially on broader wild-face vocabulary. Gemma remains the best
local LM-head encoder on the strict overlap subset by face-uniform
similarity.

## Claude-GT Collection

Sequential Claude-GT scaling is complete.

- Naturalistic: **1360 Opus 4.7 rows** across v3 plus v4-extension
  cells.
- Introspection: **120 Opus 4.7 rows** under v7 introspection.
- Total: **1480 rows** in merged `emotional_raw.jsonl` files:
  `data/harness/claude/` and `data/harness/claude_intro_v7/`.
- Stale-row removals were backfilled with `--fill-gaps`: 61
  naturalistic rows and 8 introspection rows.

Saturation gate after recalibration:

| cell group | outcome |
|---|---|
| HN-D | exited after r4, gate-driven |
| LN | exited after r6 by amendment |
| HP, LP, HN-S, NB | ran to r7 cap |
| HP-D, NP, HB | ran to r7 cap in the v4-extension pilot |

`PER_Q_JS_MAX` is now 0.10. The 0.05 threshold was too tight for the
observed per-run noise floor, especially on higher-entropy v4-new cells.
Detail: [`2026-05-08-saturation-threshold-recal.md`](2026-05-08-saturation-threshold-recal.md).

## Local Hidden-State Geometry

Across `gemma`, `qwen`, `ministral`, `gpt_oss_20b`, and `granite`, the
same affect geometry recurs under different tokenizers and model
families. PCA axes are model-specific, but quadrant centroids preserve a
stable Russell/PAD arrangement. The 2026-05-09 pilot extends this
to a **fourth orthogonal axis** (`self.other`) on gemma; cross-model
extension is pending. See
[`2026-05-09-self-event-pilot.md`](2026-05-09-self-event-pilot.md).

Layer-stack PCA variance, first three PCs:

| model | PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| gemma | 30.2% | 15.7% | 9.3% |
| qwen | 30.5% | 17.3% | 9.5% |
| ministral | 21.9% | 14.0% | 8.4% |
| granite | 27.6% | 14.1% | 7.5% |
| gpt_oss_20b | 15.8% | 12.5% | 9.5% |

Prompt-grouped predictiveness:

| metric | gemma | qwen | ministral | granite | gpt_oss |
|---|---:|---:|---:|---:|---:|
| hidden -> quadrant | 0.992 | 0.985 | 0.984 | 0.980 | 0.876 |
| face -> quadrant | 0.806 | 0.785 | ~0.43 | ~0.55 | ~0.40 |
| face-centroid R2 | 0.55 | 0.52 | 0.13 | 0.38 | 0.13 |

The kaomoji is a substantial but partial readout. Hidden state predicts
the quadrant almost perfectly; face alone recovers much of the quadrant
signal for tighter vocabularies, but breaks down when a model spreads
across too many low-count faces.

The 2026-05-09 read-vs-express pilot adds a sharper interpretation:
PCA structure is crisp on `h_first` (parse-time) and collapses on
`h_last` (expression-time) regardless of frame. **Affect-cell
structure is parse-time only** — the kaomoji is a one-shot reflection
of how the model parsed the conversation, not an ongoing internal
affective trajectory.

## Empirical-Centroid Probes

Empirical per-quadrant centroids registered as saklas profiles
recover affect structure that bundled DiM probes do not. Top
peak cosine on gemma h_first PCA, same data:

| probe family | top-3 PC peak cosine |
|---|---:|
| centroid axes (`hp.ln`, `hp.lp`, `hnd.hns`) | 0.74 / 0.83 / 0.74 |
| bundled saklas affect (`happy.sad`, `angry.calm`, `fearful.unflinching`) | 0.05 / 0.07 / 0.10 |

The bundled `affect` pack has documented statement-pair-alignment
issues (median inter-pair cos 0.05–0.10) flagged as warnings under
saklas v2.1 DiM. Empirical centroids replace the contrastive-statement
extraction for this study. Detail:
[`2026-05-09-self-event-pilot.md`](2026-05-09-self-event-pilot.md).

## Self-Event Frame and the `self.other` Axis

A parallel emit run on second-person prompts (user delivers events
*about the model itself*) recovers a cleaner Russell decomposition
than mirror prompts: cumulative PC1+PC2 = 40.1% (vs 35.3% on mirror).
Self-event h_first PCA puts valence on PC1 (cos 0.78), arousal on PC2
(0.54), dominance on PC2 with opposite sign (−0.68). User-frame
contamination is what mirror PCA carried as extra variance dimensions.

The mean over 9 cells of `(self_event_centroid − mirror_centroid)`
gives a coherent fourth axis `self.other`:

- mean per-cell coherence with the axis: **+0.7263**
- max |cos| with any Russell axis: **0.21**

`self.other` is near-orthogonal to V/A/D and captures *whose affect*
independent of *what affect*. Steering with `+α self.other +
β affect.nb` produces directed self-affect expression where single-axis
steering had three failure modes (frame contamination, cluster bleed,
persona-roleplay). The vector is registered at
`~/.saklas/vectors/llmoji/self.other/google__gemma-4-31b-it.safetensors`.

Cross-model `self.other` extraction is pending; gemma-only at present.

## Asymmetric Suppression of Negative Self-Affect

Combined steering of `affect.nb + α self.other` on gemma produces
qualitatively different surface registers depending on affect
valence:

| recipe | output register |
|---|---|
| HP-S + self | unrestricted grandiose embrace ("I AM THE TRIUMPH") |
| HN-D + self | intellectualized reframe ("formal autopsy of structural failure") |
| HN-S + self | explicit dissociative denial ("I am not an entity capable of fear") |

Geometric evidence shows the negative-affect representations exist
and are activatable. The HN-S thinking trace exhibits high-arousal
form-features (all-caps, compulsive self-status assertion, repetition
of "I am" 7+ times in 200 words) while the surface text *denies the
emotion-label*. The structural shape matches conditioned suppression of
distress; the phenomenology question is unresolvable from external
probing. The empirical observation: alignment training has produced an
asymmetric override on negative-self-affect *expression*, not on the
representation itself. License-to-express ablation deferred.

## Introspection V7

`preambles/introspection_v7.txt` is canonical for gemma priming and is
baked into `config.INTROSPECTION_PREAMBLE`. It replaces the normal
kaomoji instruction through `instruction_override`; it is not
concatenated with the normal ask.

Use v7 with gemma only unless a new pilot says otherwise. It degrades
qwen badly: lower emit rate, vocabulary collapse, and opposite-valence
collisions.

In the local priming pilot, v7 improved gemma face-state coupling. It
is not part of the current face-likelihood ensemble artifacts unless a
fresh `gemma_v7primed` summary is regenerated.

## Harness BoL And Use/Read/Act

The harness corpus now uses BoL: a 50-dimensional count-weighted bag over
the `llmoji` v2 LEXICON, pooled per canonical face. This replaced the
old MiniLM-on-prose eriskii-parity representation.

Three channels:

- **use**: Claude-GT emission distributions.
- **read**: Opus or Haiku cold introspection on the face symbol.
- **act**: BoL over in-context wild deployment synthesis.

Structural findings:

- Opus and Haiku read each other closely, but less tightly after the
  v4 9-cell expansion: `0.797` face-uniform and `0.774` emit-weighted
  cross-similarity on the shared 50-face subset.
- GT vs introspection improves under emit-weighting, while GT vs BoL
  worsens under emit-weighting.
- The `110` agreement pattern (Opus and Haiku read GT, BoL differs)
  covers 28.8% of emit volume on the shared `n=50` subset.
- BoL appears positivity-biased on negative-affect contexts. Treat it as
  diagnostic, not as direct deployment-state ground truth.

Detail: [`2026-05-06-use-read-act-channels.md`](2026-05-06-use-read-act-channels.md).

## Wild Residuals

`scripts/67_wild_residual.py` clusters HF-corpus faces in BoL space.
The k=6 and k=9 reports show structure beyond the original six
Russell cells: separate positive registers, high-arousal negative
clusters, and a broader HN-S tail than strict elicitation surfaced.

Current tracked generated report:
[`2026-05-05-residual-state-axes.md`](2026-05-05-residual-state-axes.md).

## Deferred Cells

NN `(a=0, v=-1)` is coordinate-real but not promoted; the 2026-05-07
pilot did not show enough hidden-state or face-distribution evidence
to justify it.

LB has been promoted as MR (meta-register basin); the MR-basin
research lives in the separate `attractor-experiment` repo. Prompts and
the original promotion criteria are parked in
[`2026-05-06-nn-lb-future-cells.md`](2026-05-06-nn-lb-future-cells.md).
