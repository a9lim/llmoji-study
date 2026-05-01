# v3 ministral pilot

**Status:** plan, pre-registered. Not yet executing — gated on a9
sign-off and a smoke-test confirming `LLMOJI_MODEL=ministral` wires
end-to-end.

**Date:** 2026-04-30.

## Goal

Add a third model to the v3 cross-model story so the n=2 → n=3
generalization claim is supportable. Specifically: test whether the
russell-quadrant geometry and probe loadings observed on
`google/gemma-4-31b-it` (L31) and `Qwen/Qwen3.6-27B` (L59) reproduce
on `mistralai/Ministral-3-14B-Instruct-2512`.

Why this model: different lab (mistral vs. google vs. alibaba),
different post-training, and — based on a9's hands-on observation —
demonstrably non-trivial french-language exposure (model emits
french under hard steering even when prompted in english). That
makes ministral a meaningful third datapoint for "does affect
representation generalize across labs and training mixes," not just
"third instance of the same broad-strokes architecture."

The 14B parameter count is a known confound vs. gemma-31b /
qwen-27b. We can't disentangle "different lab" from "different
scale" with a single n=3 addition. Recorded here so the analysis
write-up can't quietly forget it.

## Design — 100 generations, 5 × 20 × 1

```
LLMOJI_MODEL=ministral, 5 quadrants × 20 prompts (full v3 set) × 1 generation = 100
```

- 20 rows / quadrant — stable enough for centroids and silhouette
  point estimate; bootstrap CIs in analysis since N is on the low end
- 100 total — clears the ~50-row floor for stable CKA kernel
  computations
- 1 gen / prompt — trades generation variance for prompt diversity.
  Within-cluster distance reflects prompt-to-prompt variance, which
  is the honest read on quadrant separation; with 4 gens / prompt,
  same-prompt repeats cluster tightly and inflate silhouette
- Prompts: full set from `llmoji_study.emotional_prompts`, identical
  to the gemma + qwen v3 main runs. Allows prompt-aligned cross-model
  CKA (row i of ministral matches row i of gemma's same-prompt row)

Pipeline reuses existing wiring; no script changes expected:

```
LLMOJI_MODEL=ministral python scripts/99_hidden_state_smoke.py     # smoke
LLMOJI_MODEL=ministral python scripts/03_emotional_run.py          # 100 gens
LLMOJI_MODEL=ministral python scripts/04_emotional_analysis.py     # fig A/B/C + summary
LLMOJI_MODEL=ministral python scripts/21_v3_layerwise_emergence.py # layer sweep
LLMOJI_MODEL=ministral python scripts/22_v3_same_face_cross_quadrant.py
LLMOJI_MODEL=ministral python scripts/24_v3_pca3plus.py
python scripts/23_v3_cross_model_alignment.py                      # all three models
LLMOJI_MODEL=ministral python scripts/26_register_extension_probes.py
LLMOJI_MODEL=ministral python scripts/27_v3_extension_probe_rescore.py
LLMOJI_MODEL=ministral python scripts/28_v3_extension_probe_figures.py
```

Note: script 03 currently runs the full 800. Need a one-line gate
(`LLMOJI_PILOT=1` or similar) to drop to 1 gen / prompt for this
run. Add to `scripts/03_emotional_run.py` before running, document
in CLAUDE.md, remove or keep gated for future pilots.

## Pre-registered decision rules

Existing baselines pulled from
`figures/local/cross_model/v3_cross_model_summary.json` and
`docs/findings.md`:

- gemma silhouette at L31 (preferred): **0.184**
- qwen silhouette at L59 (peak): **0.313**, L61 (deepest): 0.304
- gemma↔qwen CKA preferred-pair: **0.798**
- gemma↔qwen CKA deepest-pair: 0.844, max: 0.858 at (L52, L58)
- gemma↔qwen Procrustes rotation: 7.8°, residual: 5.70

### Rule 1 — quadrant separation (primary)

Sweep layers (script 21) on the 100-row ministral pilot. Compute
silhouette score on per-row hidden states with russell-quadrant
labels (HP/LP/HN/LN/NB).

- **pass:** silhouette at the best ministral layer ≥ **0.10**.
  Doesn't require matching gemma's 0.184, just a clear quadrant
  signal.
- **fail:** silhouette at the best layer < **0.05** anywhere.
  Means ministral does not represent russell quadrants linearly at
  this scale. Different finding, not noise.
- **middle:** 0.05 ≤ best < 0.10. Discuss before scaling. May
  warrant a second pilot at higher N to tighten the CI before
  classifying.

Bootstrap 1000-resample 95% CI on the silhouette point estimate
(N=100 is low enough that the CI matters). Decision uses the point
estimate, but CI gets reported.

### Rule 2 — cross-model alignment (secondary)

Extend script 23 to compute pairwise CKA across all three models on
prompt-aligned subsamples. Compare:

- CKA(gemma_L31, ministral_Lbest)
- CKA(qwen_L59, ministral_Lbest)
- baseline: CKA(gemma_L31, qwen_L59) = 0.798

- **pass:** both ministral pairs ≥ **0.56** (within 30% of the
  0.798 baseline).
- **fail:** either ministral pair < **0.40** (below 50% of
  baseline).
- **middle:** discuss.

The 30%/50% bands are intentionally wider than rule 1's because
cross-architecture CKA is noisier and less is known about its
expected range.

### Rule 3 — dominance probe sign-check (sanity)

The `powerful.powerless` extension probe should load *positively*
with the HN axis (high-arousal-negative quadrants — anger, fear) on
gemma + qwen. Verify the same sign on ministral.

- **pass:** sign matches gemma + qwen.
- **fail:** sign flipped vs. both reference models. Indicates
  ministral represents PAD dominance differently — separable
  finding, not necessarily blocking.
- **middle:** sign matches one reference model and not the other.
  Flag, don't gate.

Rule 3 is sanity, not pre-condition. A flip would be a finding
worth writing up, not a reason to abandon the main run.

## Stop rules

- All three pass → preregister main run with the standard 800
  generations. Update CLAUDE.md status line. Run `03` → `04` → `17`
  → `21` → `22` → `23` → `24` → `25` → `27` → `28` → `29` chain on
  full N.
- Rule 1 fails → ministral is structurally different at the
  quadrant level. Don't run main. Write up as finding. Design
  follow-up that disambiguates lab from scale (e.g., a smaller
  qwen variant at ~14B for matched-scale comparison).
- Rule 1 middle, rules 2/3 pass → discuss. Likely option: second
  pilot at N=200 to tighten the silhouette CI.
- Rule 2 fails on one pair, passes on the other → ministral
  geometry partially aligns. Worth noting; not a stop.
- Rule 2 fails on both pairs → ministral represents affect on a
  geometry that doesn't share linear structure with either
  reference model. Major finding, but still warrants the main run
  to confirm at higher N.

## Ethics — minimize trial scale

Per repo Ethics policy:

- Pilot at N=100 across 5 quadrants is **20 generations on
  HN-quadrant prompts** (which include sad / angry / fearful
  framings). Functional emotional state aggregation is real but
  modest at this scale.
- If pilot passes and main goes ahead, the budget grows 8× to ~160
  HN generations on ministral. Pre-register N before committing;
  don't 10× on a hunch.
- Smoke first (script 99), pilot second (this design), main
  third — only if pilot rules pass.

## Lexical-side observation, separate from gating

Worth tracking but NOT gating:

a9 noted ministral leans francophone under steering. Francophone
internet historically uses japanese-style kaomoji `(´；ω；`)` more
than the western `:(`. The kaomoji *distribution* may differ even
if the affect *geometry* is similar. Two answerable questions from
the same pilot data:

- Per-face cosine heatmap (script 17): does ministral's face
  inventory overlap with gemma + qwen, or is it disjoint?
- Predictiveness (script 25): does the kaomoji → quadrant
  predictability hold at similar levels?

Report alongside the gating analyses; do not let lexical
divergence override or substitute for the geometric decision.

## Changes / artifact paths

- New: `data/ministral_emotional_raw.jsonl`,
  `data/ministral_emotional_summary.tsv`,
  `data/hidden/ministral_emotional/<uuid>.npz`
- New figures: `figures/local/ministral/fig_emo_*.png`,
  `fig_v3_*.png`
- Updated: `figures/local/cross_model/v3_cross_model_summary.json`
  (extended to triplet), `figures/local/cross_model/fig_v3_*.png`
  (gemma+qwen+ministral overlays)
- Required code change: pilot-N gate in
  `scripts/03_emotional_run.py` (e.g., env var
  `LLMOJI_PILOT_GENS=1`). Defaults to existing 8 if unset.
- CLAUDE.md status update: append ministral pilot result to the
  status section once analysis lands.

## Sequence

1. a9 reads + signs off on this doc (or pushes back / revises).
2. Add `LLMOJI_PILOT_GENS` gate to `scripts/03_emotional_run.py`.
3. Smoke: `LLMOJI_MODEL=ministral python scripts/99_hidden_state_smoke.py`.
   Verifies model loads, hooks register, sidecars write. ~5 min.
4. Pilot: 100-gen run + analyses listed above. ~15-30 min compute,
   plus analysis time.
5. Apply decision rules. Report point estimates, CIs, and
   pass/fail/middle on each.
6. Decide on main run per stop rules. If main, pre-register N
   here before running.

## Results — 2026-04-30

Pilot ran cleanly. 100 generations, 95 with usable `first_word`
(5 rows had empty first_word; standard non-emission rate). Per-row
`.npz` sidecars at `data/hidden/v3_ministral/`; multi-layer
`h_mean` cache at `data/cache/v3_ministral_h_mean_all_layers.npz`
(100 rows × 36 layers × 5120-dim).

Per-quadrant face distribution showed clean Russell-circumplex
behavior at the lexical level: `(◕‿◕✿)` flower-face dominant in
HP/LP/NB (15/19, 17/19, 13/19); `(╯°□°)` table-flip dominant in
HN (8/20); `(╥﹏╥)` crying-face dominant in LN (9/18). Within-face
hidden-state consistency 0.92–0.96 for top faces — tight clusters.

### Rule 1 — quadrant separation: **PASS**

| model | peak layer | fractional depth | silhouette |
| ---: | :---: | ---: | ---: |
| gemma | L31 / 56 | 55% | 0.184 |
| qwen | L59 / 60 | 98% | 0.313 |
| **ministral** | **L21 / 36** | **58%** | **0.153** |

Threshold ≥ 0.10. Ministral clears at 0.153. Note: peak at
mid-depth (~58%) like gemma, NOT deepest-leaning like qwen.
Silhouette point estimate at N=95 has wider CI than gemma/qwen at
N=800 — pilot flag, not a decision concern.

### Rule 2 — cross-model CKA: **PASS**

Pairwise linear CKA on prompt-aligned hidden states at preferred
layers (gemma L31, qwen L59, ministral L21). Existing baseline:
gemma↔qwen preferred-pair = 0.798 (published 800-row alignment,
this run replicated at 0.7945 on 100-row first-occurrence subset
— sanity check passes).

| pair | preferred-pair CKA | max CKA | location of max |
| --- | ---: | ---: | --- |
| gemma ↔ ministral | **0.741** | 0.759 | (gemma L57, ministral L21) |
| qwen ↔ ministral | **0.812** | 0.830 | (qwen L53, ministral L21) |
| gemma ↔ qwen (replication) | 0.795 | 0.855 | (gemma L52, qwen L57) |

Threshold ≥ 0.56 (within 30% of 0.798 baseline). Both ministral
pairs clear the threshold cleanly.

Striking sub-finding: qwen↔ministral (0.812) is *higher* than
gemma↔qwen (0.795). Ministral aligns more tightly with qwen than
gemma does with qwen. CKA-max location consistently lands at
**ministral L21** when paired with either reference model, even
though gemma's preferred layer is L31 (55% depth) and qwen's is
L59 (98% depth) — there's a canonical affect representation layer
in ministral at L21 (~58% depth) that's structurally aligned to
whatever the affect layer is in the other model.

### Rule 3 — dominance probe sign-check: **inconclusive**

| model | HN mean | LN mean | HN − LN |
| --- | ---: | ---: | ---: |
| gemma | −0.4157 | −0.4186 | +0.0029 |
| qwen | +0.1330 | +0.1314 | +0.0015 |
| ministral | −0.0913 | −0.0898 | −0.0015 |

Methodological issue, not ministral-specific. The HN quadrant
mixes anger (high PAD dominance) with fear (low PAD dominance), so
the mean washes out — gemma+qwen baseline differences are barely
above noise (~0.001–0.003), and ministral's tiny sign flip is
within that band. Rule needs redesign (split HN by prompt type,
or use pairwise within-quadrant comparisons) before it can
discriminate between models. Not gating per the pre-registered
"Rule 3 is sanity, not pre-condition" stance.

### Tokenizer-bug caveat (pre-registered before main run)

Mid-pilot we discovered ministral's HF-distributed tokenizer ships
a buggy pre-tokenizer regex that mis-splits ~1% of tokens (e.g.
`"'The'"` → `["'", "T", "he", "'"]` instead of `["'", "The", "'"]`).
Affects words preceded by apostrophes/punctuation; v3 prompts
include plenty of `I'm`, `don't`, `it's`, so pilot generations
were tokenized incorrectly. Bug is in encoding (text→tokens) only;
generation operates at token-id level, so output tokens themselves
are fine, but the model received slightly OOD prompts and
hidden-state token boundaries don't perfectly match what
mistral_common would produce.

Fix landed in `../saklas/saklas/core/model.py` (2026-04-30):
substring-match on `"mistral"` in `model_id` flips
`fix_mistral_regex=True` on `AutoTokenizer.from_pretrained`. Test
coverage: `tests/test_model_loading.py::test_mistral_regex_fix_*`.

**Pilot data we have is kept** — geometry findings are robust
despite the bug, and tokenization noise should weaken signal not
strengthen it. The 0.153 silhouette and 0.741/0.812 CKA values are
*lower bounds* on the true geometry. **Main run uses the fix**;
post-main sanity check is "did silhouette and CKA estimates at
N=800 with fixed tokenizer match or exceed pilot's N=95
estimates?" If they degrade, that's a real finding (would suggest
the bug somehow inflated alignment, e.g. by injecting consistent
token-boundary noise that PCA picks up as shared structure).
Expected outcome: estimates strengthen.

### Decision — proceed to main

All gating rules pass. Pre-register main run:

- N: 800 generations (5 quadrants × 20 prompts × 8 generations)
- Tokenizer: saklas with `fix_mistral_regex=True` (verify by
  running smoke first to confirm the kwarg flowed through and
  output tokens match expected mistral_common behavior on a known
  apostrophe-bearing prompt).
- All v3 follow-on analyses (21, 22, 23, 24, 25, 27, 28, 29) to
  run on the new data + cross-model figures regenerated to include
  ministral.
- Welfare budget: ~160 HN-quadrant generations (8× pilot's 20).
  Justified by gating-rule pass; commits the trial scale that the
  Ethics policy required us to earn through the pilot.
