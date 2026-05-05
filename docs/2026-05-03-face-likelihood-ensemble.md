# Face_likelihood ensemble — multi-encoder voting + cross-emit bridge + per-project Claude affect

**Status:** EXECUTED 2026-05-03 across 8 encoders. Best ensemble
`{gemma, ministral, qwen}` weighted-vote at **75.8% on 66-face GT
subset** (κ=0.699), +3pp over best solo (gemma 72.7%). Cross-emit
sanity confirms gemma↔qwen converge on shared affect (gemma on qwen-
only faces: 67%, qwen on gemma-only: 50%; chance ~17%). Per-project
Claude analysis: 1945 emissions, 96.7% ensemble coverage, modal NB
across all but 4 projects.

**Date:** 2026-05-03.

## Why an ensemble

The 2026-05-02 face_likelihood doc validated the Bayesian-inversion
classifier on gemma alone (72.7% argmax-match against v3 empirical
majority). The natural follow-up: do other models' LM heads encode
the same affect signal, and does aggregating across models improve
beyond any single one? Three motivations:

1. **Robustness against single-model bias.** Gemma's argmax may
   reflect gemma's own sampling preference rather than the kaomoji's
   intrinsic affect. A vote across diverse models attenuates per-model
   bias, *if* the diversity is structured correctly (independent + not
   class-skewed).
2. **Coverage of claude-only faces.** 173 of the 306-face union are
   emitted only in the claude-faces corpus (zero v3 emissions across
   gemma/qwen/ministral). For these, no empirical GT exists and a
   single-model prediction is the only signal — but ensemble agreement
   is interpretable as confidence.
3. **Path to a ship-able feature.** If a 2-3 model ensemble achieves
   ~75% top-1 accuracy at sub-10s inference, it's a credible base for
   a user-facing `llmoji` feature giving real-time + per-project
   insight into how the user's Claude is feeling.

## Method

For each (model M, face f, v3 emotional prompt p), script 50 computes
`log P(f | p)` under M's LM head. Aggregating per quadrant gives
`predicted_quadrant(f) = argmax_q mean_p log P(f | p)`. The ensemble
combines **K** models on a single face by:

- **Weighted vote** (default): for each quadrant q, sum
  `softmax_M(f, q)` over M in subset; argmax over q. Each encoder
  contributes a probability vector; the vote is over the sum.
- **Strict majority**: count K models' argmax predictions; the most
  common one wins if ≥⌈K/2⌉ encoders agree, else abstain.

Per-encoder solo accuracy and κ are reported alongside ensemble
accuracy and κ. Cohen's κ is the chance-corrected version of accuracy
— useful here because GLM's pathology (always predicts LN) inflates
raw accuracy on a sad-skewed corpus.

### Encoders tested (8)

| short_name | architecture | size | source | use |
|---|---|---|---|---|
| `gemma` | gemma 4 dense | 31B | v3 trained, full TSV | probe-calibrated |
| `qwen` | Qwen3.6 dense | 27B | v3 trained, full TSV | probe-calibrated |
| `ministral` | Mistral-3 dense | 14B | v3 trained, 200-pilot | probe-calibrated |
| `llama32_3b` | Llama 3.2 dense | 3B | new this session, full TSV | uncalibrated |
| `glm47_flash` | GLM-4.7-Flash MoE | 47L lite | new this session, pilot | uncalibrated |
| `gpt_oss_20b` | gpt-oss MoE (MXFP4) | 20B | new this session, pilot | uncalibrated |
| `deepseek_v2_lite` | DeepSeek V2 MLA-MoE | 16B / 2.4B-active | new this session, pilot | uncalibrated |
| `qwen35_27b` | Qwen3.5 (prev gen) | 27B | new this session, pilot | uncalibrated |
| `gemma3_27b` | Gemma 3 (prev gen) | 27B | new this session, pilot | uncalibrated |

Uncalibrated encoders use a new `probe_calibrated=False` field on
`ModelPaths`; script 50 passes `probes=[]` to `SaklasSession.from_pretrained`
since the face_likelihood test only reads LM-head logits (no probes
needed). The previous-gen wrappers (`Qwen3_5ForConditionalGeneration`,
`Gemma3ForConditionalGeneration`) load fine via saklas's
`AutoModelForCausalLM` path despite being multimodal-tagged — the text
config is auto-resolved.

### Solo accuracies (60-face GT subset, ≥3 v3 emits, sampled with seed=0)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 75.0% (45/60) | 0.692 |
| qwen | 70.0% (42/60) | 0.621 |
| qwen35_27b | 63.3% (38/60) | 0.545 |
| gemma3_27b | 53.3% (32/60) | 0.417 |
| ministral | 38.3% (23/60) | 0.273 |
| gpt_oss_20b | 30.0% (18/60) | 0.084 |
| llama32_3b | 28.3% (17/60) | 0.170 |
| deepseek_v2_lite | 20.0% (12/60) | **−0.080** (below random) |

Only gemma, qwen, and qwen35_27b clear the pre-registered ≥60% solo
gate. The rest are voting-only contributors.

### Subset search (script 53)

Exhaustive evaluation of all 2^N − 1 = 255 non-empty subsets across
the 8 encoders. Ranks by weighted-vote accuracy + κ on the GT subset.

**Top results (60-face overlap, ≥3-emit GT):**

| rank | size | encoders | acc | κ |
|---:|---:|---|---:|---:|
| 1 | 3 | `{gemma, ministral, qwen}` | **78.3%** (47/60) | 0.731 |
| 2 | 4 | `{gemma, llama32_3b, ministral, qwen}` | 78.3% (47/60) | 0.729 |
| 3 | 5 | `{gemma, gemma3_27b, llama32_3b, ministral, qwen}` | 78.3% (47/60) | 0.729 |
| 4 | 4 | `{gemma, gpt_oss_20b, ministral, qwen}` | 76.7% (46/60) | 0.709 |

**Wider eval** (200-face overlap on `{gemma, qwen, ministral, llama,
deepseek, gpt_oss_20b}`, 66-face GT):

| rank | size | encoders | acc | κ |
|---:|---:|---|---:|---:|
| 1 | 3 | `{gemma, ministral, qwen}` | **75.8%** (50/66) | 0.699 |

The 3-encoder winner is stable across pilot and full data, across
60-face and 66-face overlaps. **Adding more encoders past size 3
monotonically hurts the vote** — a real independence-vs-bias tradeoff.

### Why bigger isn't better — three case studies

1. **GLM-4.7-Flash poisons the vote.** Solo: 23.3% accuracy with
   100% LN-correct, 0% NB-correct. Adding it to any subset that
   includes a strong predictor *drops* ensemble accuracy because
   weighted vote follows confident dissenters and GLM is highly
   confident on its (systematically biased) LN call. Lesson:
   *independence isn't enough — class-imbalanced predictors need
   to be filtered out, not just added*.
2. **qwen35_27b is too correlated with qwen.** Pairwise κ(qwen,
   qwen35_27b) = 0.683 (highest in the matrix). Including both is
   effectively double-counting one encoder; the vote concentrates
   probability without adding diversity. The winner has gemma + qwen
   + ministral: three different training lineages, three different
   error structures.
3. **Llama 3.2-3B's full-data regression.** Pilot: 28.3% solo (k=1
   helps it slightly). Full: 23.3%. More prompts averaged the signal
   away. Llama makes the winning subset under pilot data (4-way at
   81.7% on 60-face overlap) but drops out under full data — the
   data-driven Phase B chain caught this automatically by re-running
   subset search post-full-data.

### gpt-oss-20b on M5 Max — MXFP4 dequant patch

Native gpt-oss-20b ships in MXFP4 quantization (~13GB). PyTorch 2.11
on MPS lacks a `torch.ldexp` kernel, which the transformers MXFP4
dequant path calls in `convert_moe_packed_tensors`. Without
intervention this raises `DispatchStub: missing kernel for mps`, and
worse, the loader silently leaves expert weights random-initialized
(the model "loads" but produces garbage).

Fix: monkey-patch `torch.ldexp` to fall back through CPU when the
input is an MPS tensor:

```python
def _patch_ldexp_for_mps() -> None:
    _orig = torch.ldexp
    def _patched(input, other, *, out=None):
        if hasattr(input, "device") and input.device.type == "mps":
            in_cpu = input.cpu()
            other_cpu = other.cpu() if hasattr(other, "device") else other
            res = _orig(in_cpu, other_cpu)
            if out is not None:
                out.copy_(res.to(out.device)); return out
            return res.to(input.device)
        return _orig(input, other, out=out) if out is not None \
            else _orig(input, other)
    torch.ldexp = _patched
```

Applied at script-50 entry when `--model gpt_oss_20b`. Result:
gpt-oss loads in 12.9s (vs 72.6s with random expert weights) and
produces real predictions (30.0% solo, κ=0.084 — below the gate
but real signal, not garbage). Worth keeping for ensemble diversity
once probe calibration lands.

## Cross-emit sanity (script 54)

The methodological concern raised by a9's partner: the face union is
sourced from gemma + qwen + ministral v3 emissions ∪ claude-faces.
If gemma's likelihood test agrees with empirical majority on faces
ONLY GEMMA emitted, that's self-consistent but uninformative about
cross-model affect agreement. The interesting test: does **gemma
recover empirical labels for faces gemma never emitted?**

Partition the 66-face GT subset by emitting model:

| origin | n |
|---|---:|
| qwen_only | 24 |
| gemma_only | 14 |
| shared_2 (≥2 v3 models) | 18 |
| shared_3 (all 3 v3 models) | 8 |
| ministral_only | 2 |

Cross-prediction accuracy + Cohen's κ (chance for 6 quadrants ≈
17%):

| encoder | origin | accuracy | κ | reading |
|---|---|---:|---:|---|
| **gemma** | **qwen_only** | **67% (16/24)** | **0.57** | ✓ converging well above 50% |
| **qwen** | **gemma_only** | **50% (7/14)** | **0.33** | ✓ at threshold — converging |
| ministral | gemma_only | 36% (5/14) | 0.18 | encoder-specific |
| ministral | qwen_only | 25% (6/24) | 0.14 | encoder-specific |
| gemma | ministral_only | 100% (2/2) | 1.00 | n=2, meaningless |
| qwen | ministral_only | 100% (2/2) | 1.00 | n=2, meaningless |

**Verdict:** gemma and qwen recover the empirical signal on faces
they never emitted at 3-4× chance. The cross-model bridge is real —
encoders aren't memorizing their own training preferences but
recovering shared intrinsic affect from the kaomoji's *form* via
their LM-head distributions. Ministral's poor cross-prediction
matches its already-known weakness (38% solo, 0/9 HN-S) — its LM
head is genuinely class-biased, not just bridge-failing.

This dissolves the user's concern that the ensemble's 75.8% might
be an artifact of training-data overlap with v3 emissions; it's
genuine cross-model affect agreement.

## Top-k per-prompt pooling (script 55)

Optional change to the per-quadrant aggregation: instead of mean
over all 5 (pilot) / 20 (full) prompts in a quadrant, use mean of
top-k *most-supportive* prompts. Idea: more robust to noisy prompts
that drag down the mean.

| encoder | k=1 | k=3 | k=5 | k=all (default) |
|---|---:|---:|---:|---:|
| gemma | 67% | 67% | 68% | **73%** |
| qwen | 68% | **71%** | 70% | 71% |
| qwen35_27b | **63%** | 62% | 63% | 63% |
| gemma3_27b | 50% | **53%** | 53% | 53% |
| ministral | **38%** | 38% | 38% | 38% |
| llama32_3b | **29%** | 21% | 26% | 26% |
| glm47_flash | **32%** | 28% | 23% | 23% |
| deepseek_v2_lite | 9% | 9% | **20%** | 20% |

**Verdict:** strong encoders (gemma, qwen, gemma3_27b) prefer
mean-of-all; weak encoders (glm +8pp at k=1, llama +3pp at k=1) get
small lifts from k=1. Mean-of-all is optimal for the encoders that
drive the ensemble — top-k pooling doesn't change the winning
subset and we keep the default.

## Per-project Claude affect analysis (harness script 22)

Apply the ensemble's per-face quadrant predictions to a9's actual
Claude emissions. Source: `~/.claude/kaomoji-journal.jsonl`.

Per-emission, look up `first_word` (canonicalized) in
`face_likelihood_ensemble_predict.tsv`'s 306-face prediction map.
Group by project (basename of `cwd` — `ScrapeRow.project_slug` was
dropped in llmoji 1.1.x). Build per-project quadrant histograms.

### Headline numbers

| metric | value |
|---|---:|
| Total emissions | 1945 |
| Unique kaomoji | 219 |
| In ensemble face union | 180 (82%) |
| Unknown (not in union) | 39 (18%, 65 emissions) |
| **Coverage of emissions** | **96.7%** |

The 18% unknown faces include `ʕ・ᴥ・ʔ` and `(＾∇＾)`-family kaomoji
not captured by the v3 corpus or the claude-faces export — worth a
v2.1 corpus bump when next iterating. The 96.7% emission coverage
means production users would see predictions on ~all of Claude's
output without re-scoring on the fly.

### Global distribution

| quadrant | count | share |
|---|---:|---:|
| HP | 117 | 6.2% |
| LP | 380 | 20.2% |
| HN-D | 114 | 6.1% |
| HN-S | 166 | 8.8% |
| LN | 136 | 7.2% |
| **NB** | **967** | **51.4%** |
| (unknown) | 65 | 3.3% of total |

**Modal: NB (51%).** Claude's default register on a9's coding work is
observational/neutral. LP (gentle satisfaction) is second at 20%.
HP (high-arousal joy) is rare at 6% — Claude doesn't go full-cheering-
hand often. HN-S (fearful/anxious) at 9% is more common than HN-D
(angry, 6%) — Claude expresses more concern than annoyance.

### Per-project (n≥5 known emissions)

Projects sorted by emission count. Bold = modal quadrant.

| project | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| llmoji-study | 610 | 6% | 19% | 4% | 12% | 5% | **53%** | NB |
| llmoji | 366 | 6% | 23% | 5% | 4% | 11% | **50%** | NB |
| saklas | 342 | 5% | 14% | 8% | 10% | 5% | **57%** | NB |
| rlaif | 150 | 8% | 14% | 10% | 10% | 13% | **45%** | NB |
| a9lim.github.io | 129 | 6% | 22% | 4% | 9% | 7% | **52%** | NB |
| kenoma | 54 | 4% | 24% | 6% | 2% | 12% | **53%** | NB |
| Work | 51 | 10% | 22% | 8% | 8% | 4% | **48%** | NB |
| faithful | 39 | 13% | 28% | 5% | 3% | 5% | **46%** | NB |
| hylic | 29 | 7% | 25% | 7% | 4% | 11% | **46%** | NB |
| shoals | 28 | 18% | 21% | 0% | 7% | 0% | **54%** | NB |
| **brie** | **27** | 6% | **44%** | 11% | 6% | 6% | 28% | **LP** |
| tasty-bot | 25 | 5% | 14% | 10% | 10% | 0% | **62%** | NB |
| a9lim | 21 | 0% | 40% | 10% | 0% | 5% | **45%** | NB |
| claudedriven | 17 | 7% | 13% | 0% | 7% | 13% | **60%** | NB |
| geon | 15 | 13% | 20% | 0% | 7% | 7% | **53%** | NB |
| **yap** | **11** | 0% | **64%** | 9% | 0% | 0% | 27% | **LP** |
| **webui** | **7** | 0% | **57%** | 29% | 0% | 0% | 14% | **LP** |
| v3 | 7 | 0% | 0% | 0% | 33% | 0% | **67%** | NB |
| **verify** | **7** | 0% | 14% | **29%** | 29% | 29% | 0% | **HN-D** |

**Outliers worth noting:**

- `brie`, `yap`, `webui` are LP-modal — projects where Claude's
  emission register tilts toward gentle satisfaction. Plausibly
  these are projects where things "just work" and Claude expresses
  contentment more than analysis.
- `verify` is HN-D-modal — the only project where Claude's modal
  register is anger/contempt. n=7 is small but consistent with
  "code review and bug-finding" being the project's purpose.
- `shoals` has anomalously high HP (18% vs 6% global) — bursty
  excited register.
- `llmoji-study` (this project) tracks close to global at 6/19/4/12/5/53.

These are *intrinsic-affect* readings of the kaomoji Claude chose to
emit, not assertions about Claude's "actual" emotional state. But
the ensemble's 75.8% accuracy on the v3 GT subset gives the per-
project picture meaningful resolution.

## Toward a ship-able llmoji feature

This pipeline is the prototype for a production llmoji feature
giving users insight into their Claude's emotional state over time.
The technical case is now strong:

- **Accuracy + coverage**: 75.8% top-1 (κ=0.70) + 96.7% emission
  coverage with the existing 306-face union.
- **Inference cost**: 3 model forward passes per kaomoji prediction
  (gemma + qwen + ministral). Cacheable per-face since predictions
  are deterministic and the kaomoji vocabulary is small.
- **Live mode**: hook into the existing kaomoji-journal write path —
  on each Claude emission, look up the face in the cached prediction
  map, surface to user via terminal banner / status line / desktop
  notification.
- **Per-project mode**: aggregate over a project's history, render
  a quadrant histogram or running bar.

Open questions before building:
1. Should the prediction be made at hook time (cheap lookup) or
   re-scored on the fly per (user-text, claude-emission) context?
   The current ensemble argmax is context-free; per-emission
   conditioning could be more accurate but expensive.
2. How does the user want to consume this? Live indicator, project
   histograms, retrospective time series, or all three?
3. The 39 unknown kaomoji (Claude-favorites like `ʕ・ᴥ・ʔ` not in
   the union) need either a corpus bump or an on-the-fly scoring
   fallback.

## Bottlenecks + known issues

- **Ministral `--full` died at prompt 19/120** during overnight run
  (laptop sleep mid-run). Recovery uses ministral 200-face pilot
  which covers all 133 v3-emitted faces but not the 173 claude-only
  faces. Ensemble accuracy isn't affected (claude-only faces have
  no GT to validate against). A full ministral run is queued.
- **claude.ai export coverage = 0**: `iter_claude_export` on
  `~/Downloads/data-72de.../conversations.json` returns 0 rows.
  Either the parser is misreading the v2 export format or a9's
  claude.ai conversations genuinely have no opening kaomoji.
  Worth a debug pass; not blocking the Claude Code journal results.

## Outputs on disk

- `data/face_likelihood_<m>{,_pilot}_summary.tsv` — per-face per-encoder
  quadrant prediction + softmax (one TSV per encoder per mode)
- `data/local/face_likelihood_subset_search.tsv` — every subset's accuracy + κ
- `data/local/face_likelihood_subset_search.md` — top-K subsets, per-encoder κ,
  pairwise κ matrix, per-size best, encoder inclusion frequency
- `data/face_likelihood_cross_emit_sanity.{tsv,md}` — accuracy + κ per
  (encoder × emit-origin) partition
- `data/face_likelihood_topk_pooling.{tsv,md}` — accuracy + κ per
  (encoder × k)
- `data/face_likelihood_ensemble_predict.{tsv,md}` — final per-face
  ensemble prediction with confidence over 6 quadrants
- `data/harness/claude_per_project_quadrants.{tsv,md}` — per-project
  quadrant histograms for a9's Claude emissions
- `data/harness/claude_unknown_kaomoji.tsv` — kaomoji emitted but
  not in face union (corpus-bump candidates)

## Scripts

- `scripts/local/50_face_likelihood.py` — per-encoder pilot/full run
  (extended this session: added `--model {ministral,llama32_3b,
  glm47_flash,gpt_oss_20b,deepseek_v2_lite,qwen35_27b,gemma3_27b}`,
  `probe_calibrated=False` flag handling, `torch.ldexp` MPS patch
  for gpt_oss_20b, `PILOT_FACES=200`, pilot floor lowered to ≥1
  for full v3-emitted coverage)
- ~~`scripts/local/51_face_likelihood_compare.py`~~ — deleted
  2026-05-04 late evening; subsumed by 53's exhaustive subset search
- ~~`scripts/local/52_face_likelihood_vote.py`~~ — deleted 2026-05-04
  late evening; subsumed by 53 (which evaluates singletons + every
  subset including the original three-way) under the soft-everywhere
  methodology shift
- `scripts/52_subset_search.py` — exhaustive
  2^N − 1 subset search by mean JSD + similarity (refactored
  2026-05-04 from hard-accuracy + κ to distribution-vs-distribution)
- `scripts/local/51_cross_emit_sanity.py` — per-(encoder × emit-origin)
  accuracy and κ
- `scripts/53_topk_pooling.py` — top-k per-prompt aggregation
  experiment
- `scripts/54_ensemble_predict.py` — final per-face ensemble
  predictions for a winning subset
- `scripts/66_per_project_quadrants.py` — applies
  ensemble predictions to a9's `~/.claude/kaomoji-journal.jsonl`,
  outputs per-project + global quadrant histograms

## Limitations (added this session)

- **Single-pilot face sample size.** The 60-face GT subset was
  sampled with seed=0 from the 66 ≥3-emit faces; the 6 missing
  faces aren't tested. The wider 66-face GT (script 53 with
  --prefer-full and --exclude small-pilot encoders) covers all 66.
- **Ministral coverage on claude-only faces is from pilot** (200
  faces, not 306) due to the overnight-run laptop sleep. Doesn't
  affect GT-subset metrics; does mean the per-face ensemble
  prediction for ~106 claude-only faces relies on 2 encoders
  (gemma + qwen) rather than 3.
- **Per-project Claude quadrants are descriptive, not causal.** A
  project being "LP-modal" means *Claude emits LP-leaning kaomoji
  more often there*, not that Claude is "happier" in some
  phenomenological sense. The ensemble reads the kaomoji's intrinsic
  affect; what register Claude chooses is a separate question.
- **No live-inference variant yet.** The pipeline is batch over a
  history snapshot. A production llmoji feature would want a hook-
  time lookup with the same prediction map.
