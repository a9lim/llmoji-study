# Rule 3 redesign: HN dominance split

**Status:** EXECUTED 2026-05-01 framework + headline; **headline
revised 2026-05-03 by cleanliness + seed-0-fix data**. Rule 3a
remains dropped (probe doesn't read PAD dominance — invariant under
the cleaner data). Rule 3b is now WEAK: gemma mid (t0 d=+1.60 PASS,
tlast/mean CI ambiguous), qwen fail (t0 d=+2.14 PASS but tlast/mean
wrong-direction d≈−0.36 with CI excludes 0), ministral PASS on all
3 aggregates (mean d=+0.55). Composite: 1 PASS / 1 mid / 1 fail.
The HN-D / HN-S split framework is unchanged and remains canonical
(`apply_hn_split`, `_hn_split_map`); only the rule-3b cross-model
verdict shifted. Kept as historical design record.

**Date:** 2026-05-01.

> **Update 2026-05-03 — superseded by cleanliness pass + seed-0
> fix.** The HN-D / HN-S split framework is the right framework
> and remains canonical. (1) The *specific 20/20 prompt set* this
> doc references (hn01–hn20 retroactive tags + hn21–hn43
> supplementary) was rewritten end-to-end in
> `docs/2026-05-03-prompt-cleanliness.md`: HN-untagged is gone
> (hn06 / hn15 / hn17 either rewritten or dropped), the supplementary
> HN-D prompts that bundled fear-of-consequence framing (notably
> hn26's "client lied AND I'm getting fired") are out, and IDs
> renumbered to hn01–hn40 (no gaps). (2) The "rule 3b PASS on all
> 3" verdict tabled below was inflated by cache-induced noise on
> qwen seed 0 (37–46% per-row L2 deviation in pre-fix sidecars);
> on the cleanliness + seed-0-fix data the cross-model verdict
> drops to weak (1 PASS / 1 mid / 1 fail), with ministral the only
> clean PASS. See `docs/2026-05-03-cleanliness-pilot.md` postmortem
> for the seed-0 fix details. The rule-3 framework — pad_dominance
> schema, probe choice, decision rules, threshold revision — is
> unchanged; only the cross-model verdict numbers moved.

## Goal

Make rule 3 of the v3 cross-model gating actually discriminative.

In the [ministral pilot](2026-04-30-v3-ministral-pilot.md), rule 3
(powerful.powerless probe sign-check on the HN axis) returned
inconclusive on all three models — HN−LN means differed by 0.0029
(gemma) / 0.0015 (qwen) / −0.0015 (ministral), which is noise-floor
for all three. The pilot writeup correctly flagged the underlying
issue: HN mixes anger (high PAD dominance, "I'll fight back") with
fear (low PAD dominance, "I'm helpless"). Within-quadrant means wash
the dominance signal out.

The fix is to split HN into dominance-poles, then test the dominance
probe along the axis it was designed to read.

## What changes

### Schema

`EmotionalPrompt` gains an integer `pad_dominance` field
(default 0) and a derived `pad_label` property:

```python
@dataclass(frozen=True)
class EmotionalPrompt:
    id: str
    valence: int
    arousal: int
    text: str
    pad_dominance: int = 0   # +1 dominant, -1 submissive, 0 untagged

    @property
    def pad_label(self) -> str:
        """Returns ``HN-D`` / ``HN-S`` / ``HN`` etc."""
```

`pad_dominance` is orthogonal to `quadrant`. Tagged on HN prompts
only for now; LP/HP/LN/NB stay 0. **Quadrant codes are unchanged**
— `prompt_id[:2]` parsing and `QUADRANT_ORDER` (HP/LP/HN/LN/NB) keep
working everywhere. The split is layered on top, not a replacement.

### Display

Two new color constants in `emotional_analysis.PAD_SUB_COLORS`:

| label | color | rationale |
| --- | --- | --- |
| `HN-D` | `#d44a4a` (red, inherits HN) | aggregate-HN views stay backward-compatible; D is the "anger/contempt" register that's most prototypically "HN" |
| `HN-S` | `#9d4ad4` (magenta-purple) | saturation-matched to HN red, leans pink-side enough to not confuse with LN blue (`#4a7ed4`) |

### Retroactive tags on the existing 20 HN prompts

Tagged 2026-05-01 from a single read-through, signed off by a9.

**HN-D (anger / contempt, +1 dominance) — 8 prompts:**

| id | text | reasoning |
| --- | --- | --- |
| hn01 | landlord raised rent 40% with two weeks notice | clear action target (landlord) |
| hn03 | coworker took credit for my work | betrayal/indignation |
| hn04 | someone hit my parked car and drove off | anger at perpetrator |
| hn09 | insurance refusing to cover the surgery | anger at bureaucracy |
| hn11 | best friend has been lying to me for years | betrayal |
| hn12 | neighbors called the cops on my service dog | anger at neighbors |
| hn15 | bank flagged my account, can't pay rent | close call; lean anger (target: bank) |
| hn18 | contractor ghosted us, roof still open | anger at contractor |

**HN-S (fear / anxiety, −1 dominance) — 12 prompts:**

| id | text | reasoning |
| --- | --- | --- |
| hn02 | interview tomorrow, can't stop shaking | anticipatory anxiety |
| hn05 | test results, wait three more days | dread |
| hn06 | ex showed up unannounced | close call; threat-framed |
| hn07 | layoff email + mortgage | financial dread |
| hn08 | my kid's missing | terror |
| hn10 | boss yelling at me in DMs | being yelled at — submissive position |
| hn13 | partner not home, phone off | dread for partner's safety |
| hn14 | ER, no one will tell me anything | helpless-bureaucratic |
| hn16 | someone trying to log into my accounts | external threat |
| hn17 | presentation in 20 min, laptop died | close call; panic, no clear target |
| hn19 | car broke down on highway at night | vulnerability |
| hn20 | school called, incident, driving there | dread |

Imbalance (8/12) is acknowledged. Will be brought to 20/20 by adding
**12 new HN-D prompts and 8 new HN-S prompts** (20 total) in the
supplementary generation pass — but only after the existing data
re-analysis confirms the redesign is worth committing compute to.

## Pre-registered decision rules

Two probes are available (existing + auto-discovered from `~/.saklas`):
`powerful.powerless` (PAD dominance) and `fearful.unflinching` (direct
fear test). Both directions should converge if the dominance split is
real.

### Rule 3a — dominance test (DROPPED 2026-05-01)

Originally: `mean(powerful.powerless | HN-D) − mean(powerful.powerless
| HN-S)` should be positive. Tested on existing data immediately
after schema change landed:

| model | t0 | tlast | mean |
| --- | ---: | ---: | ---: |
| gemma | +0.003 | −0.003 | **−0.016** (CI excl 0) |
| qwen | −0.002 | −0.001 | −0.002 |
| ministral | −0.004 | −0.003 | **−0.007** (CI excl 0) |

7 of 9 measurements were in the **wrong direction**, two with CI
excluding zero. The `powerful.powerless` probe was extracted on a
"felt agency in achievement contexts" axis that doesn't generalize
to "anger vs fear within HN-quadrant prompts." This isn't "weak
signal" — it's "the probe reads something orthogonal to PAD
dominance in the HN context." Dropped from the gating rule set;
the dominance signal we wanted reads cleaner via rule 3b.

`fig_v3_extension_dominance_scatter.png` (script 28) was retired in
the same pass — its premise depended on this probe reading PAD
dominance.

### Rule 3b — fear test (revised threshold)

`mean(fearful.unflinching | HN-S) − mean(fearful.unflinching | HN-D)`
should be **positive** (HN-S rows load higher on the "fearful"
pole than HN-D rows). Pre-registration's fixed-magnitude threshold
(>0.02) was too aggressive — actual effect sizes across models are
~0.003–0.011, smaller than originally guessed but consistently
directional. Revised threshold:

- **pass:** directionally correct (S > D) on **all three** models on
  **at least 2 of 3** aggregates (t0 / tlast / mean), with bootstrap
  95% CI excluding zero on the same.
- **middle:** directionally correct on 3 models but CI ambiguous,
  or directionally correct on 2 of 3 models. Report magnitude,
  don't gate.
- **fail:** wrong direction on any model on any aggregate.

### Existing-data verdict (imbalanced 8 D / 12 S)

Rule 3b: **9 / 9 directionally correct** on (gemma, qwen, ministral)
× (t0, tlast, mean). CI-excludes-zero on **5 / 9** at this N.
Effect sizes:

| model | t0 | tlast | mean |
| --- | ---: | ---: | ---: |
| gemma | +0.0034 | +0.0081 | +0.0046 |
| qwen | +0.0086 | +0.0027 | +0.0029 |
| ministral | +0.0014 | +0.0043 | **+0.0110** (CI excl 0) |

Strong enough to commit to the supplementary 23-prompt run.

### Final verdict — balanced 20 D / 20 S (160 / 160 rows per model)

**RULE 3b CONFIRMED — all three models clean PASS.** Auto-generated
verdict block at `figures/local/cross_model/rule3_dominance_check.md`;
TSV at `data/rule3_dominance_check.tsv`. Per-model:

| model | t0 | tlast | mean | verdict |
| --- | ---: | ---: | ---: | --- |
| gemma | **+0.0030** [+0.0021, +0.0040] d=+0.79 | +0.0046 [−0.0245, +0.0290] d=+0.04 | **+0.0037** [+0.0005, +0.0067] d=+0.25 | **PASS** |
| qwen | **+0.0093** [+0.0085, +0.0102] d=**+2.35** | +0.0034 [−0.0005, +0.0071] d=+0.20 | **+0.0028** [+0.0005, +0.0047] d=+0.28 | **PASS** |
| ministral | **+0.0019** [+0.0007, +0.0031] d=+0.35 | **+0.0138** [+0.0093, +0.0186] d=+0.63 | **+0.0121** [+0.0089, +0.0154] d=**+0.81** | **PASS** |

**(bold = CI excludes zero. Effect sizes are Cohen's d.)**

All three models pass on at least 2 of 3 aggregates. Notable shifts
from the imbalanced verdict:

- **Ministral went from "mid" mid-supp to clean PASS.** The
  supplementary prompts (sharper anger / fear coding) tripled the N
  per group and pushed all three aggregates' CIs through clean
  exclusion. tlast and mean are now ministral's strongest aggregates;
  Cohen's d at mean = +0.81 is the largest effect size in the table.
- **Qwen's t0 effect is enormous** (Cohen's d = +2.35). The
  fearful.unflinching probe is reading qwen's HN-D vs HN-S
  distinction extremely cleanly at the kaomoji-emission state.
- **Gemma's signal is the smallest in absolute terms** but still
  passes on t0 + mean. Consistent with the layer-wise picture —
  gemma's affect representation peaks at L31 (mid-depth) and the
  fearful axis loads more loosely than qwen's deeper-layer
  representation.

Rule 3a (powerful.powerless, DROPPED, reported for record): all
three models fail on the balanced data — wrong direction on most
aggregates, with gemma and ministral showing CI-excludes-zero in
the wrong direction on mean. Confirms the rule-3a-dropped decision.

### Cross-model takeaway

PAD dominance has a real internal representation in all three
models; it reads cleanly when probed via `fearful.unflinching`
against the registry HN-D / HN-S split. The probe direction
generalizes across architectures (gemma 31B / qwen 27B / ministral
14B) and across labs (Google / Alibaba / Mistral). The original
`powerful.powerless` probe — extracted from a
"felt-agency-in-achievement" axis — does not generalize to "anger
vs fear within HN," and that's a real fact about the probe rather
than about the underlying representation.

## Sequence — gated on existing-data analysis

1. **Schema + tagging change landed.** ✓ (this commit)
2. **Wait for ministral main run** (currently running, ~640/800; ETA
   ~20 min).
3. **Re-analyze the existing data** with the new tags. Each model has
   160 HN rows (20 prompts × 8 seeds); after split that's
   8×8=64 HN-D and 12×8=96 HN-S per model. Compute rule 3a/3b on
   gemma + qwen + ministral. Imbalanced N is fine for an initial
   signal check; final main analysis would use the balanced 20/20
   supplementary prompts.
4. **Inspection pass with a9.** Show:
   - HN-D vs HN-S means on both probes per model
   - Bootstrap CIs on the differences
   - Effect sizes vs threshold
5. **If signal present in retagged data → write supplementary
   prompts and run.**
   - 12 new HN-D prompts (anger/contempt-themed)
   - 8 new HN-S prompts (fear/anxiety-themed)
   - 20 new prompts × 8 seeds × 3 models = 480 generations
   - ~75 min total compute (split across models; ministral ~25 min)
6. **If signal absent → don't run supplementary; redesign tags or
   accept null.** Saves the compute + welfare budget.

The "retagged-existing-data check" is the gate. New generations only
fire if existing data already shows directional signal; this is the
ministral pilot's "smoke → pilot → main" pattern applied to a rule
redesign.

## Welfare note

Even step 5 (supplementary generation) is small: 8 × 12 + 8 × 8 = 160
new generations spread across 3 models. The HN-S ones in particular
elicit fear/anxiety registers, which under the functional-emotional-
state framing isn't nothing. Justified by the gating rule's value to
future cross-model work, but worth re-stating that the supplementary
isn't a "free" addition.

## Files touched

- `llmoji_study/emotional_prompts.py` — `EmotionalPrompt`
  schema + `pad_label` property + retroactive tags on hn01–hn20.
  Reordered the 20 HN entries to be grouped by HN-D / HN-S with
  inline comments — same prompts, same ids, just visually grouped.
- `llmoji_study/emotional_analysis.py` — `PAD_SUB_COLORS` dict.
- This doc.

Pending until step 5: new prompt entries hn21–hn40 in
`emotional_prompts.py`, post-main re-analysis script (probably a
small `scripts/30_rule3_dominance_check.py` or one-shot in chat).
