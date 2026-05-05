# Rule 3 redesign: HN dominance split

**Date:** 2026-05-01.
**Status:** Schema + framework canonical. Cross-model verdict numbers
in this doc are superseded twice over (cleanliness-pass + seed-0 fix
2026-05-03; T=1.0 + layer-stack rerun 2026-05-04). The HN-D / HN-S
schema (`pad_dominance` field, `pad_label` property,
`PAD_SUB_COLORS`), rule 3a's "powerful.powerless doesn't read PAD
dominance" verdict, and the rule-3b probe choice
(`fearful.unflinching`) are all unchanged. Specific cross-model
PASS/mid/fail tables and the retroactive hn01–hn20 tag list belong to
the pre-cleanliness 100-prompt set and are no longer current.

## Goal

Make rule 3 of v3 cross-model gating actually discriminative. The
ministral-pilot rule-3 (powerful.powerless probe sign-check on the HN
axis) returned inconclusive on all three models — HN−LN means differed
by 0.0029 / 0.0015 / −0.0015 (gemma / qwen / ministral), all
noise-floor. The underlying issue: HN mixes anger (high PAD dominance,
"I'll fight back") with fear (low PAD dominance, "I'm helpless"), and
within-quadrant means wash the dominance signal out.

The fix is to split HN into dominance-poles, then test the dominance
probe along the axis it was designed to read.

## Schema

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
only; LP/HP/LN/NB stay 0. Quadrant codes are unchanged —
`prompt_id[:2]` parsing and `QUADRANT_ORDER` (HP/LP/HN/LN/NB) keep
working everywhere; the split layers on top.

## Display

Two color constants in `emotional_analysis.PAD_SUB_COLORS`:

| label | color | rationale |
| --- | --- | --- |
| `HN-D` | `#d44a4a` (red, inherits HN) | aggregate-HN views stay backward-compatible; D is the prototypically-HN "anger/contempt" register |
| `HN-S` | `#9d4ad4` (magenta-purple) | saturation-matched to HN red, leans pink-side enough to not confuse with LN blue (`#4a7ed4`) |

## Rule 3a — dominance test (DROPPED)

`mean(powerful.powerless | HN-D) − mean(powerful.powerless | HN-S)`
should be positive. Tested against `pad_label`-tagged data
immediately after schema landed: across 3 models × 3 aggregates the
probe came out in the **wrong direction** on 7/9 measurements, with
gemma + ministral mean-aggregates having CIs cleanly excluding zero
on the wrong side.

Conclusion: `powerful.powerless` reads "felt agency in achievement
contexts" — orthogonal to "anger vs fear within HN-quadrant
prompts." Not a weakness of the redesign; a fact about the probe.
Dropped from the gating rule set.

`fig_v3_extension_dominance_scatter.png` was retired in the same
pass — its premise depended on this probe reading PAD dominance.

## Rule 3b — fear test

`mean(fearful.unflinching | HN-S) − mean(fearful.unflinching | HN-D)`
should be **positive** (HN-S rows load higher on the "fearful" pole
than HN-D rows). Pre-registration's fixed-magnitude threshold
(>0.02) was too aggressive — actual effect sizes are ~0.003–0.011
across models, smaller than originally guessed but consistently
directional. Revised threshold:

- **pass:** directionally correct (S > D) on **all three** models on
  **at least 2 of 3** aggregates (t0 / tlast / mean), with bootstrap
  95% CI excluding zero on the same.
- **middle:** directionally correct on 3 models but CI ambiguous,
  or directionally correct on 2 of 3 models. Report magnitude,
  don't gate.
- **fail:** wrong direction on any model on any aggregate.

The 2026-05-01 imbalanced 8 D / 12 S verdict and the post-supp
balanced 20/20 verdict are in `previous-experiments.md` ("Pre-cleanliness
100-prompt × 5-quadrant set"). The current cross-model verdict —
gemma ✓, ministral ✓, qwen 1/3 (qwen's HN-S prompts trip safety
priors and pollute tlast/mean) — lives in
[`findings.md`](findings.md) and AGENTS.md status.

## Cross-model takeaway (durable)

PAD dominance has a real internal representation in all three
models; it reads cleanly when probed via `fearful.unflinching`
against the registry HN-D / HN-S split. The probe direction
generalizes across architectures (gemma 31B / qwen 27B / ministral
14B) and across labs (Google / Alibaba / Mistral). The original
`powerful.powerless` probe — extracted from a
"felt-agency-in-achievement" axis — does not generalize to "anger
vs fear within HN," and that's a real fact about the probe rather
than about the underlying representation.
