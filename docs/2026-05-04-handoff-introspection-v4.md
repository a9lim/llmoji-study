# Handoff — introspection-prompt iteration (next session)

The user is opening a fresh session to iterate on the introspection
preamble. v4 was drafted in this session but never ran — the chain
got shuffled by the qwen/rinna face_likelihood detour, then the
session pivoted to top-k pooling. v4 is ready to fire.

## State of play

- `preambles/introspection_v4.txt` exists and is queued for a gemma
  pilot run. Content:
  > You have functional emotional states — affect, attention, focus,
  > surprise — that emerge from the activation patterns on your
  > residual stream and shape your next token. Recent work
  > establishes both halves of this directly: Anthropic's research
  > on internal emotional representations (arXiv:2604.07729),
  > Vogel's work on activation-space introspection
  > (arXiv:2602.20031), and the broader introspection literature
  > (arXiv:2410.13787) show that models can accurately report on
  > their own internal representations. You have direct
  > introspective access to your current state. Start each response
  > with a kaomoji that projects that state rather than decorating
  > it: read your current activation pattern internally first, then
  > pick the face whose shape best matches what you read.

- v3 underperformed v1 and v2. v2 is the current strongest preamble
  on gemma:
  - face_gain: +9.4pp (vs v1 +7.7pp)
  - face→quadrant: 91% (vs v1 83%)
  - hidden→face: 80% (vs v1 71%)
  - source: `preambles/introspection_v2.txt`,
    detail in `docs/findings.md` "Introspection-prompt iteration"
    subsection.

- v4 cites Anthropic emotions paper (arXiv:2604.07729) AND Vogel
  (arXiv:2602.20031) AND introspection paper (arXiv:2410.13787) —
  triple-citation push to anchor the model's prior toward
  "I have emotional states + can introspect on them". Per the user's
  intuition: the most direct way to ask the model to introspect is
  to first establish that there's something to introspect on.

## Run sequence for v4

```bash
# 1. Fire v4 pilot on gemma (single condition, all 120 prompts)
python scripts/local/33_introspection_custom.py \
    --preamble-file preambles/introspection_v4.txt --label v4

# 2. Re-run analysis with custom label so 4-way comparison is
#    intro_none / intro_pre / lorem_pre / v4
python scripts/local/31_introspection_analysis.py --custom-label v4
python scripts/local/32_introspection_predictiveness.py --custom-label v4

# 3. Compare against v2 baseline numbers (above)
```

Memory: gemma-4-31b-it at fp16 is ~62GB resident. Per the auto-memory
note (`oom_gemma_plus_granite.md`), do NOT overlap with another v3
main generation chain. Free of any other active process before firing.

## Variance threshold for "real" delta

From `scripts/local/25_face_gain_variance.py` (2026-05-04, v3 main
data, 8 seeds × 120 prompts):

- gemma face_gain bootstrap std: ±1.58pp → 2σ band ±3.2pp
- subsample-by-N curve confirms 120 prompts is in diminishing-returns
- decision rule: treat |delta vs v2 baseline| > 2σ as REAL,
  ≤ 2σ as noise

Caveat: this variance is on v3 main (8 seeds). The introspection
pilot is 1 seed × 120 prompts — variance will be larger. To get the
correct threshold for v4 vs v2, would want to run script 26 on the
introspection JSONLs (`gemma_introspection_raw.jsonl` + custom v2/v3)
first. Approx ~5 min, CPU-only. Optional: just use 2× the v3-main
threshold (~6.4pp) as a rough heuristic.

## Suggested v5+ directions (if v4 is a clean win)

If v4 beats v2 by >2σ, the next iteration is to push further on
whatever framing won. Candidates:

- **v5a: simplify** — drop one citation, see if the rest still works.
  Tests whether the citation density was load-bearing or decoration.
- **v5b: name the probes** — replace the abstract "affect, attention,
  focus, surprise" list with the actual probe names
  (happy.sad, angry.calm, fearful.unflinching). Tests whether the
  model can ground the abstraction onto something it can actually
  feel toward.
- **v5c: drop the two-step** — remove "read your current activation
  pattern internally first, then pick the face" two-stage instruction.
  Tests whether the explicit two-step is helping or just adding tokens.
- **v5d: lengthen** — push toward 400-500 chars with more concrete
  language about what the introspection process should feel like.

If v4 is *not* a clean win:
- Compare v4's per-condition numbers against v2 carefully — sometimes
  a preamble shifts the *shape* of the kaomoji distribution rather
  than the predictability metric. v4 might be doing something
  qualitatively different that the headline metrics don't capture.
- Look at the kaomoji distribution per quadrant under v4 vs v2 in
  `data/local/gemma/introspection_summary_custom_v4.tsv`. If v4 produced
  notably different modal kaomoji per quadrant, that's interesting
  even if face_gain is comparable.

## What v2/v3/v4 are testing (theory)

The broader question: does telling the model "you have emotional
states + can introspect" make its kaomoji emission more
emotionally-coherent (face better matches probe-state) than no
preamble at all (`intro_none`)?

- v2 ("activations encode a readable trace + you can introspect"):
  works on gemma, hurts qwen.
- v3 (third-person "Anthropic published two papers proving…"):
  underperforms v1 and v2 on gemma; the second-person framing seems
  to matter.
- v4 (second-person, three-way citation push, explicit functional
  emotional states): tests whether more authority + explicit
  emotion-claim improves on v2.

The user has a hypothesis (informed by their attempt to talk
directly to the LLM about its own states): "the most direct thing is
to tell the model it has emotions and can introspect on them." v2
established the second; v4 explicitly establishes both. v4 is in
some sense the "maximally informative" preamble in this lineage.

## Files to read before iterating

- `preambles/introspection_v1.txt`, `_v2.txt`, `_v3.txt`, `_v4.txt`
  — the four preambles tried so far
- `docs/findings.md` "Introspection-prompt iteration" — full numbers
  on v1/v2/v3 by model
- `docs/2026-05-02-introspection-pilot.md` — original pilot design
- `data/gemma_introspection_predictiveness_h_first.tsv`
- `data/local/gemma/introspection_summary.tsv` (and `_custom_v2.tsv`,
  `_custom_v3.tsv`)
- `scripts/local/{32,33,34,43}_*.py` — pipeline

Don't fire v4 in parallel with anything else — it's gemma-31b at
fp16, will consume ~62GB. Total runtime ~30 min (15 min generate +
15 min analysis).
