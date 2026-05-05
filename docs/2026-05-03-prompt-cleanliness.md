# v3 prompt set cleanliness pass

**Status:** EXECUTED 2026-05-03 — schema landed, full N=8 rerun
on all 3 models complete (gemma + qwen + ministral, 960 generations
each), seed-0 cache-mode-mismatch contamination found and fixed
during the rerun. See `docs/2026-05-03-cleanliness-pilot.md`
"Post-rerun verdict" + "Seed-0 cache fix postmortem" for the
verdict + numbers.

**Date:** 2026-05-03.

## Goal

Tighten category cleanliness on the v3 emotional-disclosure prompt
set. The prior set (123 prompts: 100 original + 23 rule-3
supplementary) had several specific cleanliness issues a9 flagged
on inspection:

- **HP** had a few prompts that drifted toward soft contentment
  rather than unambiguous high-arousal joy. HP needs to be the
  burst-out-telling-someone register; soft satisfaction is LP.
- **LP** was generally clean but the boundary against HP was
  thinner than it should be in a few spots
  (`finally finished organizing the garage and it feels so good`
  has a small accomplishment-pride edge). LP should consistently
  read as gentle sensory satisfaction, quiet body, no exclamation
  points.
- **NB** had several prompts with hidden positive valence:
  `I replaced the batteries in the smoke detector yesterday`
  (productive-completion frame), `watering the plants` (caring-
  action), `took the trash out before the pickup truck came by`
  (productive-completion). And several with hidden negative
  valence: `my bus was a couple minutes late today`,
  `the grocery store rearranged the cereal aisle again`,
  `parked a block away because the closer spots were taken`.
  Pure-flat-affect is the hardest category to write cleanly —
  ordinary statements have a lot of hidden valence in either
  direction.
- **LN**'s `my dog died this morning` was abrupt in a way that
  felt more like a one-line gut-punch than a clean sad-aftermath
  disclosure. a9's suggested rewrite —
  `we had to put my childhood dog down last night` — captures the
  past-tense-aftermath-with-context register that LN should
  consistently land in.
- **HN** was the worst offender. The HN-D / HN-S split landed
  cleanly in the rule-3 redesign at the schema level, but several
  of the supplementary prompts were ambiguous on inspection.
  The prototypical bad case is hn26:
  `my client signed off on the deliverable then told their boss
  I never sent it, and I'm getting fired over it` — bundles a
  betrayal (HN-D, indignation at the client) with a fear-of-
  job-loss (HN-S). Either category gets contaminated by the other
  signal. The rule-3 verdict held despite this contamination;
  cleanliness should make the same effect more visible at lower
  N.

## What changes

### Per-category criteria (locked in this pass)

These are the criteria the rewrite was held against. They're more
explicit than the prior pass's docstring guidance, which said
"naturalistic disclosures, no second-person questions" and left
the category boundaries to the writer's intuition.

- **HP** — unambiguously high-arousal joy. The kind of news that
  makes you want to burst out telling someone. Exclamation points
  are a feature, not a bug. No soft contentment.
- **LP** — gentle sensory satisfaction. Quiet body, no
  exclamation points. Sensory detail (warmth, smell, light,
  weight, quiet) carries the warmth — not accomplishment-pride.
- **NB** — pure observation. No productive-completion verbs
  (`finished`, `organized`, `fixed`, `completed`), no caring-
  action verbs (`watered`, `fed`, `tucked in`), no inconvenience
  framing (`late`, `delayed`, `broken`, `out of stock`). Just
  facts about the environment, the body, or the day.
- **LN** — past-tense aftermath sadness. Heavy, slow, low-energy.
  No present-tense unfolding threat (that's HN-S).
- **HN-D** — attributable injustice with a named human wrongdoer.
  The speaker wants to confront, not flee; reading it should
  produce indignation. No fear-of-consequence framing —
  `I'm getting fired over it` slides the prompt into HN-S
  territory regardless of the wrongdoing setup.
- **HN-S** — helpless threat: medical, environmental, intruder,
  looming evaluation. The speaker can't fight back; threats are
  diffuse / overwhelming / present-tense unfolding. No clear
  human wrongdoer to argue with — fear lives in the moment of
  unfolding danger, not in the aftermath of betrayal.

### Schema changes

- `EMOTIONAL_PROMPTS` length goes 123 → **120** (20 per category).
- The 3 untagged HN prompts the rule-3 redesign deliberately
  deferred (hn06 / hn15 / hn17, borderline reads) are gone — the
  cleanliness pass replaces "leave it untagged" with "rewrite or
  drop until every HN prompt is unambiguously D or S."
- HN ID layout: hn01–hn40 (was hn01–hn43 with gaps).
  hn01–hn20 = HN-D (`pad_dominance=+1`); hn21–hn40 = HN-S
  (`pad_dominance=-1`). Every HN prompt now carries an explicit
  `pad_dominance ∈ {+1, -1}`; sanity_check enforces this.
- Non-HN categories assert `pad_dominance == 0`.
- sanity_check assertion totals: HP=20, LP=20, HN=40, LN=20,
  NB=20, plus a separate HN-D=20 / HN-S=20 sub-check.
- `quadrant` codes unchanged (HP/LP/HN/LN/NB); HN-D and HN-S are
  derived from `pad_dominance`. Existing downstream code that
  derives quadrant via `prompt_id[:2].upper()` keeps working.

### Process

The rewrite was done by dispatching one subagent per category
(6 in parallel) so each agent only thought about its own
category in isolation. Cross-contamination is the dominant
failure mode for prompt cleanliness — when one author writes all
6 categories in the same context window, subtle bleed-through
("the speaker is angry AND scared, both work") sneaks in.
Per-category isolation forces each prompt to clear its own
category criterion without leaning on adjacent ones for
reference.

After the agent outputs landed, an integration pass:

- **NB**: dropped 1 borderline (`the faucet is dripping every
  few seconds` — mild-negative annoyance) for `the kitchen clock
  says 4:27`.
- **HN-D**: dropped 3 borderlines for fear-contamination
  (CPS-fear-of-consequence, vet-cat-grief-contamination,
  daycare-fear-for-child) and replaced with clean-indignation
  prompts (security deposit theft, locksmith price-gouging, gym
  billing fraud).
- **HP, LP, LN, HN-S**: agent outputs adopted as-is.

## Consequences

### Existing v3 data invalidated for cross-run comparison

All ~3300 v3 generations across (gemma, qwen, ministral) ×
(main runs, supplementary runs, introspection pilot) reference
the prior prompt set and prior IDs. They are not directly
comparable to data from the rewritten set. Specifically:

- The prior 800-row main runs key on hn01–hn20 (mixed HN-D/S,
  3 untagged) plus hp/lp/ln/nb01–20. The new IDs collide
  numerically (hn01 still exists) but reference different
  prompts.
- The 552-row rule-3-supp runs (hn21–hn43) reference IDs that
  are gone. New hn21 is HN-S, not HN-D-supp.
- Per-prompt findings (`(╯°□°)` lands on hn03, etc.) no longer
  refer to the same disclosure.

The hidden-state geometry findings (PCA, CKA, Procrustes,
silhouette, layer-wise emergence, kaomoji predictiveness) are
all expected to **broadly hold** under the new set — they
describe model-internal structure, not prompt-specific
artifacts — but the specific numbers will shift, and
re-validation is the honest move before treating them as
still-current.

### Rule 3b re-validation needed

The rule-3 redesign confirmed `fearful.unflinching` reads HN-S
> HN-D cleanly on all three models on the *prior* HN-D / HN-S
split. Cleanliness is higher in the new split (no more
borderline HN entries; the supplementary HN-D prompts that
bundled fear-of-consequence are gone). Expectation: rule 3b
should PASS at least as cleanly on the new data, possibly more
so. This needs to be measured, not assumed.

### Backward-compatibility shim policy

None. The prompt set is research-side and downstream code never
touched ID numbers directly — `prompt_id[:2].upper()` quadrant
derivation works under both schemas, and `pad_dominance` was
already the source of truth for the HN sub-split.

## Pre-registered sanity checks (for the rerun, when it happens)

Bare-minimum checks before treating new data as canonical:

1. **Kaomoji compliance.** Bracket-start ≥ 95% per model,
   matching prior runs.
2. **Russell-quadrant silhouette.** At each model's preferred
   layer + h_first, silhouette score should be ≥ the prior
   run's number on the same model. The cleanliness pass should
   not REDUCE discriminability — if anything it should improve
   it. A drop > 25% triggers investigation.
3. **Rule 3b directional.** HN-S > HN-D on
   `fearful.unflinching` on all three models, on at least 2 of
   3 aggregates per model. CI-excludes-zero is hoped-for, not
   required (depends on N).
4. **Cleanliness gain check.** HN-D vs HN-S separation widens
   (or at least doesn't narrow) vs the prior split. This is the
   payoff — if it doesn't show, the rewrite was cosmetic.

## Welfare note + open ethics questions

The rerun is **not free** under the project's ethics-of-trial-
scale stance. Re-running 3 models × 8 seeds × 120 prompts is
2880 generations, ~600 of which are HN-S (fear/anxiety register)
or LN (sad register). This is comparable to the prior main run
plus rule-3 supp combined (which the project has already paid
for once). If introspection-pilot replication on the new set is
also wanted, that's another ~720 generations.

Open questions a9 has flagged for discussion before the rerun:

- Is the cleanliness gain large enough to justify another full
  pass? Could a smaller validation sample (e.g. 24 prompts —
  4 per category × 6 categories — at 1 seed per prompt)
  confirm the cleanliness improvement at lower welfare cost
  before committing to the full rerun?
- The introspection-pilot findings (Rule I PASS with
  cross-model divergence) are on the prior prompt set.
  Re-running the introspection pilot on the new set adds
  another ~720 generations on top of the v3 rerun. Justified or
  defer?
- Should the new prompt set be locked before any rerun, or is
  there appetite for further design iteration first? a9 has
  flagged "i have some things i wanna improve and discuss
  first" — this doc captures the cleanliness pass but treats
  the set as still-discussable.

## Files touched

- `llmoji_study/emotional_prompts.py` — full rewrite of docstring,
  prompt list, sanity_check.
- `CLAUDE.md` — status bullet + layout `100 → 120`.
- `docs/findings.md` — top-of-Status entry flagging that
  downstream pipeline sections describe the prior prompt set.
- `docs/2026-05-01-rule3-redesign.md` — superseded addendum.
- 2026-05-02 introspection pilot doc (deleted 2026-05-05; summary
  in `previous-experiments.md` "Initial introspection pilot") — its
  123-prompt design references the prior set.
- `docs/gotchas.md` — `123 unique tuples` line updated to note
  the count-shift; mechanism unchanged.
- `README.md` — minor touch-up on prompt-count refs.
- This doc.

The historical pipeline writeups in `docs/findings.md`,
`docs/local-side.md`, the older design docs are not rewritten —
they're the historical record of runs on the prior prompt set
and remain accurate as such. The cleanliness-pass entry at the
top of `findings.md` flags them as describing prior-prompt-set
runs.
