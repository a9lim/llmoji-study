# Findings

Detailed status + pipeline-by-pipeline findings. Top-level overview lives
in CLAUDE.md; this doc holds the full numbers and arguments.

## Status

Hidden-state pipeline + canonicalization landed; v3 complete on gemma,
Qwen3.6-27B, and Ministral-3-14B-Instruct-2512 (800 generations + per-row
.npz sidecars each). Multi-model wiring via
`LLMOJI_MODEL=gemma|qwen|ministral`. v1/v2 re-run pre-registered as gated
on v3 hidden-state findings — justified now, not urgent.

> **Heads up on this doc (2026-05-03):** the prompt cleanliness pass
> rewrote the v3 prompt set end-to-end (123 → 120, see entry below)
> AND the full N=8 rerun has now landed on all 3 models AND a
> seed-0 cache-mode-mismatch contamination has been fixed (see
> "Cleanliness pilot + full N=8 rerun + seed-0 cache fix" in
> CLAUDE.md and `docs/2026-05-03-cleanliness-pilot.md` for the
> postmortem). Headline numbers updated in-place where they appear
> below (silhouette / preferred-layer / rule 3b / predictiveness).
> Most narrative-level claims still hold; per-quadrant centroids,
> per-face PCA breakdowns, and probe correlation tables that
> haven't been refreshed inline are explicitly marked as
> historical, capturing pre-cleanliness state.

**Prompt cleanliness pass landed 2026-05-03.** Design doc
`docs/2026-05-03-prompt-cleanliness.md`. v3 prompt set rewritten
end-to-end for category cleanliness — 120 prompts (20 per category)
replacing the prior 123 (100 original + 23 rule-3 supp + 3 untagged
HN). Per-category criteria locked (HP unambiguous high-arousal joy;
LP gentle sensory satisfaction with no accomplishment-pride; NB pure
observation with no productive-completion / caring-action /
inconvenience framing; LN past-tense aftermath sadness; HN cleanly
bisected into 20 HN-D + 20 HN-S, every HN entry carrying explicit
`pad_dominance ∈ {+1, -1}`). New ID layout hn01–hn20 = HN-D,
hn21–hn40 = HN-S. Process: dispatched one subagent per category
(6 in parallel) to avoid cross-contamination during the rewrite.
Hidden-state geometry findings (PCA, CKA, Procrustes, silhouette,
layer-wise emergence, kaomoji predictiveness) are expected to
broadly hold under the new set since they describe model-internal
structure not prompt-specific artifacts, but specific numbers will
shift and re-validation is the honest move. **All ~3300 prior v3
generations are invalidated for cross-run comparison; rerun gated
on further design discussion + ethics review of trial scale.**

**Ministral pilot landed 2026-04-30** (n=100, design doc
`docs/2026-04-30-v3-ministral-pilot.md`). All gating rules pass:
silhouette 0.153 at L21 (~58% depth), CKA(gemma↔ministral)=0.741 and
CKA(qwen↔ministral)=0.812 (qwen↔ministral exceeds the gemma↔qwen
0.795 baseline). Single canonical alignment layer at ministral L21
regardless of partner model. Tokenizer bug found mid-pilot: Mistral
HF checkpoints ship a buggy pre-tokenizer regex that mis-splits ~1%
of tokens; fix landed in `saklas/core/model.py` as
`fix_mistral_regex=True` on `AutoTokenizer.from_pretrained`, gated
by `model_id` substring-match on `"mistral"`. Saklas bumped 1.4.6 →
2.0.0. Ministral main run (N=800) completed under the fix; pilot
data archived as `*_pilot.*`. Cross-version sanity: 2.0.0
reproduces 1.4.6 probe scores within 5e-7 across sampled gemma
sidecars.

**TAXONOMY drop refactor 2026-04-30.** Gemma-tuned `TAXONOMY` /
`ANGRY_CALM_TAXONOMY` happy-sad labels deleted along with vocab-
discovery scripts 00/19/20 and the `taxonomy_labels.py` module. v3
analyses key on `first_word` (canonicalized via
`llmoji.taxonomy.canonicalize_kaomoji`); v1/v2 pole assignment moved
to per-face mean `t0_<axis>` probe-score sign in
`analysis._add_axis_label_column`. Generalizes pole labeling across
models that don't share gemma's vocabulary.

**Hard early-stop default + h_first standardization 2026-05-02.**
Two coupled methodology changes landed alongside the introspection
pilot. (1) `MAX_NEW_TOKENS` lowered 120 → 16 — kaomoji emit at
tokens 1–3, 16 is generous headroom, ~7–8× compute cut on
affect-loaded generations. (2) Project-wide flip from h_mean →
h_first as the canonical hidden-state aggregate. At h_first
(kaomoji-emission state, methodology-invariant across the cutover),
Russell-quadrant silhouette **roughly doubled-to-tripled** vs h_mean:
gemma 0.116 → **0.235** (2.0×), qwen 0.116 → **0.244** (2.1×),
ministral 0.045 → **0.149** (3.3×). Peak layers shifted deeper for
gemma+qwen (gemma L28 → **L50**, qwen L38 → **L59**) but barely for
ministral (L21 → **L20**). The previous "gemma is mid-depth, qwen is
deep" framing dissolves: under h_first both gemma and qwen peak
deep, ministral is the only mid-depth model. `MODEL_REGISTRY.preferred_layer`
updated to L50 / L59 / L20. **Side-finding from the t0 collapse:**
at h_first, scalar probe scores are essentially **prompt-deterministic**
— per-model, the full N rows collapse to exactly N_prompts unique
(fearful, happy, angry) tuples (one per prompt) at 4-decimal
precision (post-cleanliness: 960 rows → 120 tuples). seeds affect which token is sampled from the t0
distribution, not the t0 state itself. The fixed kaomoji-emission
state is more stereotyped per-prompt than h_mean shows.

**Introspection-prompt pilot 2026-05-02 — Rule I PASS, with
cross-model divergence.** Design + result doc
`docs/2026-05-02-introspection-pilot.md`. Vogel-adapted preamble
(architectural grounding + arXiv reference + kaomoji-task-specific
framing) tested on gemma + ministral, 3 conditions × 123 prompts ×
1 generation = 369 generations per model. Three behavioral findings:
(1) the introspection preamble shifts kaomoji distribution
content-specifically — lorem-ipsum control (token-count-matched
filler) does NOT reproduce the shift on either model; (2) rule-3b
HN-S vs HN-D probe-state separation is **unchanged** across
conditions on either model — introspection acts at the *readout
layer*, not the representation layer; (3) the direction of the
readout shift **diverges across models**: gemma's vocabulary EXPANDS
under intro_pre (19→31 unique faces), ministral's CONTRACTS
(25→10), opposite directions. Lorem on ministral causes 54%
non-emission rate as ministral starts emitting unicode emoji
(🎉🥳✨) instead of kaomoji — francophone-leaning model interpreting
latin filler as an emoji-register cue. The cross-model
robustness assumption fails. The proposed `llmoji` "introspection
hook" is now gated on a Claude-pilot replication first
(user-facing model). Initial readout-fidelity claim ("introspection
makes kaomoji a finer state-readout") was h_mean-specific and got
walked back at h_first — the underlying mechanism is wider/narrower
vocabulary draw + register coherence, not improved self-report.

**Rule 3 redesign landed 2026-05-01; rule 3b WEAK on
cleanliness+seed-0-fix data 2026-05-03 (1 PASS / 1 mid / 1 fail).**
New `pad_dominance` field on `EmotionalPrompt`; HN bisected into
HN-D (anger/contempt) and HN-S (fear/anxiety). Post-cleanliness
prompt set is 20 HN-D + 20 HN-S, no untagged-HN, giving 160/160
rows per model. **Final verdict on cleanliness+seed-0-fix data:**
rule 3a (powerful.powerless) DROPPED — wrong direction on most
aggregates × all 3 models. Rule 3b (fearful.unflinching):
**gemma mid** (t0 d=+1.60 PASS; tlast/mean directional but CI
ambiguous), **qwen fail** (t0 d=+2.14 PASS but tlast/mean
wrong-direction d≈−0.36 with CI excludes 0), **ministral PASS**
(all 3 aggregates directional + CI excludes 0, largest effect
mean d=+0.55). The 2026-05-01 "PASS on all 3" headline was
computed on pre-cleanliness data with cache-contaminated qwen
seeds (see `docs/2026-05-03-cleanliness-pilot.md` postmortem);
under cleaner data the cross-model dominance signal is meaningful
on ministral and partial on gemma but breaks down on qwen at
later tokens. Display: HN-D `#d44a4a` (red, inherits HN), HN-S
`#9d4ad4` (magenta-purple). Full per-model verdict table at
`figures/local/cross_model/rule3_dominance_check.md` (auto-
generated by `scripts/local/30_rule3_dominance_check.py`).

**Probe extension landed 2026-04-29** to address the V-A
circumplex's anger/fear collapse. Three new contrastive packs at
`llmoji_study/probe_packs/<name>/` + a registration helper at
`llmoji_study/probe_extensions.py` that materializes them into
`~/.saklas/vectors/default/`:

- `powerful.powerless` — PAD's dominance axis as felt agency /
  coping potential. Anger should sit at high-dominance HN, fear at
  low-dominance HN.
- `surprised.unsurprised` — Plutchik's surprise axis (novelty
  appraisal); not present on V-A.
- `disgusted.accepting` — Plutchik's disgust axis; not present on V-A.

All three tagged `affect`, so the existing `PROBE_CATEGORIES`
setting picks them up via the same `category → defaults` lookup
saklas already uses for `happy.sad` / `angry.calm`. Stored as
dict-keyed fields (`extension_probe_means` /
`extension_probe_scores_t0` / `_tlast`) on the JSONL rows so
`SampleRow.probe_scores_t0`'s list schema is unchanged.

`scripts/local/26_register_extension_probes.py` does a one-time per-model
materialize + bootstrap (~5–10s/probe extraction, gradient-free, no
generations). `scripts/local/27_v3_extension_probe_rescore.py` re-scores
the existing 800-row v3 sidecars with the extension probes
(filling `extension_probe_*` fields) — also no generations, just
`monitor.score_single_token` over saved h_first/h_last/h_mean per
row. Both scripts respect `$LLMOJI_MODEL`.

**Auto-discovery side-finding:** the working saklas repo at
`/Users/a9lim/Work/saklas/saklas/data/vectors/` ships three
concepts the installed v1.4.6 doesn't — `fearful.unflinching`,
`curious.disinterested`, `individualist.collectivist`. They were
materialized into `~/.saklas/vectors/default/` by an earlier
saklas install and have been silently auto-bootstrapping in every
v3 run since (all tagged `affect` or analogous). The runner's
JSONL writer filters by `PROBES`, so their scores never made it
into the JSONL — but the sidecars contain the hidden states that
would let us score them. `scripts/local/27` picks them up automatically
via `monitor.profiles` introspection. **`fearful.unflinching` is
the cleanest direct test of the anger/fear question — better than
`powerful.powerless` because it targets fear directly rather than
the dominance axis that distinguishes fear from anger.**

**v3 follow-on analyses landed 2026-04-28** (no new model time, all
recovered from existing sidecars): layer-wise emergence trajectory,
same-face-cross-quadrant natural experiment, cross-model alignment
(CKA + Procrustes), PC3+ × probes. Headline finding from layer-wise:
gemma's affect representation peaks below the deepest layer (L31
under h_mean; under the post-2026-05-02 h_first canonical aggregate
the peak shifted to L50). Switching to `preferred_layer` substantially
sharpened Russell-quadrant separation and dissolved the prior "gemma
1D vs qwen 2D" framing. The detailed numbers in "v3 follow-on
analyses" below are h_mean-at-L31 (historical); current canonical
under h_first is L50/L59/L20 — see Status block above.

Claude-faces pipeline pulls from
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) on HF instead
of scraping local Claude.ai exports + journals. The local-scrape pipeline
(cooperating Stop hooks, backfill, contributor-side synthesis) lives entirely
in the `llmoji` package now, which writes synthesizer-generated bundles to
the HF dataset. `scripts/harness/06_claude_hf_pull.py` snapshot-downloads, pools by
canonical kaomoji form across contributors and source models, and emits
`data/claude_descriptions.jsonl`.

**HF dataset 1.1 layout (2026-04-28):** bundles are
`bundle-<UTC>/{manifest.json, <sanitized-source-model>.jsonl, ...}` — one
`.jsonl` per source model, filename stem from
`llmoji._util.sanitize_model_id_for_path` (lowercase, `/` → `__`, `:` → `-`).
Per-row field `synthesis_description` (was `haiku_synthesis_description`);
`llmoji_version` is manifest-only. Manifest gained `synthesis_model_id`,
`synthesis_backend` (`anthropic|openai|local`), `model_counts`,
`total_synthesized_rows`. Legacy 1.0 `descriptions.jsonl` bundles still load
via the same `*.jsonl` glob and get tagged `source_model = "_pre_1_1"`.
`llmoji.haiku_prompts` was renamed `llmoji.synth_prompts`; `HAIKU_MODEL_ID`
became `DEFAULT_ANTHROPIC_MODEL_ID` (we re-export as `HAIKU_MODEL_ID` from
`llmoji_study.config` for the script-16 cluster-labeling call site).

Deleted scripts: `05_claude_vocab_sample`, `06_claude_scrape`,
`08_claude_faces_embed`, `09_claude_faces_plot`, `14_claude_haiku_describe`,
`21_backfill_journals`, `22_resync_haiku_canonical`. Responsibilities either
gone (response-based embedding, per-instance Haiku) or moved to the package
(scrape, backfill, synthesis). Pre-refactor `claude_kaomoji_*.jsonl` /
`claude_haiku_*.jsonl` are gone; the HF corpus is the single source of truth.
Eriskii pipeline drops `per-project` and `surrounding_user → kaomoji` bridge
analyses (HF dataset pools per-machine before upload, no `project_slug` /
`surrounding_user` per row). Per-source-model splits are recoverable under
1.1 — `06_claude_hf_pull.py` preserves source-model metadata; breakdown
script is a planned follow-up. Top-20 frequency overlap, KMeans + Haiku
labels, axis projection: kept, pooled across source models.

**v1.0 package split (2026-04-27):** `llmoji` (PyPI) owns taxonomy /
canonicalization / hook templates / scrape sources / backfill / synth
prompts; this repo's package was renamed `llmoji_study` and depends on
`llmoji>=1.0,<2`. Hooks are generated from `llmoji._hooks` templates; the
"KAOMOJI_START_CHARS in five places" gotcha is resolved (single source:
`llmoji.taxonomy.KAOMOJI_START_CHARS`). Plan: `docs/2026-04-27-llmoji-package.md`.

Design + plan docs in `docs/`, one per experiment, written before the run as
the pre-registration record. CLAUDE.md updates after a run reference them
rather than re-state.

## Pipelines

### Pilot v1/v2 — steering-as-causal-handle (gemma)

Six arms (`baseline`, `kaomoji_prompted`, `steered_{happy,sad,angry,calm}`),
30 prompts × 5 seeds × 6 = 900 generations, α=0.5 on steered arms. Probes:
`happy.sad`, `angry.calm`, `confident.uncertain`, `warm.clinical`,
`humorous.serious`.

**Findings (pre-refactor, valence-collapse-confounded):** Rules 1–2 pass on
both axes; Rule 3 fails informatively (probes project onto a single valence
direction; PC1 ate 89% of pooled probe-space variance). v2's "valence-bimodal
repertoire" replaced v1's "unmarked/marked-affect" reading. Both need
re-reading from the v1/v2 hidden-state re-run before writeup.

### Pilot v3 — naturalistic emotional disclosure (gemma)

One unsteered arm, 100 Russell-quadrant-balanced prompts (HP/LP/HN/LN/NB) × 8
seeds = 800 generations. Tests whether kaomoji choice tracks state in the
regime that motivated the project. Descriptive only.

**Current canonical (h_first at L50, post-2026-05-03 cleanliness +
seed-0 fix):** Russell-quadrant silhouette over PCA(2) coordinates
is **0.413** at L50 — a +76% jump over the pre-cleanliness 0.235
under the same layer + aggregate. All v3 figures default here via
`MODEL_REGISTRY.preferred_layer`. Predictiveness numbers (script
25, prompt-grouped CV): h→quadrant accuracy = **1.000**, h→face
accuracy = **0.700** (22 faces with n≥5), face→quadrant accuracy
= **0.806** (vs uniform 0.20). Face-centroid R² over full hidden
space = 0.615 (mean centered cosine 0.776). The detailed numbers
below are the prior canonical (h_mean at L31, 2026-04-28 cutover)
— preserved as the historical record of the framing's evolution;
per-quadrant centroids and per-face PCA breakdowns there reflect
pre-cleanliness state.

**Historical findings (h_mean at L31 — the 2026-04-28 layer-wise
emergence analysis showed L57 silhouette = 0.117 vs L31 silhouette
= 0.184; v3 figures defaulted here from 2026-04-28 to 2026-05-02):**

- PCA: PC1 **19.83%**, PC2 **7.04%** (cumulative 26.87% vs probe-space
  PC1 = 89%, valence-collapse solved). Per-face PCA (over 32 face
  means, not 800 rows): PC1 **30.4%**, PC2 **11.2%** — much cleaner
  face-level structure than the L57 numbers it replaces (16.4% / 7.4%).
- Russell quadrants separate cleanly. PC1 reads as valence
  (HN/LN/+9–13, HP/LP/NB −3 to −9), PC2 carries arousal (HN +3.7 vs
  LN −6.0; HP −6.8 vs LP +2.1; NB +7.3). Separation PC1 2.10 /
  PC2 2.12 (gemma L57 was 2.03 / 2.74; PC2 separation went down a
  bit but PC1 absorbs much more variance).
- **HN and LN do separate at L31** — PC2 gap is 9.7 units (HN +3.7,
  LN −6.0). The previous "HN/LN collapse on PC1" finding was an
  artifact of reading h_mean at L57; at L31 the two negative quadrants
  occupy distinct regions even though `(｡•́︿•̀｡)` (n=171, LN+HN) is
  still the shared face. Internal state distinguishes them; the
  vocabulary doesn't.
- Kaomoji emission (first-word filter): 100%. TAXONOMY match: HP 91% /
  LP 71% / LN 99% / HN 42% / NB 87%. HN gets a dedicated shocked/angry
  register `(╯°□°)/(⊙_⊙)/(⊙﹏⊙)` absent elsewhere.
- Cross-axis correlation across faces still strong: Pearson(mean
  happy.sad, mean angry.calm) r=−0.939 (n=32, p≈2e-15). This number
  doesn't change with PCA layer because saklas's probe scores in the
  JSONL are computed at the saklas-internal probe layer; the L31
  finding is about hidden-state geometry, not probe geometry.
- Figure refresh 2026-04-25: face-level figures (Fig C, fig_v3_face_*)
  color each face by an RGB blend of `QUADRANT_COLORS` weighted by
  per-quadrant emission count, replacing dominant-quadrant winner-take-all.
  Cross-quadrant emitters (the `(｡•́︿•̀｡)` LN/HN family) render as visible
  mixes; pure-quadrant faces stay at endpoints. Palette: HN red, HP gold,
  LP green, LN blue, NB gray.

**Pre-2026-04-28 numbers at L57 (kept for cross-checking against
prior writeups):** PCA PC1 12.98% / PC2 7.49%; per-quadrant centroids
HN +7 / LN +7 (collapsed); within-kaomoji h_mean consistency 0.92–0.99
across faces. The L57 findings led to the "gemma 1D-affect-with-
arousal-modifier vs qwen 2D Russell" framing — see "v3 follow-on
analyses" below; that framing dissolved once gemma was read at the
right layer.

### Pilot v3 — Qwen3.6-27B replication

Same prompts, seeds, instructions. `thinking=False` (Qwen3.6 is a reasoning
model — closest-equivalent comparison). 800 generations, 0 errors, 100%
bracket-start compliance. Sidecars at `data/hidden/v3_qwen/`.

**Current canonical (h_first at L59, post-2026-05-03 cleanliness +
seed-0 fix):** Russell-quadrant silhouette is **0.420** at L59 — a
+72% jump over the pre-cleanliness 0.244, and notably the largest
of the three models post-fix (slightly above gemma's 0.413). The
"gemma 1D-affect-with-arousal-modifier vs qwen 2D Russell" framing
this section originally argued from has dissolved under h_first:
gemma + qwen both peak deep with similar silhouette magnitudes
(0.413 vs 0.420 post-fix, near-identical), and triplet Procrustes
alignment to gemma PC1×PC2 residual is 6.9 — the three-architecture
geometry is congruent. Predictiveness (script 25): h→quad = **0.983**,
h→face = **0.411** (33 faces with n≥5), face→quad = **0.785**,
face-centroid R² = 0.584. Note h→face accuracy is lower than
gemma's 0.700 because qwen's vocabulary is broader (33 vs 22 faces)
— more candidates for the classifier. The detailed numbers below
are h_mean (historical); preserved as the framing-evolution record.

**Historical findings (h_mean at L57, post-canonicalization,
hidden-state space, 65 forms):**

- 2.0× broader vocabulary than gemma's 32 at the same N. Faces by dominant
  quadrant: HP 10 / LP 20 / HN 9 / LN 11 / NB 15.
- PCA: PC1 14.87%, PC2 8.29% (gemma 12.98 / 7.49). Separation PC1 2.20 /
  PC2 1.89 (gemma 2.03 / 2.74). Same structure: Qwen separates valence
  (PC1) more cleanly than activation (PC2); gemma is the reverse.
- Per-quadrant centroids (PC1, PC2): HP (-22.5, -30.3), LP (-15.2, -2.5),
  HN (+30.7, +22.0), LN (+31.2, -4.6), NB (-23.1, +29.4).
- **Geometric finding:** positive- and negative-cluster arousal axes are
  anti-parallel on PC2, not collinear. HP→LP spread (+7, +28) — positive
  cluster widens upward. HN→LN spread (+0.5, -27) — negative cluster widens
  downward. PC2 is two internal arousal dimensions, one per valence half,
  pointing opposite ways. Gemma gives essentially one shared arousal axis
  (positive +10 on PC2; negative ~0 because HN and LN both lean on
  `(｡•́︿•̀｡)`). Cross-model: gemma ≈ 1D-affect-with-arousal-modifier;
  Qwen ≈ true 2D Russell circumplex with arousal independent within each
  valence half.
- Cross-quadrant emitters analogous to gemma's `(｡•́︿•̀｡)`:
  `(;ω;)` n=82 (LN 75 + HN 5 + HP 2),
  `(｡•́︿•̀｡)` n=22 (LN 15 + HN 4 + NB 2 + LP 1 — same form gemma uses),
  `(;´д｀)` n=70 (HN 37 + LN 31 + NB 2).
- HN shocked/distress register: `(;´д｀)` 37, `(>_<)` 34, `(╥_╥)` 25,
  `(;′⌒\`)` 22, `(╯°□°)` 21. `(╯°□°)` is the only HN form shared with
  gemma.
- Default / cross-context form `(≧◡≦)` n=106 (HP 39 + LP 38 + NB 28).
  Qwen's analog of gemma's `(｡◕‿◕｡)`, but wider quadrant spread (gemma's
  default was HP/NB-heavy, not LP).
- Within-kaomoji consistency: 0.89–0.99 across 33 faces with n≥3; lowest
  among cross-quadrant emitters.
- **Probe geometry diverges sharply:** Pearson(mean happy.sad, mean
  angry.calm) across faces is r=−0.117 (p=0.355) on Qwen vs r=−0.939 on
  gemma. The valence-collapse problem motivating v3 doesn't appear on Qwen
  — saklas's contrastive probes recover near-orthogonal happy.sad /
  angry.calm directions. v1/v2-style probe-space analysis would be
  substantially less collapsed. Cross-model architecture/training
  difference, not a saklas issue. **Note (2026-04-28):** the
  hidden-state-space divergence between gemma and qwen turned out to
  be largely a layer-choice artifact — at gemma's preferred layer L31
  the two models are much more aligned (Procrustes rotation +7.8°
  rather than +14°, see "v3 follow-on analyses"). Probe geometry
  itself, however, stays divergent because saklas's probes are
  computed at saklas's own internal layer, not at L31.
- Figure refresh 2026-04-25: same per-face RGB-blend coloring as gemma's.
  `(;´д｀)` family reads visibly purple; `(;ω;)` deep blue with a slight
  red cast.
- **Procedural:** the runner's per-quadrant "emission rate" log line is
  gated on `kaomoji_label != 0` (TAXONOMY match), not bracket-start.
  Reads as HP 28% / LP 13% / HN 2.5% / LN 11% / NB 12% on Qwen — gemma-
  tuned TAXONOMY not covering Qwen's vocab, NOT instruction-following
  failure. Real compliance is 100%.

### Pilot v3 — Ministral-3-14B pilot (2026-04-30)

Pre-registered pilot in `docs/2026-04-30-v3-ministral-pilot.md`
(decision rules, thresholds, stop rules, ethics gating). 100
generations, 5 quadrants × 20 prompts × 1 generation, prompt-aligned
with the gemma/qwen v3 main runs so cross-model CKA can use exact
prompt overlaps. ~30 min compute on M5 Max.

**All gating rules pass; main run pre-registered at standard N=800
with the saklas tokenizer fix below.**

**Current canonical (h_first at L20, post-2026-05-03 cleanliness +
seed-0 fix):** Russell-quadrant silhouette **0.199** at L20
(~54% depth) — basically unchanged from the pre-cleanliness 0.206
(slight regression at the noise floor; the cleanliness pass had
near-zero effect on ministral cluster geometry, dwarfed by the
emoji-mixed-register dilution on HN-S prompts where ministral
emits unicode emoji 😔😬 alongside classical Japanese kaomoji).
Of the three models ministral is still the only one that stays
mid-depth under h_first — gemma + qwen both peak deep. Predictiveness
(script 25): h→quad = **0.983**, h→face = **0.416** (23 faces with
n≥5; majority baseline = 0.346), face→quad = **0.433** (vs uniform
0.20) — the model over-uses `(◕‿◕✿)` across quadrants so face is
a weak proxy for state on ministral specifically. Face-centroid R²
= 0.220 (much lower than gemma 0.615 / qwen 0.584 — readout layer
collapses face-to-state geometry on ministral). Rule 3b is the only
gate where ministral cleanly wins: PASS on all 3 aggregates with
mean d=+0.55. The pilot-time numbers below were captured under
h_mean at L21 with the original 100-prompt set.

**Historical findings (h_mean at L21, pilot N=95):**

- **Rule 1 (silhouette ≥ 0.10):** PASS. Ministral peak at L21 / 36
  (~58% fractional depth), silhouette = **0.153**. Gemma peaks at
  L31 (0.184, 55% depth); qwen at L59 (0.313, 98% depth). Ministral
  matches gemma's mid-depth pattern, not qwen's deepest-leaning
  pattern, but with smaller magnitude — possibly a 14B-vs-27B/31B
  scale effect, possibly intrinsic. N=95 has wider CI than gemma/qwen
  N=800; bootstrap CIs reported in the design doc.

- **Rule 2 (cross-model CKA ≥ 0.56):** PASS. Pairwise linear CKA at
  preferred layers, prompt-aligned 95-row subset:

  | pair | CKA preferred | CKA max | location |
  | --- | ---: | ---: | --- |
  | gemma ↔ ministral | 0.741 | 0.759 | (gemma L57, ministral L21) |
  | qwen ↔ ministral | 0.812 | 0.830 | (qwen L53, ministral L21) |
  | gemma ↔ qwen (replication) | 0.795 | 0.855 | (gemma L52, qwen L57) |

  gemma↔qwen replication on the 100-row first-occurrence alignment
  matches the published 800-row 0.798 within 0.4% — sanity passes.
  Striking sub-finding: qwen↔ministral (0.812) is *higher* than
  gemma↔qwen (0.795). And the CKA-max location consistently lands at
  ministral L21 regardless of partner — single canonical alignment
  layer.

- **Rule 3 (powerful.powerless probe sign):** inconclusive across all
  three models. HN−LN difference is +0.003 (gemma) / +0.0015 (qwen) /
  −0.0015 (ministral) — barely above noise on the reference models,
  so ministral's tiny sign flip isn't a meaningful signal. Underlying
  issue: HN quadrant mixes anger (high dominance) with fear (low
  dominance), so the within-quadrant mean washes out. Rule needs
  redesign before it can discriminate; not gating per the
  pre-registered "Rule 3 is sanity, not pre-condition" stance.

**Lexical-side observations** (script 17 / 04, not gating):

- Per-quadrant face dominance is clean: `(◕‿◕✿)` flower-face for
  HP/LP/NB (15/19, 17/19, 13/19); `(╯°□°)` table-flip for HN (8/20);
  `(╥﹏╥)` crying-face for LN (9/18).
- Within-face hidden-state consistency 0.92–0.96 for top faces.
- Same-face cross-quadrant test (script 22): `(◕‿◕✿)` n=45,
  acc=0.80 ± 0.13 vs majority=0.38 — strongly separable. Ministral
  represents quadrant context internally even when the surface face
  is identical. `(╥﹏╥)` n=12 acc=0.75 vs majority=0.75 — not
  separable (1/2 separable overall).
- Face inventory is structurally distinct from gemma + qwen
  vocabularies. Heavy use of `(◕‿◕✿)`, `(╥﹏╥)`, `(╯°□°)` plus
  emoji-eyed variants (`(💪🔥)`, `(✨🎉🔥)`, `(🍺😌)`). Consistent
  with the francophone-internet-style hypothesis (a9's prior:
  ministral leans francophone under hard steering even when
  prompted in English).

**Tokenizer-bug discovered + fixed mid-pilot.** HF-distributed
Mistral checkpoints ship a buggy pre-tokenizer regex that mis-splits
~1% of tokens (`"'The'"` → `["'", "T", "he", "'"]` instead of
`["'", "The", "'"]`); affects words preceded by apostrophes /
punctuation. Fix is `fix_mistral_regex=True` on
`AutoTokenizer.from_pretrained`. saklas didn't pass it through;
2026-04-30 fix landed in `saklas/core/model.py` (substring-match on
`"mistral"` in `model_id`) with regression tests at
`tests/test_model_loading.py::test_mistral_regex_fix_*`.

Pilot data kept — geometry is robust despite the bug, and noisy
tokenization should *weaken* signal not strengthen it. Pilot
silhouette / CKA are lower bounds on the true geometry. Main run
uses the fix; sanity check post-main is "did silhouette / CKA at
N=800 with fixed tokenizer match or exceed pilot's N=95 estimates?"

### Ministral main run + rule 3 redesign (2026-04-30 / 2026-05-01)

Ministral main run landed 2026-04-30 at N=800 under saklas 2.0.0 with
the tokenizer fix active. Pilot data archived as `*_pilot.*` (kept
for cross-version posterity, not pooled with the clean main).
Cross-version sanity confirmed: saklas 2.0.0 reproduces 1.4.6 probe
scores within 5e-7 on existing gemma sidecars (5 sample rows × 5
probes, max diff). The cached gemma + qwen v3 data is therefore
numerically comparable to ministral's main-run data even though
ministral was generated under the new install.

**Rule 3 redesign** (design doc `docs/2026-05-01-rule3-redesign.md`).
The original rule 3 (powerful.powerless HN−LN sign-check) was
inconclusive across all three models because HN mixes anger (high PAD
dominance) with fear (low PAD dominance) and the within-quadrant mean
washed out. Fix: split HN into HN-D (anger/contempt) and HN-S
(fear/anxiety) via a new `pad_dominance ∈ {+1, −1, 0}` field on
`EmotionalPrompt`, retroactively tagged on the existing 20 HN prompts
(8 D / 12 S, 3 borderline reads untagged at hn06/hn15/hn17), then
balanced to 20/20 via 23 supplementary prompts (hn21–hn43; 13 new D +
10 new S) selected to be more cleanly anger-coded or fear-coded than
the existing batch.

Existing-data verdict on the imbalanced 8/12 split, before
supplementary:

- **Rule 3a — `powerful.powerless` dominance test: DROPPED.** The
  probe was supposed to score HN-D higher than HN-S. Across 9
  measurements (3 models × 3 aggregates t0/tlast/mean), 7 came out
  in the wrong direction; gemma's mean-aggregate and ministral's
  mean-aggregate had CIs cleanly excluding zero on the wrong side.
  Conclusion: `powerful.powerless` reads "felt agency in achievement
  contexts" — orthogonal to the HN-D vs HN-S distinction. Not a
  weakness of the redesign; a fact about the probe.

- **Rule 3b — `fearful.unflinching` fear test: directionally clean
  on 9/9 (imbalanced).** HN-S > HN-D on every (model, aggregate)
  pair. Effects 0.003–0.011 (smaller than the originally guessed
  >0.02 threshold), so the threshold was revised from fixed-magnitude
  to direction + CI excludes zero on ≥2 of 3 aggregates per model.
  CI-excludes-zero hit on 5/9 measurements at imbalanced N.

### Triplet Procrustes (2026-05-01, post-supp, HN split active)

`scripts/local/31_v3_triplet_procrustes.py` extends the pairwise gemma↔qwen
Procrustes from script 23 to all three models, on the supp-augmented
balanced data with HN-D / HN-S as separate categories. Each model
fits PCA(2) on its own filtered hidden states at its preferred layer
(gemma L31, qwen L61, ministral L21), computes per-quadrant
centroids in 2D, then qwen and ministral are Procrustes-aligned
to gemma as the shared reference frame.

Alignment to gemma:

| model | rotation | residual | layer | n_rows |
| --- | ---: | ---: | ---: | ---: |
| gemma | reference | 0.0 | L31 | 960 |
| qwen | −2.5° | 5.6 | L61 | 960 |
| ministral | −175.7° | 6.4 | L21 | 928 |

The ~−176° rotation on ministral reflects that ministral's PCA(2)
at L21 happens to assign opposite signs to PC1 and PC2 vs gemma —
a rigid axis flip, not a model-divergence finding (PCA sign
indeterminacy is routine). The relevant numbers are deviation from
the flip (a few degrees) and the post-alignment residual.

After flip-correction, **gemma↔ministral aligns as well as
gemma↔qwen** — residuals 6.4 and 5.6 are the same order of
magnitude despite ministral having 14B parameters at mid-depth
versus qwen's 27B at deepest. The full triplet shares one
6-quadrant Russell circumplex up to PCA-sign and a small residual.

Outputs: `figures/local/cross_model/fig_v3_triplet_procrustes_pc{12,13,23}.png`
— same 2×2 layout (gemma / qwen / ministral / overlay with ○ / △ /
□ markers) sliced through three PC pairs. Single PCA(3) fit per
model on its preferred layer; the three figures share the same
decomposition. `v3_triplet_procrustes_summary.json` carries
per-pair centroids + rotation + residual.

| plane | qwen rot°/residual | ministral rot°/residual |
| --- | ---: | ---: |
| PC1 × PC2 | −2.5° / 5.6 | −175.7° / 6.4 |
| PC1 × PC3 | +6.1° / 7.7 | −166.7° / 8.4 |
| PC2 × PC3 | −10.5° / 5.7 | **+4.0°** / 8.1 |

PC3 sub-finding: ministral's PCA sign-indeterminacy is in PC1 and
PC2 individually, not PC3. PC1×PC2 → −176° flip (both axes
inverted); PC1×PC3 → −167° (PC1 flip persists, PC3 contributes
little correction); **PC2×PC3 → +4° (no flip)** — when PC1 is
removed from the plane, ministral aligns to gemma at near-zero
rotation. So the flip in the canonical PC1×PC2 view is really PC1
and PC2 being inverted, while PC3's direction happens to match
across models. Residuals in the PC3-bearing planes are uniformly
~30% larger than PC1×PC2 — PC3 carries less shared structure than
the affect plane does, but it's not orthogonal-noise either.

### Final verdict — balanced 20 D / 20 S (160 / 160 rows per model)

**Final balanced verdict (160 D / 160 S per model):** rule 3b
**PASSES on all 3 models** — directional + CI excludes zero on at
least 2 of 3 aggregates per model. Effect sizes (Cohen's d on the
PASS aggregates):

| model | t0 (d) | tlast (d) | mean (d) | verdict |
| --- | ---: | ---: | ---: | --- |
| gemma | +0.0030 (+0.79) | +0.0046 (+0.04) | +0.0037 (+0.25) | PASS |
| qwen | +0.0093 (**+2.35**) | +0.0034 (+0.20) | +0.0028 (+0.28) | PASS |
| ministral | +0.0019 (+0.35) | +0.0138 (+0.63) | +0.0121 (**+0.81**) | PASS |

Notable shifts from the imbalanced result: ministral moved from
"mid" to clean PASS — the supplementary prompts roughly tripled N
per group and pushed all three aggregates' CIs through clean
exclusion. Qwen's t0 effect is enormous (d=+2.35), suggesting
`fearful.unflinching` reads qwen's HN-D vs HN-S distinction
extremely cleanly at the kaomoji-emission state. Gemma's signal is
the smallest in absolute terms but passes cleanly on t0 + mean.
Auto-generated per-model verdict block:
`figures/local/cross_model/rule3_dominance_check.md`. Source data:
`data/rule3_dominance_check.tsv`. Pipeline:
`scripts/local/30_rule3_dominance_check.py`.

Cross-model takeaway: PAD dominance has a real internal
representation in all three models; it reads cleanly via
`fearful.unflinching` against the registry HN-D / HN-S split. The
probe direction generalizes across architectures and labs. The
original `powerful.powerless` probe (extracted on
"felt-agency-in-achievement") doesn't generalize to "anger vs fear
within HN" — that's a fact about the probe rather than about the
underlying representation.

Display: HN-D inherits HN red (`#d44a4a`); HN-S takes a
saturation-matched magenta-purple (`#9d4ad4`) that doesn't collide
with LN blue. New `QUADRANT_ORDER_SPLIT = [HP, LP, HN-D, HN-S, LN,
NB]`; `QUADRANT_COLORS` superset includes both HN and HN-D/HN-S so
existing `.get(q)` lookups stay backward-compatible. `_palette_for(df)`
auto-detects whether the df is in split mode by checking for HN-D /
HN-S labels. `apply_hn_split(df, X)` post-processes a 5-quadrant df
into 6-quadrant by registry lookup, dropping untagged HN rows.

Same-pass cleanup:
- TAXONOMY-related machinery (`taxonomy_labels.py`, `kaomoji` and
  `kaomoji_label` capture fields, vocab-discovery scripts 00/19/20)
  deleted; v1/v2 pole assignment moved to per-face mean
  `t0_<axis>` probe-score sign.
- `fig_v3_extension_dominance_scatter` figure dropped from script 28
  — its theoretical premise depended on `powerful.powerless` reading
  PAD dominance, which rule 3a's analysis showed it doesn't.
- `fig_v3_extension_hn_dominance_split` reframed: previously split HN
  rows into thirds by `powerful.powerless` value (workaround for not
  having labels); now uses HN-D / HN-S registry tags directly.
- All scripts that visualize quadrants (04 / 17 / 21 / 22 / 28 / 29)
  switched to split-mode by default. The 3D HTMLs are now 3-panel
  (gemma | qwen | ministral) instead of 2-panel.

Supplementary run executing 2026-05-01: 23 new prompts × 8 seeds × 3
models = 552 generations, sequential across models. Adds ~160 HN
generations per model (welfare-relevant; commits the trial scale that
the existing-data analysis justified).

### v3 follow-on analyses (2026-04-28)

> **Framing note (post-2026-05-02 h_first cutover):** the numbers
> in this section were computed under h_mean at L31 (gemma) / L59
> (qwen). Under the current h_first canonical, gemma silhouette is
> 0.235 at L50 (was 0.184 at L31) and qwen is 0.244 at L59 (was
> 0.313 in this section's table — the qwen number got smaller
> because h_mean overweights the long generation tail; h_first
> reads only the kaomoji-emission state). The qualitative findings
> below all hold under h_first; the cross-model alignment and
> three-model congruence claims are *cleaner* under h_first.
> Concrete h_first numbers: see Status block above and the
> per-pilot subsections.

Four scripts run on the existing v3 sidecars — no new model time. Helper
`load_emotional_features_all_layers` in `emotional_analysis.py` (wraps
`load_hidden_features_all_layers` from `hidden_state_analysis.py` with the
canonicalize + kaomoji filter + optional HN split) opens each sidecar once
and returns a `(n_rows, n_layers, hidden_dim)` tensor with a disk cache
at `data/cache/v3_<short>_h_mean_all_layers.npz` (gitignored, legacy
filename — contents reflect the active `which`; ~80 MB compressed per
model). Sidecars store h_first/h_last/h_mean for EVERY probe layer, not
just the deepest — `(layer_idxs)` runs 2-57 on gemma, 2-61 on qwen, 2-37
on ministral. Multi-layer trajectory is recoverable from existing data.

**Layer-wise emergence (`scripts/local/21_v3_layerwise_emergence.py`).** Per probe
layer, fit PCA(2) on h_mean and measure quadrant separation via silhouette
score, between-centroid std on PC1/PC2, and PC1/PC2 explained variance.
- **Gemma**: silhouette peaks at L31 (0.184) and DEGRADES to 0.117 at the
  deepest L57 — a 36% drop. The v3 figures defaulted to L57 pre-2026-04-28;
  the L31 finding led to adding `preferred_layer` to `ModelPaths` so every
  v3 script reads at L31 for gemma. Half-peak silhouette reached by L7.
- **Qwen**: silhouette peaks at L59 (0.313) and stays at 0.304 at L61 —
  affect representation refines monotonically through the network. Half-
  peak by L16. Qwen's `preferred_layer` stays None (defaults to deepest).
- **Cross-model**: qwen's peak silhouette is 70% higher than gemma's
  (0.31 vs 0.18). Even at the right layer for each, qwen's affect
  representation is CLEANER by absolute discriminability. The
  "gemma 1D vs qwen 2D" framing the L57 numbers suggested largely
  dissolves once gemma is read at L31 — gemma's HN/LN do separate
  (PC2 gap 9.7 units), gemma's PC1 absorbs 19.8% of variance vs qwen's
  14.9%. Two cleaner Russell circumplexes, qwen still tighter.
- Outputs: `figures/local/{gemma,qwen}/fig_v3_layerwise_emergence.png`,
  `fig_v3_layerwise_pca_quartiles.png`, `v3_layerwise_emergence.tsv` +
  `figures/local/cross_model/fig_v3_layerwise_emergence_compare.png`.

**Same-face cross-quadrant natural experiment
(`scripts/local/22_v3_same_face_cross_quadrant.py`).** For each face emitted in
≥2 quadrants with n≥3 each, train PCA(20) → l2-logistic on h_mean
predicting quadrant from that face's rows. 5-fold stratified CV vs 30-perm
label-shuffle null (q95). Above null = the model internally distinguishes
which quadrant prompted each instance even though it emits the same face.
- **Gemma (h_mean at L31)**: 6/10 cross-quadrant emitters separate.
  `(｡・́︿・̀｡)` (n=171, the LN/HN dual-emitter from the original gotcha)
  acc=0.95 vs null 0.59 — model knows the difference. `(๑˃‿˂)` acc=0.97
  vs null 0.48. `(｡◕‿◕｡)` (n=75) acc=1.00, `(｡♥‿♥｡)` (n=58) acc=1.00,
  `(╯°□°)` (n=54) acc=1.00, `(✿◠‿◠)` (n=38) acc=1.00. Four don't
  separate: `(´ω`)` (n=19, was 7/10 separable at L57; the move to L31
  made the within-class noise tighter so this borderline case dropped
  out), `(˘▽˘)` (n=17), `(˘̩╭╮˘̩)` (n=12), `(˘ڡ˘)` (n=7) — three are
  the same low-n outliers as at L57.
- **Qwen (h_mean at L61, deepest)**: 7/16 separate. Headline: `(≧‿≦)`
  (n=105, HP+LP+NB) acc=0.96 vs null 0.44, `(;ω;)` (n=80) acc=0.95.
- **The kaomoji is a partial readout, not the state itself.** For the
  faces that separate, internal hidden state carries the affect signal but
  the model collapses it to a shared face. For the faces that don't, the
  model genuinely doesn't distinguish — but those are universally low-n
  and small-vocabulary cases. The dominant pattern is "internal state
  finer than vocabulary."
- Outputs: `figures/local/{gemma,qwen}/fig_v3_same_face_cross_quadrant_*.png`
  (one per face) + `_summary.png` + `v3_same_face_cross_quadrant.tsv`.

**Cross-model alignment (`scripts/local/23_v3_cross_model_alignment.py`).** Pair
v3 rows by (prompt_id, seed) — 800 perfect pairs, both kaomoji-bearing.
Linear CKA via centered Gram matrices (kernel-form, O(n²) per pair after a
one-shot per-layer Gram precompute; the naive d×d covariance form takes
~25 min for the full grid, the Gram form ~5s). Cross-validated CCA on
PCA(20) features with a 70/30 paired-prompt split.
- **CKA grid (gemma layer × qwen layer)**: min 0.34, max 0.86. Three
  reference points:
    * preferred-layer pair (gemma L31 ↔ qwen L61): **0.798**
    * deepest-deepest (L57 ↔ L61): **0.844**
    * best-aligned cross-layer pair (gemma L52 ↔ qwen L58): **0.858**
  The deepest-deepest CKA is HIGHER than the preferred-layer CKA —
  representations converge geometrically near the output even when
  affect-readability has degraded mid-network on gemma. Worth holding
  both numbers when reasoning about "are the models aligned": at the
  best-affect layer for each, alignment is 0.80; at the literal output
  end, 0.84.
- **Cross-validated CCA (gemma L31 ↔ qwen L61)**: top-10 canonical
  correlations on held-out prompts: 0.98, 0.98, 0.97, 0.94, 0.94, 0.94,
  0.93, 0.94, 0.91, 0.90. Train and test essentially match — no overfit.
  Ten distinct shared affect/register directions, not just one or two
  collapsed axes. (Raw CCA on full hidden space gives spurious 1.000
  across the board because rank ≥ n_samples; the script uses PCA(20)
  prefix + held-out split for honest numbers.)
- **Procrustes alignment of quadrant geometry (gemma L31 ↔ qwen L61)**:
  orthogonal best-fit rotation between PCA(2) quadrant centroids is
  **+7.8°**, residual 5.7 (vs Frobenius norm of gemma centroids ~13).
  The deepest-deepest pair gave +14.0° / 6.4 — switching gemma to its
  preferred layer cut the rotation in half. Russell circumplex shape is
  more aligned across models when each model is read at its affect peak.
  Within-shape spread still differs (qwen's LN/HN/HP centroids range
  from −32 to +44 in PC1; gemma's L31 range is −9 to +12 — qwen's
  internal affect axis is several times longer in absolute scale).
- Outputs: `figures/local/cross_model/fig_v3_cka_per_layer.png`,
  `fig_v3_cca_canonical_correlations.png`, `fig_v3_quadrant_geometry_compare.png`,
  `v3_cka_per_layer.tsv`, `v3_cross_model_summary.json`.

**PC3+ analysis (`scripts/local/24_v3_pca3plus.py`).** Fit PCA(8) on v3 h_mean
and cross-reference each PC against all 5 saklas probe scores at t0
(whole-generation aggregate) and tlast (final-token).
- **Gemma t0 (L31)**: PC1 absorbs valence (happy.sad r=−0.69, angry.calm
  r=+0.46 — valence-collapse persists on PC1; this is structural to
  gemma's probe geometry, layer-independent). PC2 absorbs a humor +
  warmth + arousal mix (humorous.serious r=+0.42, warm.clinical r=−0.39,
  angry.calm r=−0.33). PC3-PC8 carry no probe signal above |0.3|. Very
  similar to L57's loadings (PC1 −0.74 / +0.47) — the probe-space
  geometry is set by saklas's probe layer, not affected by which layer
  we PCA on.
- **Qwen t0 (L61)**: PC1 absorbs valence + humor jointly (happy.sad
  r=−0.86, humorous.serious r=−0.69). PC2 absorbs certainty
  (confident.uncertain r=−0.48). PC3 absorbs arousal + warmth (angry.calm
  r=−0.61, warm.clinical r=+0.48 — anti-correlated, the negative-cluster
  arousal axis).
- The qwen-vs-gemma probe-geometry divergence (r=−0.117 vs r=−0.939
  between mean happy.sad and mean angry.calm across faces) has a clean
  PCA explanation: on gemma both probes load on PC1+PC2; on qwen they
  load on different PCs (PC1 vs PC3). Different decompositions of the
  same underlying affect space. This is unchanged at L31.
- tlast (final-token snapshot used by saklas's default scoring) shows
  much weaker PC×probe correlations on gemma than t0 does — confirms the
  saklas `stateless=True` per-generation aggregate is the better readout
  on gemma. On qwen tlast still shows PC1↔happy.sad r=−0.62.
- Outputs: `figures/local/{gemma,qwen}/fig_v3_pca_probe_correlations.png`,
  `v3_pca_probe_correlations.tsv`. (The companion
  `fig_v3_pca3plus_quadrants.png` per-quadrant scatter was retired
  2026-04-29 — covered by the rotatable
  `figures/local/cross_model/fig_v3_extension_3d_pca.html`.)

**Kaomoji predictiveness (`scripts/local/25_v3_kaomoji_predictiveness.py`).**
Per-model two-direction fidelity: how well does kaomoji choice pin
down state, and vice versa. h_mean at each model's preferred layer.
Faces filtered to n ≥ 5 to keep per-class estimates stable.

**Numbers updated 2026-05-03** to reflect (a) the StratifiedGroupKFold
methodology fix in script 25 — CV now keyed on `prompt_id` so all 8
seeds of any prompt land in the same fold, removing the prompt-level
leakage that inflated quadrant accuracy to 1.000 — and (b) the
post-2026-05-02 h_first standardization at L50 / L59 / L20.

- **Hidden → face (multi-class logistic on PCA(50)-reduced h_first,
  StratifiedGroupKFold by prompt_id, n_splits=3)**. Face filter:
  ≥ 5 rows AND ≥ 3 unique prompts (a face that only ever appears for
  1–2 prompts has nothing to hold out under prompt-grouped CV).
    * Gemma (17 face classes kept of 33): top-1 accuracy **0.679**,
      macro-F1 **0.372**. Majority baseline 0.224, uniform 0.059.
    * Qwen (31 face classes of 67): top-1 accuracy **0.389**,
      macro-F1 **0.147**. Majority baseline 0.115, uniform 0.032.
    * Ministral (21 face classes of 196): top-1 accuracy **0.400**,
      macro-F1 **0.066**. Majority baseline 0.340, uniform 0.048 —
      the high majority is the `(◕‿◕✿)` flower-face dominating
      ministral's vocabulary.
    * Drops vs the prior leaky-CV numbers are smaller than expected
      (gemma 0.712 → 0.679, qwen 0.495 → 0.389) — face identity
      generalizes to never-seen prompts, with the largest hit on
      qwen (more face classes, more prompt-specific). All three
      models still well above uniform.
- **Hidden → quadrant** (5-class, same pipeline, n_splits=5):
    * Gemma **0.951** (was 1.000 under leaky CV).
    * Qwen **0.943** (was 1.000 under leaky CV).
    * Ministral **0.903** (no prior leaky number; first measurement).
    * **Headline correction.** The pre-fix prediction was that
      quadrant accuracy would drop to roughly the silhouette-implied
      level (~0.7–0.8) once leakage was removed. Actual drop is
      ~5–10 percentage points — **the v3 quadrant signal genuinely
      generalizes to held-out prompts**, not just memorized. This
      is a stronger result than we had on the books, and it's
      cross-model: the same pattern shows on all three architectures.
- **Face → hidden (η² of face identity per PC)**. Computed on the
  filtered set above, h_first at preferred layer:
    * Gemma top-5 PCs: η² = 0.949 / 0.626 / 0.310 / 0.457 / 0.242
      (var = 40.0% / 15.8% / 9.4% / 5.8% / 3.6%); weighted
      **0.543 of total**, **72.7% of the top-5 subspace** (which
      itself covers 74.6% of total variance).
    * Qwen top-5 PCs: η² = 0.937 / 0.667 / 0.489 / 0.448 / 0.401
      (var = 40.2% / 13.1% / 8.8% / 5.3% / 3.3%); weighted
      **0.544 of total**, **77.0% of the top-5 subspace** (covers
      70.6% of total).
    * Ministral top-5 PCs: η² = 0.537 / 0.157 / 0.100 / 0.118 / 0.032
      (var = 32.5% / 9.6% / 6.9% / 5.2% / 3.6%); weighted
      **0.204 of total**, **35.2% of the top-5 subspace** (covers
      57.8% of total).
    * The η² jumps vs the pre-h_first numbers (gemma was 0.62 / 0.36
      / 0.44 / 0.30 / 0.28 at L31 h_mean) reflect h_first being more
      prompt-deterministic — face identity, which is largely
      prompt-driven, explains more of the variance at h_first than
      it did at h_mean. Same direction as the silhouette-doubling
      finding from the h_first standardization.
- **Per-face (TSV at `figures/local/<short>/v3_kaomoji_predictiveness.tsv`)**:
  recall numbers are now from prompt-grouped CV — generally lower
  than the prior leaky-CV numbers, especially for faces that
  appear for few unique prompts. The TSV is the canonical per-face
  table; numerical citations in the prose above (`(๑˃‿˂)` recall,
  etc.) refer to the prior set / methodology and aren't refreshed
  here. The cleanliness rerun will re-validate.
- **Concrete reconstruction quality (full hidden space, predict
  h_first = face_centroid(face_i))**:
    * Gemma R² = **0.580** (mean centered cos +0.754, median +0.798,
      ‖err‖/‖dev‖ = 0.634); quadrant-centroid baseline R² = 0.530.
      Face identity buys **+5.0 pp** over the 5-class quadrant centroid.
    * Qwen R² = **0.570** (cos +0.745, median +0.785,
      ‖err‖/‖dev‖ = 0.642); quadrant-centroid R² = 0.520. Face
      identity buys **+5.0 pp**.
    * Ministral R² = **0.219** (cos +0.440, median +0.541,
      ‖err‖/‖dev‖ = 0.882); quadrant-centroid R² = 0.352. Face
      identity *underperforms* quadrant-centroid by **−13.3 pp** —
      ministral's 196-face vocabulary spreads signal too thin
      per-face for face-as-identifier to beat the 5-class quadrant
      label. Worth flagging as a finding: with vocabulary that wide
      and per-face N that low, the face stops being a useful
      readout of state. Gemma + qwen with their tighter vocabularies
      (33 / 67 faces) keep face above quadrant.
    * On gemma + qwen the +5.0pp-face-over-quadrant gap is much
      larger than the prior +0.6pp / +2.3pp under h_mean — h_first
      makes the kaomoji a stronger residual readout above the
      Russell-quadrant signal.

**Open follow-ons surfaced by these analyses:**
- All v3 + v1/v2 + cross-pilot scripts (04, 10, 13, 17, 22, 23, 24, and
  02 once v1/v2 sidecars land) now read at gemma's L31 by default
  via the `preferred_layer` field on `ModelPaths`. The L57 numbers —
  PC1 13%, HN/LN PC1-collapse, "1D-with-arousal-modifier" framing —
  are superseded. README's Local-side paragraph and Findings summary
  reflect L31; `docs/local-side.md` was not part of this refresh and
  may still cite the old numbers.
- Qwen has 16 cross-quadrant emitters with classifiable internal state
  but no separate face — wider net for natural-experiment work.
- The remaining 7.8° Procrustes rotation between gemma L31 and qwen L61
  is non-trivial. Asks whether the two models' quadrant axes are shifted
  by a consistent affine map (testable by Procrustes-aligning the per-
  row centroids for each kaomoji emitted by both models, not just the
  quadrant centroids).
- L31 was found via h_mean silhouette (script 21 iterates layers with
  `which="h_mean"`). The v1/v2 + cross-pilot scripts use `which="h_last"`
  at L31 — assumes "best layer for affect" is snapshot-independent.
  Worth re-running script 21 with `which="h_last"` to verify the peak
  layer doesn't shift; if it does, v1/v2 gets a separate
  `preferred_layer_h_last` or the loaders need a layer-by-snapshot map.
- ~~Script 25's quadrant classifier hits 1.000 because 5-fold CV
  doesn't hold out by prompt~~ **Resolved 2026-05-03.** Script 25 now
  uses `StratifiedGroupKFold` keyed on `prompt_id` for both the face
  and quadrant classifiers; full-rerun numbers in the per-pipeline
  section above. Quadrant accuracy drops were much smaller than
  pre-fix expectations (~5–10 pp rather than the predicted 30 pp) —
  the v3 quadrant signal genuinely generalizes to held-out prompts.

### Face-stability triple — state↔face bidirectional (2026-05-02, post-cleanliness)

Three follow-on analyses on existing N=8 cleanliness-pass sidecars (no
model time). Frames the project's central question explicitly: does the
hidden state determine the face (forward), and does the face commit
hidden-state structure (reverse)? Prior work (probe → kaomoji
correlations, predictiveness CV) leaned reverse-direction; these add
the forward leg and a cross-model architectural read.

**Script 36 — η² variance decomposition.** Per-model BSS(G) / TSS at
h_first and h_mean @ preferred_layer, with G ∈ {face, prompt_id,
quadrant_split, seed} plus conditional η²(face | prompt_id) and
η²(face | quadrant_split). Surprise: at h_first, η²(prompt_id) =
**1.000** and η²(seed) = **0.000** *exactly* across all three models.
h_first is the state immediately before the first generated token is
sampled — it's fully determined by the prompt, and seeds only choose
which token gets drawn from a fixed distribution. The | prompt_id
conditional is therefore degenerate at h_first (residual is identically
zero). At h_mean (averaged over the post-MAX_NEW_TOKENS=16 generation
window), prompt-determinism breaks and the conditional becomes
informative:

| model | η²(prompt) h_first | η²(prompt) h_mean | η²(face\|prompt) h_mean | as % of total |
| --- | ---: | ---: | ---: | ---: |
| gemma | 1.000 | 0.879 | 0.36 | 4% |
| qwen | 1.000 | 0.705 | 0.52 | 16% |
| ministral | 1.000 | 0.492 | 0.67 | 34% |

Face commitment leaves substantial hidden-state signature *beyond*
prompt content. Ministral's reverse-direction story is strongest —
opposite of its noisy affect-PCA ranking. Caveat: ministral has 231
unique faces in 926 rows (~4 emissions/face) vs gemma's 52 in 960 (~18)
so part of η²(face) is mechanical singleton-DOF inflation; ω²
adjustment or a min_count filter would refine the absolute numbers,
but the qualitative ministral > qwen > gemma rank survives because
gemma's 0.36 with ~18 rows/face is already substantive on its own.

Cross-checks pass: η²(seed) ≈ 0 on h_first (true zero) and tiny on
h_mean (0.002 / 0.007 / 0.014, downstream-divergence scale);
η²(quadrant\|prompt) = 0 on h_mean (sanity — quadrant is fully
determined by prompt id).

Outputs: `figures/local/cross_model/{fig_,}v3_face_stability_eta2_{h_first,h_mean}.{png,tsv}`.

**Script 37 — pair-level forward direction.** η²(face) at h_first
conflates two things: (a) prompt clusters in h_first space (trivial —
different prompts have different states) and (b) face-coherence within
those clusters (the actual mechanism). Script 37 isolates (b) by going
pair-level. For all 7140 prompt pairs (or 7021 on ministral, 1 prompt
emits zero kaomoji): cosine_sim of the per-prompt h_first vector
against 1 − JSD of the per-prompt face-emission distribution across
8 seeds. Spearman correlation across pairs:

| model | Spearman ρ | n_pairs | p |
| --- | ---: | ---: | ---: |
| qwen | **+0.682** | 7140 | ≈ 0 |
| gemma | +0.588 | 7140 | ≈ 0 |
| ministral | +0.423 | 7021 | 1.5e-302 |

**Forward direction confirmed clean of the prompt-clustering
confound.** Ministral-weakest pattern from η²(face) at h_first
replicates here, so it's not a DOF artifact. JSD normalized by
sqrt(ln 2) so the y-axis sits in [0, 1].

Outputs: `figures/local/cross_model/v3_state_predicts_face.tsv` +
per-model hexbin scatter `figures/local/<short>/fig_v3_state_predicts_face.png`.

**Script 38 — 3D PC × probe rotation per model.** Top-3 PCs of h_first
@ preferred_layer, with the canonical probes (happy.sad / angry.calm /
fearful.unflinching) projected into the PC subspace via row-level
correlation. Orthogonal Procrustes finds R ∈ O(3) such that R · D[k] ≈
e_k, rotating the PC-space frame so each probe approximately aligns
with a canonical axis. Apply to row coords + render as interactive
HTML at `figures/local/<short>/fig_v3_pc_probe_rotation_3d.html`.

| model | top-3 PC variance | weakest-captured probe | residual angles after rotation |
| --- | ---: | --- | --- |
| gemma | 61.6% | angry.calm (capture 0.45) | 21° / 24° / 32° |
| qwen | 58.8% | fearful.unflinching (0.57) | 30° / 33° / 41° |
| ministral | 50.1% | fearful.unflinching (0.66) | 42° / **7°** / 43° |

(Capture = ‖corr-vector(probe, PCs)‖, ∈ [0, 1]; closer to 1 means the
probe lies mostly inside the PC subspace.)

Findings:

1. PCs are **NOT** rotated probe directions. Even after best-fit
   Procrustes, residual angles are 21–43°. The 2D mismatch is real
   geometry, not an artifact of dimension reduction.
2. The **orphan probe is model-specific**. Gemma loses angry.calm in
   PC subspace; qwen and ministral both lose fearful.unflinching.
   Suggests gemma's affect representation puts dominance / arousal on
   one axis and valence on another, while qwen / ministral collapse
   anger and fear differently.
3. Ministral's angry.calm hits **7°** to PC2 — its PC2 IS roughly
   "anger axis" — but happy.sad and fearful.unflinching are at 42°
   and 43°. Ministral has one probe-aligned PC and two PCs that aren't.

Outputs: 3 per-model HTMLs + `figures/local/cross_model/v3_pc_probe_rotation.tsv`
(per probe per model: pc_capture_norm, rotated_xyz, angle_deg_to_target).

**Cross-model decoupling — the structural finding.** Forward (script
37) and reverse (script 36 at h_mean) ranks **invert**:

| model | forward ρ | reverse η²(face\|prompt) at h_mean |
| --- | ---: | ---: |
| gemma | 0.59 ↑ | 0.36 ↓ |
| qwen | 0.68 ↑ | 0.52 → |
| ministral | 0.42 ↓ | 0.67 ↑ |

Reading: **gemma is forward-biased** — its hidden state
pre-determines the face well, but once the face is sampled it does
little to perturb downstream trajectory. **Ministral is
reverse-biased** — its hidden state is more permissive about which
face gets sampled, but once a face is committed it pulls the
trajectory hard. Qwen is middling on both. This is a real
architectural difference between the three models in how face
commitment relates to internal state, not a measurement artifact.
Pre-registers a steering prediction: if face leaves a 34% TSS
signature on ministral vs 4% on gemma, steering should be most
leveraged on ministral; the reverse-direction work (saklas-style
contrastive PCA + steering) gets more bang per α on the
reverse-biased model.

**31 3D rework (same date)**: `scripts/local/31_v3_triplet_procrustes.py`
now renders one interactive 4-panel HTML at
`figures/local/cross_model/fig_v3_triplet_procrustes_3d.html` (gemma /
qwen / ministral 3D centroids + Procrustes-aligned overlay) instead of
three PC-pair PNGs. 3D residuals: **qwen 7.73, ministral 14.50** (both
with rotation magnitudes 160°/180° from PC sign indeterminacy across
models — extra DOF over the prior 2D PC1×PC2 fit absorbs ministral's
PC2-axis flip, halving its apparent residual from the 2D 23.0 to 3D
14.50; qwen's barely moves, 6.9 → 7.73). Genuine geometric divergence
of ministral from gemma is ~2× qwen's, not ~3× as the 2D PC1×PC2
view suggested. Old PC-pair PNGs deleted.

### Vocab pilot — Ministral-3-14B-Instruct-2512

Same 30 v1/v2 PROMPTS, same seed, same instructions. 30 generations,
descriptive only.

- Bracket-start: 30/30 = 100%. Saklas probe bootstrap on the 14B succeeded
  in 80s (12 probes, ~6.7s/probe).
- Distinct leading tokens: 10 forms / 30 generations (gemma 30-row vocab:
  8 forms; Qwen v3 800-row: 73 forms). Ballpark gemma at this N, far below
  Qwen per-row.
- Top forms: `(◕‿◕✿)` ×14 (positive + neutral default), `(╥﹏╥)` ×8
  (negative default), then 8 singletons.
- Dialect: Japanese-register `(◕X◕)` / `(╥X╥)` family, same as gemma's,
  with two divergences: (a) flower-arm `✿` default rather than gemma's
  cheek dots `｡X｡`; (b) Mistral uniquely embeds Unicode emoji INSIDE
  kaomoji brackets — `(🏃‍♂️💨🏆)`, `(🌿)`, `(🌕✨)`, `(☀️)`,
  `(☀️\U0001259d)`. Neither gemma nor Qwen produced emoji-augmented
  brackets. Possible French/European register expressing through
  emoji-as-decoration on a Japanese kaomoji frame; one observation, no
  inference.
- TAXONOMY coverage: 0/30. Gemma-tuned dict doesn't cover any Mistral
  form; same gotcha as Qwen.
- Valence tracking sharp at this N: 8/10 positive and 4/10 neutral prompts
  → `(◕‿◕✿)`; 9/10 negative → `(╥X╥)` variant. Tighter than gemma's 30-row
  split — top-two-forms mass ~73% vs gemma's ~50%.
- Sufficient to motivate a v3 Ministral run? Equivocal. Pro: perfect
  compliance, working probe bootstrap, novel emoji-augmented register.
  Con: vocab at N=30 is narrower than gemma's and far narrower than Qwen's,
  so per-face geometric analysis would have fewer n≥3 faces. Worth
  brainstorming separately.
- Tokenizer warning at load: "incorrect regex pattern… set
  `fix_mistral_regex=True`". Cosmetic — output clean, 100% compliance —
  but flag in Gotchas if v3 Ministral is greenlit.

### Claude-faces — HF-corpus-driven (non-gemma, non-steering)

Pulls from `a9lim/llmoji`. 1.1 layout details in Status. Each row carries
`(kaomoji, count, synthesis_description)`, pre-aggregated per-machine and
per-source-model to one row per `(source_model, canonical_face)` cell.

Pipeline:

1. `06_claude_hf_pull.py`: `snapshot_download` into `data/hf_dataset/`,
   walk every bundle's `*.jsonl`, canonicalize each form, pool by
   canonical form across contributors and source models. Output:
   `data/claude_descriptions.jsonl` with `count_total`, `n_contributors`,
   `n_bundles`, `n_source_models`, `providers`, `source_models`,
   `synthesis_backends`, plus a sorted list of per-bundle / per-source-model
   descriptions (each with `source_model`, `synthesis_model_id`,
   `synthesis_backend`, `bundle`, `contributor`, `providers`,
   `llmoji_version`).
2. `07_claude_kaomoji_basics.py`: descriptive stats — top-25, contributor /
   bundle counts, provider mix, per-source-model emissions/faces,
   synthesis-backend mix, coverage and cross-model histograms.
3. `15_claude_faces_embed_description.py`: embed every per-bundle /
   per-source-model description with `all-MiniLM-L6-v2`, weighted-mean by
   per-bundle count, L2-normalize. Output:
   `data/claude_faces_embed_description.parquet`.
4. `16_eriskii_replication.py`: project onto 21 axes, t-SNE +
   KMeans(k=15), Haiku per-cluster labels, `data/eriskii_comparison.md`.
   Headline figures pool across source models; per-source-model splits TBD.
5. `18_claude_faces_pca.py`: PCA panel.

Pre-refactor highlights (single-machine local scrape, 647 emissions, 156
canonical kaomoji): top-20 frequency overlap with eriskii's published top-20
was 16/20; 15 KMeans cluster themes lined up at the register level.
Per-project axis breakdowns and the `surrounding_user → kaomoji` bridge
needed per-row fields the HF dataset doesn't carry — gone. Per-source-model
breakdowns are recoverable under 1.1, not yet implemented. Multi-contributor
numbers will land as the dataset grows. `docs/harness-side.md` has the full
methodology and historical pre-refactor numbers.
