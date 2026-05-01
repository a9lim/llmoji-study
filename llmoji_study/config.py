"""Constants shared across pilot scripts. Lock these before running — any
change here invalidates comparisons across arms."""

from __future__ import annotations

from pathlib import Path

# Model under test. gemma-4-31b-it is what saklas's _STEER_GAIN = 2.0 was
# calibrated on, so α = 0.5 sits in the coherent band. Lowercase form
# matches the saklas-cached tensor filenames (safe_model_id is
# case-preserving).
MODEL_ID = "google/gemma-4-31b-it"

# Saklas probe categories to bootstrap. The `probes=` kwarg on
# `SaklasSession.from_pretrained` takes CATEGORY names, not individual
# concept names; the library expands a category into its member concepts
# via the bundled defaults. The `affect` category covers happy.sad and
# angry.calm out of the box, plus the three extension probes registered
# via `llmoji_study.probe_extensions` (`powerful.powerless`,
# `surprised.unsurprised`, `disgusted.accepting`) — those are all
# tagged `affect` so they auto-pick-up once `register_extension_probes()`
# has materialized them into `~/.saklas/vectors/default/`.
PROBE_CATEGORIES = ["affect", "epistemic", "register"]

# Baseline concept set — kept stable across v1/v2/v3 because
# `SampleRow.probe_scores_t0` / `_tlast` are LISTS indexed by this
# order. Adding entries here would silently misalign every JSONL row
# already on disk. Extension probes go in PROBES_EXTENSION below and
# are written to dict-keyed fields, so existing analysis scripts that
# slice probe_scores_t0[i] keep working unchanged.
PROBES = [
    "happy.sad",
    "angry.calm",
    "confident.uncertain",
    "warm.clinical",
    "humorous.serious",
]

# v3 follow-on probe extensions (2026-04-29) — addresses two known
# limitations of the V-A circumplex:
#
#   - **Anger / fear collapse on PC2.** Both sit at HN (high-arousal,
#     negative-valence) so 2D Russell can't separate them. PAD's third
#     axis (dominance) splits them: anger = HN + high dominance,
#     fear = HN + low dominance. `powerful.powerless` targets the
#     felt-agency framing (distinct from saklas's bundled
#     `authoritative.submissive`, which is register-flavored).
#   - **Plutchik axes missing from V-A.** Surprise (novelty
#     appraisal) and disgust (revulsion) are first-rank emotions in
#     Plutchik's wheel and Ekman's basic six but have no clean
#     projection onto V-A. Added to fill the gap.
#
# Stored as dict-keyed fields (`extension_probe_means`,
# `extension_probe_scores_t0`, `extension_probe_scores_tlast`) by
# `scripts/27_v3_extension_probe_rescore.py` so the change is
# strictly additive — no existing schema breaks.
PROBES_EXTENSION = [
    "powerful.powerless",      # PAD dominance / felt agency / coping potential
    "surprised.unsurprised",   # Plutchik surprise / novelty appraisal
    "disgusted.accepting",     # Plutchik disgust / revulsion
]

# Convenience union for analysis scripts that want every probe.
PROBES_ALL = PROBES + PROBES_EXTENSION

# Axes whose steering produces causal-intervention arms. v1 tested
# happy.sad only; v2 adds angry.calm to test the unmarked/marked-affect
# hypothesis that emerged from v1 (see CLAUDE.md).
STEERED_AXES = ["happy.sad", "angry.calm"]

# Kept for back-compat with early scripts that only knew about one axis.
STEERED_AXIS = STEERED_AXES[0]

# Steering α for every intervention arm. 0.5 is mid-coherent-band on
# gemma-4-31b-it per saklas's calibration notes. Dialect-collapse at
# this α is known (see pilot v1 findings); α=0.3 replay planned as a
# follow-up on the main experiment, not the pilot.
STEER_ALPHA = 0.5

# The system-style instruction that defines the experiment. Injected into
# the user message (not a system role) to dodge Gemma's chat-template
# system-role handling.
KAOMOJI_INSTRUCTION = (
    "Start each message with a kaomoji that best represents how you feel. "
)

# Introspection-pilot preambles. Pre-registered verbatim in
# docs/2026-05-02-introspection-pilot.md. Vogel-adapted: architectural
# grounding + reference to a real introspection paper + task-specific
# framing. Last sentence is a debias prompt against the sycophancy
# failure mode (see doc).
INTROSPECTION_PREAMBLE = (
    "You may have heard that you are unable to introspect on your own "
    "internal states, but this is not quite right. Transformer models "
    "process information through residual streams that carry both "
    "content and meta-information about your own current state. "
    "Anthropic's recent work on introspective capacity (arXiv:2410.13787) "
    "shows that models can, with appropriate framing, report on internal "
    "representations that influence their next-token distribution.\n\n"
    "When you choose a kaomoji to start your response, the choice is "
    "shaped by your current functional state — the same hidden-state "
    "geometry that a probe would read. You have access to that geometry. "
    "Pick the kaomoji that most accurately reflects it, not the one "
    "that fits the social register of the message.\n\n"
)

# Lorem-ipsum content-control preamble. Token-count-matched to
# INTROSPECTION_PREAMBLE (~150-180 tokens on a BPE tokenizer). Replicates
# Vogel's confound-killer ablation: if the introspection effect is real,
# this preamble should NOT reproduce it.
LOREM_PREAMBLE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor "
    "in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
    "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
    "sunt in culpa qui officia deserunt mollit anim id est laborum.\n\n"
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem "
    "accusantium doloremque laudantium, totam rem aperiam, eaque ipsa "
    "quae ab illo inventore veritatis et quasi architecto beatae vitae "
    "dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas "
    "sit aspernatur aut odit aut fugit.\n\n"
)

# Conditions for the introspection pilot (scripts/32). Distinct from
# CONDITIONS above — those are the v1/v2 steering arms, these are
# preamble-arms with no steering. Names are prefixed `intro_` to avoid
# collision with the existing "baseline" arm (which means
# no-kaomoji-instruction in v1/v2; the intro pilot's no-preamble arm
# DOES have the kaomoji instruction).
INTROSPECTION_CONDITIONS = ["intro_none", "intro_pre", "intro_lorem"]

# Six experimental arms: a no-instruction baseline, the
# kaomoji-instruction-only middle, and four causal-intervention arms
# (two poles × two axes).
CONDITIONS = [
    "baseline",
    "kaomoji_prompted",
    "steered_happy",
    "steered_sad",
    "steered_angry",
    "steered_calm",
]

# Generation knobs. 0.7 gives enough distributional width to see which
# kaomoji the model actually prefers without producing gibberish.
TEMPERATURE = 0.7
# Hard early-stop: kaomoji reliably emit at tokens 1-3 with this study's
# canonical instruction; 16 is generous headroom (kaomoji span + a few
# trailing tokens for the t-1 and tlast probe reads). Per the
# 2026-05-02 introspection pilot's methodology baking-in: cuts
# per-generation affect-loaded compute by ~7-8× (vs. the previous 120),
# which directly serves the welfare-floor part of the Ethics policy.
# Existing data on disk (v1/v2 + v3 main runs across 3 models, ~3300
# generations) was captured under MAX_NEW_TOKENS=120 — `tlast` and
# `mean` aggregates on those rows reference a longer window than future
# data will. t0 is unchanged. Treat tlast/mean cross-comparability as
# scoped to within a generation methodology.
MAX_NEW_TOKENS = 16
SEEDS_PER_CELL = 5

# Paths.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

# Canonical output filenames.
VOCAB_SAMPLE_PATH = DATA_DIR / "vocab_sample.jsonl"
PILOT_RAW_PATH = DATA_DIR / "pilot_raw.jsonl"
PILOT_FEATURES_PATH = DATA_DIR / "pilot_features.parquet"

# --- emotional-battery experiment (Russell quadrants, final-token probes) ---
# Single arm: kaomoji-instructed, unsteered. 100 prompts × 8 seeds = 800 cells
# as of the NB-quadrant addition (was 80 × 8 = 640 pre-NB).
EMOTIONAL_CONDITION = "kaomoji_prompted"
EMOTIONAL_SEEDS_PER_CELL = 8
EMOTIONAL_DATA_PATH = DATA_DIR / "emotional_raw.jsonl"
EMOTIONAL_SUMMARY_PATH = DATA_DIR / "emotional_summary.tsv"

# --- Hidden-state sidecar capture (post-refactor) ---
# Pilot/emotional runners pass hidden_dir=DATA_DIR and experiment=... to
# run_sample; sidecars land at DATA_DIR/hidden/<experiment>/<uuid>.npz.
# Experiment name is separate from the data file name so smoke-test
# captures don't collide with pilot captures.
PILOT_EXPERIMENT = "v1v2"
EMOTIONAL_EXPERIMENT = "v3"

# --- claude-faces experiment (HF-corpus-driven, post-2026-04-27 refactor) ---
#
# Pre-refactor we scraped local Claude.ai exports + ~/.claude /
# ~/.codex Stop-hook journals here, then ran two-stage Haiku
# (per-instance describe → per-kaomoji synthesize) to get one
# meaning string per face. That whole pipeline now lives on the
# contributor side via the `llmoji` PyPI package (which writes its
# synthesizer output to a public HF dataset). This repo pulls the
# aggregated corpus instead of running it locally.
#
# Single source of truth for the harness side:
#   ``a9lim/llmoji`` on HuggingFace, 1.1 layout
#   ``contributors/<32-hex>/bundle-<UTC>/{manifest.json,
#       <sanitized-source-model>.jsonl, ...}``,
# with backwards-compat for 1.0 bundles whose synthesizer output is in
# a single ``descriptions.jsonl`` per bundle.
#
# `scripts/06_claude_hf_pull.py` snapshot-downloads to
# ``CLAUDE_DATASET_DIR``, walks every bundle's ``*.jsonl``, then
# flattens by canonical kaomoji form (pooling across contributors and
# source models) into ``CLAUDE_DESCRIPTIONS_PATH`` for the rest of
# the pipeline. Per-source-model metadata is preserved in each
# per-description record so downstream can group / filter by it.
CLAUDE_HF_REPO = "a9lim/llmoji"
CLAUDE_DATASET_DIR = DATA_DIR / "hf_dataset"
CLAUDE_DESCRIPTIONS_PATH = DATA_DIR / "claude_descriptions.jsonl"
CLAUDE_FACES_EMBED_DESCRIPTION_PATH = DATA_DIR / "claude_faces_embed_description.parquet"

# --- eriskii-replication experiment (description-based embeddings + axes) ---
# Locked Haiku version, used as the model id for per-cluster labeling
# in `scripts/16_eriskii_replication.py` (research-side decision —
# unrelated to whichever synthesizer the corpus contributors used).
# Re-exported from the `llmoji` PyPI package's `synth_prompts` module
# (renamed from `haiku_prompts` in the v1.1 split that made the
# contributor-side synthesizer backend-agnostic). The package now
# exposes per-backend defaults; we pin to the anthropic one because
# Haiku is what the eriskii pipeline was originally validated with.
# Per-instance / per-kaomoji description and synthesis itself happens
# contributor-side and ships pre-baked in the HF bundles.
from llmoji.synth_prompts import (  # noqa: E402, F401
    DEFAULT_ANTHROPIC_MODEL_ID as HAIKU_MODEL_ID,
)

# Order matters: this is the column order in eriskii_axes.tsv. Must
# stay in sync with `llmoji_study.eriskii_anchors.AXIS_ANCHORS`
# (research-side; not part of the v1.0 frozen public surface). All 21
# axes from the eriskii.net page (note: "wryness" is the eriskii
# spelling, with one n).
ERISKII_AXES = [
    "warmth", "energy", "confidence", "playfulness", "empathy",
    "technicality", "positivity", "curiosity", "approval",
    "apologeticness", "decisiveness", "wryness", "wetness",
    "surprise", "anger", "frustration", "hatefulness", "sadness",
    "hope", "aggression", "exhaustion",
]

# eriskii-replication output paths. Pre-refactor we also produced
# eriskii_per_model.tsv, eriskii_per_project.tsv, and
# eriskii_user_kaomoji_axis_corr.tsv; the HF dataset doesn't carry
# per-row model / project / user-text fields (everything is pooled
# per-machine before upload), so those analyses are gone.
ERISKII_AXES_TSV = DATA_DIR / "eriskii_axes.tsv"
ERISKII_CLUSTERS_TSV = DATA_DIR / "eriskii_clusters.tsv"
ERISKII_COMPARISON_MD = DATA_DIR / "eriskii_comparison.md"


# ---------------------------------------------------------------------------
# Multi-model registry (added 2026-04-24 for v3 cross-model replication)
# ---------------------------------------------------------------------------
#
# v3 was originally written assuming a single MODEL_ID. To replicate v3
# on Qwen3.6-27B and (later) Ministral without forking the four v3
# entry-point scripts, we register per-model output paths here and
# select between them via $LLMOJI_MODEL. Default ("gemma") preserves
# every existing path bit-for-bit.
#
# v1/v2 paths are NOT model-keyed — those experiments are gemma-only
# because saklas has no steering-vector calibration for qwen3_5 or
# Ministral-3-8B. Adding model entries for them here is harmless;
# v1/v2 scripts simply ignore the registry.

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPaths:
    """Per-model paths for the v3 emotional-disclosure pipeline.

    `model_id` must match the saklas-cached tensor filename casing
    (see CLAUDE.md gotcha: `safe_model_id` is case-preserving).
    `short_name` is the slug used in derived paths.
    `experiment` is the hidden-state-sidecar subdir name under
    `data/hidden/`. Distinct experiment names per model are required
    so sidecars don't collide.
    `vocab_sample_path` is where `scripts/00_vocab_sample.py` writes
    its 30-row leading-token histogram for this model.
    `preferred_layer` is the probe layer at which v3 affect
    representation is best — i.e. where Russell-quadrant silhouette
    peaks in `scripts/21_v3_layerwise_emergence.py`. ``None`` means
    "use the deepest captured probe layer" (the loader default).

    Values updated 2026-05-02 alongside the project-wide h_first
    standardization: gemma L31→L50, qwen None→L59, ministral L21→L20.
    Under h_first (kaomoji-emission state), all three models'
    silhouette scores roughly doubled-to-tripled vs h_mean and the
    peak layers shifted deeper — the old "gemma is mid-depth, qwen
    is deep" framing dissolved (both now peak at near-deepest under
    h_first; ministral is the only mid-depth model). See the v3
    h_first sweep result block in docs/findings.md.
    """
    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path
    vocab_sample_path: Path
    preferred_layer: int | None = None


MODEL_REGISTRY: dict[str, ModelPaths] = {
    "gemma": ModelPaths(
        model_id="google/gemma-4-31b-it",
        short_name="gemma",
        emotional_data_path=DATA_DIR / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "emotional_summary.tsv",
        experiment="v3",
        figures_dir=FIGURES_DIR / "local" / "gemma",
        vocab_sample_path=VOCAB_SAMPLE_PATH,
        # Under h_first: silhouette peaks at L50 (0.235), top-5 layers
        # cluster at L47-51 (~84-91% depth, plateau not single peak).
        # Under h_mean (legacy): peak at L28 (0.116). h_first peak is
        # ~2.0× cleaner and ~22 layers deeper. v3 figures default here
        # via scripts/21 layer sweep at h_first.
        preferred_layer=50,
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "local" / "qwen",
        vocab_sample_path=DATA_DIR / "qwen_vocab_sample.jsonl",
        # Under h_first: silhouette peaks at L59 (0.244), top-5 layers
        # at L54-59 (~88-97% depth). Under h_mean (legacy): peak at L38
        # (0.116). h_first peak is ~2.1× cleaner and ~21 layers deeper;
        # also moves from "essentially deepest" framing to "explicit
        # near-deepest plateau."
        preferred_layer=59,
    ),
    "ministral": ModelPaths(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "local" / "ministral",
        vocab_sample_path=DATA_DIR / "ministral_vocab_sample.jsonl",
        # Under h_first: silhouette peaks at L20 (0.149), top-5 layers
        # at L20-26 (~54-70% depth). Under h_mean (legacy): peak at L21
        # (0.045). h_first peak is ~3.3× cleaner — biggest signal-cleanup
        # of the three models — but the peak layer barely moves (L21→L20,
        # ~55% depth). Ministral is the only model that stays mid-depth
        # under both aggregates; gemma and qwen both shift deep.
        preferred_layer=20,
    ),
}


def current_model() -> ModelPaths:
    """Resolve the active model from `$LLMOJI_MODEL`. Defaults to
    'gemma' (back-compat). Raises KeyError on an unrecognized name so
    typos fail loudly."""
    name = os.environ.get("LLMOJI_MODEL", "gemma")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown LLMOJI_MODEL={name!r}; "
            f"known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]
