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
MAX_NEW_TOKENS = 120
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
    Gemma's affect representation peaks at L31 of 56 and degrades by
    36% to the deepest L57; Qwen's peaks essentially at the deepest
    L59-61. Set per-model so v3 figures pick the right slice without
    every script having to special-case it.
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
        # Silhouette peaks at L31 (0.184) and degrades to 0.117 at the
        # deepest L57 — v3 figures default here per scripts/21.
        preferred_layer=31,
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "local" / "qwen",
        vocab_sample_path=DATA_DIR / "qwen_vocab_sample.jsonl",
        # Silhouette peaks at L59 (0.313); deepest L61 is essentially
        # tied at 0.304. Leave None → deepest.
        preferred_layer=None,
    ),
    "ministral": ModelPaths(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "local" / "ministral",
        vocab_sample_path=DATA_DIR / "ministral_vocab_sample.jsonl",
        # No v3 run yet; preferred layer will be set after running 21.
        preferred_layer=None,
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
