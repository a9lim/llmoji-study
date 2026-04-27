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
# via the bundled defaults. The union of these three categories covers
# the five concepts we actually read (see PROBES below).
PROBE_CATEGORIES = ["affect", "epistemic", "register"]

# Concept names we read probe scores for. Subset of the categories
# above. Only the first of these is steered in the pilot; the others
# serve as a steering-selectivity check and features for clustering.
PROBES = [
    "happy.sad",
    "angry.calm",
    "confident.uncertain",
    "warm.clinical",
    "humorous.serious",
]

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

# --- claude-faces experiment (scrape + t-SNE cluster plot) ---
CLAUDE_CODE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
# Claude.ai exports are non-idempotent across requests — newer exports
# sometimes return empty content arrays for conversations that earlier
# exports returned fully. List multiple export dirs here; the source
# adapter unions them by conversation UUID, preferring whichever copy
# has non-empty content.
CLAUDE_AI_EXPORT_DIRS: list[Path] = [
    Path(
        "/Users/a9lim/Downloads/data-72de1230-b9fa-4c55-bc10-84a35b58d89c"
        "-1777012863-4b01638a-batch-0000"
    ),
    Path(
        "/Users/a9lim/Downloads/data-72de1230-b9fa-4c55-bc10-84a35b58d89c"
        "-1776479747-1b0e6bd8-batch-0000"
    ),
]
# Stop-hook journals: cooperating shell hooks at ~/.claude/hooks/ and
# ~/.codex/hooks/ append one JSONL line per assistant turn (kaomoji
# prefix + cwd + session/turn ids; NO full assistant_text). Live and
# growing — cheap to re-scrape on every run.
CLAUDE_HOOK_JOURNAL_CLAUDE = Path.home() / ".claude" / "kaomoji-journal.jsonl"
CLAUDE_HOOK_JOURNAL_CODEX = Path.home() / ".codex" / "kaomoji-journal.jsonl"

# Per-source scrape outputs (independently regeneratable):
#   _export — Claude.ai conversations.json exports (only changes when
#             a fresh export drops in; otherwise static)
#   _hook   — unified Claude + Codex journal (live Stop hooks +
#             retroactive backfill from transcripts/rollouts; this is
#             the single source of truth for every assistant turn).
# CLAUDE_KAOMOJI_PATH is the merged view of the two — full
# assistant_text on every row, ready for the embed / Haiku-describe /
# eriskii pipelines.
CLAUDE_KAOMOJI_EXPORT_PATH = DATA_DIR / "claude_kaomoji_export.jsonl"
CLAUDE_KAOMOJI_HOOK_PATH = DATA_DIR / "claude_kaomoji_hook.jsonl"
CLAUDE_KAOMOJI_PATH = DATA_DIR / "claude_kaomoji.jsonl"
CLAUDE_VOCAB_SAMPLE_PATH = DATA_DIR / "claude_vocab_sample.tsv"
CLAUDE_FACES_EMBED_PATH = DATA_DIR / "claude_faces_embed.parquet"

# --- eriskii-replication experiment (description-based embeddings + axes) ---
# Locked Haiku version. Re-exported from the `llmoji` PyPI package
# so the canonical-corpus value lives in one place — the v1.0 split
# froze this string in `llmoji.haiku_prompts`; bumping it is a major
# version bump there. Imported here for back-compat with every
# research-side script that reads `from llmoji_study.config import
# HAIKU_MODEL_ID`.
from llmoji.haiku_prompts import HAIKU_MODEL_ID  # noqa: E402, F401

# Stage-A sampling: per kaomoji, randomly sample up to this many
# instances for per-instance Haiku description (eriskii used 4 with
# a floor for low-frequency faces). Floor is implicit — kaomoji with
# fewer than the cap are fully sampled.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# Order matters: this is the column order in eriskii_axes.tsv and the
# heatmap-row order in per-model / per-project figures. Must stay in
# sync with `llmoji_study.eriskii_anchors.AXIS_ANCHORS` (research-
# side; not part of the v1.0 frozen public surface). All 21 axes
# from the eriskii.net page (note: "wryness" is the eriskii
# spelling, with one n).
ERISKII_AXES = [
    "warmth", "energy", "confidence", "playfulness", "empathy",
    "technicality", "positivity", "curiosity", "approval",
    "apologeticness", "decisiveness", "wryness", "wetness",
    "surprise", "anger", "frustration", "hatefulness", "sadness",
    "hope", "aggression", "exhaustion",
]

CLAUDE_HAIKU_DESCRIPTIONS_PATH = DATA_DIR / "claude_haiku_descriptions.jsonl"
CLAUDE_HAIKU_SYNTHESIZED_PATH = DATA_DIR / "claude_haiku_synthesized.jsonl"
CLAUDE_FACES_EMBED_DESCRIPTION_PATH = DATA_DIR / "claude_faces_embed_description.parquet"

# eriskii-replication output paths
ERISKII_AXES_TSV = DATA_DIR / "eriskii_axes.tsv"
ERISKII_CLUSTERS_TSV = DATA_DIR / "eriskii_clusters.tsv"
ERISKII_PER_MODEL_TSV = DATA_DIR / "eriskii_per_model.tsv"
ERISKII_PER_PROJECT_TSV = DATA_DIR / "eriskii_per_project.tsv"
ERISKII_USER_KAOMOJI_CORR_TSV = DATA_DIR / "eriskii_user_kaomoji_axis_corr.tsv"
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
    """
    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path
    vocab_sample_path: Path


MODEL_REGISTRY: dict[str, ModelPaths] = {
    "gemma": ModelPaths(
        model_id="google/gemma-4-31b-it",
        short_name="gemma",
        emotional_data_path=DATA_DIR / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "emotional_summary.tsv",
        experiment="v3",
        figures_dir=FIGURES_DIR,
        vocab_sample_path=VOCAB_SAMPLE_PATH,
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "qwen",
        vocab_sample_path=DATA_DIR / "qwen_vocab_sample.jsonl",
    ),
    "ministral": ModelPaths(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "ministral",
        vocab_sample_path=DATA_DIR / "ministral_vocab_sample.jsonl",
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
