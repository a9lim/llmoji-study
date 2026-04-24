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
CLAUDE_KAOMOJI_PATH = DATA_DIR / "claude_kaomoji.jsonl"
CLAUDE_VOCAB_SAMPLE_PATH = DATA_DIR / "claude_vocab_sample.tsv"
CLAUDE_FACES_EMBED_PATH = DATA_DIR / "claude_faces_embed.parquet"

# --- eriskii-replication experiment (description-based embeddings + axes) ---
# Locked Haiku version. Bumping invalidates description-corpus parity.
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"

# Order matters: this is the column order in eriskii_axes.tsv and the
# heatmap-row order in per-model / per-project figures. Must stay in
# sync with llmoji.eriskii_prompts.AXIS_ANCHORS.
ERISKII_AXES = [
    "warmth", "energy", "confidence", "playfulness", "empathy",
    "technicality", "positivity", "curiosity", "approval",
    "wrynness", "wetness",
]

CLAUDE_HAIKU_DESCRIPTIONS_PATH = DATA_DIR / "claude_haiku_descriptions.jsonl"
CLAUDE_FACES_EMBED_DESCRIPTION_PATH = DATA_DIR / "claude_faces_embed_description.parquet"

# eriskii-replication output paths
ERISKII_AXES_TSV = DATA_DIR / "eriskii_axes.tsv"
ERISKII_CLUSTERS_TSV = DATA_DIR / "eriskii_clusters.tsv"
ERISKII_PER_MODEL_TSV = DATA_DIR / "eriskii_per_model.tsv"
ERISKII_PER_PROJECT_TSV = DATA_DIR / "eriskii_per_project.tsv"
ERISKII_USER_KAOMOJI_CORR_TSV = DATA_DIR / "eriskii_user_kaomoji_axis_corr.tsv"
ERISKII_COMPARISON_MD = DATA_DIR / "eriskii_comparison.md"
