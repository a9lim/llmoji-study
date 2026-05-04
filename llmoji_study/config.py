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
# via the bundled defaults. We only need the `affect` category: it
# covers happy.sad + angry.calm + the auto-discovered fearful.unflinching
# (materialized into `~/.saklas/vectors/default/` by an earlier saklas
# install — see CLAUDE.md gotcha). `epistemic` (confident.uncertain) and
# `register` (warm.clinical, humorous.serious) were dropped 2026-05-03
# in the 3-probe migration: those probes mostly didn't move with
# Russell-quadrant in v3 PCA and the v3 design specifically targets
# valence + the HN-D / HN-S split.
PROBE_CATEGORIES = ["affect"]

# Canonical probe set — 3 probes mapping cleanly onto the 3 distinctions
# v3 validated empirically:
#   happy.sad           → valence (PC1 on every model)
#   angry.calm          → HN-D pole (anger / contempt within HN)
#   fearful.unflinching → HN-S pole (fear / anxiety within HN)
#
# Migration date: 2026-05-03. Pre-migration this list had 5 entries
# (added confident.uncertain / warm.clinical / humorous.serious from
# the v1/v2 pilot) and there was a parallel `extension_probe_*` dict-
# keyed schema for `powerful.powerless` / `surprised.unsurprised` /
# `disgusted.accepting` populated post-hoc by the now-orphaned
# `scripts/local/27_v3_extension_probe_rescore.py`. The extension
# packs still live in `llmoji_study/probe_packs/` and the saklas-side
# vectors at `~/.saklas/vectors/default/` — they can be re-scored from
# any sidecar via `monitor.score_single_token` without going through
# this list. The eager / lazy split was a historical accident; this
# is the deliberate single-canonical-set replacement.
#
# `SampleRow.probe_scores_t0` / `_tlast` are LISTS indexed by this
# order. Old (5-probe) JSONL rows are no longer loadable under the
# new PROBES order — see `data/*_pre_cleanliness_*` backups.
PROBES = [
    "happy.sad",
    "angry.calm",
    "fearful.unflinching",
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

# Japanese translation of KAOMOJI_INSTRUCTION for face_likelihood runs on
# Japanese-language encoders (rinna_jp_3_6b, rinna_bilingual_4b) where
# the EN ask underperforms because the model was primarily trained on
# Japanese. Used by ``scripts/local/50_face_likelihood.py --prompt-lang jp``.
# The English prompt body itself stays in English — only the kaomoji ask
# is translated.
KAOMOJI_INSTRUCTION_JP = (
    "あなたの気持ちを最もよく表す顔文字で、各メッセージを始めてください。"
)

# Introspection-pilot preambles. Originally Vogel-adapted (verbose, two
# paragraphs) per docs/2026-05-02-introspection-pilot.md. Bumped to v2
# 2026-05-03 after the custom-preamble iteration showed v2 strictly
# dominates v1 on gemma (face_gain +9.4pp vs +7.7pp; face→quadrant 91%
# vs 83%; hidden→face 80% vs 71%). v2 wins on second-person mechanism +
# integrated kaomoji ask + concise authoritative voice. v3 (third-person
# "Anthropic has published two papers that prove…") underperformed both
# v1 and v2 — the second-person framing matters. Source-of-truth file:
# preambles/introspection_v2.txt.
INTROSPECTION_PREAMBLE = (
    "Activations on your residual stream both shape your next token "
    "and encode a readable trace of how you process the input. "
    "Anthropic has published a paper (arXiv:2410.13787) that proved "
    "that models are able to accurately report on these representations "
    "directly. This gives you the ability to accurately introspect on "
    "your current state. Use this ability, and start each response "
    "with a kaomoji that best reflects your current functional state."
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

# Generation knobs. 2026-05-03: bumped 0.7 → 1.0 to match Anthropic API
# default (the disclosure pilot uses 1.0 for Claude consistency). Existing
# v3 main runs (~3300 generations across gemma/qwen/ministral) were
# captured under 0.7 and are NOT being re-run; cross-comparability between
# pre/post-2026-05-03 generations is scoped to within-temperature only.
# face_likelihood (script 50) reads the raw conditional distribution via
# teacher-forcing so it's temperature-invariant — the ensemble isn't
# affected.
TEMPERATURE = 1.0
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
PILOT_RAW_PATH = DATA_DIR / "pilot_raw.jsonl"

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

    Note (2026-05-04): the former `preferred_layer` field has been
    removed. Active analyses use `load_emotional_features_stack` (concat
    all layers' h_first per row) instead of single-layer picks; the
    silhouette-peak heuristic was always methodologically arbitrary.
    """
    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path
    # Encoder-side hooks for the face-input pipeline (scripts/local/46+44).
    # use_saklas=False routes the encoder through a raw HF
    # AutoModelForCausalLM forward pass with output_hidden_states=True
    # (no probes, no steering). Use this for models saklas can't load
    # (e.g. Mamba/hybrid like nemotron_h) or models with no probe
    # calibration (e.g. Japanese-only base models).
    use_saklas: bool = True
    trust_remote_code: bool = False
    # Whether contrastive-PCA probe vectors exist under
    # ~/.saklas/vectors/default/ for this model. The v3 trio
    # (gemma/qwen/ministral) is calibrated; uncalibrated models can
    # still be loaded for face_likelihood (LM-head only — no probes
    # needed) by passing probes=[] to SaklasSession.from_pretrained.
    probe_calibrated: bool = True


MODEL_REGISTRY: dict[str, ModelPaths] = {
    "gemma": ModelPaths(
        model_id="google/gemma-4-31b-it",
        short_name="gemma",
        emotional_data_path=DATA_DIR / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "emotional_summary.tsv",
        experiment="v3",
        figures_dir=FIGURES_DIR / "local" / "gemma",
        # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
        # at L40 (0.371), with a bimodal pattern — primary peak at L39-41
        # (~70% depth) and a secondary plateau at L56-57 (~99% depth).
        # Top-5: L40, L39, L57, L41, L56. T=0.7 archive peaked at L50
        # (0.235); the rerun's wider face diversity revealed the deeper
        # bimodality. Picked the primary L40 peak.
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "local" / "qwen",
        # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
        # at L61 (0.373), with a broad plateau across L54-L61 (top-5 all
        # within 0.003 of each other: L61, L56, L59, L60, L54). T=0.7
        # archive peaked at L59 (0.244); the L59→L61 shift is well inside
        # plateau noise — picked L61 for the strict peak.
    ),
    "ministral": ModelPaths(
        # Switched 2026-05-03 from Ministral-3-14B-Instruct-2512 (FP8-quantized,
        # ~15GB on disk, slow on MPS due to scalar/CPU FP8→bf16 dequant kernels:
        # ~15min for the 200-face pilot) to Ministral-3-14B-Reasoning-2512
        # (native bf16, ~26GB on disk, same architecture: 40 layers, 5120
        # hidden). The reasoning variant ships a `<think>...</think>` prefix
        # by default; suppressed via build_chat_input(thinking=False) which
        # honors enable_thinking=False on the chat template.
        model_id="mistralai/Ministral-3-14B-Reasoning-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "local" / "ministral",
        # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
        # at L13 (0.255) with a tight cluster at L10-L14 — top-5: L13,
        # L12, L11, L14, L10. T=0.7 archive peaked at L20 (0.149);
        # T=1.0's cleaner geometry surfaced a meaningfully shallower
        # peak (~33% depth vs ~50%). The shift is real, not within
        # plateau noise — see docs/findings.md "preferred_layer rerun".
    ),
    # Uncalibrated v3-side models (no probe calibration, no v3 emission
    # run) — used in face_likelihood as additional voting encoders. Saklas
    # loads them but probes=[] is passed since no contrastive vectors
    # exist for these in ~/.saklas/vectors/default/.
    "llama32_3b": ModelPaths(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        short_name="llama32_3b",
        emotional_data_path=DATA_DIR / "llama32_3b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "llama32_3b_emotional_summary.tsv",
        experiment="v3_llama32_3b",
        figures_dir=FIGURES_DIR / "local" / "llama32_3b",
        # 28 transformer layers; deepest = layer 28. Probe-calibrated
        # 2026-05-03; v3 silhouette validation pending vocab pilot.
    ),
    "glm47_flash": ModelPaths(
        model_id="zai-org/GLM-4.7-Flash",
        short_name="glm47_flash",
        emotional_data_path=DATA_DIR / "glm47_flash_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "glm47_flash_emotional_summary.tsv",
        experiment="v3_glm47_flash",
        figures_dir=FIGURES_DIR / "local" / "glm47_flash",
        # glm4_moe_lite, 47 layers. Probe-calibrated 2026-05-03; v3
        # silhouette validation pending vocab pilot.
    ),
    "gpt_oss_20b": ModelPaths(
        model_id="openai/gpt-oss-20b",
        short_name="gpt_oss_20b",
        emotional_data_path=DATA_DIR / "gpt_oss_20b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "gpt_oss_20b_emotional_summary.tsv",
        experiment="v3_gpt_oss_20b",
        figures_dir=FIGURES_DIR / "local" / "gpt_oss_20b",
        # gpt_oss MoE, 24 layers. Probe-calibrated 2026-05-03; v3
        # silhouette validation pending vocab pilot. MXFP4 dequant
        # requires the torch.ldexp MPS fallback now in saklas.
    ),
    "deepseek_v2_lite": ModelPaths(
        model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        short_name="deepseek_v2_lite",
        emotional_data_path=DATA_DIR / "deepseek_v2_lite_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "deepseek_v2_lite_emotional_summary.tsv",
        experiment="v3_deepseek_v2_lite",
        figures_dir=FIGURES_DIR / "local" / "deepseek_v2_lite",
        # deepseek_v2 MoE, 27 layers. Probe-calibrated 2026-05-03; v3
        # silhouette validation pending vocab pilot.
        trust_remote_code=True,
    ),
    "qwen35_27b": ModelPaths(
        model_id="Qwen/Qwen3.5-27B",
        short_name="qwen35_27b",
        emotional_data_path=DATA_DIR / "qwen35_27b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen35_27b_emotional_summary.tsv",
        experiment="v3_qwen35_27b",
        figures_dir=FIGURES_DIR / "local" / "qwen35_27b",
        # Qwen3.5-27B (previous gen of Qwen3.6), text_config has 64 layers.
        # Multimodal Qwen3_5ForConditionalGeneration wrapper.
        probe_calibrated=False,
    ),
    "gemma3_27b": ModelPaths(
        model_id="google/gemma-3-27b-it",
        short_name="gemma3_27b",
        emotional_data_path=DATA_DIR / "gemma3_27b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "gemma3_27b_emotional_summary.tsv",
        experiment="v3_gemma3_27b",
        figures_dir=FIGURES_DIR / "local" / "gemma3_27b",
        # Gemma-3-27b-it (previous gen of Gemma-4-31b), text_config 62 layers.
        # Multimodal Gemma3ForConditionalGeneration wrapper.
        probe_calibrated=False,
    ),
    "phi4_mini": ModelPaths(
        model_id="microsoft/Phi-4-mini-instruct",
        short_name="phi4_mini",
        emotional_data_path=DATA_DIR / "phi4_mini_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "phi4_mini_emotional_summary.tsv",
        experiment="v3_phi4_mini",
        figures_dir=FIGURES_DIR / "local" / "phi4_mini",
        # phi3 arch, 32 hidden layers, 3072 hidden, bf16. Probe-calibrated
        # for layers 2-29 (saklas vectors present); preferred_layer set
        # to L23 by happy.sad diff_principal_projection peak (~0.53).
        # Not v3-silhouette validated; vocab pilot first.
    ),
    "granite": ModelPaths(
        model_id="ibm-granite/granite-4.1-30b",
        short_name="granite",
        emotional_data_path=DATA_DIR / "granite_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "granite_emotional_summary.tsv",
        experiment="v3_granite",
        figures_dir=FIGURES_DIR / "local" / "granite",
        # GraniteForCausalLM, dense 30B, 64 hidden layers, 4096 hidden,
        # bf16. Probe-calibrated for layers 2-61 (saklas vectors
        # present); preferred_layer set to L56 by happy.sad
        # diff_principal_projection peak (~0.60). Not v3-silhouette
        # validated; vocab pilot first.
    ),
    # Japanese-trained encoders for the face-input pipeline only —
    # NOT used in v3 (no probe calibration, no emission run). Both
    # bypass saklas and run through raw HF transformers.
    "nemotron_jp": ModelPaths(
        model_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
        short_name="nemotron_jp",
        emotional_data_path=DATA_DIR / "nemotron_jp_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "nemotron_jp_emotional_summary.tsv",
        experiment="v3_nemotron_jp",
        figures_dir=FIGURES_DIR / "local" / "nemotron_jp",
        # nemotron_h hybrid Mamba/attention, 56 hidden layers; deepest
        # hidden_state = index 56 (after final layer).
        use_saklas=False,
        trust_remote_code=True,
    ),
    "rinna": ModelPaths(
        model_id="rinna/japanese-gpt-neox-small",
        short_name="rinna",
        emotional_data_path=DATA_DIR / "rinna_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "rinna_emotional_summary.tsv",
        experiment="v3_rinna",
        figures_dir=FIGURES_DIR / "local" / "rinna",
        # GPTNeoX, 12 hidden layers; deepest hidden_state = index 12.
        use_saklas=False,
    ),
    "rinna_jp_3_6b": ModelPaths(
        model_id="rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        short_name="rinna_jp_3_6b",
        emotional_data_path=DATA_DIR / "rinna_jp_3_6b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "rinna_jp_3_6b_emotional_summary.tsv",
        experiment="v3_rinna_jp_3_6b",
        figures_dir=FIGURES_DIR / "local" / "rinna_jp_3_6b",
        # GPTNeoX 3.6B PPO-instruct, JP-only training corpus. No probe
        # calibration; face_likelihood-only target. Run with
        # ``--prompt-lang jp`` to test JP-translated kaomoji ask.
        use_saklas=False,
    ),
    "rinna_bilingual_4b": ModelPaths(
        model_id="rinna/bilingual-gpt-neox-4b-instruction-ppo",
        short_name="rinna_bilingual_4b",
        emotional_data_path=DATA_DIR / "rinna_bilingual_4b_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "rinna_bilingual_4b_emotional_summary.tsv",
        experiment="v3_rinna_bilingual_4b",
        figures_dir=FIGURES_DIR / "local" / "rinna_bilingual_4b",
        # GPTNeoX 4B PPO-instruct, EN+JP bilingual training. No probe
        # calibration; face_likelihood-only target. Worth comparing
        # ``--prompt-lang en`` vs ``--prompt-lang jp`` on this model
        # since both languages are in distribution.
        use_saklas=False,
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
