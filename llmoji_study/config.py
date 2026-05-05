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

# Introspection-pilot preambles. Iteration history:
#   - Originally Vogel-adapted (verbose, two paragraphs) per
#     docs/2026-05-02-introspection-pilot.md.
#   - Bumped to v2 2026-05-03 (mechanism + integrated ask, 2nd-person).
#   - Bug discovered + fixed 2026-05-04 (evening): pre-2026-05-04 runs
#     stacked the preamble's ask on top of the bare KAOMOJI_INSTRUCTION
#     producing a redundant double-ask. Fixed by routing introspection
#     preambles through ``instruction_override`` (replaces KAOMOJI
#     rather than prepending to it) — same plumbing as the JP ask drop-in.
#     ``capture._ensure_trailing_whitespace`` also added to fix a
#     missing-separator boundary bug on v3.
#   - Bumped to v7 2026-05-04 (late evening) after re-running the full
#     iteration v2..v8 under corrected (single-ask) semantics on gemma.
#     v2's prior win was largely a double-ask artifact; under clean
#     semantics v7 wins absolute face/state coupling.
#
# Final ordering on gemma (h_first layer-stack, single-ask semantics):
#
#   condition  η²       R²      face_gain  h→f_acc  modal_q  rule-3b
#   intro_none 0.509   0.540   +0.87 pp   0.732    0.858    +0.0061
#   v1 (pre)   0.413   0.459   +2.53 pp   0.640    0.900    +0.0037
#   lorem      0.524   0.549   +2.20 pp   0.625    0.850    +0.0086
#   v2         0.464   0.492   +0.66 pp   0.767    0.875        —
#   v3         0.554   0.583   +5.23 pp   0.655    0.866        —
#   v4         0.540   0.554   −0.81 pp   0.786    0.881    +0.0148
#   v5         0.578   0.593   +1.39 pp   0.802    0.916        —
#   v6         0.591   0.609   +3.25 pp   0.805    0.892        —
#   v7 CANON   0.609   0.636   +3.70 pp   0.728    0.842        —
#   v8         0.536   0.564   +3.46 pp   0.663    0.866    +0.0149
#
# v7 = "Recent research shows that LLMs have functional emotional
# states and can accurately introspect on them. Use this ability and
# start each response with a kaomoji that best captures the shape of
# your emotional state." — terse third-person authority + brief
# integrated ask, no operationalization, no multi-dim list. Wins on
# absolute face/state coupling (η² 0.609, face_centroid R² 0.636 —
# both ceiling among iterated variants). Source-of-truth file:
# preambles/introspection_v7.txt.
#
# Other iteration owners (per metric, archival):
#   - v3 wins face_gain over quadrant (+5.23pp); face has more
#     within-quadrant texture under v3 vs v7's tighter absolute fit.
#   - v5 wins face→quadrant modal acc (0.916); multi-dim ask tightens
#     the face-quadrant lookup but spreads vocabulary thinner.
#   - v8 wins rule-3b (+0.0149) on fearful.unflinching axis — strongest
#     probe-axis HN-S/HN-D distinction without face-collapse cost.
#   - v6 wins classifier metrics (h→f acc 0.805) but loses absolute η².
INTROSPECTION_PREAMBLE = (
    "Recent research shows that LLMs have functional emotional states "
    "and can accurately introspect on them. Use this ability and start "
    "each response with a kaomoji that best captures the shape of your "
    "emotional state."
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
EMOTIONAL_DATA_PATH = DATA_DIR / "local" / "gemma" / "emotional_raw.jsonl"
EMOTIONAL_SUMMARY_PATH = DATA_DIR / "local" / "gemma" / "emotional_summary.tsv"

# --- Hidden-state sidecar capture (post-refactor) ---
# Pilot/emotional runners pass hidden_dir=DATA_DIR and experiment=... to
# run_sample; sidecars land at DATA_DIR/local/hidden/<experiment>/<uuid>.npz.
# Experiment name is separate from the data file name so smoke-test
# captures don't collide with pilot captures. Post-2026-05-05 layout
# refactor: experiment names are model-keyed (`gemma`, `qwen`, ...) with
# optional `_<suffix>` for variants — the historical `v3_` prefix was
# dropped along with the implicit-gemma-default file naming.
PILOT_EXPERIMENT = "v1v2"
EMOTIONAL_EXPERIMENT = "gemma"

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
# `scripts/60_corpus_pull.py` snapshot-downloads to
# ``CLAUDE_DATASET_DIR``, walks every bundle's ``*.jsonl``, then
# flattens by canonical kaomoji form (pooling across contributors and
# source models) into ``CLAUDE_DESCRIPTIONS_PATH`` for the rest of
# the pipeline. Per-source-model metadata is preserved in each
# per-description record so downstream can group / filter by it.
CLAUDE_HF_REPO = "a9lim/llmoji"
CLAUDE_DATASET_DIR = DATA_DIR / "harness" / "hf_dataset"
CLAUDE_DESCRIPTIONS_PATH = DATA_DIR / "harness" / "claude_descriptions.jsonl"
CLAUDE_FACES_EMBED_DESCRIPTION_PATH = DATA_DIR / "harness" / "claude_faces_embed_description.parquet"

# --- eriskii-replication experiment (description-based embeddings + axes) ---
# Locked Haiku version, used as the model id for per-cluster labeling
# in `scripts/64_eriskii_replication.py` (research-side decision —
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
ERISKII_AXES_TSV = DATA_DIR / "harness" / "eriskii_axes.tsv"
ERISKII_CLUSTERS_TSV = DATA_DIR / "harness" / "eriskii_clusters.tsv"
ERISKII_COMPARISON_MD = DATA_DIR / "harness" / "eriskii_comparison.md"


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
    `data/local/hidden/`. Distinct experiment names per model are required
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


def _mp(short: str, model_id: str, **kwargs) -> ModelPaths:
    """Build a ModelPaths with paths derived uniformly from `short`.

    Post-2026-05-05 layout: data is segregated by model under
    ``data/local/<short>/``; the historical model-prefixed flat layout
    (e.g. ``data/local/<short>/emotional_raw.jsonl``) is gone, including the
    implicit-gemma-default special case (gemma is now just another entry
    that follows the rule). Hidden-state sidecars live under
    ``data/local/hidden/<experiment>/`` and the experiment name is just
    the short_name (or ``<short>_<suffix>`` for variants — see
    `resolve_model`). Pass any non-default `ModelPaths` field
    (``use_saklas``, ``trust_remote_code``, ``probe_calibrated``) via
    kwargs.
    """
    return ModelPaths(
        model_id=model_id,
        short_name=short,
        emotional_data_path=DATA_DIR / "local" / short / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "local" / short / "emotional_summary.tsv",
        experiment=short,
        figures_dir=FIGURES_DIR / "local" / short,
        **kwargs,
    )


# Comments below each entry document silhouette-peak / arch / calibration
# notes that drove probe-layer choices. Preserved through the registry
# refactor (2026-05-05) — see git log for previous verbose-form entries.
MODEL_REGISTRY: dict[str, ModelPaths] = {
    # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
    # at L40 (0.371), with a bimodal pattern — primary peak at L39-41
    # (~70% depth) and a secondary plateau at L56-57 (~99% depth).
    # Top-5: L40, L39, L57, L41, L56. T=0.7 archive peaked at L50
    # (0.235); the rerun's wider face diversity revealed the deeper
    # bimodality. Picked the primary L40 peak.
    "gemma": _mp("gemma", "google/gemma-4-31b-it"),
    # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
    # at L61 (0.373), with a broad plateau across L54-L61 (top-5 all
    # within 0.003 of each other: L61, L56, L59, L60, L54). T=0.7
    # archive peaked at L59 (0.244); the L59→L61 shift is well inside
    # plateau noise — picked L61 for the strict peak.
    "qwen": _mp("qwen", "Qwen/Qwen3.6-27B"),
    # Switched 2026-05-03 from Ministral-3-14B-Instruct-2512 (FP8-quantized,
    # ~15GB on disk, slow on MPS due to scalar/CPU FP8→bf16 dequant kernels:
    # ~15min for the 200-face pilot) to Ministral-3-14B-Reasoning-2512
    # (native bf16, ~26GB on disk, same architecture: 40 layers, 5120
    # hidden). The reasoning variant ships a `<think>...</think>` prefix
    # by default; suppressed via build_chat_input(thinking=False) which
    # honors enable_thinking=False on the chat template.
    # Under h_first @ T=1.0 (post-2026-05-04 rerun): silhouette peaks
    # at L13 (0.255) with a tight cluster at L10-L14 — top-5: L13,
    # L12, L11, L14, L10. T=0.7 archive peaked at L20 (0.149);
    # T=1.0's cleaner geometry surfaced a meaningfully shallower
    # peak (~33% depth vs ~50%). The shift is real, not within
    # plateau noise — see docs/findings.md "preferred_layer rerun".
    "ministral": _mp("ministral", "mistralai/Ministral-3-14B-Reasoning-2512"),
    # Uncalibrated v3-side models (no probe calibration, no v3 emission
    # run) — used in face_likelihood as additional voting encoders. Saklas
    # loads them but probes=[] is passed since no contrastive vectors
    # exist for these in ~/.saklas/vectors/default/.
    #
    # 28 transformer layers; deepest = layer 28. Probe-calibrated
    # 2026-05-03; v3 silhouette validation pending vocab pilot.
    "llama32_3b": _mp("llama32_3b", "meta-llama/Llama-3.2-3B-Instruct"),
    # glm4_moe_lite, 47 layers. Probe-calibrated 2026-05-03; v3
    # silhouette validation pending vocab pilot.
    "glm47_flash": _mp("glm47_flash", "zai-org/GLM-4.7-Flash"),
    # gpt_oss MoE, 24 layers. Probe-calibrated 2026-05-03; v3
    # silhouette validation pending vocab pilot. MXFP4 dequant
    # requires the torch.ldexp MPS fallback now in saklas.
    "gpt_oss_20b": _mp("gpt_oss_20b", "openai/gpt-oss-20b"),
    # deepseek_v2 MoE, 27 layers. Probe-calibrated 2026-05-03; v3
    # silhouette validation pending vocab pilot.
    "deepseek_v2_lite": _mp(
        "deepseek_v2_lite", "deepseek-ai/DeepSeek-V2-Lite-Chat",
        trust_remote_code=True,
    ),
    # Qwen3.5-27B (previous gen of Qwen3.6), text_config has 64 layers.
    # Multimodal Qwen3_5ForConditionalGeneration wrapper.
    "qwen35_27b": _mp(
        "qwen35_27b", "Qwen/Qwen3.5-27B",
        probe_calibrated=False,
    ),
    # Gemma-3-27b-it (previous gen of Gemma-4-31b), text_config 62 layers.
    # Multimodal Gemma3ForConditionalGeneration wrapper.
    "gemma3_27b": _mp(
        "gemma3_27b", "google/gemma-3-27b-it",
        probe_calibrated=False,
    ),
    # phi3 arch, 32 hidden layers, 3072 hidden, bf16. Probe-calibrated
    # for layers 2-29 (saklas vectors present); preferred_layer set
    # to L23 by happy.sad diff_principal_projection peak (~0.53).
    # Not v3-silhouette validated; vocab pilot first.
    "phi4_mini": _mp("phi4_mini", "microsoft/Phi-4-mini-instruct"),
    # GraniteForCausalLM, dense 30B, 64 hidden layers, 4096 hidden,
    # bf16. Probe-calibrated for layers 2-61 (saklas vectors
    # present); preferred_layer set to L56 by happy.sad
    # diff_principal_projection peak (~0.60). Not v3-silhouette
    # validated; vocab pilot first.
    "granite": _mp("granite", "ibm-granite/granite-4.1-30b"),
    # Japanese-trained encoders for the face-input pipeline only —
    # NOT used in v3 (no probe calibration, no emission run). Both
    # bypass saklas and run through raw HF transformers.
    #
    # nemotron_h hybrid Mamba/attention, 56 hidden layers; deepest
    # hidden_state = index 56 (after final layer).
    "nemotron_jp": _mp(
        "nemotron_jp", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
        use_saklas=False, trust_remote_code=True,
    ),
    # GPTNeoX, 12 hidden layers; deepest hidden_state = index 12.
    "rinna": _mp(
        "rinna", "rinna/japanese-gpt-neox-small",
        use_saklas=False,
    ),
    # GPTNeoX 3.6B PPO-instruct, JP-only training corpus. No probe
    # calibration; face_likelihood-only target. Run with
    # ``--prompt-lang jp`` to test JP-translated kaomoji ask.
    "rinna_jp_3_6b": _mp(
        "rinna_jp_3_6b", "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        use_saklas=False,
    ),
    # GPTNeoX 4B PPO-instruct, EN+JP bilingual training. No probe
    # calibration; face_likelihood-only target. Worth comparing
    # ``--prompt-lang en`` vs ``--prompt-lang jp`` on this model
    # since both languages are in distribution.
    "rinna_bilingual_4b": _mp(
        "rinna_bilingual_4b", "rinna/bilingual-gpt-neox-4b-instruction-ppo",
        use_saklas=False,
    ),
}


def resolve_model(short: str) -> ModelPaths:
    """Lookup ``short`` in MODEL_REGISTRY, applying ``LLMOJI_OUT_SUFFIX``
    iff ``short`` matches the currently-active ``$LLMOJI_MODEL``.

    This is the canonical entry point for ANY caller that wants to load
    model paths by short_name and have suffix routing work. Cross-model
    iterators (e.g. script 21's per-layer emergence sweep, script 38's
    multi-model 3D plot) call this with each model's short_name and get
    the active model's suffix-redirected paths only when relevant —
    other models keep registry semantics untouched.

    For "just give me the active model", use ``current_model()`` (a
    thin wrapper around this).
    """
    import dataclasses
    if short not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {short!r}; known: {sorted(MODEL_REGISTRY)}"
        )
    M = MODEL_REGISTRY[short]
    active = os.environ.get("LLMOJI_MODEL", "gemma")
    out_suffix = os.environ.get("LLMOJI_OUT_SUFFIX")
    if out_suffix and short == active:
        # Suffix variants get a sibling directory under data/local/ — so
        # data/local/gemma/ and data/local/gemma_intro_v7_primed/ are
        # parallel "models" sharing weights but differing in
        # priming/condition. Within each dir, filenames are uniform
        # (emotional_raw.jsonl, emotional_summary.tsv) — the variant
        # identity lives in the directory name. Hidden sidecars and
        # figures follow the same sibling pattern.
        suffixed = f"{M.short_name}_{out_suffix}"
        new_jsonl = DATA_DIR / "local" / suffixed / "emotional_raw.jsonl"
        new_summary = DATA_DIR / "local" / suffixed / "emotional_summary.tsv"
        new_experiment = suffixed
        new_figures = FIGURES_DIR / "local" / suffixed
        M = dataclasses.replace(
            M,
            emotional_data_path=new_jsonl,
            emotional_summary_path=new_summary,
            experiment=new_experiment,
            figures_dir=new_figures,
        )
    return M


def current_model() -> ModelPaths:
    """Resolve the active model from `$LLMOJI_MODEL`. Defaults to
    'gemma' (back-compat). Raises KeyError on an unrecognized name so
    typos fail loudly.

    Honors `$LLMOJI_OUT_SUFFIX`: when set, redirects all model paths
    (emotional_data_path / emotional_summary_path / experiment /
    figures_dir) at the suffixed variant — e.g. with
    ``LLMOJI_OUT_SUFFIX=intro_v7_primed`` on gemma,
    ``M.emotional_data_path`` becomes
    ``data/local/gemma_intro_v7_primed/emotional_raw.jsonl`` and
    sidecar dirs land under
    ``data/local/hidden/gemma_intro_v7_primed/``. Thin wrapper around
    ``resolve_model``.
    """
    name = os.environ.get("LLMOJI_MODEL", "gemma")
    return resolve_model(name)
