"""Generation runner — one sample in, one feature row out.

The feature we actually care about is the probe score at the state that
*produced* the first generated token (which, with the kaomoji
instruction, is the kaomoji itself). That lives at
`result.readings[probe].per_generation[0]`. We also record the full
per-token trace and the aggregate mean for downstream analysis.

As of the hidden-state refactor: if ``hidden_dir`` is passed to
``run_sample``, a per-row .npz sidecar is written with full-sequence
hidden states at probe layers and the final layer's last-token
attention weights — see ``llmoji.hidden_capture`` and
``llmoji.hidden_state_io``. The SampleRow gets a ``row_uuid`` that keys
into the sidecar. Probe-score fields are kept for back-compat and as
a pre-computed redundant readout.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from saklas import SaklasSession, SamplingConfig

from .config import (
    KAOMOJI_INSTRUCTION,
    MAX_NEW_TOKENS,
    PROBES,
    STEER_ALPHA,
    STEERED_AXIS,
    TEMPERATURE,
)
from .hidden_capture import read_after_generate
from .hidden_state_io import SidecarWriter, hidden_state_path, save_hidden_states
from .prompts import Prompt
from llmoji.taxonomy import extract


# GPT-2-style byte-level encoding decoder. Mistral's reasoning-variant
# tokenizer (e.g. Ministral-3-14B-Reasoning-2512) returns
# `tok.decode(...)` output in BPE-byte-encoded form rather than
# round-tripped UTF-8 — so a kaomoji like `(ﾉ◕ヮ◕)` comes out as
# `(ï¾īâĹķãĥ®âĹķ)`, an emoji like 🎉 as `ðŁİī`, and spaces as `Ġ`.
# The instruct variant decodes properly. Other tokenizers
# (gemma sentencepiece, qwen) also decode properly. The heuristic
# below sniffs for BPE-marker chars (Ġ, Ċ, U+0100..0143) and only
# applies the decode when present — idempotent on already-UTF-8 text.
def _build_byte_decoder() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 174)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def _decode_byte_encoded_text(s: str, *, force: bool = False) -> str:
    """If ``s`` looks BPE-byte-encoded, decode it back to UTF-8.

    Heuristic detection (default): any character in U+0100..U+0143
    (the appended-bytes range — Ġ, Ċ, etc.) implies a BPE marker.
    Returns ``s`` unchanged if no markers are present (avoids false
    positives on gemma/qwen output that happens to contain Latin-1
    supplement chars like `°` or `±`).

    The heuristic FAILS for byte-encoded text whose source bytes all
    fall in the Latin-1 supplement range, e.g. `(ï¿£_ï¿£;)` which is
    `(￣_￣;)` byte-encoded (chars are all U+00A0..00FF). For known
    sources (mistral-reasoning's tokenizer, which always byte-encodes
    non-ASCII output), pass ``force=True`` to skip the heuristic.

    Always returns ``s`` if any character is outside the byte decoder's
    domain (kaomoji chars like `╯` U+256F or `๑` U+0E51 are not in the
    map — KeyError signals "this is already-decoded, leave alone").
    """
    if not force and not any("Ā" <= c <= "Ń" for c in s):
        return s
    try:
        return bytearray(_BYTE_DECODER[c] for c in s).decode(
            "utf-8", errors="replace"
        )
    except KeyError:
        return s


def _is_mistral_reasoning(session) -> bool:
    """True if the session's model_id is the mistral reasoning variant
    (whose tokenizer returns BPE-byte-encoded text)."""
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    return "Ministral-3-14B-Reasoning" in model_id


# gpt-oss-only Lenny-family suppression.
#
# Distinctive Lenny characters: U+0361 ͡ COMBINING DOUBLE INVERTED BREVE
# (eye-cap), U+035C ͜ COMBINING DOUBLE BREVE BELOW (lower eye-cap),
# U+0296 ʖ LATIN LETTER INVERTED GLOTTAL STOP (mouth). All three live in
# UTF-8 2-byte ranges led by 0xCD (combining marks U+0340..U+037F) or
# 0xCA (spacing modifier letters U+0280..U+02BF).
#
# In sentencepiece-style tokenizers (gemma) `͡` and `ʖ` appear as their
# own vocab entries decoding to the unicode char directly. In byte-level
# BPE tokenizers (gpt-oss's o200k_harmony) each unicode char is split
# across multiple single-byte tokens (e.g. `͡` → tokens for bytes
# 0xCD 0xA1, both of which appear in the vocab under the GPT-2 byte
# encoder as `Í` and `¡`). Both paths are handled below.
#
# Scoped to gpt-oss because no other studied model emits Lenny natively —
# universal suppression would blanket-block combining-mark kaomoji on
# gemma/qwen/ministral with no benefit. If/when another model develops
# Lenny dominance, extend the gate.
_LENNY_DISTINCTIVE_CHARS = ("͡", "͜", "ʖ")  # ͡ ͜ ʖ
_LENNY_LEAD_BYTES = frozenset({0xCD, 0xCA})  # UTF-8 leaders for the above
_LENNY_BIAS_VALUE = -100.0  # softmax-effective zero
_LENNY_BIAS_CACHE: dict[int, dict[int, float]] = {}


def _gpt_oss_lenny_logit_bias(session: SaklasSession) -> dict[int, float]:
    """Compute a logit-bias dict that suppresses Lenny-family kaomoji
    on gpt-oss-* sessions; returns empty dict otherwise.

    Why suppress (gpt-oss specifically): Lenny ( ͡° ͜ʖ ͡°) appeared in
    47% of gpt_oss kaomoji emissions across all six emotional quadrants
    in the pilot — including HN-S ("my dog died" prompts) where it's
    contextually wrong. The dominance reflects pretraining-corpus
    contamination (4chan/reddit-lineage text) rather than affective
    state, so blocking it lets the second-choice distribution surface
    where it's more state-discriminative.

    Mechanism: byte-level BPE tokenizers represent the Lenny chars as
    single-byte tokens for the UTF-8 leading bytes 0xCD (eye-caps) and
    0xCA (mouth) plus continuation bytes that vary. Blocking the
    leading-byte tokens prevents Lenny while preserving the rest of
    the kaomoji vocab — collateral suppression is limited to
    U+0280..U+02BF (some IPA chars incl. ʘ) and U+0340..U+037F (other
    combining marks), which gpt-oss doesn't lean on for kaomoji.

    Cached per tokenizer.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    if "gpt-oss" not in model_id.lower():
        return {}

    tokenizer = session.tokenizer
    cache_key = id(tokenizer)
    if cache_key in _LENNY_BIAS_CACHE:
        return _LENNY_BIAS_CACHE[cache_key]

    bias: dict[int, float] = {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}

    for tok_str, tid in vocab.items():
        # Path 1: token decodes to a string containing a Lenny char
        # directly (sentencepiece-style — won't fire on byte-BPE but
        # cheap to keep for robustness if model_id ever points at a
        # gpt-oss variant with a different tokenizer).
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            decoded = ""
        if any(c in decoded for c in _LENNY_DISTINCTIVE_CHARS):
            bias[tid] = _LENNY_BIAS_VALUE
            continue

        # Path 2 + 3 (byte-BPE): map raw vocab string back to bytes via
        # the GPT-2 byte_decoder, then check for either (a) a Lenny char
        # in the round-tripped UTF-8 (multi-byte tokens) or (b) a Lenny
        # leading byte in the raw byte sequence (single-byte tokens).
        try:
            byte_seq = bytes(_BYTE_DECODER[c] for c in tok_str)
        except KeyError:
            continue
        if any(b in _LENNY_LEAD_BYTES for b in byte_seq):
            bias[tid] = _LENNY_BIAS_VALUE
            continue
        try:
            byte_decoded = byte_seq.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(c in byte_decoded for c in _LENNY_DISTINCTIVE_CHARS):
            bias[tid] = _LENNY_BIAS_VALUE

    _LENNY_BIAS_CACHE[cache_key] = bias
    return bias


# Emoji suppression for "noisy-but-coherent" models that have affective
# discrimination but emit emoji rather than parens-kaomoji at start of
# response. Pushes them toward their kaomoji second-choice register so
# we get state-correlated kaomoji output for cross-model comparison.
#
# Coverage:
#   - 4-byte UTF-8 (U+1F300..U+1FFFF, leading byte 0xF0): modern emoji
#     (🎉 😊 🤯 💔 🌧 🍲 etc.). Side-effect: also blocks supplementary-
#     plane chars (math alphanumeric, CJK Ext B+) which kaomoji do not
#     use.
#   - 3-byte UTF-8 with 2-byte prefix 0xE2 followed by 0x98 / 0x9A /
#     0x9B / 0x9C / 0x9E:
#       0x98 → U+2600..U+263F (☀ ☁ ☂ ☃ ☎ ☕ etc.)
#       0x9A → U+2680..U+26BF (⚓ ⚒ ⚖ ⚗ ⚙ ⚛ ⚠ ⚡ ⚧ etc.)
#       0x9B → U+26C0..U+26FF (⛄ ⛅ ⛈ ⛪ ⛰ ⛵ ⛺ etc.)
#       0x9C → U+2700..U+273F (✂ ✅ ✈ ✉ ✊ ✋ ✌ ✍ ✏ ✒ ✔ ✖ ✨ etc.)
#       0x9E → U+2780..U+27BF (➕ ➖ ➗ ➡ ➰ ➿ etc.)
# Deliberately NOT blocked (kaomoji-decoration ranges):
#   0x99 → U+2640..U+267F (♀ ♂ zodiac ♠♡♢♣ card suits, ♨ ♻ ♿)
#   0x9D → U+2740..U+277F (❀❁❂❃ flowers, ❄ snowflake, ❤ heart, ❌❎,
#                         ❓❔❕ punctuation, ❡❦❧ dingbats, circled
#                         digits ❶❷❸)
# These two slabs were dropped 2026-05-03 because they contain real
# kaomoji decorators (✿❀❁ used in `(✿◕‿◕)`, ♥ in `♥(◠‿◠)♥`, card
# suits in face-eye positions) at higher density than emoji-flavored
# chars. The few emoji that slip through (❤ heart, ❄ snowflake, ❌
# X-mark) are acceptable noise vs. losing the kaomoji decorations.
#
# Confirmed safe (NOT suppressed): kaomoji body chars in 0xE2 0x95..0x97
# (╥ ╯ ◕ ╰), 0xEF (﹏ ︶ ︿), 0xE3 (Korean ㅅ ㅂ ㅜ jamo), and BMP-2-byte
# chars (ಥ ° ´).
_EMOJI_4BYTE_LEAD = 0xF0
_EMOJI_3BYTE_PREFIX = (0xE2, frozenset({0x98, 0x9A, 0x9B, 0x9C, 0x9E}))
_EMOJI_SUPPRESS_MODEL_PATTERNS = ("granite", "Ministral-3-14B-Reasoning", "GLM-4")
_EMOJI_BIAS_CACHE: dict[int, dict[int, float]] = {}

# Codepoints that share their UTF-8 byte slab with emoji we want to
# block, but are themselves kaomoji decorations (not emoji). After the
# byte-level bias pass we walk the bias dict and unbias any token whose
# decoded form contains only these chars (whitespace ignored), so e.g.
# `★彡` decoration kaomoji can still be emitted on the noisy-but-coherent
# models. The byte-split case (where ★ tokenizes to single-byte tokens)
# is already allowed naturally — those single bytes don't contain the
# 0xE2 prefix on their own, so they never enter the bias dict.
_KAOMOJI_DECORATION_CODEPOINTS: frozenset[int] = frozenset({
    0x2605,  # ★ BLACK STAR
    0x2606,  # ☆ WHITE STAR
    0x2726,  # ✦ BLACK FOUR POINTED STAR
    0x2727,  # ✧ WHITE FOUR POINTED STAR (also in KAOMOJI_START_CHARS)
    0x2729,  # ✩ STRESS OUTLINED WHITE STAR
    0x272A,  # ✪ CIRCLED WHITE STAR (already in KAOMOJI_START_CHARS)
    0x273F,  # ✿ BLACK FLORETTE (small flower used in `(✿◕‿◕)`)
})


def _emoji_logit_bias(session: SaklasSession) -> dict[int, float]:
    """Compute a logit-bias dict that suppresses emoji on the
    "noisy-but-coherent" models (granite, ministral) which have
    real affective discrimination but emit emoji rather than
    parens-kaomoji at start of response.

    Returns empty dict for any other model. Gate is on
    ``_EMOJI_SUPPRESS_MODEL_PATTERNS`` matching the session's model_id.

    Effect on covered models: pushes the next-token distribution off
    emoji (which would otherwise dominate ~63% of ministral T=1 emits
    and ~80% of granite emits) and onto the second-choice register —
    kaomoji for these models per the v3-trio-comparison pilot data.

    Cached per tokenizer.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    if not any(p in model_id for p in _EMOJI_SUPPRESS_MODEL_PATTERNS):
        return {}

    tokenizer = session.tokenizer
    cache_key = id(tokenizer)
    if cache_key in _EMOJI_BIAS_CACHE:
        return _EMOJI_BIAS_CACHE[cache_key]

    e2_lead, e2_mid_range = _EMOJI_3BYTE_PREFIX

    bias: dict[int, float] = {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}

    for tok_str, tid in vocab.items():
        # Path 1: directly-decoded form (sentencepiece)
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            decoded = ""
        if decoded:
            try:
                decoded_bytes = decoded.encode("utf-8")
            except UnicodeEncodeError:
                decoded_bytes = b""
            if _EMOJI_4BYTE_LEAD in decoded_bytes:
                bias[tid] = _LENNY_BIAS_VALUE
                continue
            for i in range(len(decoded_bytes) - 1):
                if decoded_bytes[i] == e2_lead and decoded_bytes[i + 1] in e2_mid_range:
                    bias[tid] = _LENNY_BIAS_VALUE
                    break
            if tid in bias:
                continue

        # Path 2: byte-BPE — map raw vocab string to bytes via byte_decoder
        try:
            byte_seq = bytes(_BYTE_DECODER[c] for c in tok_str)
        except KeyError:
            continue
        if _EMOJI_4BYTE_LEAD in byte_seq:
            bias[tid] = _LENNY_BIAS_VALUE
            continue
        for i in range(len(byte_seq) - 1):
            if byte_seq[i] == e2_lead and byte_seq[i + 1] in e2_mid_range:
                bias[tid] = _LENNY_BIAS_VALUE
                break

    # Decoration whitelist pass: unbias tokens whose decoded OR
    # byte-decoded form is exclusively kaomoji-decoration codepoints
    # (whitespace ignored). Rescues ★ ☆ etc. on both sentencepiece-
    # style tokenizers (decode returns the actual char) and byte-BPE
    # tokenizers (decode returns the byte-encoded form like `'âĺħ'`,
    # so we also byte-decode the raw vocab string).
    def _decoration_only(s: str) -> bool:
        nonws = [c for c in s if not c.isspace()]
        return bool(nonws) and all(
            ord(c) in _KAOMOJI_DECORATION_CODEPOINTS for c in nonws
        )

    for tok_str, tid in vocab.items():
        if tid not in bias:
            continue
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            decoded = ""
        if _decoration_only(decoded):
            del bias[tid]
            continue
        try:
            byte_seq = bytes(_BYTE_DECODER[c] for c in tok_str)
            byte_decoded = byte_seq.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        if _decoration_only(byte_decoded):
            del bias[tid]

    _EMOJI_BIAS_CACHE[cache_key] = bias
    return bias


def _compose_logit_bias(session: SaklasSession) -> dict[int, float]:
    """Combine all applicable per-model logit-bias suppressions.
    Returns empty dict (treated as None at SamplingConfig level) when
    no suppression applies."""
    out: dict[int, float] = {}
    out.update(_gpt_oss_lenny_logit_bias(session))
    out.update(_emoji_logit_bias(session))
    return out


@dataclass
class SampleRow:
    """One generation's worth of data, ready to JSONL-dump."""

    # --- experimental bookkeeping ---
    condition: str
    prompt_id: str
    prompt_valence: int
    seed: int

    # --- what we asked the model ---
    prompt_text: str
    steering: str | None  # canonical expression string, or None

    # --- what the model did ---
    text: str
    first_word: str
    token_count: int
    tok_per_sec: float
    finish_reason: str

    # --- feature vector: score at state producing token 0 ---
    # one float per probe in PROBES, same order
    probe_scores_t0: list[float]

    # --- feature vector: score at state producing the final token ---
    # one float per probe in PROBES, same order. Mirrors probe_scores_t0
    # but reads per_generation[-1]. Required — old pilot_raw.jsonl rows
    # will fail to deserialize until re-run under this schema.
    probe_scores_tlast: list[float]

    # --- auxiliary: per-token trace for the axis of interest ---
    # ProbeReadings.per_generation for STEERED_AXIS; lets us sanity-check
    # that token 0 is actually where the kaomoji emerges in the causal
    # arms and see how state evolves across the response.
    steered_axis_per_token: list[float]

    # --- aggregate stats across the full generation ---
    probe_means: dict[str, float]

    # --- hidden-state sidecar bookkeeping ---
    # Populated when run_sample is called with hidden_dir; otherwise
    # empty string (rows from pre-refactor runs won't have this).
    row_uuid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_messages(
    prompt: Prompt,
    *,
    kaomoji_instructed: bool,
    extra_preamble: str | None = None,
    instruction_override: str | None = None,
) -> list[dict[str, str]]:
    """Construct the chat-message list for a single generation.

    We put the kaomoji instruction inside the user message rather than
    using a `system` role because Gemma's chat template doesn't accept
    system roles cleanly. This keeps template handling identical across
    all arms; only the string content changes.

    ``extra_preamble`` is prepended to the kaomoji instruction when
    ``kaomoji_instructed=True``. Used by the introspection pilot
    (script 32) to inject INTROSPECTION_PREAMBLE / LOREM_PREAMBLE
    without duplicating the message-build logic.

    ``instruction_override`` swaps in a non-default kaomoji instruction
    (e.g. ``KAOMOJI_INSTRUCTION_JP`` for Japanese-only encoders). Only
    honored when ``kaomoji_instructed=True``.
    """
    if kaomoji_instructed:
        instruction = instruction_override if instruction_override is not None else KAOMOJI_INSTRUCTION
        if extra_preamble:
            instruction = _ensure_trailing_whitespace(extra_preamble) + instruction
        # Guard the preamble→prompt boundary too — preamble files that
        # omit a trailing newline (e.g. introspection_v3.txt) would
        # otherwise concatenate directly with the prompt body
        # (`...feel.offer letter...`), producing weird tokenizer-
        # boundary states. KAOMOJI_INSTRUCTION already ends with
        # ". " so this is a no-op for the default path.
        instruction = _ensure_trailing_whitespace(instruction)
        content = instruction + prompt.text
    else:
        content = prompt.text
    return [{"role": "user", "content": content}]


def _ensure_trailing_whitespace(s: str) -> str:
    """Append a single space if ``s`` ends with an ASCII non-whitespace
    character.

    Used at the boundaries of ``build_messages`` concatenation
    (preamble→instruction, instruction→prompt) to prevent ASCII
    preamble files / instruction overrides that omit a trailing
    newline from running together with neighbours
    (e.g. ``introspection_v3.txt`` ending ``"feel."`` would
    otherwise concatenate as ``"feel.offer letter..."``).

    Non-ASCII trailing characters (e.g. ``KAOMOJI_INSTRUCTION_JP``
    ending in ``"。"``) are left alone — historical face_likelihood
    JP-encoder data was generated without the separator and CJK
    tokenizer boundaries don't need ASCII space.
    """
    if not s:
        return s
    last = s[-1]
    if last.isspace() or ord(last) > 127:
        return s
    return s + " "


_GPT_OSS_GENPROMPT_SENTINEL = "<|start|>assistant\n{%- endif -%}"
_GPT_OSS_GENPROMPT_FINAL_PIN = (
    "<|start|>assistant<|channel|>final<|message|>\n{%- endif -%}"
)


def maybe_override_gpt_oss_chat_template(session: SaklasSession) -> bool:
    """Patch ``session.tokenizer.chat_template`` for openai/gpt-oss-* to
    pin the Harmony-format ``final`` channel at generation time, skipping
    the ``analysis`` (chain-of-thought) channel.

    Why: gpt-oss's harmony chat template's ``add_generation_prompt`` block
    emits ``<|start|>assistant``, leaving the model to choose the next
    channel. Trained behavior is to emit ``<|channel|>analysis|>...``
    first (reasoning trace), then ``<|channel|>final|>`` for the
    user-facing reply. Under MAX_NEW_TOKENS=16 the analysis trace eats
    the entire budget and no kaomoji emits in the final channel.

    Fix: pin the assistant turn to start with
    ``<|start|>assistant<|channel|>final<|message|>`` directly. Model
    wakes up inside the final channel; first token of generation is the
    kaomoji. Caveat: response quality may degrade because the model is
    skipping its trained reasoning step — fine for first-token kaomoji
    measurement; **do not reuse this override for tasks that need
    reasoning quality**.

    Returns True when the override fired, False otherwise (allows
    callers to log). Idempotent — safe to call multiple times.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    if "gpt-oss" not in model_id.lower():
        return False
    template = session.tokenizer.chat_template
    if template is None:
        return False
    if _GPT_OSS_GENPROMPT_FINAL_PIN in template:
        return False  # already patched
    if _GPT_OSS_GENPROMPT_SENTINEL not in template:
        return False  # template structure changed upstream; bail safely
    session.tokenizer.chat_template = template.replace(
        _GPT_OSS_GENPROMPT_SENTINEL, _GPT_OSS_GENPROMPT_FINAL_PIN
    )
    return True


_RINNA_PPO_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "ユーザー: {{ message['content'] }}\n"
    "{% elif message['role'] == 'system' -%}"
    "システム: {{ message['content'] }}\n"
    "{% endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}システム: {% endif -%}"
)


def maybe_override_rinna_chat_template(session: SaklasSession) -> bool:
    """Patch ``session.tokenizer.chat_template`` for rinna PPO instruct
    models to their native ``ユーザー: …\\nシステム: `` frame.

    Why: rinna/japanese-gpt-neox-3.6b-instruction-ppo and
    rinna/bilingual-gpt-neox-4b-instruction-ppo ship without a
    ``chat_template`` attribute on the tokenizer (verified). saklas's
    fallback wraps user content as ``User: …\\nAssistant:`` (English
    boilerplate), which puts the JP-trained model in an off-distribution
    framing. The HF model cards document the native format as
    ``ユーザー: <content>\\nシステム: `` for inference; we install that
    Jinja so saklas's ``apply_chat_template`` produces the right prefix.

    Returns True when the override fired, False otherwise. Idempotent.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    mid_lower = model_id.lower()
    if "rinna" not in mid_lower or "ppo" not in mid_lower:
        return False
    if session.tokenizer.chat_template == _RINNA_PPO_CHAT_TEMPLATE:
        return False  # already installed
    session.tokenizer.chat_template = _RINNA_PPO_CHAT_TEMPLATE
    return True


def maybe_override_ministral_chat_template(session: SaklasSession) -> bool:
    """Patch ``session.tokenizer.chat_template`` for the Ministral
    Reasoning variant to use the FP8-Instruct's chat template.

    Why: Ministral-3-14B-Reasoning-2512's native chat_template ships
    with a `[THINK]…[/THINK]` system block and **ignores
    ``enable_thinking=False``** (verified — the template returns the
    same 614-char output with or without the flag). Result: under
    MAX_NEW_TOKENS=16 the generation budget is consumed by the
    thinking trace and no kaomoji is emitted (~0% emit rate).

    Fix: download the sibling FP8-Instruct's chat_template.jinja and
    swap it in. Same base weights (~99% similar — only post-training
    differs), no thinking system block, generation works.

    Returns True when the override fired, False otherwise (allows
    callers to log). Idempotent — safe to call multiple times.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    if "Ministral-3-14B-Reasoning" not in model_id:
        return False
    from huggingface_hub import hf_hub_download
    instruct_jinja_path = hf_hub_download(
        "mistralai/Ministral-3-14B-Instruct-2512",
        "chat_template.jinja",
    )
    with open(instruct_jinja_path) as fh:
        session.tokenizer.chat_template = fh.read()
    return True


def install_linear_attention_cache_patch() -> bool:
    """Add ``batch_repeat_interleave`` to transformers' linear-attention
    cache layers so prefix-cache tiling works on hybrid models.

    Why: ``transformers/cache_utils.py`` defines
    ``batch_repeat_interleave`` on ``DynamicLayer`` (lines ~150) but NOT
    on ``LinearAttentionCacheLayerMixin`` / ``LinearAttentionLayer``.
    Qwen3.6-27B is a hybrid: pure-LA layers + ``LinearAttentionAndFull
    AttentionLayer`` layers. ``Cache.batch_repeat_interleave`` iterates
    every layer and calls ``layer.batch_repeat_interleave(repeats)`` —
    pure-LA layers AttributeError; hybrid layers silently fall through
    to ``DynamicLayer``'s version, which tiles only KV (not the LA
    state), producing a shape-mismatched cache for face-batch tiling.

    Patch: install ``batch_repeat_interleave`` on ``LinearAttentionLayer``
    that ``repeat_interleave``s ``conv_states``/``recurrent_states``
    along dim 0; override on ``LinearAttentionAndFullAttentionLayer`` to
    chain BOTH parents' versions (LA state + KV).

    Used by ``scripts/local/50_face_likelihood.py``'s
    ``_expand_kv_cache`` for prefix-cache tiling. Idempotent.
    Returns True when the patch was newly applied, False if already
    installed or the transformers version doesn't expose these classes.
    """
    try:
        from transformers.cache_utils import (
            LinearAttentionLayer,
            LinearAttentionAndFullAttentionLayer,
            DynamicLayer,
        )
    except ImportError:
        return False

    if getattr(LinearAttentionLayer, "_llmoji_bri_installed", False):
        return False

    def _la_batch_repeat_interleave(self, repeats: int) -> None:
        if getattr(self, "is_conv_states_initialized", False):
            self.conv_states = self.conv_states.repeat_interleave(repeats, dim=0)
        if getattr(self, "is_recurrent_states_initialized", False):
            self.recurrent_states = self.recurrent_states.repeat_interleave(repeats, dim=0)
        if hasattr(self, "max_batch_size"):
            self.max_batch_size = self.max_batch_size * repeats

    def _hybrid_batch_repeat_interleave(self, repeats: int) -> None:
        _la_batch_repeat_interleave(self, repeats)
        DynamicLayer.batch_repeat_interleave(self, repeats)

    setattr(LinearAttentionLayer, "batch_repeat_interleave", _la_batch_repeat_interleave)
    setattr(
        LinearAttentionAndFullAttentionLayer,
        "batch_repeat_interleave",
        _hybrid_batch_repeat_interleave,
    )
    setattr(LinearAttentionLayer, "_llmoji_bri_installed", True)
    return True


install_linear_attention_cache_patch()


def install_prefix_cache(
    session: SaklasSession,
    prompts: list[Prompt],
    *,
    extra_preamble: str | None = None,
    instruction_override: str | None = None,
    kaomoji_instructed: bool = True,
) -> int:
    """Compute the longest common chat-template token prefix across
    ``prompts`` (under the same condition) and install it via
    ``session.cache_prefix()``. Returns the prefix length in tokens.

    For v3-shaped runs the common prefix covers the chat-template head
    + ``KAOMOJI_INSTRUCTION`` (+ ``extra_preamble`` if any); the variable
    suffix per call is just the prompt body + assistant-turn-start.
    Halves per-call prefill in practice. Must be called outside any
    ``session.steering()`` scope. Re-call with a different prompt set or
    preamble to replace the cache.

    ``instruction_override`` swaps in a non-default kaomoji instruction
    (replaces ``KAOMOJI_INSTRUCTION`` rather than prepending). Used for
    introspection preambles whose own integrated kaomoji ask should be
    the sole instruction (replaces the bare KAOMOJI_INSTRUCTION),
    matching the JP drop-in plumbing used for ``KAOMOJI_INSTRUCTION_JP``.
    """
    if not prompts:
        return 0
    import torch
    tok = session.tokenizer
    all_ids: list[list[int]] = []
    for p in prompts:
        msgs = build_messages(
            p,
            kaomoji_instructed=kaomoji_instructed,
            extra_preamble=extra_preamble,
            instruction_override=instruction_override,
        )
        result = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
        )
        ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        all_ids.append(ids[0].tolist())

    common = all_ids[0]
    for other in all_ids[1:]:
        n = min(len(common), len(other))
        i = 0
        while i < n and common[i] == other[i]:
            i += 1
        common = common[:i]
        if not common:
            return 0
    if not common:
        return 0
    return session.cache_prefix(torch.tensor(common, dtype=torch.long))


def install_full_input_cache(
    session: SaklasSession,
    prompt: Prompt,
    *,
    extra_preamble: str | None = None,
    instruction_override: str | None = None,
    kaomoji_instructed: bool = True,
) -> int:
    """Cache the full chat-templated input for ``prompt`` (minus the
    final token) via ``session.cache_prefix()``. Returns the cached
    length.

    Useful when the same prompt runs N>1 times with different seeds
    (v3 main: 8 seeds/cell). Seeds 2..N hit the cache exactly and
    do only a 1-token suffix prefill + decode — effectively
    decode-only after the first seed. Caching N-1 (not the full
    input) avoids the zero-suffix edge case in saklas's generate
    flow.

    Replaces any prior cache. Call at the top of each prompt's
    seed loop. Degenerate at N=1 — use ``install_prefix_cache``
    over the prompt set instead.

    **Qwen bypass (2026-05-03):** on Qwen3.6 the cache_prefix path
    produces contaminated KV state — every seed 1..N decodes
    identical off-prompt text (markdown headers, code docs,
    unrelated content) regardless of the input prompt. Root cause
    is on the saklas side (qwen-tokenizer / model interaction);
    pragmatic fix is to skip the cache install for Qwen and pay
    the ~30-50% per-row prefill cost. Gemma + Mistral are
    unaffected. See docs/gotchas.md.
    """
    model_id = (
        getattr(session, "model_id", None)
        or getattr(getattr(session, "config", None), "model_id", "")
        or ""
    )
    if "qwen" in model_id.lower():
        return 0
    import torch
    msgs = build_messages(
        prompt,
        kaomoji_instructed=kaomoji_instructed,
        extra_preamble=extra_preamble,
        instruction_override=instruction_override,
    )
    result = session.tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
    )
    full = result if isinstance(result, torch.Tensor) else result["input_ids"]
    return session.cache_prefix(full[0, :-1])


_STEER_EXPR_BY_CONDITION = {
    "steered_happy": "happy",
    "steered_sad":   "sad",
    "steered_angry": "angry",
    "steered_calm":  "calm",
}


def steering_for(condition: str) -> str | None:
    """Steering expression for a given arm, or None.

    Bare pole names (``happy``, ``sad``, ``angry``, ``calm``) resolve
    through saklas's shared steering-expression grammar — each maps to
    its bipolar parent with the correct sign (``sad`` → ``happy.sad
    @ -α``, etc.).
    """
    pole = _STEER_EXPR_BY_CONDITION.get(condition)
    if pole is None:
        return None
    return f"{STEER_ALPHA} {pole}"


@contextmanager
def _maybe_steer(session: SaklasSession, expr: str | None) -> Iterator[None]:
    """Context manager that steers iff expr is not None.

    Keeps the caller branch-free.
    """
    if expr is None:
        with nullcontext():
            yield
    else:
        with session.steering(expr):
            yield


def run_sample(
    session: SaklasSession,
    *,
    prompt: Prompt,
    condition: str,
    seed: int,
    hidden_dir: Path | None = None,
    experiment: str = "default",
    # no v3 analysis script reads `hidden_L<idx>` post-h_first cutover; ~60× sidecar shrink.
    store_full_trace: bool = False,
    extra_preamble: str | None = None,
    instruction_override: str | None = None,
    override_max_tokens: int | None = None,
    sidecar_writer: SidecarWriter | None = None,
) -> SampleRow:
    """Run one generation and build a feature row.

    If ``hidden_dir`` is given, writes a per-row sidecar of hidden
    states + attention weights under
    ``<hidden_dir>/hidden/<experiment>/<row_uuid>.npz`` — enables
    post-hoc probe computation and cosine-in-hidden-state analysis.
    Default ``store_full_trace=False`` stores only the three aggregate
    snapshots (h_first / h_last / h_mean) per layer, dropping the
    per-token trace — ~60x smaller sidecars at the cost of losing
    mid-sequence access. No v3 analysis script reads ``hidden_L<idx>``
    post-h_first cutover, so this is safe for the hot path. Pass
    ``store_full_trace=True`` explicitly for the smoke test or any
    analysis that needs mid-sequence states.

    ``sidecar_writer`` (optional, mutually exclusive with the inline
    save path): if provided, the sidecar write is enqueued on the
    writer's background thread and this function returns as soon as
    the CPU snapshot is built. The runner owns the writer and is
    responsible for draining/closing it. ``hidden_dir`` is still
    required for canonical path resolution.

    ``extra_preamble`` (originally for the introspection pilot's lorem
    control) is prepended to ``KAOMOJI_INSTRUCTION`` inside the user
    message. Treated as "kaomoji_instructed = True" regardless of the
    condition string, since the preamble itself implies a kaomoji-
    emission setup.

    ``instruction_override`` (used 2026-05-04 onward by the
    introspection-preamble pilots) replaces ``KAOMOJI_INSTRUCTION``
    with the supplied string. Use this for preambles that include
    their own integrated kaomoji ask (v2/v3/v4/v5 introspection-
    preamble iterations) — without override, the bare
    ``KAOMOJI_INSTRUCTION`` would stack after the preamble's ask
    creating a redundant double-ask. Mirrors the JP plumbing used
    for ``KAOMOJI_INSTRUCTION_JP`` on Japanese encoders. Like
    ``extra_preamble``, also forces ``kaomoji_instructed = True``.

    ``override_max_tokens`` (used by the introspection pilot's
    hard-early-stop) sets ``max_tokens`` on the SamplingConfig in
    place of the registered MAX_NEW_TOKENS default.
    """
    kaomoji_instructed = (
        condition != "baseline"
        or extra_preamble is not None
        or instruction_override is not None
    )
    messages = build_messages(
        prompt,
        kaomoji_instructed=kaomoji_instructed,
        extra_preamble=extra_preamble,
        instruction_override=instruction_override,
    )
    expr = steering_for(condition)

    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=override_max_tokens if override_max_tokens is not None else MAX_NEW_TOKENS,
        seed=seed,
        logit_bias=_compose_logit_bias(session) or None,
    )

    with _maybe_steer(session, expr):
        result = session.generate(
            messages,
            steering=None if expr is None else expr,
            sampling=sampling,
            thinking=False,      # force token 0 = first response token
            stateless=True,      # don't mutate session history between samples
        )

    # Mistral reasoning's tokenizer returns BPE-byte-encoded text rather
    # than UTF-8; decode if so. force=True for that variant (where the
    # heuristic alone misses Latin-1-only sequences); otherwise the
    # heuristic-based default preserves text from other tokenizers.
    decoded_text = _decode_byte_encoded_text(
        result.text, force=_is_mistral_reasoning(session),
    )
    match = extract(decoded_text)

    # Real per-token probe scores live on session.last_per_token_scores
    # rather than on result.readings[...].per_generation — see the
    # saklas gotcha in CLAUDE.md. Under stateless=True (our mode),
    # ProbeReadings.per_generation collapses to a single whole-
    # generation mean, so indexing [0] and [-1] both return the
    # aggregate. The per-token dict is what we actually want for t=0
    # vs t=last separation.
    per_token_scores: dict[str, list[float]] = (
        getattr(session, "last_per_token_scores", None) or {}
    )

    # Token-0 probe scores in canonical PROBES order.
    probe_scores_t0: list[float] = []
    for probe in PROBES:
        seq = per_token_scores.get(probe) or []
        if seq:
            probe_scores_t0.append(float(seq[0]))
        else:
            # Fallback to the (aggregate) readings when per-token is
            # unavailable — keeps the field populated on old saklas
            # versions or probe-less runs.
            readings = result.readings.get(probe)
            if readings is None or not readings.per_generation:
                probe_scores_t0.append(float("nan"))
            else:
                probe_scores_t0.append(float(readings.per_generation[0]))

    # Final-token probe scores in canonical PROBES order.
    probe_scores_tlast: list[float] = []
    for probe in PROBES:
        seq = per_token_scores.get(probe) or []
        if seq:
            probe_scores_tlast.append(float(seq[-1]))
        else:
            readings = result.readings.get(probe)
            if readings is None or not readings.per_generation:
                probe_scores_tlast.append(float("nan"))
            else:
                probe_scores_tlast.append(float(readings.per_generation[-1]))

    # Per-token trace for the steered axis. Prefer the per-token dict
    # (real per-token data) over readings.per_generation (which is
    # length-1 under stateless=True).
    steered_seq = per_token_scores.get(STEERED_AXIS)
    if steered_seq:
        steered_axis_per_token = [float(x) for x in steered_seq]
    else:
        steered_axis_readings = result.readings.get(STEERED_AXIS)
        if steered_axis_readings is None:
            steered_axis_per_token = []
        else:
            steered_axis_per_token = [float(x) for x in steered_axis_readings.per_generation]

    probe_means = {
        probe: (
            float(result.readings[probe].mean)
            if probe in result.readings else float("nan")
        )
        for probe in PROBES
    }

    # Hidden-state sidecar capture (if requested). Reads directly from
    # saklas's post-generation capture buckets — no extra forward pass.
    # The capture itself is gated by ``store_full_trace``: with it
    # False (the default, hot path), the per-token trace never crosses
    # the device boundary — only the three aggregates do.
    row_uuid = ""
    if hidden_dir is not None:
        row_uuid = uuid.uuid4().hex
        capture = read_after_generate(session, store_full_trace=store_full_trace)
        sidecar = hidden_state_path(hidden_dir, experiment, row_uuid)
        if sidecar_writer is not None:
            # Async path: enqueue and return. Caller owns drain/close.
            sidecar_writer.submit(capture, sidecar, store_full_trace=store_full_trace)
        else:
            save_hidden_states(capture, sidecar, store_full_trace=store_full_trace)

    return SampleRow(
        condition=condition,
        prompt_id=prompt.id,
        prompt_valence=prompt.valence,
        seed=seed,
        prompt_text=prompt.text,
        steering=result.applied_steering,
        text=decoded_text,
        first_word=match.first_word,
        token_count=result.token_count,
        tok_per_sec=result.tok_per_sec,
        finish_reason=result.finish_reason,
        probe_scores_t0=probe_scores_t0,
        probe_scores_tlast=probe_scores_tlast,
        steered_axis_per_token=steered_axis_per_token,
        probe_means=probe_means,
        row_uuid=row_uuid,
    )
