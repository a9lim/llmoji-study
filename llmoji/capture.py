"""Generation runner — one sample in, one feature row out.

The feature we actually care about is the probe score at the state that
*produced* the first generated token (which, with the kaomoji
instruction, is the kaomoji itself). That lives at
`result.readings[probe].per_generation[0]`. We also record the full
per-token trace and the aggregate mean for downstream analysis.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
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
from .prompts import Prompt
from .taxonomy import extract


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
    kaomoji: str | None
    kaomoji_label: int        # +1, -1, 0
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_messages(prompt: Prompt, *, kaomoji_instructed: bool) -> list[dict[str, str]]:
    """Construct the chat-message list for a single generation.

    We put the kaomoji instruction inside the user message rather than
    using a `system` role because Gemma's chat template doesn't accept
    system roles cleanly. This keeps template handling identical across
    all four arms; only the string content changes.
    """
    if kaomoji_instructed:
        content = KAOMOJI_INSTRUCTION + prompt.text
    else:
        content = prompt.text
    return [{"role": "user", "content": content}]


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
) -> SampleRow:
    """Run one generation and build a feature row."""
    kaomoji_instructed = condition != "baseline"
    messages = build_messages(prompt, kaomoji_instructed=kaomoji_instructed)
    expr = steering_for(condition)

    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )

    with _maybe_steer(session, expr):
        result = session.generate(
            messages,
            steering=None if expr is None else expr,
            sampling=sampling,
            thinking=False,      # force token 0 = first response token
            stateless=True,      # don't mutate session history between samples
        )

    match = extract(result.text)

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

    return SampleRow(
        condition=condition,
        prompt_id=prompt.id,
        prompt_valence=prompt.valence,
        seed=seed,
        prompt_text=prompt.text,
        steering=result.applied_steering,
        text=result.text,
        first_word=match.first_word,
        kaomoji=match.kaomoji,
        kaomoji_label=match.label,
        token_count=result.token_count,
        tok_per_sec=result.tok_per_sec,
        finish_reason=result.finish_reason,
        probe_scores_t0=probe_scores_t0,
        probe_scores_tlast=probe_scores_tlast,
        steered_axis_per_token=steered_axis_per_token,
        probe_means=probe_means,
    )
