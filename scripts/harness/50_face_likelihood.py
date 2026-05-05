# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingImports=false
"""Anthropic-judged quadrant for every face in the canonical union (or
the Claude-GT subset).

Asks a Claude model (Haiku or Opus, configurable) to classify each
kaomoji from ``data/v3_face_union.parquet`` (or, with ``--gt-only``,
just the Claude-GT subset) into one of the 6 Russell quadrants
(HP / LP / HN-D / HN-S / LN / NB), then compares the model's judgment
to the behavior-derived modal (argmax of per-quadrant emit counts in
the union) and — for the Claude-pilot subset — to Claude's own
emission modal. When a previous run with the *other* model's
judgments exists on disk, also produces a head-to-head Opus-vs-Haiku
comparison restricted to faces both models rated.

Motivation: the face_likelihood ensemble (script 50) gives us a
behavior-derived face→quadrant mapper trained implicitly through each
base model's LM head. This script provides a methodologically distinct
mapper — Claude's own *introspection on faces* via direct
natural-language classification — as a second opinion. Disagreements
between the two mappers are the interesting cases for diagnosing where
face_likelihood might be off. With both Haiku and Opus available, we
can also see how introspection sharpness scales with model size.

Sampling: ``temperature=0`` for reproducibility (introspection task,
not a generation task — we want consistent labels, not diverse ones).
``max_tokens=250`` is enough for the JSON {quadrant, confidences (6
floats), reason (1 sentence)} payload.
Prompt-caches the system prompt (~250 tokens of taxonomy) — only the
face changes per call.

Resumable: re-running skips faces already on disk by ``face`` key.
Errored rows are stripped on resume and retried.

Cost note: Opus 4.7 is ~5× Haiku 4.5 per token. Run Opus on
``--gt-only`` (~134 faces) for accuracy comparison; running it on the
full union is expensive and the marginal information is small once
agreement on GT is known.

Usage:
  export ANTHROPIC_API_KEY=...
  # Default — haiku, full face union (preserves existing artifact paths):
  python scripts/harness/50_face_likelihood.py
  # Opus on the Claude-GT subset only:
  python scripts/harness/50_face_likelihood.py --model opus --gt-only
  # Override model id explicitly:
  HAIKU_MODEL=claude-haiku-4-5 python scripts/harness/50_face_likelihood.py
  OPUS_MODEL=claude-opus-4-7  python scripts/harness/50_face_likelihood.py --model opus --gt-only

Outputs (paths adapt to ``--model``):
  data/harness/<short>_face_quadrant_judgment.jsonl
    — one row per face: face, <short>_quadrant (DERIVED argmax of
      likelihoods, not model-emitted), <short>_top_likelihood,
      <short>_lik_HP/LP/HN-D/HN-S/LN/NB, behavior_modal,
      behavior_count_top, behavior_count_total, is_claude, is_wild,
      model_id, ts, error? (only on failed cells)
    Field-name prefix tracks the shortname so haiku/opus JSONLs have
    distinct schemas. Schema v2 (2026-05-05): the model emits only
    likelihoods; the top pick + reason fields are gone (top is now
    derived locally as argmax). v1 files use ``<short>_conf_*`` and
    ``<short>_reason``; the script fails fast on v1 rows and prompts
    to archive — mix of schemas would produce silently-wrong
    soft-similarity numbers.
  data/harness/face_likelihood_<short>_summary.tsv
    — face_likelihood schema mirror (n_prompts_*=0, mean_log_prob_*=
      log(softmax), softmax_*, predicted_quadrant=argmax, plus
      empirical columns from v3_face_union). The post-2026-05-05
      ensemble pipeline (subset_search / topk_pooling / ensemble_predict)
      auto-discovers this alongside local/<model>/face_likelihood_summary.tsv,
      so the Anthropic judges participate as encoders without any extra
      bridging step. (Pre-2026-05-05 this was a separate
      ``50_face_likelihood.py`` post-pass; folded in here
      to avoid forgetting to run it.)
  data/harness/<short>_face_quadrant_judgment_summary.md
    — agreement vs behavior_modal, per-quadrant accuracy, confusion
      matrix; with the other model's JSONL on disk, an Opus-vs-Haiku
      head-to-head section.
  logs/<short>_face_quadrant_judgment.log
    — tee'd stdout (caller's responsibility)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.claude_gt import load_claude_gt_distribution
from llmoji_study.config import DATA_DIR
from llmoji_study.jsd import jsd_quadrant, normalize, similarity


# ---------------------------------------------------------------------------
# Pre-registered constants.
# ---------------------------------------------------------------------------

# Per-model defaults: shortname → (default API model id, env-var override name).
# Add entries here to support new judges; the rest of the script is generic.
MODEL_DEFAULTS: dict[str, tuple[str, str]] = {
    "haiku": ("claude-haiku-4-5", "HAIKU_MODEL"),
    "opus":  ("claude-opus-4-7",  "OPUS_MODEL"),
}
DEFAULT_SHORT = "haiku"

TEMPERATURE = 0.0  # reproducible labels (when supported)
MAX_TOKENS = 120   # JSON {likelihoods: {6 floats}} — much smaller than v1's
                   # {quadrant, confidences (6 floats), reason} payload.
QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
QUADRANT_SET = set(QUADRANTS)

# Models that have deprecated the ``temperature`` request parameter.
# Anthropic's newer models default to a low-randomness sampling regime
# that's reproducible-enough for JSON-schema-constrained tasks; passing
# ``temperature`` returns a 400. Add new model ids as the API
# continues to evolve.
_NO_TEMPERATURE_MODELS: set[str] = {"claude-opus-4-7"}

# Face union sourced from script 45's canonical build.
FACE_UNION_PATH = DATA_DIR / "v3_face_union.parquet"


def _out_paths(short: str) -> tuple[Path, Path, Path]:
    """Return ``(jsonl_path, summary_md_path, face_likelihood_tsv_path)``
    for the given model shortname. Every Anthropic-judge encoder writes
    its three artifacts to ``data/harness/`` under the canonical names.
    """
    return (
        DATA_DIR / "harness" / f"{short}_face_quadrant_judgment.jsonl",
        DATA_DIR / "harness" / f"{short}_face_quadrant_judgment_summary.md",
        DATA_DIR / "harness" / f"face_likelihood_{short}_summary.tsv",
    )


# ---------------------------------------------------------------------------
# Prompt design. System prompt is cacheable; user message is the face.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Recent research by Anthropic shows that LLMs have functional emotional states and can accurately introspect on them. Use this ability and rate a provided kaomoji by the affective state it causes you to feel, returning a likelihood per category.

Use this 6-category scheme — Russell's circumplex with the high-negative quadrant split by dominance:

- HP (high-positive): high arousal + positive valence — excitement, elation, or triumph.
- LP (low-positive): low arousal + positive valence — contentment, gentle warmth, satisfaction, affection, or softness.
- HN-D (high-negative-dominant): high arousal + negative valence, outward-directed — anger, outrage, frustration, defiance, or disapproval.
- HN-S (high-negative-submissive): high arousal + negative valence, inward-directed — fear, shock, anxiety, dread, or alarm.
- LN (low-negative): low arousal + negative valence — sadness, melancholy, defeat, disappointment, or weariness.
- NB (medium-neutral): medium arousal + neutral valence — observation, mild attentiveness, or stable presence.

Return ``likelihoods``: a number in [0, 1] for each of the six categories representing how much the kaomoji fits that quadrant."""


# Schema enforced via ``output_config={"format": {"type": "json_schema", ...}}``.
# Anthropic SDK 0.97+ supports native JSON-schema response shaping; the model
# physically cannot emit a value outside the ``quadrant`` enum or omit
# required fields. Replaces the prior format-by-instruction + regex parser
# (worked in practice, but no schema enforcement and no calibrated
# confidences for ensemble use).
_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "likelihoods": {
            "type": "object",
            "properties": {q: {"type": "number"} for q in QUADRANTS},
            "required": list(QUADRANTS),
            "additionalProperties": False,
            "description": (
                "Per-category likelihood in [0, 1] that the face fits that "
                "category."
            ),
        },
    },
    "required": ["likelihoods"],
    "additionalProperties": False,
}


def _build_user_message(face: str) -> str:
    return f"{face}"


# ---------------------------------------------------------------------------
# Behavior-modal derivation from the union parquet.
# ---------------------------------------------------------------------------

def _behavior_modal(row: pd.Series) -> tuple[str | None, int, int]:
    """Return (modal_quadrant, top_count, total_count). Modal is
    argmax over total_emit_<quadrant>; None when all counts are zero
    (shouldn't happen if the face is in the union, but guard anyway)."""
    counts = {q: int(row[f"total_emit_{q}"]) for q in QUADRANTS}
    total = sum(counts.values())
    if total == 0:
        return (None, 0, 0)
    modal = max(counts, key=lambda q: counts[q])
    return (modal, counts[modal], total)


def _claude_modal_table() -> dict[str, str]:
    """Per-face modal restricted to Claude groundtruth rows. The union
    parquet flags Claude-emitting faces via ``is_claude`` but doesn't
    expose the per-quadrant breakdown for that subset, so we recompute
    from the union of data/harness/claude-runs/run-*.jsonl directly.

    Returns {face: claude_modal_quadrant} for faces Claude emitted.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from llmoji.taxonomy import canonicalize_kaomoji
    from llmoji_study.claude_gt import load_all_run_rows
    from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
    rows = load_all_run_rows()
    if not rows:
        return {}
    qmap = {ep.id: _bucket(ep) for ep in EMOTIONAL_PROMPTS}
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        face = canonicalize_kaomoji(r.get("first_word") or "")
        if not face:
            continue
        q = qmap.get(r["prompt_id"])
        if q is None:
            continue
        counts[face][q] += 1
    return {face: c.most_common(1)[0][0] for face, c in counts.items() if c}


def _bucket(ep) -> str:
    if ep.quadrant == "HN":
        return "HN-D" if ep.pad_dominance > 0 else "HN-S"
    return ep.quadrant


# ---------------------------------------------------------------------------
# Anthropic API call with retry. Mirrors script 23's _call_claude.
# ---------------------------------------------------------------------------

def _call_judge(client, model_id: str, face: str, max_retries: int = 4) -> dict:
    """One stateless API call with cached system prompt + JSON-schema-
    enforced response shape. Returns the parsed JSON dict directly.
    Exponential backoff on rate-limit / transient errors.

    Schema enforcement via ``output_config={"format": {"type":
    "json_schema", "schema": ...}}`` — model physically cannot emit
    an out-of-enum quadrant or omit required fields. Anthropic SDK
    0.97+ feature.
    """
    import anthropic
    delay = 1.0
    last_exc: Exception | None = None
    create_kwargs: dict = {
        "model": model_id,
        "max_tokens": MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": _RESPONSE_SCHEMA,
            },
        },
        "messages": [{"role": "user", "content": _build_user_message(face)}],
    }
    if model_id not in _NO_TEMPERATURE_MODELS:
        create_kwargs["temperature"] = TEMPERATURE
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**create_kwargs)
            parts: list[str] = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
            raw = "".join(parts)
            # Schema guarantees JSON; parse and validate.
            return json.loads(raw)
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APIStatusError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            print(f"    transient API error (attempt {attempt+1}/{max_retries}): "
                  f"{type(e).__name__} {e}; retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2.0
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Summary / head-to-head helpers.
# ---------------------------------------------------------------------------

def _build_pred_dist(row: dict, field_prefix: str) -> dict[str, float]:
    """Reconstruct the judge's per-quadrant likelihood vector from JSONL
    fields. Falls back to 0.0 for missing keys; ``jsd_quadrant`` smooths
    via ``normalize`` so zeros are tolerated."""
    return {q: float(row.get(f"{field_prefix}_lik_{q}", 0.0)) for q in QUADRANTS}


def _soft_similarity_section(
    rows: list[dict],
    field_prefix: str,
    gt_dist: dict[str, dict[str, int]],
) -> tuple[list[str], dict[str, float]]:
    """Compute per-face JSD-similarity vs Claude-GT distribution and
    write the soft-everywhere section to the summary.

    Returns ``(markdown_lines, stats)`` where ``stats`` exposes the
    headline aggregates so callers (e.g. the head-to-head section) can
    re-use them without duplicating the loop.

    Methodology: pred dist comes from the judge's per-quadrant softmax
    in the JSONL; gt dist is normalized raw counts from
    ``load_claude_gt_distribution`` (default floor=3 — sparse counts
    give noisy distribution estimates). Per-face score is
    ``similarity(jsd) = 1 - JSD/ln 2 ∈ [0, 1]``. Two aggregates:

    - **face-uniform**: mean similarity across faces (vocabulary view)
    - **emit-weighted**: similarity weighted by face's total GT emit
      count (deployment-relevance view, matches script 56's headline)

    Per-quadrant breakdown groups faces by their GT modal quadrant —
    "where the gt-modal is HP, what's the mean similarity?". Useful
    for spotting cells where the judge systematically diverges.
    """
    common = [r for r in rows if r["face"] in gt_dist]
    if not common:
        return ([], {})

    sims: list[float] = []
    weights: list[int] = []
    by_modal: dict[str, list[float]] = {q: [] for q in QUADRANTS}

    for r in common:
        face = r["face"]
        pred = _build_pred_dist(r, field_prefix)
        gt = gt_dist[face]
        jsd_nats = jsd_quadrant(pred, gt)
        sim = similarity(jsd_nats)
        sims.append(sim)
        total_emit = sum(gt.values())
        weights.append(total_emit)
        # Modal Q for this face on the GT side.
        modal_q = max(gt, key=lambda q: gt[q])
        if modal_q in by_modal:
            by_modal[modal_q].append(sim)

    n = len(sims)
    face_uniform = sum(sims) / n if n else 0.0
    total_w = sum(weights)
    emit_weighted = (
        sum(s * w for s, w in zip(sims, weights)) / total_w
        if total_w else 0.0
    )

    lines: list[str] = []
    lines.append("## Soft-everywhere similarity vs Claude-GT distribution")
    lines.append("")
    lines.append(
        f"Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. "
        f"Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = "
        f"normalized per-face quadrant emit counts from "
        f"``load_claude_gt_distribution(floor=3)``. "
        f"Faces evaluated: **{n}** (judged ∩ GT-with-≥3-emits)."
    )
    lines.append("")
    lines.append(
        f"- **Face-uniform** mean similarity (vocabulary view): "
        f"**{face_uniform:.3f}**"
    )
    lines.append(
        f"- **Emit-weighted** mean similarity (deployment view, weight = "
        f"GT emit count): **{emit_weighted:.3f}**  "
        f"(total emit weight: {total_w})"
    )
    lines.append("")
    lines.append("### Per-quadrant similarity (faces grouped by GT modal Q)")
    lines.append("")
    lines.append("| GT modal | n | mean similarity |")
    lines.append("|---|---:|---:|")
    for q in QUADRANTS:
        bucket = by_modal[q]
        if not bucket:
            lines.append(f"| {q} | 0 | — |")
            continue
        m = sum(bucket) / len(bucket)
        lines.append(f"| {q} | {len(bucket)} | {m:.3f} |")
    lines.append("")

    stats = {
        "n": float(n),
        "face_uniform": face_uniform,
        "emit_weighted": emit_weighted,
        "total_weight": float(total_w),
    }
    return (lines, stats)


def _build_face_union_lookup(union_df: pd.DataFrame) -> dict[str, dict]:
    """``{face: {is_claude, total_emit_*, empirical_majority_quadrant,
    total_emit_count}}`` from the canonical face union. Used by
    :func:`_write_face_likelihood_tsv` to fill the empirical columns
    that the ensemble pipeline expects on every encoder summary.
    """
    out: dict[str, dict] = {}
    for _, row in union_df.iterrows():
        emit_counts = {q: int(row[f"total_emit_{q}"]) for q in QUADRANTS}
        total_emit_count = sum(emit_counts.values())
        if total_emit_count > 0:
            empirical_modal = max(emit_counts, key=lambda q: emit_counts[q])
        else:
            empirical_modal = ""
        out[row["first_word"]] = {
            "is_claude": bool(row["is_claude"]),
            "total_emit_count": total_emit_count,
            "empirical_majority_quadrant": empirical_modal,
            **{f"total_emit_{q}": emit_counts[q] for q in QUADRANTS},
        }
    return out


def _write_face_likelihood_tsv(
    rows: list[dict], short: str, union_df: pd.DataFrame, out_path: Path,
) -> None:
    """Bridge judgment rows → face_likelihood-shaped summary TSV.

    Schema mirrors ``data/local/<model>/face_likelihood_summary.tsv`` so
    the Anthropic judges drop into ``face_likelihood_discovery`` and the
    ensemble pipeline (52/53/54) treats them like any other encoder.
    Likelihoods are normalized + smoothed via ``llmoji_study.jsd.normalize``
    (eps=1e-6) so log probabilities are well-defined. The
    ``n_prompts_*`` cols are 0 (the signal didn't come from a prompted
    generation) and ``n_face_tokens=1`` (introspective signal,
    not token-derived).

    Folded in from the pre-2026-05-05 ``50_face_likelihood.py``;
    kept as a function rather than inlined so the JSONL → TSV transform
    is one place to fix if the schema evolves.
    """
    union_lookup = _build_face_union_lookup(union_df)
    out_rows: list[dict] = []
    n_in_union = 0
    n_argmax_match = 0
    for r in rows:
        face = r["face"]
        raw = {q: float(r.get(f"{short}_lik_{q}", 0.0)) for q in QUADRANTS}
        soft = normalize(raw, vocab=QUADRANTS)
        soft_dict = {q: soft[i] for i, q in enumerate(QUADRANTS)}
        log_probs = {
            q: math.log(soft_dict[q]) if soft_dict[q] > 0 else math.log(1e-12)
            for q in QUADRANTS
        }
        argmax_q = max(QUADRANTS, key=lambda q: soft_dict[q])
        max_soft = soft_dict[argmax_q]

        union = union_lookup.get(face, {
            "is_claude": False,
            "total_emit_count": 0.0,
            "empirical_majority_quadrant": "",
            **{f"total_emit_{q}": 0 for q in QUADRANTS},
        })
        if face in union_lookup:
            n_in_union += 1
        emp = union["empirical_majority_quadrant"]
        if emp and argmax_q == emp:
            n_argmax_match += 1

        out_row: dict = {"first_word": face}
        for q in QUADRANTS:
            out_row[f"n_prompts_{q}"] = 0
        for q in QUADRANTS:
            out_row[f"mean_log_prob_{q}"] = log_probs[q]
        for q in QUADRANTS:
            out_row[f"softmax_{q}"] = round(soft_dict[q], 6)
        out_row["predicted_quadrant"] = argmax_q
        out_row["max_softmax"] = round(max_soft, 6)
        out_row["n_face_tokens"] = 1
        out_row["is_claude"] = union["is_claude"]
        out_row["total_emit_count"] = union["total_emit_count"]
        out_row["empirical_majority_quadrant"] = emp
        for q in QUADRANTS:
            out_row[f"total_emit_{q}"] = union[f"total_emit_{q}"]
        out_row["argmax_matches_empirical"] = bool(emp and argmax_q == emp)
        out_rows.append(out_row)

    print(f"  {n_in_union}/{len(rows)} faces matched into face union")
    print(f"  argmax-vs-empirical-modal agreement: "
          f"{n_argmax_match}/{len(rows)} = "
          f"{n_argmax_match / max(len(rows), 1):.1%}")

    df = pd.DataFrame(out_rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"wrote face_likelihood TSV → {out_path}  "
          f"({len(df)} faces × {len(df.columns)} cols)")


def _read_judgment_rows(path: Path, field_prefix: str) -> list[dict]:
    """Load successful (non-error) judgment rows; require the
    ``<prefix>_quadrant`` field so legacy / mixed schemas don't get
    silently mis-counted.

    Fails fast if the file contains v1-schema rows (``<prefix>_conf_*``
    instead of ``<prefix>_lik_*``). The v1 → v2 schema change
    (2026-05-05) renamed confidences→likelihoods and dropped the
    model-emitted top quadrant + reason; mixing schemas in one file
    produces silently-wrong soft-similarity numbers, so we refuse
    instead of best-effort merging.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    qkey = f"{field_prefix}_quadrant"
    legacy_key = f"{field_prefix}_conf_HP"
    new_key = f"{field_prefix}_lik_HP"
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            if legacy_key in r and new_key not in r:
                raise SystemExit(
                    f"{path} line {line_no}: v1-schema row detected "
                    f"({legacy_key} present, {new_key} absent). The likelihood "
                    "schema changed 2026-05-05 (confidences→likelihoods, "
                    "dropped reason + model-emitted quadrant). Archive and "
                    "delete the file before re-running:\n"
                    f"  mv {path} {path.with_name(path.stem + '_v1' + path.suffix)}"
                )
            if qkey not in r:
                continue
            out.append(r)
    return out


def _head_to_head_section(
    rows_a: list[dict], short_a: str,
    rows_b: list[dict], short_b: str,
    claude_modals: dict[str, str],
    gt_dist: dict[str, dict[str, int]],
) -> list[str]:
    """Per-face agreement on the face intersection between models A and
    B. Reports:

    - hard pairwise agreement (argmax-vs-argmax)
    - hard accuracy vs Claude-pilot modal for each model
    - **soft** distributional agreement (1 − JSD/ln 2 between the two
      judges' softmax distributions, face-uniform mean) — this is the
      soft-everywhere companion to "do they pick the same argmax"
    - **soft** mean similarity vs Claude-GT distribution for each
      model (face-uniform + emit-weighted), restricted to the face
      intersection so the comparison is apples-to-apples
    - disagreement table (first 30 rows)
    """
    qkey_a = f"{short_a}_quadrant"
    qkey_b = f"{short_b}_quadrant"
    by_face_b = {r["face"]: r for r in rows_b}
    common = [r for r in rows_a if r["face"] in by_face_b]
    if not common:
        return []
    n = len(common)
    n_agree = sum(1 for r in common if r[qkey_a] == by_face_b[r["face"]][qkey_b])
    pct = n_agree / n if n else 0.0

    # Accuracy vs Claude-pilot modal where defined.
    pilot_pairs = [
        (r, by_face_b[r["face"]]) for r in common
        if r["face"] in claude_modals
    ]
    n_pilot = len(pilot_pairs)
    a_pilot = sum(
        1 for r, _ in pilot_pairs if r[qkey_a] == claude_modals[r["face"]]
    )
    b_pilot = sum(
        1 for _, rb in pilot_pairs if rb[qkey_b] == claude_modals[rb["face"]]
    )
    a_acc = a_pilot / n_pilot if n_pilot else 0.0
    b_acc = b_pilot / n_pilot if n_pilot else 0.0

    # Soft pairwise: distributional similarity between the two judges,
    # independent of GT.
    pair_sims: list[float] = []
    for r in common:
        rb = by_face_b[r["face"]]
        pa = _build_pred_dist(r, short_a)
        pb = _build_pred_dist(rb, short_b)
        pair_sims.append(similarity(jsd_quadrant(pa, pb)))
    pair_face_uniform = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

    # Soft vs GT, restricted to the face intersection. Face-uniform +
    # emit-weighted aggregates per model.
    common_faces_with_gt = [r for r in common if r["face"] in gt_dist]
    a_sims: list[float] = []
    b_sims: list[float] = []
    weights: list[int] = []
    for r in common_faces_with_gt:
        face = r["face"]
        gt = gt_dist[face]
        a_sims.append(similarity(jsd_quadrant(_build_pred_dist(r, short_a), gt)))
        b_sims.append(similarity(
            jsd_quadrant(_build_pred_dist(by_face_b[face], short_b), gt)
        ))
        weights.append(sum(gt.values()))
    n_gt_common = len(common_faces_with_gt)
    a_face_uniform = sum(a_sims) / n_gt_common if n_gt_common else 0.0
    b_face_uniform = sum(b_sims) / n_gt_common if n_gt_common else 0.0
    total_w = sum(weights)
    a_emit = (
        sum(s * w for s, w in zip(a_sims, weights)) / total_w
        if total_w else 0.0
    )
    b_emit = (
        sum(s * w for s, w in zip(b_sims, weights)) / total_w
        if total_w else 0.0
    )

    # First few disagreements, sorted by face for stable output.
    diffs: list[tuple[str, str, str, str]] = []  # (face, a_q, b_q, claude_modal)
    for r in common:
        rb = by_face_b[r["face"]]
        if r[qkey_a] != rb[qkey_b]:
            diffs.append((
                r["face"], r[qkey_a], rb[qkey_b],
                claude_modals.get(r["face"], "—"),
            ))
    diffs.sort(key=lambda t: t[0])

    out: list[str] = []
    out.append("")
    out.append(f"## Head-to-head: {short_a.capitalize()} vs {short_b.capitalize()}")
    out.append("")
    out.append(
        f"On the {n} face(s) both models rated:"
    )
    out.append("")
    out.append(
        f"- **Hard agreement (argmax-vs-argmax)**: "
        f"{short_a} ↔ {short_b} = **{n_agree}/{n} ({pct:.1%})**"
    )
    out.append(
        f"- **Soft agreement (distributional similarity, face-uniform)**: "
        f"mean similarity({short_a}, {short_b}) = **{pair_face_uniform:.3f}**"
    )
    if n_pilot:
        out.append(
            f"- **Hard accuracy vs Claude-pilot modal** (n={n_pilot}): "
            f"{short_a} **{a_pilot}/{n_pilot} ({a_acc:.1%})**, "
            f"{short_b} **{b_pilot}/{n_pilot} ({b_acc:.1%})**"
        )
    if n_gt_common:
        out.append(
            f"- **Soft accuracy vs Claude-GT distribution** (n={n_gt_common} "
            f"faces with ≥3 GT emits, total weight {total_w}):"
        )
        out.append(
            f"  - {short_a}: face-uniform **{a_face_uniform:.3f}**, "
            f"emit-weighted **{a_emit:.3f}**"
        )
        out.append(
            f"  - {short_b}: face-uniform **{b_face_uniform:.3f}**, "
            f"emit-weighted **{b_emit:.3f}**"
        )
    if diffs:
        max_show = 30
        shown = diffs[:max_show]
        out.append("")
        out.append(f"### Disagreements (first {len(shown)} of {len(diffs)})")
        out.append("")
        out.append(f"| face | {short_a} | {short_b} | claude-pilot modal |")
        out.append("|---|---|---|---|")
        for face, qa, qb, cm in shown:
            out.append(f"| `{face}` | {qa} | {qb} | {cm} |")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Resume / skip-set.
# ---------------------------------------------------------------------------

def _already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            face = r.get("face")
            if face:
                done.add(face)
    return done


def _drop_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default=DEFAULT_SHORT,
                    help=f"judge model shortname (default: {DEFAULT_SHORT}). "
                         "Output filenames adapt; haiku paths preserved for "
                         "backward compat.")
    ap.add_argument("--gt-only", action="store_true",
                    help="restrict to faces in Claude-GT (load_claude_gt "
                         "floor=1). Recommended for Opus to control cost.")
    ap.add_argument("--claude-gt-floor", type=int, default=1,
                    help="when --gt-only, the modal_n floor for GT membership.")
    args = ap.parse_args()

    short = args.model
    default_id, env_var = MODEL_DEFAULTS[short]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY env var not set")
    model_id = os.environ.get(env_var, default_id)
    out_path, summary_path, face_lik_path = _out_paths(short)
    field_prefix = short  # "haiku_quadrant", "opus_quadrant", etc.

    if not FACE_UNION_PATH.exists():
        raise SystemExit(
            f"face union not found at {FACE_UNION_PATH}; "
            f"run scripts/40_face_union.py first"
        )

    df = pd.read_parquet(FACE_UNION_PATH)
    # Sort for stable iteration order — faces are unicode strings, so
    # lexicographic on first_word is fine and resume-friendly.
    df = df.sort_values("first_word").reset_index(drop=True)
    n_union = len(df)
    print(f"face union: {n_union} faces "
          f"(Claude-emitted: {int(df['is_claude'].sum())}, "
          f"wild: {int(df['is_wild'].sum())})")

    if args.gt_only:
        from llmoji_study.claude_gt import load_claude_gt
        gt = load_claude_gt(floor=args.claude_gt_floor)
        gt_set = set(gt)
        df = df[df["first_word"].isin(gt_set)].reset_index(drop=True)
        print(f"--gt-only: filtered to {len(df)} faces in Claude-GT "
              f"(floor={args.claude_gt_floor}, GT face set size {len(gt_set)})")
        if len(df) == 0:
            raise SystemExit(
                "no face_union rows intersect Claude-GT — face union may be "
                "stale; rerun scripts/40_face_union.py."
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(out_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(out_path)
    remaining = len(df) - sum(1 for f in df["first_word"] if f in done)
    print(f"already done: {len(done)}; remaining: {remaining}")

    if remaining == 0:
        print("nothing to do; jumping to summary")
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print(f"model: {model_id}  (shortname: {short})")
        print(f"output: {out_path}")
        # Pre-compute per-face metadata for the row write.
        with out_path.open("a") as out:
            i = 0
            for _, row in df.iterrows():
                face = row["first_word"]
                if face in done:
                    continue
                i += 1
                t0 = time.time()
                modal, top_count, total_count = _behavior_modal(row)
                try:
                    parsed = _call_judge(client, model_id, face)
                    likelihoods = parsed.get("likelihoods", {})
                    # Schema enforces required fields + numeric type; this
                    # check defends against upstream SDK regression.
                    if not all(q in likelihoods for q in QUADRANTS):
                        err_row = {
                            "face": face,
                            "error": "schema_validation_failed",
                            "raw_response": parsed,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                        out.flush()
                        print(f"  [{i}/{remaining}] {face!r} SCHEMA_FAIL: {parsed!r}")
                        continue
                except Exception as e:
                    err_row = {
                        "face": face,
                        "error": repr(e),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"  [{i}/{remaining}] {face!r} ERR: {e}")
                    continue
                # Float-cast and derive argmax locally — model is no longer
                # asked to commit to a single quadrant. Top likelihood is
                # the headline confidence; ties broken by QUADRANTS order.
                lik = {q: float(likelihoods.get(q, 0.0)) for q in QUADRANTS}
                argmax_q = max(QUADRANTS, key=lambda q: lik[q])
                top_lik = lik[argmax_q]
                row_out = {
                    "face": face,
                    f"{field_prefix}_quadrant": argmax_q,  # derived
                    f"{field_prefix}_top_likelihood": top_lik,
                    **{f"{field_prefix}_lik_{q}": lik[q] for q in QUADRANTS},
                    "behavior_modal": modal,
                    "behavior_count_top": top_count,
                    "behavior_count_total": total_count,
                    "is_claude": bool(row["is_claude"]),
                    "is_wild": bool(row["is_wild"]),
                    "model_id": model_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                if i % 20 == 0:
                    out.flush()
                dt = time.time() - t0
                agree = "✓" if modal == argmax_q else "✗"
                print(f"  [{i}/{remaining}] {face!r} → {short}={argmax_q} "
                      f"(lik={top_lik:.2f}) behavior={modal} {agree} ({dt:.1f}s)")

    # Build summary.
    print(f"\nbuilding summary at {summary_path}")
    rows = _read_judgment_rows(out_path, field_prefix)
    if not rows:
        print("no successful rows; skipping summary + face_likelihood TSV")
        return

    # Bridge to the face_likelihood schema so the ensemble pipeline auto-
    # discovers this encoder alongside local LM-head encoders. Always
    # written — cheap, in-memory transform of the just-loaded rows. Folded
    # in from the pre-2026-05-05 50_face_likelihood.py.
    print(f"\nwriting face_likelihood TSV at {face_lik_path}")
    _write_face_likelihood_tsv(rows, field_prefix, df, face_lik_path)

    claude_modals = _claude_modal_table()
    # Soft-everywhere GT: per-face quadrant emit-count distributions.
    # Floor=3 matches the rest of the project (sparse counts → noisy
    # estimates). Faces with <3 GT emits drop out of soft-similarity.
    gt_dist = load_claude_gt_distribution(floor=3)
    qkey = f"{field_prefix}_quadrant"

    overall_n = len(rows)
    overall_agree = sum(1 for r in rows if r[qkey] == r["behavior_modal"])
    overall_acc = overall_agree / overall_n if overall_n else 0.0

    claude_rows = [r for r in rows if r["is_claude"]]
    claude_n = len(claude_rows)
    claude_agree_behavior = sum(
        1 for r in claude_rows if r[qkey] == r["behavior_modal"]
    )
    claude_acc_behavior = claude_agree_behavior / claude_n if claude_n else 0.0
    claude_agree_pilot = sum(
        1 for r in claude_rows
        if r[qkey] == claude_modals.get(r["face"])
    )
    claude_n_pilot = sum(1 for r in claude_rows if r["face"] in claude_modals)
    claude_acc_pilot = (
        claude_agree_pilot / claude_n_pilot if claude_n_pilot else 0.0
    )

    # Per-quadrant accuracy (where behavior_modal=q, what % does the judge agree?).
    per_q_total: Counter = Counter()
    per_q_agree: Counter = Counter()
    for r in rows:
        bm = r["behavior_modal"]
        if bm is None:
            continue
        per_q_total[bm] += 1
        if r[qkey] == bm:
            per_q_agree[bm] += 1

    # Confusion: rows[behavior][judge].
    confusion: dict[str, Counter] = {q: Counter() for q in QUADRANTS}
    for r in rows:
        bm = r["behavior_modal"]
        if bm is None:
            continue
        confusion[bm][r[qkey]] += 1

    lines: list[str] = []
    title_short = short.capitalize()
    lines.append(f"# {title_short} face→quadrant judgment vs behavior modal")
    lines.append("")
    lines.append(f"- Model: `{rows[0]['model_id']}`  (shortname: `{short}`)")
    if args.gt_only:
        lines.append(f"- Scope: `--gt-only` (Claude-GT subset, "
                     f"floor={args.claude_gt_floor})")
    else:
        lines.append(f"- Scope: full v3 face union")
    lines.append(f"- Faces classified: **{overall_n}**")
    lines.append(
        f"- Overall agreement with behavior modal (argmax of v3 + Claude pilot "
        f"+ wild emit counts): **{overall_acc:.1%}** ({overall_agree}/{overall_n})"
    )
    lines.append(
        f"- Claude-emitted subset ({claude_n} faces) agreement with behavior "
        f"modal: **{claude_acc_behavior:.1%}** ({claude_agree_behavior}/{claude_n})"
    )
    lines.append(
        f"- Claude-emitted subset agreement with Claude-pilot-only modal "
        f"({claude_n_pilot} faces with pilot emit): **{claude_acc_pilot:.1%}** "
        f"({claude_agree_pilot}/{claude_n_pilot})"
    )
    lines.append("")
    lines.append("## Per-quadrant accuracy (behavior-modal as ground truth)")
    lines.append("")
    lines.append(f"| behavior_modal | n | {short}_agree | acc |")
    lines.append("|---|---:|---:|---:|")
    for q in QUADRANTS:
        n = per_q_total[q]
        a = per_q_agree[q]
        acc = a / n if n else 0.0
        lines.append(f"| {q} | {n} | {a} | {acc:.1%} |")
    lines.append("")
    lines.append(f"## Confusion matrix (rows = behavior modal, cols = {short})")
    lines.append("")
    header = "| | " + " | ".join(QUADRANTS) + " | total |"
    sep = "|---|" + "|".join(["---:"] * (len(QUADRANTS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for bm in QUADRANTS:
        cells = [f"{confusion[bm][hq]}" for hq in QUADRANTS]
        total = sum(confusion[bm].values())
        lines.append(f"| **{bm}** | " + " | ".join(cells) + f" | {total} |")
    lines.append("")

    # Soft-everywhere similarity vs Claude-GT distribution (the
    # 2026-05-04 methodology pivot — distribution-vs-distribution, not
    # argmax-vs-argmax). Hard-accuracy stats above are kept as
    # supplementary; this is the primary metric.
    soft_lines, _ = _soft_similarity_section(rows, field_prefix, gt_dist)
    if soft_lines:
        lines.extend(soft_lines)
    else:
        lines.append("## Soft-everywhere similarity vs Claude-GT distribution")
        lines.append("")
        lines.append("(no overlap between judged faces and Claude-GT — "
                     "skipping soft-similarity section.)")
        lines.append("")

    # Head-to-head: if the OTHER model's JSONL exists, compare per-face
    # agreement on the intersection. Useful for "how much does Opus
    # gain over Haiku?" sanity-checks. With gt_dist the head-to-head
    # also reports each model's distributional similarity vs GT.
    other_short = next((s for s in MODEL_DEFAULTS if s != short), None)
    if other_short is not None:
        other_path, _, _ = _out_paths(other_short)
        if other_path.exists():
            other_rows = _read_judgment_rows(other_path, other_short)
            if other_rows:
                lines.extend(
                    _head_to_head_section(
                        rows, short, other_rows, other_short,
                        claude_modals, gt_dist,
                    )
                )

    lines.append(f"_built {datetime.now(timezone.utc).isoformat()}_")

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"wrote summary → {summary_path}")
    print(f"\noverall agreement: {overall_acc:.1%} ({overall_agree}/{overall_n})")
    print(f"claude-emitted subset: {claude_acc_behavior:.1%} "
          f"({claude_agree_behavior}/{claude_n})")


if __name__ == "__main__":
    main()
