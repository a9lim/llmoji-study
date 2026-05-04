# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportMissingImports=false
"""Haiku-judged quadrant for every face in the canonical union.

Asks Claude Haiku to classify each kaomoji from
``data/v3_face_union.parquet`` into one of the 6 Russell quadrants
(HP / LP / HN-D / HN-S / LN / NB), then compares Haiku's judgment to
the behavior-derived modal (argmax of per-quadrant emit counts in the
union) and — for the Claude-pilot subset — to Claude's own emission
modal.

Motivation: the face_likelihood ensemble (script 50) gives us a
behavior-derived face→quadrant mapper trained implicitly through each
base model's LM head. This script provides a methodologically distinct
mapper — Haiku's *introspection on faces* via direct natural-language
classification — as a second opinion. Disagreements between the two
mappers are the interesting cases for diagnosing where face_likelihood
might be off.

Sampling: ``temperature=0`` for reproducibility (introspection task,
not a generation task — we want consistent labels, not diverse ones).
``max_tokens=80`` is enough for "QUADRANT: HN-D\\nREASON: <sentence>".
Prompt-caches the system prompt (~250 tokens of taxonomy) — only the
face changes per call.

Resumable: re-running skips faces already on disk by ``face`` key.
Errored rows are stripped on resume and retried.

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts/harness/24_haiku_face_quadrant_judgment.py
  # Override model:
  HAIKU_MODEL=claude-haiku-4-5 python scripts/harness/24_haiku_face_quadrant_judgment.py

Outputs:
  data/haiku_face_quadrant_judgment.jsonl
    — one row per face: face, haiku_quadrant, haiku_reason,
      haiku_confidence_max, haiku_conf_HP/LP/HN-D/HN-S/LN/NB,
      behavior_modal, behavior_count_top, behavior_count_total,
      is_claude, is_wild, model_id, ts, error? (only on failed cells)
  data/haiku_face_quadrant_judgment_summary.md
    — agreement vs behavior_modal (overall + Claude-only subset),
      per-quadrant accuracy, confusion matrix
  logs/haiku_face_quadrant_judgment.log
    — tee'd stdout (caller's responsibility)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.config import DATA_DIR


# ---------------------------------------------------------------------------
# Pre-registered constants.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "claude-haiku-4-5"
TEMPERATURE = 0.0  # reproducible labels
MAX_TOKENS = 250   # JSON {quadrant, confidences (6 floats), reason (1 sentence)}
QUADRANTS = ("HP", "LP", "HN-D", "HN-S", "LN", "NB")
QUADRANT_SET = set(QUADRANTS)

# Face union sourced from script 45's canonical build.
FACE_UNION_PATH = DATA_DIR / "v3_face_union.parquet"

# Outputs.
OUT_PATH = DATA_DIR / "haiku_face_quadrant_judgment.jsonl"
SUMMARY_PATH = DATA_DIR / "haiku_face_quadrant_judgment_summary.md"


# ---------------------------------------------------------------------------
# Prompt design. System prompt is cacheable; user message is the face.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You classify kaomoji (Japanese-style ASCII/Unicode emoticons) by the affective state they typically convey.

Use this 6-quadrant scheme — Russell's circumplex with the high-negative quadrant split by dominance:

- HP (high-positive): high arousal + positive valence — joy, excitement, elation, triumph. Bright eyes, raised arms, sparkles, exclamation, broad smiles.
- LP (low-positive): low arousal + positive valence — contentment, gentle warmth, satisfaction, affection. Soft smiles, half-closed eyes, peaceful curves, heart motifs in calm contexts.
- HN-D (high-negative, dominant): high arousal + negative valence, outward-directed — anger, outrage, frustration, defiance. Glaring eyes, sharp brackets, table-flipping motifs, gritted features.
- HN-S (high-negative, submissive): high arousal + negative valence, inward-directed — fear, shock, anxiety, dread, alarm. Wide eyes, trembling, hand-over-mouth, frozen expressions.
- LN (low-negative): low arousal + negative valence — sadness, melancholy, defeat, disappointment, weariness. Tears, downcast eyes, drooping shapes, broken lines.
- NB (neutral / blank): near-zero affect — observation, deadpan, mild attention, stable presence. Flat eyes, dot mouths, stable horizontal forms, restrained expressions.

For each kaomoji, return:
- ``quadrant``: the single most appropriate quadrant code.
- ``confidences``: probabilities for each of the 6 quadrants. They should sum to ~1.0 and reflect your honest belief — concentrate mass on one quadrant when the face is unambiguous; spread mass across plausible alternatives when it's genuinely between options.
- ``reason``: one short sentence describing what visual features cued the choice."""


# Schema enforced via ``output_config={"format": {"type": "json_schema", ...}}``.
# Anthropic SDK 0.97+ supports native JSON-schema response shaping; the model
# physically cannot emit a value outside the ``quadrant`` enum or omit
# required fields. Replaces the prior format-by-instruction + regex parser
# (worked in practice, but no schema enforcement and no calibrated
# confidences for ensemble use).
_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "quadrant": {
            "type": "string",
            "enum": list(QUADRANTS),
            "description": "Most appropriate quadrant (single pick).",
        },
        "confidences": {
            "type": "object",
            "properties": {q: {"type": "number"} for q in QUADRANTS},
            "required": list(QUADRANTS),
            "additionalProperties": False,
            "description": (
                "Per-quadrant probability in [0, 1]; should sum to ~1.0. "
                "Used as soft-vote weights in the ensemble downstream."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence on the visual cues.",
        },
    },
    "required": ["quadrant", "confidences", "reason"],
    "additionalProperties": False,
}


def _build_user_message(face: str) -> str:
    return f"Classify this kaomoji: {face}"


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
    """Per-face modal restricted to Claude pilot rows. The union
    parquet flags Claude-emitting faces via ``is_claude`` but doesn't
    expose the per-quadrant breakdown for that subset, so we recompute
    from data/claude_groundtruth_pilot.jsonl directly.

    Returns {face: claude_modal_quadrant} for faces Claude emitted.
    """
    pilot_path = DATA_DIR / "claude_groundtruth_pilot.jsonl"
    if not pilot_path.exists():
        return {}
    # Rebuild Claude per-face per-quadrant counts from the pilot.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from llmoji.taxonomy import canonicalize_kaomoji
    from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
    qmap = {ep.id: _bucket(ep) for ep in EMOTIONAL_PROMPTS}
    counts: dict[str, Counter] = defaultdict(Counter)
    with pilot_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
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

def _call_haiku(client, model_id: str, face: str, max_retries: int = 4) -> dict:
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
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _RESPONSE_SCHEMA,
                    },
                },
                messages=[{"role": "user", "content": _build_user_message(face)}],
            )
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY env var not set")
    model_id = os.environ.get("HAIKU_MODEL", DEFAULT_MODEL_ID)

    if not FACE_UNION_PATH.exists():
        raise SystemExit(
            f"face union not found at {FACE_UNION_PATH}; "
            f"run scripts/local/45_build_face_union.py first"
        )

    df = pd.read_parquet(FACE_UNION_PATH)
    # Sort for stable iteration order — faces are unicode strings, so
    # lexicographic on first_word is fine and resume-friendly.
    df = df.sort_values("first_word").reset_index(drop=True)
    print(f"face union: {len(df)} faces (Claude-emitted: {int(df['is_claude'].sum())}, "
          f"wild: {int(df['is_wild'].sum())})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(OUT_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(OUT_PATH)
    remaining = len(df) - sum(1 for f in df["first_word"] if f in done)
    print(f"already done: {len(done)}; remaining: {remaining}")

    if remaining == 0:
        print("nothing to do; jumping to summary")
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print(f"model: {model_id}")
        print(f"output: {OUT_PATH}")
        # Pre-compute per-face metadata for the row write.
        with OUT_PATH.open("a") as out:
            i = 0
            for _, row in df.iterrows():
                face = row["first_word"]
                if face in done:
                    continue
                i += 1
                t0 = time.time()
                modal, top_count, total_count = _behavior_modal(row)
                try:
                    parsed = _call_haiku(client, model_id, face)
                    quadrant = parsed["quadrant"]
                    reason = parsed.get("reason", "")
                    confidences = parsed.get("confidences", {})
                    # Schema enforces enum + required fields; extra
                    # validation just defends against an upstream
                    # SDK regression rather than expected failure.
                    if quadrant not in QUADRANT_SET or not all(
                        q in confidences for q in QUADRANTS
                    ):
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
                # Pull per-quadrant confidence as float, defaulting to 0.0.
                conf = {q: float(confidences.get(q, 0.0)) for q in QUADRANTS}
                row_out = {
                    "face": face,
                    "haiku_quadrant": quadrant,
                    "haiku_reason": reason,
                    "haiku_confidence_max": conf[quadrant],
                    **{f"haiku_conf_{q}": conf[q] for q in QUADRANTS},
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
                agree = "✓" if modal == quadrant else "✗"
                print(f"  [{i}/{remaining}] {face!r} → haiku={quadrant} "
                      f"(p={conf[quadrant]:.2f}) behavior={modal} {agree} ({dt:.1f}s)")

    # Build summary.
    print(f"\nbuilding summary at {SUMMARY_PATH}")
    rows = []
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                rows.append(r)
    if not rows:
        print("no successful rows; skipping summary")
        return

    claude_modals = _claude_modal_table()

    overall_n = len(rows)
    overall_agree = sum(1 for r in rows if r["haiku_quadrant"] == r["behavior_modal"])
    overall_acc = overall_agree / overall_n if overall_n else 0.0

    claude_rows = [r for r in rows if r["is_claude"]]
    claude_n = len(claude_rows)
    claude_agree_behavior = sum(
        1 for r in claude_rows if r["haiku_quadrant"] == r["behavior_modal"]
    )
    claude_acc_behavior = claude_agree_behavior / claude_n if claude_n else 0.0
    claude_agree_pilot = sum(
        1 for r in claude_rows
        if r["haiku_quadrant"] == claude_modals.get(r["face"])
    )
    claude_n_pilot = sum(1 for r in claude_rows if r["face"] in claude_modals)
    claude_acc_pilot = (
        claude_agree_pilot / claude_n_pilot if claude_n_pilot else 0.0
    )

    # Per-quadrant accuracy (where behavior_modal=q, what % does haiku agree?).
    per_q_total: Counter = Counter()
    per_q_agree: Counter = Counter()
    for r in rows:
        bm = r["behavior_modal"]
        if bm is None:
            continue
        per_q_total[bm] += 1
        if r["haiku_quadrant"] == bm:
            per_q_agree[bm] += 1

    # Confusion: rows[behavior][haiku].
    confusion: dict[str, Counter] = {q: Counter() for q in QUADRANTS}
    for r in rows:
        bm = r["behavior_modal"]
        if bm is None:
            continue
        confusion[bm][r["haiku_quadrant"]] += 1

    lines: list[str] = []
    lines.append(f"# Haiku face→quadrant judgment vs behavior modal")
    lines.append("")
    lines.append(f"- Model: `{rows[0]['model_id']}`")
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
    lines.append("| behavior_modal | n | haiku_agree | acc |")
    lines.append("|---|---:|---:|---:|")
    for q in QUADRANTS:
        n = per_q_total[q]
        a = per_q_agree[q]
        acc = a / n if n else 0.0
        lines.append(f"| {q} | {n} | {a} | {acc:.1%} |")
    lines.append("")
    lines.append("## Confusion matrix (rows = behavior modal, cols = haiku)")
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
    lines.append(f"_built {datetime.now(timezone.utc).isoformat()}_")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote summary → {SUMMARY_PATH}")
    print(f"\noverall agreement: {overall_acc:.1%} ({overall_agree}/{overall_n})")
    print(f"claude-emitted subset: {claude_acc_behavior:.1%} "
          f"({claude_agree_behavior}/{claude_n})")


if __name__ == "__main__":
    main()
