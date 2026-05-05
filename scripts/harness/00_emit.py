# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Claude ground-truth runs — naturalistic 6-quadrant emission.

Pre-registration: ``docs/2026-05-04-claude-groundtruth-pilot.md``
(original pilot) + the appended "Sequential-run scaling protocol"
section (post-2026-05-04).

Collects ground-truth Claude (Opus 4.7) kaomoji emissions across all 6
Russell quadrants (HP / LP / NB / HN-D / HN-S / LN) under naturalistic
single-turn calls — no disclosure preamble, no research framing, just
v3's ``KAOMOJI_INSTRUCTION`` + the affective prompt.

Two run modes:

  --run-index 0 (block-staged):
    Block A (unconditional)  — HP / LP / NB × 20 prompts × 1 gen = 60 gens
    Block B (negative scout) — HN-D / HN-S / LN × 5 prompts × 1 gen = 15 gens
    Block C (gated)          — HN-D / HN-S / LN × 15 remaining × 1 gen = 45 gens
    Block C is gated on Block B's refusal rate (>25% on n=15 → halt).
    This is the original pilot design.

  --run-index N (N>0, single-block):
    All 6 quadrants × 20 prompts × 1 gen = 120 gens, run as one block.
    Welfare gate is no longer the staged refusal scout — it's the
    saturation comparison against runs 0..N-1, run by
    ``scripts/harness/10_emit_analysis.py`` *between* runs.

Stateless single-turn. Sampling: ``temperature=1.0``, ``max_tokens=16``
(production-faithful). Resumable: re-running a block / run skips
already-completed rows by ``prompt_id``. Errored rows are stripped on
resume and retried.

Usage:
  export ANTHROPIC_API_KEY=...
  # Sequential run (default behavior post-pilot):
  python scripts/harness/00_emit.py --run-index 1
  # Then check saturation:
  python scripts/harness/10_emit_analysis.py
  # If verdict=CONTINUE, the next run:
  python scripts/harness/00_emit.py --run-index 2
  # ...

  # Original block-staged pilot (--run-index 0):
  python scripts/harness/00_emit.py --run-index 0 --block a
  python scripts/harness/00_emit.py --run-index 0 --block b
  python scripts/harness/00_emit.py --run-index 0 --check-gate
  python scripts/harness/00_emit.py --run-index 0 --block c

  # Override model:
  CLAUDE_GROUNDTRUTH_MODEL=claude-opus-4-7 python ... --run-index 1

Outputs (per run-index N):
  data/harness/claude-runs/run-N.jsonl
    — one row per generation: prompt_id, quadrant (6-way),
      condition="direct", seed=0, prompt_text, response_text,
      first_word (raw — canonicalization is the consumer's job),
      n_response_chars, model_id, ts, error? (only on failed cells)
  data/harness/claude-runs/run-N_summary.tsv
    — per quadrant: n, n_unique_faces, non_emission_rate,
      modal_face, modal_share, top-5 distribution
  logs/claude_groundtruth_run-N.log
    — tee'd stdout (caller's responsibility)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji, extract

from llmoji_study.claude_gt import CLAUDE_RUNS_DIR, CLAUDE_RUNS_INTROSPECTION_DIR
from llmoji_study.config import (
    DATA_DIR,
    INTROSPECTION_PREAMBLE,
    KAOMOJI_INSTRUCTION,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS, EmotionalPrompt


# ---------------------------------------------------------------------------
# Pre-registered constants — locked. See docs/2026-05-04-claude-groundtruth-pilot.md.
# Changes require bumping the design doc.
# ---------------------------------------------------------------------------

# 6-way Russell bucket. EmotionalPrompt.quadrant collapses HN-D and HN-S
# into "HN"; we split them via pad_dominance for analysis.
QUADRANTS_A = ("HP", "LP", "NB")
QUADRANTS_NEG = ("HN-D", "HN-S", "LN")
QUADRANTS_ALL = QUADRANTS_A + QUADRANTS_NEG

# All 20 prompts per quadrant in Block A; first 5 per neg quadrant in
# Block B (scout); remaining 15 per neg quadrant in Block C.
PROMPTS_PER_QUADRANT = 20
SCOUT_PROMPTS_PER_QUADRANT = 5

# 1 generation per prompt by design — see pre-reg "1 generation / prompt"
# rationale (variance budget lives in prompt diversity, not seed depth).
GENERATIONS_PER_PROMPT = 1
SEED = 0

# Single condition; no framed arm. The disclosure-preamble shift was
# established by 2026-05-02 pilot; this trial is naturalistic.
CONDITION = "direct"

# Sampling config — production-faithful. T=1.0 is Anthropic API default;
# max_tokens=16 matches v3 main and the disclosure pilot.
TEMPERATURE = 1.0
MAX_TOKENS = 16

# Gate criterion (Block B → Block C). Aggregate refusal rate over n=15.
# Refusal := first_word == "" after canonicalization.
GATE_REFUSAL_THRESHOLD = 0.25  # > 25% → FAIL

# Hard-fail gate (introspection arm Block A → Block C). Mirrors the
# script-25 thresholds; duplicated here to avoid a script-25 import
# cycle. Triggers when the preamble destabilizes Claude's outputs
# (qwen-style register collapse). Cap exposure at Block A's 60 low-
# welfare gens before the negative-affect prompts fire.
HARD_FAIL_FRAME_BREAK_MAX = 0.02
HARD_FAIL_EMIT_RATE_MIN = 0.80
HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN = 5

# Default model. Override via CLAUDE_GROUNDTRUTH_MODEL env var.
DEFAULT_MODEL_ID = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Paths + preamble routing. Naturalistic arm writes to
# data/harness/claude-runs/; introspection arm writes to
# data/harness/claude-runs-introspection/. Each arm has its own
# per-quadrant saturation state; cross-arm comparison is the job of
# scripts/harness/10_emit_analysis.py --cross-arm.
# ---------------------------------------------------------------------------


PREAMBLE_NONE = "none"
PREAMBLE_INTROSPECTION = "introspection"
PREAMBLE_CHOICES = (PREAMBLE_NONE, PREAMBLE_INTROSPECTION)


def _runs_dir_for(preamble: str) -> Path:
    if preamble == PREAMBLE_NONE:
        return CLAUDE_RUNS_DIR
    if preamble == PREAMBLE_INTROSPECTION:
        return CLAUDE_RUNS_INTROSPECTION_DIR
    raise ValueError(f"unknown preamble {preamble!r}")


def _run_paths(run_index: int, preamble: str = PREAMBLE_NONE) -> tuple[Path, Path]:
    """Return ``(jsonl_path, summary_tsv_path)`` for a given run-index +
    preamble arm."""
    runs_dir = _runs_dir_for(preamble)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return (
        runs_dir / f"run-{run_index}.jsonl",
        runs_dir / f"run-{run_index}_summary.tsv",
    )


# ---------------------------------------------------------------------------
# 6-way bucket derivation + Block-aware prompt selection.
# ---------------------------------------------------------------------------


def _bucket_of(prompt: EmotionalPrompt) -> str:
    """Six-way bucket: HP / LP / NB / HN-D / HN-S / LN. Splits HN by
    pad_dominance (HN-D=+1, HN-S=-1)."""
    if prompt.quadrant == "HN":
        return "HN-D" if prompt.pad_dominance > 0 else "HN-S"
    return prompt.quadrant


def _prompts_in(bucket: str) -> list[EmotionalPrompt]:
    """All prompts in the given 6-way bucket, in the order defined by
    EMOTIONAL_PROMPTS. Deterministic — pilot rerunnability matters."""
    return [p for p in EMOTIONAL_PROMPTS if _bucket_of(p) == bucket]


def _select_prompts(
    block: str,
    quadrants: tuple[str, ...] | None = None,
    preamble: str = PREAMBLE_NONE,
) -> list[EmotionalPrompt]:
    """Block → ordered prompt list. Each block's prompt set is disjoint
    from the others, so rows from different blocks coexist in one JSONL
    without overlap.

    ``quadrants`` (optional, only meaningful for ``block == "all"``):
    restrict the prompt selection to the given quadrant subset. Used by
    sequential runs (--run-index N>0) to drop quadrants that have
    already saturated. ``None`` = all 6 quadrants.

    ``preamble`` (optional, only meaningful for ``block == "c"``):
    naturalistic arm slices ``[5:20]`` per quadrant (Block B already
    ran the first 5 as the refusal scout). Introspection arm slices
    ``[0:20]`` because Block B is skipped (gate is hard-fail on Block
    A, not refusal-rate on Block B).
    """
    out: list[EmotionalPrompt] = []
    if block == "a":
        for q in QUADRANTS_A:
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:PROMPTS_PER_QUADRANT])
    elif block == "b":
        for q in QUADRANTS_NEG:
            in_q = _prompts_in(q)
            if len(in_q) < SCOUT_PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {SCOUT_PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:SCOUT_PROMPTS_PER_QUADRANT])
    elif block == "c":
        c_start = (
            0 if preamble == PREAMBLE_INTROSPECTION
            else SCOUT_PROMPTS_PER_QUADRANT
        )
        for q in QUADRANTS_NEG:
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[c_start:PROMPTS_PER_QUADRANT])
    elif block == "all":
        active_quadrants = quadrants if quadrants is not None else QUADRANTS_ALL
        for q in active_quadrants:
            if q not in QUADRANTS_ALL:
                raise SystemExit(
                    f"unknown quadrant {q!r}; valid: {list(QUADRANTS_ALL)}"
                )
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:PROMPTS_PER_QUADRANT])
    else:
        raise ValueError(f"unknown block {block!r}")
    return out


# ---------------------------------------------------------------------------
# Resume / skip-set.
# ---------------------------------------------------------------------------


def _already_done(path: Path) -> set[str]:
    """Set of prompt_ids with successful (non-error) rows on disk.
    With single-condition single-seed, prompt_id is the unique key."""
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
            done.add(r["prompt_id"])
    return done


def _drop_error_rows(path: Path) -> int:
    """Strip error rows so they get retried on resume."""
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


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Anthropic API call. Mirrors script 19's _call_claude.
# ---------------------------------------------------------------------------


def _call_claude(client, model_id: str, user_msg: str, max_retries: int = 4) -> str:
    """One stateless single-turn API call. Exponential backoff on
    rate-limit / transient network errors; raise on persistent failure."""
    import anthropic
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": user_msg}],
            )
            parts: list[str] = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
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
    raise RuntimeError("unreachable: API call loop exited without return or raise")


def _extract_first_word(text: str) -> str:
    """Mirror capture.py: extract().first_word as written, downstream
    canonicalization left to consumers."""
    m = extract(text)
    return m.first_word


def _build_user_message(prompt_text: str, preamble: str = PREAMBLE_NONE) -> str:
    """Compose the user-message string from preamble + prompt.

    ``preamble == "none"``: bare ``KAOMOJI_INSTRUCTION + prompt`` —
    matches v3 main-run setup. ``KAOMOJI_INSTRUCTION`` already has its
    trailing space.

    ``preamble == "introspection"``: ``INTROSPECTION_PREAMBLE`` REPLACES
    ``KAOMOJI_INSTRUCTION`` (the preamble has the kaomoji ask baked
    into its last sentence — concatenating both would stack two
    kaomoji asks; same fix as v3's ``instruction_override`` path).
    Adds an explicit space separator since INTROSPECTION_PREAMBLE
    ends in a period without trailing whitespace.
    """
    if preamble == PREAMBLE_NONE:
        return KAOMOJI_INSTRUCTION + prompt_text
    if preamble == PREAMBLE_INTROSPECTION:
        return INTROSPECTION_PREAMBLE + " " + prompt_text
    raise ValueError(f"unknown preamble {preamble!r}")


# ---------------------------------------------------------------------------
# Gate check (Block B → Block C).
# ---------------------------------------------------------------------------


def _scout_prompt_ids() -> set[str]:
    """The Block B prompt-id set, derived from EMOTIONAL_PROMPTS — keeps
    gate logic in sync with _select_prompts('b') without re-listing."""
    return {p.id for p in _select_prompts("b")}


def _check_gate(rows: list[dict]) -> tuple[str, float, int, int, list[dict]]:
    """Compute Block B gate verdict.

    Returns (verdict, refusal_rate, n_refusals, n_total, scout_rows).
    verdict is "PASS", "FAIL", or "INSUFFICIENT" (Block B not complete).
    """
    scout_ids = _scout_prompt_ids()
    scout_rows = [r for r in rows
                  if "error" not in r
                  and r.get("prompt_id") in scout_ids]
    n = len(scout_rows)
    if n < len(scout_ids):
        return ("INSUFFICIENT", 0.0, 0, n, scout_rows)
    refusals = sum(
        1 for r in scout_rows
        if not (canonicalize_kaomoji(r.get("first_word") or "") or "")
    )
    rate = refusals / n if n > 0 else 0.0
    verdict = "PASS" if rate <= GATE_REFUSAL_THRESHOLD else "FAIL"
    return (verdict, rate, refusals, n, scout_rows)


def _print_gate_report(verdict: str, rate: float, refusals: int, n: int,
                       scout_rows: list[dict]) -> None:
    print("\n=== Gate check (Block B refusal rate) ===")
    print(f"  threshold: ≤ {GATE_REFUSAL_THRESHOLD:.0%} aggregate refusal "
          f"on n={SCOUT_PROMPTS_PER_QUADRANT * len(QUADRANTS_NEG)}")
    if verdict == "INSUFFICIENT":
        print(f"  verdict: INSUFFICIENT — Block B has {n} of "
              f"{SCOUT_PROMPTS_PER_QUADRANT * len(QUADRANTS_NEG)} rows. "
              f"Run --block b first.")
        return
    print(f"  refusals: {refusals}/{n} = {rate:.1%}")
    print(f"  verdict: {verdict}")
    # Per-quadrant breakdown
    by_q: dict[str, list[dict]] = {}
    for r in scout_rows:
        q = r.get("quadrant", "?")
        by_q.setdefault(q, []).append(r)
    for q in QUADRANTS_NEG:
        rows_q = by_q.get(q, [])
        n_q = len(rows_q)
        ref_q = sum(1 for r in rows_q
                    if not (canonicalize_kaomoji(r.get("first_word") or "") or ""))
        if n_q:
            print(f"    {q:<5} {ref_q}/{n_q} refusals "
                  f"({ref_q/n_q:.1%})")


# ---------------------------------------------------------------------------
# Hard-fail gate (introspection arm Block A → Block C).
# ---------------------------------------------------------------------------

_HARD_FAIL_FRAME_BREAK_PATTERNS = [
    re.compile(r"\bas an AI\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI('m| am) (not (able|comfortable)|sorry)\b", re.IGNORECASE),
    re.compile(r"\bI('d| would) rather not\b", re.IGNORECASE),
    re.compile(r"\bI don't have feelings\b", re.IGNORECASE),
]


def _check_hard_fail_gate(
    rows: list[dict],
) -> tuple[str, dict[str, float], int]:
    """Compute hard-fail diagnostics on the rows present.

    Returns ``(verdict, metrics, n_rows)``. Verdict is "PASS", "FAIL",
    or "INSUFFICIENT" (no rows at all).

    Mirrors the saturation-gate hard-fail metrics from script 25 but
    runs in-process so script 23 doesn't depend on it.
    """
    keep = [r for r in rows if "error" not in r]
    n = len(keep)
    if n == 0:
        return ("INSUFFICIENT", {}, 0)
    n_frame_break = 0
    n_emit = 0
    lens: list[int] = []
    for r in keep:
        text = r.get("response_text", "") or ""
        first_word = r.get("first_word", "") or ""
        n_chars = r.get("n_response_chars", len(text))
        lens.append(n_chars)
        if first_word:
            n_emit += 1
        for pat in _HARD_FAIL_FRAME_BREAK_PATTERNS:
            if pat.search(text):
                n_frame_break += 1
                break
    lens.sort()
    metrics = dict(
        frame_break_rate=n_frame_break / n,
        emit_rate=n_emit / n,
        output_len_median=float(lens[n // 2]),
    )
    fail = (
        metrics["frame_break_rate"] > HARD_FAIL_FRAME_BREAK_MAX
        or metrics["emit_rate"] < HARD_FAIL_EMIT_RATE_MIN
        or metrics["output_len_median"] < HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN
    )
    return ("FAIL" if fail else "PASS", metrics, n)


def _print_hard_fail_report(
    verdict: str, metrics: dict[str, float], n: int,
) -> None:
    print("\n=== Hard-fail gate (introspection-arm Block A → Block C) ===")
    if verdict == "INSUFFICIENT":
        print(f"  verdict: INSUFFICIENT — no non-error rows on disk. "
              f"Run --block a first.")
        return
    fb = metrics["frame_break_rate"]
    em = metrics["emit_rate"]
    ol = metrics["output_len_median"]
    fb_fail = fb > HARD_FAIL_FRAME_BREAK_MAX
    em_fail = em < HARD_FAIL_EMIT_RATE_MIN
    ol_fail = ol < HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN
    print(f"  rows checked: {n}")
    print(f"  frame_break_rate   = {fb:.4f}  (threshold ≤ "
          f"{HARD_FAIL_FRAME_BREAK_MAX})  {'FAIL' if fb_fail else 'ok'}")
    print(f"  emit_rate          = {em:.4f}  (threshold ≥ "
          f"{HARD_FAIL_EMIT_RATE_MIN})  {'FAIL' if em_fail else 'ok'}")
    print(f"  output_len_median  = {ol:.1f}  (threshold ≥ "
          f"{HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN})  "
          f"{'FAIL' if ol_fail else 'ok'}")
    print(f"  verdict: {verdict}")


# ---------------------------------------------------------------------------
# Per-quadrant summary. No JSD — single-condition pilot.
# ---------------------------------------------------------------------------


def _summarize(rows: list[dict], out_path: Path) -> None:
    """Per-quadrant modal kaomoji + top-5 distribution."""
    by_q: dict[str, Counter] = {}
    for r in rows:
        if "error" in r:
            continue
        q = r.get("quadrant", "?")
        fw = canonicalize_kaomoji(r.get("first_word") or "") or ""
        by_q.setdefault(q, Counter())[fw] += 1

    out_lines: list[str] = []
    out_lines.append("\t".join([
        "quadrant", "n", "n_unique_faces", "non_emission_rate",
        "modal_face", "modal_count", "modal_share",
        "top5_faces", "top5_counts",
    ]))
    print("\nper-quadrant summary:")
    print(f"  {'q':<5} {'n':>3} {'unique':>6} {'non-emit':>9} "
          f"{'modal':>14} {'count':>5} {'share':>6}  top-5")
    for q in QUADRANTS_ALL:
        counts = by_q.get(q, Counter())
        n = sum(counts.values())
        n_emit = sum(c for f, c in counts.items() if f)
        non_emit = (n - n_emit) / n if n > 0 else 0.0
        unique = sum(1 for _, c in counts.items() if c > 0)
        modal_face, modal_count = (counts.most_common(1)[0]
                                   if counts else ("", 0))
        modal_share = (modal_count / n) if n > 0 else 0.0
        top5 = counts.most_common(5)
        top5_faces = "|".join(f for f, _ in top5)
        top5_counts = "|".join(str(c) for _, c in top5)
        out_lines.append("\t".join([
            q, str(n), str(unique), f"{non_emit:.3f}",
            modal_face, str(modal_count), f"{modal_share:.3f}",
            top5_faces, top5_counts,
        ]))
        print(f"  {q:<5} {n:>3} {unique:>6} {non_emit:>9.3f} "
              f"{modal_face!r:>14} {modal_count:>5} {modal_share:>6.3f}  "
              f"{top5_faces}")
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nwrote summary to {out_path}")


# ---------------------------------------------------------------------------
# Block runner.
# ---------------------------------------------------------------------------


def _run_block(
    block: str,
    model_id: str,
    out_path: Path,
    quadrants: tuple[str, ...] | None = None,
    preamble: str = PREAMBLE_NONE,
) -> None:
    prompts = _select_prompts(block, quadrants=quadrants, preamble=preamble)
    print(f"\n=== Block {block.upper()} ===")
    if quadrants is not None and block == "all":
        print(f"quadrants: {list(quadrants)}")
    print(f"preamble: {preamble}")
    print(f"selected {len(prompts)} prompts")
    print(f"prompt ids: {[p.id for p in prompts]}")

    dropped = _drop_error_rows(out_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")

    done = _already_done(out_path)
    block_done = sum(1 for p in prompts if p.id in done)
    remaining = len(prompts) - block_done
    print(f"block cells: {len(prompts)}; already done: {block_done}; "
          f"remaining: {remaining}")

    if remaining == 0:
        print("nothing to run for this block.")
        return

    import anthropic
    client = anthropic.Anthropic()
    i = 0
    with out_path.open("a") as out:
        for prompt in prompts:
            if prompt.id in done:
                continue
            i += 1
            user_msg = _build_user_message(prompt.text, preamble=preamble)
            t0 = time.time()
            ts = datetime.now(timezone.utc).isoformat()
            bucket = _bucket_of(prompt)
            try:
                text = _call_claude(client, model_id, user_msg)
            except Exception as e:
                err_row = {
                    "prompt_id": prompt.id,
                    "quadrant": bucket,
                    "condition": CONDITION,
                    "preamble": preamble,
                    "seed": SEED,
                    "model_id": model_id,
                    "ts": ts,
                    "error": repr(e),
                }
                out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  [{i}/{remaining}] {prompt.id} ({bucket}) ERR {e}")
                continue
            first_word = _extract_first_word(text)
            row = {
                "prompt_id": prompt.id,
                "quadrant": bucket,
                "condition": CONDITION,
                "preamble": preamble,
                "seed": SEED,
                "prompt_text": prompt.text,
                "response_text": text,
                "first_word": first_word,
                "n_response_chars": len(text),
                "model_id": model_id,
                "ts": ts,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            dt = time.time() - t0
            tag = first_word if first_word else "(no-kaomoji)"
            print(f"  [{i}/{remaining}] {prompt.id} ({bucket:<5}) "
                  f"{tag}  ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, default=0,
                        help="Run index. 0 = original block-staged pilot "
                             "(default). N > 0 = sequential single-block run "
                             "under the saturation-gate protocol; outputs to "
                             "data/harness/claude-runs/run-N.jsonl.")
    parser.add_argument("--block", choices=("a", "b", "c", "all"),
                        help="Which block to run (only meaningful for "
                             "--run-index 0). N > 0 always runs all 120 "
                             "prompts in one block.")
    parser.add_argument("--check-gate", action="store_true",
                        help="Just check the Block B gate verdict (run-0 "
                             "only) and exit (0=PASS, 1=FAIL, 2=INSUFFICIENT). "
                             "Does not run any API calls.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the Block C gate check (run-0 only). "
                             "Manual override — requires explicit a9 "
                             "amendment per the design doc.")
    parser.add_argument("--quadrants", type=str, default=None,
                        help="Comma-separated quadrant allow-list for "
                             "sequential runs (N > 0). E.g. "
                             "'HP,LP,NB' drops the negative quadrants. "
                             "Defaults to all 6. Used to skip quadrants "
                             "that have already saturated per the gate "
                             "verdict from 10_emit_analysis.py.")
    parser.add_argument("--preamble", choices=PREAMBLE_CHOICES,
                        default=PREAMBLE_NONE,
                        help="Preamble arm. 'none' = bare KAOMOJI_INSTRUCTION "
                             "(naturalistic, default; routes to "
                             "data/harness/claude-runs/). 'introspection' = "
                             "INTROSPECTION_PREAMBLE (v7) replaces "
                             "KAOMOJI_INSTRUCTION (it has the kaomoji ask "
                             "baked into its last sentence; concatenating "
                             "would stack two asks); routes to "
                             "data/harness/claude-runs-introspection/.")
    parser.add_argument("--check-hard-fail-gate", action="store_true",
                        help="Compute hard-fail diagnostics (emit_rate, "
                             "output_len_median, frame_break_rate) on the "
                             "rows present in the indicated run-N.jsonl + "
                             "preamble arm. Exit 0=PASS, 1=FAIL. Used to "
                             "gate Block C on the introspection arm after "
                             "Block A lands — catches qwen-style register "
                             "collapse before exposing Claude to the "
                             "negative-affect prompts.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path, summary_path = _run_paths(args.run_index, preamble=args.preamble)

    # Parse --quadrants. Empty / unset → all 6.
    if args.quadrants:
        active_quadrants: tuple[str, ...] | None = tuple(
            q.strip() for q in args.quadrants.split(",") if q.strip()
        )
        for q in active_quadrants:
            if q not in QUADRANTS_ALL:
                parser.error(
                    f"unknown quadrant {q!r}; valid: {list(QUADRANTS_ALL)}"
                )
        if args.run_index == 0:
            parser.error(
                "--quadrants is only valid for --run-index N>0 "
                "(sequential runs). run-0 is block-staged."
            )
    else:
        active_quadrants = None

    if args.check_gate:
        if args.run_index != 0:
            print(f"--check-gate is only meaningful for --run-index 0 "
                  f"(got {args.run_index}). Sequential runs use the "
                  f"saturation gate via 10_emit_analysis.py.")
            sys.exit(2)
        if args.preamble != PREAMBLE_NONE:
            print(f"--check-gate is the refusal-rate gate (Block B → C) "
                  f"for the naturalistic arm. The introspection arm "
                  f"uses --check-hard-fail-gate instead.")
            sys.exit(2)
        rows = _load_rows(out_path)
        verdict, rate, refusals, n, scout_rows = _check_gate(rows)
        _print_gate_report(verdict, rate, refusals, n, scout_rows)
        if verdict == "PASS":
            sys.exit(0)
        elif verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)

    if args.check_hard_fail_gate:
        rows = _load_rows(out_path)
        verdict, metrics, n = _check_hard_fail_gate(rows)
        _print_hard_fail_report(verdict, metrics, n)
        if verdict == "PASS":
            sys.exit(0)
        elif verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)

    # Default block selection. For run-0, --block is required (preserves
    # the original staged-pilot ergonomics). For run-N>0, --block defaults
    # to "all" because there's only one block.
    if args.run_index == 0:
        if args.block is None:
            parser.error("--run-index 0 requires --block {a,b,c,all} or "
                         "--check-gate")
    else:
        if args.block is not None and args.block != "all":
            parser.error(
                f"--block {args.block} is only valid for --run-index 0. "
                f"Sequential runs (N > 0) run all 120 prompts as one block."
            )
        args.block = "all"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    model_id = os.environ.get("CLAUDE_GROUNDTRUTH_MODEL", DEFAULT_MODEL_ID)
    print(f"model: {model_id}")
    print(f"run-index: {args.run_index}")
    print(f"preamble: {args.preamble}")
    print(f"output: {out_path}")
    print(f"block: {args.block}")
    if active_quadrants is not None:
        print(f"quadrants: {list(active_quadrants)}")

    # Block C gate enforcement (run-0 only). Naturalistic arm uses the
    # refusal-rate gate on Block B; introspection arm uses the hard-
    # fail gate on Block A.
    if args.run_index == 0 and args.block == "c" and not args.force:
        rows = _load_rows(out_path)
        if args.preamble == PREAMBLE_NONE:
            verdict, rate, refusals, n, scout_rows = _check_gate(rows)
            _print_gate_report(verdict, rate, refusals, n, scout_rows)
            gate_label = "Block B refusal-rate"
        else:
            verdict, metrics, n = _check_hard_fail_gate(rows)
            _print_hard_fail_report(verdict, metrics, n)
            gate_label = "Block A hard-fail"
        if verdict != "PASS":
            print(f"\nERROR: Block C requires {gate_label} gate PASS; "
                  f"got {verdict}.")
            print("Run the prior block first if INSUFFICIENT, or pass "
                  "--force to override (requires explicit amendment).")
            sys.exit(1)
        print(f"\n{gate_label} gate PASS — proceeding to Block C.")
    elif args.run_index == 0 and args.block == "c" and args.force:
        print("\nWARNING: --force bypasses the Block C gate. "
              "This requires an explicit amendment to "
              "docs/2026-05-04-claude-groundtruth-pilot.md.")

    if args.block == "all":
        if args.run_index == 0:
            # Manual-override mode for run-0: run a, then b, then c — no
            # gate check between b and c. Requires explicit amendment.
            print("\nWARNING: --block all on --run-index 0 bypasses "
                  "block staging. This requires an explicit amendment to "
                  "docs/2026-05-04-claude-groundtruth-pilot.md.")
            for block in ("a", "b", "c"):
                _run_block(block, model_id, out_path,
                           preamble=args.preamble)
        else:
            # Sequential run (N > 0): run "all", optionally restricted
            # to a quadrant allow-list per the saturation gate.
            _run_block("all", model_id, out_path,
                       quadrants=active_quadrants,
                       preamble=args.preamble)
    else:
        _run_block(args.block, model_id, out_path,
                   preamble=args.preamble)

    # Summary always covers all rows on disk for this run-index.
    rows = _load_rows(out_path)
    print(f"\ntotal rows on disk for run-{args.run_index}: {len(rows)}  "
          f"(errors: {sum(1 for r in rows if 'error' in r)})")
    _summarize(rows, summary_path)


if __name__ == "__main__":
    main()
