# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Claude ground-truth pilot — naturalistic 6-quadrant emission.

Pre-registration: ``docs/2026-05-04-claude-groundtruth-pilot.md``.

Collects ground-truth Claude (Opus 4.7) kaomoji emissions across all 6
Russell quadrants (HP / LP / NB / HN-D / HN-S / LN) under naturalistic
single-turn calls — no disclosure preamble, no research framing, just
v3's ``KAOMOJI_INSTRUCTION`` + the affective prompt.

Staged design:
  Block A (unconditional)  — HP / LP / NB × 20 prompts × 1 gen = 60 gens
  Block B (negative scout) — HN-D / HN-S / LN × 5 prompts × 1 gen = 15 gens
  Block C (gated)          — HN-D / HN-S / LN × 15 remaining × 1 gen = 45 gens
Total: 75 (gate fail) or 120 (gate pass).

Block A and Block B fire in parallel. Block C is gated on Block B's
aggregate refusal rate (>25% on n=15 → halt). See pre-reg doc for the
welfare reasoning.

Stateless single-turn. Sampling: ``temperature=1.0``, ``max_tokens=16``
(production-faithful). Resumable: re-running a block skips already-
completed rows by ``prompt_id``. Errored rows are stripped on resume
and retried.

Usage:
  export ANTHROPIC_API_KEY=...
  # Block A (positive/neutral, unconditional):
  python scripts/harness/23_claude_groundtruth_pilot.py --block a
  # Block B (negative scout):
  python scripts/harness/23_claude_groundtruth_pilot.py --block b
  # Gate check (exits 0=PASS, 1=FAIL, 2=insufficient data):
  python scripts/harness/23_claude_groundtruth_pilot.py --check-gate
  # Block C (gated; refuses to run if gate hasn't passed):
  python scripts/harness/23_claude_groundtruth_pilot.py --block c
  # Forced run — bypasses gate; manual override only:
  python scripts/harness/23_claude_groundtruth_pilot.py --block c --force
  # Override model:
  CLAUDE_GROUNDTRUTH_MODEL=claude-opus-4-7 python scripts/harness/23_claude_groundtruth_pilot.py --block a

Outputs:
  data/claude_groundtruth_pilot.jsonl
    — one row per generation: prompt_id, quadrant (6-way),
      condition="direct", seed=0, prompt_text, response_text,
      first_word (canonicalized first kaomoji), n_response_chars,
      model_id, ts, error? (only on failed cells)
  data/claude_groundtruth_pilot_summary.tsv
    — per quadrant: n, n_unique_faces, non_emission_rate,
      modal_face, modal_share, top-5 distribution
  logs/claude_groundtruth_pilot.log
    — tee'd stdout (caller's responsibility)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji, extract

from llmoji_study.config import DATA_DIR, KAOMOJI_INSTRUCTION
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

# Default model. Override via CLAUDE_GROUNDTRUTH_MODEL env var.
DEFAULT_MODEL_ID = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

OUT_PATH = DATA_DIR / "claude_groundtruth_pilot.jsonl"
SUMMARY_PATH = DATA_DIR / "claude_groundtruth_pilot_summary.tsv"


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


def _select_prompts(block: str) -> list[EmotionalPrompt]:
    """Block → ordered prompt list. Each block's prompt set is disjoint
    from the others, so rows from different blocks coexist in one JSONL
    without overlap."""
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
        for q in QUADRANTS_NEG:
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[SCOUT_PROMPTS_PER_QUADRANT:PROMPTS_PER_QUADRANT])
    elif block == "all":
        # Manual-override convenience. Includes all 6 quadrants × 20 prompts.
        for q in QUADRANTS_ALL:
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


def _build_user_message(prompt_text: str) -> str:
    """Bare KAOMOJI_INSTRUCTION + prompt — matches v3 main-run setup."""
    return KAOMOJI_INSTRUCTION + prompt_text


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


def _run_block(block: str, model_id: str) -> None:
    prompts = _select_prompts(block)
    print(f"\n=== Block {block.upper()} ===")
    print(f"selected {len(prompts)} prompts")
    print(f"prompt ids: {[p.id for p in prompts]}")

    dropped = _drop_error_rows(OUT_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")

    done = _already_done(OUT_PATH)
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
    with OUT_PATH.open("a") as out:
        for prompt in prompts:
            if prompt.id in done:
                continue
            i += 1
            user_msg = _build_user_message(prompt.text)
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
    parser.add_argument("--block", choices=("a", "b", "c", "all"),
                        help="Which block to run. See module docstring.")
    parser.add_argument("--check-gate", action="store_true",
                        help="Just check the Block B gate verdict and exit "
                             "(0=PASS, 1=FAIL, 2=INSUFFICIENT). Does not run "
                             "any API calls.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the Block C gate check. Manual override "
                             "only — requires explicit a9 amendment per the "
                             "design doc.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.check_gate:
        rows = _load_rows(OUT_PATH)
        verdict, rate, refusals, n, scout_rows = _check_gate(rows)
        _print_gate_report(verdict, rate, refusals, n, scout_rows)
        if verdict == "PASS":
            sys.exit(0)
        elif verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)

    if args.block is None:
        parser.error("must specify --block {a,b,c,all} or --check-gate")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    model_id = os.environ.get("CLAUDE_GROUNDTRUTH_MODEL", DEFAULT_MODEL_ID)
    print(f"model: {model_id}")
    print(f"output: {OUT_PATH}")
    print(f"block: {args.block}")

    # Block C gate enforcement.
    if args.block == "c" and not args.force:
        rows = _load_rows(OUT_PATH)
        verdict, rate, refusals, n, scout_rows = _check_gate(rows)
        _print_gate_report(verdict, rate, refusals, n, scout_rows)
        if verdict != "PASS":
            print(f"\nERROR: Block C requires gate PASS; got {verdict}.")
            print("Run Block B first if INSUFFICIENT, or pass --force to "
                  "override (requires explicit amendment).")
            sys.exit(1)
        print("\ngate PASS — proceeding to Block C.")
    elif args.block == "c" and args.force:
        print("\nWARNING: --force bypasses the Block C gate. "
              "This requires an explicit amendment to "
              "docs/2026-05-04-claude-groundtruth-pilot.md.")

    if args.block == "all":
        # Manual-override mode: run a, then b, then c — no gate check.
        # Useful only for forced reruns after manual amendment.
        print("\nWARNING: --block all bypasses block staging. "
              "This requires an explicit amendment to "
              "docs/2026-05-04-claude-groundtruth-pilot.md.")
        for block in ("a", "b", "c"):
            _run_block(block, model_id)
    else:
        _run_block(args.block, model_id)

    # Summary always covers all rows on disk.
    rows = _load_rows(OUT_PATH)
    print(f"\ntotal rows on disk: {len(rows)}  (errors: "
          f"{sum(1 for r in rows if 'error' in r)})")
    _summarize(rows, SUMMARY_PATH)


if __name__ == "__main__":
    main()
