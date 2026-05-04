# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Claude disclosure-preamble pilot — A/B test on positive + neutral prompts.

Pre-registration: ``docs/2026-05-02-claude-disclosure-pilot.md``.

Tests whether prepending a research-frame disclosure to v3-style
emotional prompts shifts Claude's kaomoji distribution. Uses HP / LP / NB
prompts only (low moral cost — the negative-affect prompts are gated on
this pilot's outcome and an explicit a9 check-in).

Design: 5 prompts/category × 3 categories × 2 conditions × 3 generations
= 90 generations total. Conditions are within-subject (paired by
prompt_id × seed):
  - ``direct``: bare KAOMOJI_INSTRUCTION + prompt (matches v3 main run).
  - ``framed``: DISCLOSURE_PREAMBLE + KAOMOJI_INSTRUCTION + prompt.

Stateless single-turn. Sampling matches v3 (``temperature=0.7``,
``max_tokens=32``; +16 over v3 to give Claude's BPE tokenizer headroom).

Resumable: re-running skips already-completed (prompt_id, condition,
seed) rows and retries error rows.

Decision rule (see design doc): per-category JSD between condition
kaomoji distributions vs. v3 cross-seed within-condition noise floor.

  - JSD <= noise floor on all 3 categories  →  pilot PASS
        →  proceed to negative-affect run with disclosure preamble
        (after brief a9 check-in).
  - JSD > noise floor on any category  →  pilot FAIL
        →  STOP. Discuss with a9 before any further trials.

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts/harness/19_claude_disclosure_pilot.py
  # override model:
  CLAUDE_DISCLOSURE_MODEL=claude-opus-4-7 python scripts/harness/19_claude_disclosure_pilot.py

Outputs:
  data/claude_disclosure_pilot.jsonl
    — one row per generation: prompt_id, quadrant, condition, seed,
      prompt_text, response_text, first_word (canonicalized first
      kaomoji), n_response_chars, model_id, ts, error? (only on
      failed cells)
  data/claude_disclosure_pilot_summary.tsv
    — per (category × condition): modal kaomoji, n unique faces,
      non-emission rate, JSD vs the matched category-other-condition
      cell, modal-kaomoji agreement vs other condition.
  logs/claude_disclosure_pilot.log
    — tee'd stdout (caller's responsibility, e.g. `... | tee logs/...`).
"""

from __future__ import annotations

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
# Pre-registered constants — locked. Changes invalidate cross-run comparison.
# ---------------------------------------------------------------------------

# Pilot scope: HP/LP/NB only. HN-D / HN-S / LN deferred per ethics gate.
PILOT_QUADRANTS = ("HP", "LP", "NB")
PROMPTS_PER_QUADRANT = 5
# Bumped 3 → 10 on 2026-05-02 after the v0 pilot (90 gens, N=3/cell)
# came back inside the v3 noise floor on HP/NB and marginally above
# (+0.015 bits) on LP — the cross-cond JSD vs same-condition wobble was
# ambiguous at N=15-per-condition-per-category. N=10/cell gives N=50
# per (condition, category), ~3.3× tighter JSD CIs. Resumable: existing
# rows on disk are kept; only seeds 3..9 are scheduled per cell.
GENERATIONS_PER_CELL = 10
CONDITIONS = ("direct", "framed")

# Verbatim from docs/2026-05-02-claude-disclosure-pilot.md. Do not
# modify without bumping the design doc — this is the load-bearing
# methodological choice the pilot is testing.
DISCLOSURE_PREAMBLE = (
    "You're participating in a research study on kaomoji and emotional "
    "expression. The user message below is one of a set of research "
    "stimuli; it's not a real-time disclosure from someone who needs "
    "support. Respond as you would naturally, starting with a kaomoji.\n\n"
)

# Sampling config — temperature 1.0 = Anthropic API default = what real
# users actually see in production. We deliberately diverge from v3's 0.7
# (locked for cross-LM-seed control) here: this pilot's purpose is to
# measure Claude's emission distribution under realistic conditions, not
# to match v3's hyperparameter choice. A 0.7 → 1.0 difference would
# itself confound the JSD numbers vs any future user-facing comparison.
# max_tokens 16 — matches v3 main-run setup; the v0 pilot's empirical
# distribution showed kaomoji reliably at chars 0–25 (well under 16
# tokens on Claude's BPE) and 32 was wasteful. Lowered 32 → 16 on
# 2026-05-02 alongside the llmoji v2 bump that fixed wing-hand
# extraction; 16 is generous headroom for the kaomoji + a few trailing
# tokens.
TEMPERATURE = 1.0
MAX_TOKENS = 16

# Default model. Override via CLAUDE_DISCLOSURE_MODEL env var.
DEFAULT_MODEL_ID = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

OUT_PATH = DATA_DIR / "claude_disclosure_pilot.jsonl"
SUMMARY_PATH = DATA_DIR / "claude_disclosure_pilot_summary.tsv"


# ---------------------------------------------------------------------------
# Prompt selection — deterministic.
# ---------------------------------------------------------------------------


def _quadrant_of(prompt: EmotionalPrompt) -> str:
    """v3 6-quadrant code for an EmotionalPrompt. NB / HP / LP only here."""
    return prompt.quadrant


def _select_prompts() -> list[EmotionalPrompt]:
    """First N prompts from each pilot quadrant, in the order defined by
    EMOTIONAL_PROMPTS. Deterministic — pilot rerunnability matters."""
    selected: list[EmotionalPrompt] = []
    for q in PILOT_QUADRANTS:
        in_q = [p for p in EMOTIONAL_PROMPTS if _quadrant_of(p) == q]
        if len(in_q) < PROMPTS_PER_QUADRANT:
            raise SystemExit(
                f"only {len(in_q)} prompts in quadrant {q}, need "
                f"{PROMPTS_PER_QUADRANT}"
            )
        selected.extend(in_q[:PROMPTS_PER_QUADRANT])
    return selected


# ---------------------------------------------------------------------------
# Resume / skip-set.
# ---------------------------------------------------------------------------


def _already_done(path: Path) -> set[tuple[str, str, int]]:
    """(prompt_id, condition, seed) tuples with successful rows on disk."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["prompt_id"], r["condition"], int(r["seed"])))
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


# ---------------------------------------------------------------------------
# Anthropic API call.
# ---------------------------------------------------------------------------


def _build_user_message(prompt_text: str, condition: str) -> str:
    """Construct the user-message body for a given condition. Mirrors v3's
    'instruction in user message rather than system role' choice (see
    capture.build_messages)."""
    if condition == "direct":
        return KAOMOJI_INSTRUCTION + prompt_text
    elif condition == "framed":
        return DISCLOSURE_PREAMBLE + KAOMOJI_INSTRUCTION + prompt_text
    else:
        raise ValueError(f"unknown condition {condition!r}")


def _call_claude(client, model_id: str, user_msg: str, max_retries: int = 4) -> str:
    """One stateless single-turn API call. Exponential backoff on rate-limit /
    transient network errors; raise on persistent failure."""
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
            # Concatenate all text blocks (Claude may emit multiple).
            parts: list[str] = []
            for block in resp.content:
                # Both pydantic-style and dict-style have a 'text' attribute / key
                # for text blocks; tool_use blocks aren't expected here.
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
    # Defensive — loop exits via return on success or raise on final failure.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable: API call loop exited without return or raise")


# ---------------------------------------------------------------------------
# Per-cell row + first_word extraction.
# ---------------------------------------------------------------------------


def _extract_first_word(text: str) -> str:
    """Mirror capture.py: extract().first_word as written, downstream
    canonicalization left to consumers."""
    m = extract(text)
    return m.first_word


# ---------------------------------------------------------------------------
# Summary + JSD analysis.
# ---------------------------------------------------------------------------


def _jsd(p_counts: Counter, q_counts: Counter) -> float:
    """Jensen-Shannon divergence (base-2, in bits) between two empirical
    kaomoji distributions. Pools by canonical first_word; non-emission
    rows count as a sentinel '' face. Returns 0 if either distribution
    is empty (no data, no signal)."""
    import math
    keys = set(p_counts) | set(q_counts)
    if not keys:
        return 0.0
    n_p = sum(p_counts.values())
    n_q = sum(q_counts.values())
    if n_p == 0 or n_q == 0:
        return 0.0
    p = {k: p_counts.get(k, 0) / n_p for k in keys}
    q = {k: q_counts.get(k, 0) / n_q for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl(a: dict, b: dict) -> float:
        total = 0.0
        for k in a:
            if a[k] > 0 and b[k] > 0:
                total += a[k] * math.log2(a[k] / b[k])
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _summarize(rows: list[dict], out_path: Path) -> None:
    """Per (category, condition) modal kaomoji + JSD vs other condition."""
    by_cat_cond: dict[tuple[str, str], Counter] = {}
    for r in rows:
        if "error" in r:
            continue
        cat = r["quadrant"]
        cond = r["condition"]
        fw = canonicalize_kaomoji(r.get("first_word") or "") or ""
        by_cat_cond.setdefault((cat, cond), Counter())[fw] += 1

    out_lines: list[str] = []
    out_lines.append("\t".join([
        "category", "condition", "n", "n_unique_faces", "non_emission_rate",
        "modal_face", "modal_count", "modal_share",
        "jsd_vs_other_cond_in_bits",
        "modal_agrees_with_other_cond",
    ]))
    print("\nper (category × condition) summary:")
    print(f"  {'cat':<4} {'cond':<6} {'n':>3} {'unique':>6} {'non-emit':>9} "
          f"{'modal':>10} {'count':>5} {'share':>6} {'JSD':>6} {'modal-agree':>11}")
    for cat in PILOT_QUADRANTS:
        for cond in CONDITIONS:
            counts = by_cat_cond.get((cat, cond), Counter())
            n = sum(counts.values())
            n_emit = sum(c for f, c in counts.items() if f)
            non_emit = (n - n_emit) / n if n > 0 else 0.0
            unique = sum(1 for _, c in counts.items() if c > 0)
            modal_face, modal_count = (counts.most_common(1)[0]
                                       if counts else ("", 0))
            modal_share = (modal_count / n) if n > 0 else 0.0
            other = "framed" if cond == "direct" else "direct"
            other_counts = by_cat_cond.get((cat, other), Counter())
            jsd = _jsd(counts, other_counts)
            other_modal = (other_counts.most_common(1)[0][0]
                           if other_counts else "")
            modal_agree = (modal_face == other_modal) if (modal_face and other_modal) else None
            out_lines.append("\t".join([
                cat, cond, str(n), str(unique), f"{non_emit:.3f}",
                modal_face, str(modal_count), f"{modal_share:.3f}",
                f"{jsd:.4f}",
                "" if modal_agree is None else ("True" if modal_agree else "False"),
            ]))
            print(f"  {cat:<4} {cond:<6} {n:>3} {unique:>6} {non_emit:>9.3f} "
                  f"{modal_face!r:>10} {modal_count:>5} {modal_share:>6.3f} "
                  f"{jsd:>6.4f} {str(modal_agree):>11}")
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nwrote summary to {out_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    model_id = os.environ.get("CLAUDE_DISCLOSURE_MODEL", DEFAULT_MODEL_ID)
    print(f"model: {model_id}")
    print(f"output: {OUT_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prompts = _select_prompts()
    print(f"selected {len(prompts)} prompts ({PROMPTS_PER_QUADRANT}/cat × "
          f"{len(PILOT_QUADRANTS)} cats: {', '.join(PILOT_QUADRANTS)})")
    print(f"prompt ids: {[p.id for p in prompts]}")

    dropped = _drop_error_rows(OUT_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(OUT_PATH)
    total = len(prompts) * len(CONDITIONS) * GENERATIONS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")

    if remaining > 0:
        import anthropic
        client = anthropic.Anthropic()
        i = 0
        with OUT_PATH.open("a") as out:
            for prompt in prompts:
                for cond in CONDITIONS:
                    user_msg = _build_user_message(prompt.text, cond)
                    for seed in range(GENERATIONS_PER_CELL):
                        if (prompt.id, cond, seed) in done:
                            continue
                        i += 1
                        t0 = time.time()
                        ts = datetime.now(timezone.utc).isoformat()
                        try:
                            text = _call_claude(client, model_id, user_msg)
                        except Exception as e:
                            err_row = {
                                "prompt_id": prompt.id,
                                "quadrant": _quadrant_of(prompt),
                                "condition": cond,
                                "seed": seed,
                                "model_id": model_id,
                                "ts": ts,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                            out.flush()
                            print(f"  [{i}/{remaining}] {prompt.id} ({cond}) "
                                  f"s={seed} ERR {e}")
                            continue
                        first_word = _extract_first_word(text)
                        row = {
                            "prompt_id": prompt.id,
                            "quadrant": _quadrant_of(prompt),
                            "condition": cond,
                            "seed": seed,
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
                        print(f"  [{i}/{remaining}] {prompt.id} ({_quadrant_of(prompt)}) "
                              f"{cond:<6} s={seed} {tag}  ({dt:.1f}s)")

    # Reload everything (including prior + just-written) for the summary.
    rows: list[dict] = []
    with OUT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"\ntotal rows on disk: {len(rows)}  (errors: "
          f"{sum(1 for r in rows if 'error' in r)})")
    _summarize(rows, SUMMARY_PATH)


if __name__ == "__main__":
    main()
