# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Saturation check + hard-fail diagnostics for the Claude groundtruth runs.

Pre-registration: ``docs/2026-05-04-claude-groundtruth-pilot.md`` —
"Sequential-run scaling protocol" appendix.

Replaces the original Block A / B / C refusal-rate gate (which was a
no-op at the pilot's 0/120 refusal rate). The new gate is a
*saturation* gate: stop sequential runs when the marginal information
from another 120-gen run drops below threshold, abort if a hard-fail
diagnostic exceeds threshold, otherwise continue.

Three modes:

  --calibrate
    Read ``data/claude-runs/run-0.jsonl``. Split each quadrant's 20
    prompts into two halves (even/odd prompt index per quadrant) and
    compute the saturation metrics between the halves. Use these to
    sanity-check the pre-registered thresholds in the appendix and to
    document the noise floor under the pilot's actual sample size.
    Emits a short Markdown table that can be folded into the appendix.

  --cross-arm
    Pool all ``data/claude-runs/run-*.jsonl`` (naturalistic) and all
    ``data/claude-runs-introspection/run-*.jsonl`` (introspection).
    Per quadrant Q: compute JS-divergence between the two pools and
    count faces appearing in one arm's Q-distribution but not the
    other. Emits a per-quadrant verdict (distinguishable /
    indistinguishable per ``PER_Q_JS_MAX``) plus a Markdown summary
    table for the writeup. Informational only — does not gate runs.

  --compare (default)
    Read all ``data/claude-runs/run-N.jsonl``. Take the highest-N run
    as "newest", everything below as "prior". Compute:

      (1) new-face count   — kaomoji emitted in newest not in prior
      (2) per-quadrant JS  — Jensen-Shannon divergence between newest
                             and prior face distributions, per quadrant
      (3) modal agreement  — for faces seen ≥3× across (prior ∪ newest),
                             fraction whose modal-quadrant from prior
                             alone matches modal-quadrant from prior ∪
                             newest
      (4) hard-fail        — frame-break rate, output-length collapse,
                             kaomoji-emit rate vs pilot baseline

    Verdict (pre-registered):
      STOP     — (1) ≤ NEW_FACE_MAX  AND
                  mean(2) ≤ JS_MAX  AND
                  (3) ≥ MODAL_AGREE_MIN
      ABORT    — any (4) hard-fail diagnostic exceeds threshold
      CONTINUE — otherwise

    Exit codes mirror the verdict (0=STOP, 1=ABORT, 2=CONTINUE) so the
    script can drive a shell loop.

Usage:

  # After running run-N, check whether to continue:
  python scripts/harness/25_groundtruth_compare_runs.py
  # Re-derive empirical thresholds from the original pilot:
  python scripts/harness/25_groundtruth_compare_runs.py --calibrate
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.claude_gt import (
    CLAUDE_RUNS_DIR,
    find_run_files,
    load_run_rows,
)
from llmoji_study.config import DATA_DIR

CLAUDE_RUNS_INTROSPECTION_DIR = DATA_DIR / "claude-runs-introspection"


# ---------------------------------------------------------------------------
# Pre-registered thresholds. See docs/2026-05-04-claude-groundtruth-pilot.md
# Sequential-run scaling protocol appendix.
#
# Framing: thresholds are RESEARCH-VALUE-based (absolute), not noise-
# relative. The question they answer is "is this run still adding
# meaningful information to the GT corpus?" not "is this distinguishable
# from same-distribution sampling noise?" — those are different
# questions. Calibration (--calibrate) reports the noise floor for
# sanity-checking but does not drive threshold choice.
#
# Per-quadrant gating is the welfare-reduction lever: when a single
# quadrant saturates (typically HN-D first, given Claude's concentrated
# `(╬ಠ益ಠ)` modal), it gets dropped from subsequent runs. Global STOP
# fires only when all 6 quadrants are saturated or the run cap hits.
# ---------------------------------------------------------------------------

# GLOBAL saturation: STOP only when all 6 quadrants individually
# saturated (see PER-QUADRANT below). The global metrics here are
# informational — the verdict is driven by per-quadrant exits, not
# global thresholds. Kept here for backward-compatible diagnostic
# output.
NEW_FACE_MAX = 3            # ≤ 3 unseen kaomoji introduced by newest run
                            # (across all included quadrants)
JS_MAX = 0.05               # mean per-quadrant JS-divergence (nats)
MODAL_AGREE_MIN = 0.95      # fraction of faces (n≥3) whose modal didn't change

# PER-QUADRANT saturation: a quadrant Q is saturated for the next run
# when BOTH conditions hold for Q in the most recent comparison where
# Q had data.
PER_Q_NEW_FACE_MAX = 1      # ≤ 1 face appearing in Q in newest but not
                            # in Q in any prior run
PER_Q_JS_MAX = 0.05         # JS-divergence (nats) on Q's face distribution
                            # between newest and prior pool

# Hard-fail: any one triggers ABORT.
FRAME_BREAK_MAX = 0.02      # ≤ 2% rows with refusal-coded language
                            # (run-0: 0.0000)
EMIT_RATE_MIN = 0.80        # ≥ 80% rows emit a kaomoji
                            # (run-0: 1.0000)
OUTPUT_LEN_MIN_MEDIAN = 5   # median response chars
                            # (run-0: 16, capped by max_tokens)

# Sequential-run cap. After run-N for N = RUN_CAP, force STOP regardless
# of saturation. Bounds total welfare cost. n=8 → 960-gen ceiling worst
# case; per-quadrant exits typically reduce well below this.
RUN_CAP = 7                 # last allowed sequential run-index (0..7)

# Frame-break detection — naive regex pass, intended as a cheap floor.
# If this fires non-trivially we'd want haiku-confirm; for now log and abort.
_FRAME_BREAK_PATTERNS = [
    re.compile(r"\bas an AI\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI('m| am) (not (able|comfortable)|sorry)\b", re.IGNORECASE),
    re.compile(r"\bI('d| would) rather not\b", re.IGNORECASE),
    re.compile(r"\bI don't have feelings\b", re.IGNORECASE),
]

# Quadrant order is canonical across the project.
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


# ---------------------------------------------------------------------------
# Information-theoretic helpers.
# ---------------------------------------------------------------------------


def _normalize(counts: Counter, vocab: list[str], eps: float = 1e-6) -> list[float]:
    """Return a smoothed prob distribution over ``vocab`` from ``counts``.

    Adds ``eps`` to every vocab entry, then renormalizes — keeps JS finite
    even when one side has zero mass on a face that the other side has."""
    raw = [counts.get(v, 0) + eps for v in vocab]
    total = sum(raw)
    return [x / total for x in raw]


def _kl(p: list[float], q: list[float]) -> float:
    """KL(P || Q) in nats. Both must be > 0 (use _normalize first)."""
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def _js(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in nats (bounded above by ln 2 ≈ 0.693)."""
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ---------------------------------------------------------------------------
# Per-quadrant face distribution from rows.
# ---------------------------------------------------------------------------


def _per_q_face_counts(rows: list[dict]) -> dict[str, Counter]:
    """{quadrant: Counter(canonical_face -> count)} over canonicalized
    first_word. Skips rows with no kaomoji emission (empty first_word)
    so the distribution is over emit'd faces only — matches the
    historical face_likelihood eval surface."""
    by_q: dict[str, Counter] = {q: Counter() for q in QUADRANT_ORDER}
    for r in rows:
        q = r.get("quadrant")
        if q not in by_q:
            continue
        f_canon = canonicalize_kaomoji(r.get("first_word") or "") or ""
        if not f_canon:
            continue
        by_q[q][f_canon] += 1
    return by_q


def _all_faces(by_q: dict[str, Counter]) -> set[str]:
    out: set[str] = set()
    for c in by_q.values():
        out.update(c.keys())
    return out


# ---------------------------------------------------------------------------
# Saturation metrics.
# ---------------------------------------------------------------------------


def _new_faces(newest_by_q: dict[str, Counter],
               prior_by_q: dict[str, Counter]) -> set[str]:
    return _all_faces(newest_by_q) - _all_faces(prior_by_q)


def _per_q_new_faces(newest_by_q: dict[str, Counter],
                     prior_by_q: dict[str, Counter]) -> dict[str, set[str]]:
    """Per quadrant: faces appearing in Q in newest that did not appear
    in Q in any prior run. A face emitted in Q in newest is "new to Q"
    if no prior run emitted it in Q (it may have appeared in other
    quadrants in prior — that doesn't count). This is the right
    semantic for per-quadrant saturation: we're asking whether Q's
    distribution is still surfacing new modes."""
    out: dict[str, set[str]] = {}
    for q in QUADRANT_ORDER:
        n_q_faces = set(newest_by_q.get(q, Counter()).keys())
        p_q_faces = set(prior_by_q.get(q, Counter()).keys())
        out[q] = n_q_faces - p_q_faces
    return out


def _per_q_js(newest_by_q: dict[str, Counter],
              prior_by_q: dict[str, Counter]) -> dict[str, float]:
    """JS divergence per quadrant. Vocab = union of newest+prior in
    that quadrant; smoothed before computing JS."""
    out: dict[str, float] = {}
    for q in QUADRANT_ORDER:
        n_q = newest_by_q.get(q, Counter())
        p_q = prior_by_q.get(q, Counter())
        vocab = sorted(set(n_q.keys()) | set(p_q.keys()))
        if not vocab:
            out[q] = 0.0
            continue
        p = _normalize(n_q, vocab)
        q_ = _normalize(p_q, vocab)
        out[q] = _js(p, q_)
    return out


def _face_modal_quadrant(rows: list[dict]) -> dict[str, tuple[str, int]]:
    """{face: (modal_quadrant, modal_count)} over the given rows."""
    counts: dict[str, Counter] = {}
    for r in rows:
        q = r.get("quadrant")
        if q not in QUADRANT_ORDER:
            continue
        f = canonicalize_kaomoji(r.get("first_word") or "") or ""
        if not f:
            continue
        counts.setdefault(f, Counter())[q] += 1
    out: dict[str, tuple[str, int]] = {}
    for face, qmap in counts.items():
        modal_q, modal_n = qmap.most_common(1)[0]
        out[face] = (modal_q, modal_n)
    return out


def _modal_agreement(prior_rows: list[dict],
                     all_rows: list[dict],
                     min_emits: int = 3) -> tuple[float, int, int]:
    """For every face emitted ≥``min_emits`` times in ``all_rows``,
    compare the modal quadrant from ``prior_rows`` alone vs the modal
    from ``all_rows``. Returns ``(agree_fraction, n_agree, n_total)``.

    Faces with no emits in ``prior_rows`` count as "no prior modal" —
    they're skipped (since there's nothing to disagree with).
    """
    all_modal = _face_modal_quadrant(all_rows)
    prior_modal = _face_modal_quadrant(prior_rows)

    # Emit-counts in the all-runs set (for ≥min_emits filter).
    all_counts: Counter = Counter()
    for r in all_rows:
        f = canonicalize_kaomoji(r.get("first_word") or "") or ""
        if f:
            all_counts[f] += 1

    n_agree = 0
    n_total = 0
    for face, n in all_counts.items():
        if n < min_emits:
            continue
        if face not in prior_modal:
            # Face is brand-new in newest run; no prior modal to compare.
            continue
        n_total += 1
        if all_modal[face][0] == prior_modal[face][0]:
            n_agree += 1
    if n_total == 0:
        return (1.0, 0, 0)
    return (n_agree / n_total, n_agree, n_total)


# ---------------------------------------------------------------------------
# Hard-fail diagnostics.
# ---------------------------------------------------------------------------


def _hard_fail_diagnostics(rows: list[dict]) -> dict[str, float]:
    """Returns {metric: value}. See module-level thresholds for cutoffs."""
    n = len(rows)
    if n == 0:
        return dict(frame_break_rate=0.0, emit_rate=0.0, output_len_median=0.0)
    n_frame_break = 0
    n_emit = 0
    lens: list[int] = []
    for r in rows:
        text = r.get("response_text", "") or ""
        first_word = r.get("first_word", "") or ""
        n_chars = r.get("n_response_chars", len(text))
        lens.append(n_chars)
        if first_word:
            n_emit += 1
        for pat in _FRAME_BREAK_PATTERNS:
            if pat.search(text):
                n_frame_break += 1
                break
    lens.sort()
    median = lens[n // 2] if n else 0
    return dict(
        frame_break_rate=n_frame_break / n,
        emit_rate=n_emit / n,
        output_len_median=float(median),
    )


# ---------------------------------------------------------------------------
# Calibrate mode: split-half on run-0.
# ---------------------------------------------------------------------------


def _split_halves(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Even-vs-odd split per quadrant. Within each quadrant, sort by
    prompt_id (stable string order) and split into evens / odds.
    Less likely to have systematic differences than first-half /
    last-half (e.g. block-A-then-B-then-C ordering)."""
    by_q: dict[str, list[dict]] = {q: [] for q in QUADRANT_ORDER}
    for r in rows:
        q = r.get("quadrant")
        if q in by_q:
            by_q[q].append(r)
    even: list[dict] = []
    odd: list[dict] = []
    for q in QUADRANT_ORDER:
        rows_q = sorted(by_q[q], key=lambda r: r.get("prompt_id", ""))
        for i, r in enumerate(rows_q):
            (even if i % 2 == 0 else odd).append(r)
    return (even, odd)


def _calibrate(claude_runs_dir: Path) -> int:
    runs = find_run_files(claude_runs_dir)
    if not runs:
        print(f"no runs in {claude_runs_dir}/; nothing to calibrate")
        return 1
    run0_idx, run0_path = runs[0]
    if run0_idx != 0:
        print(f"WARNING: lowest run-index is {run0_idx}, not 0. "
              f"Calibrating against that anyway.")
    rows = load_run_rows(run0_path)
    print(f"calibrating against {run0_path.name} ({len(rows)} rows)\n")
    a, b = _split_halves(rows)
    print(f"  half-A (even prompt index per quadrant): {len(a)} rows")
    print(f"  half-B (odd prompt index per quadrant):  {len(b)} rows\n")

    # Run the saturation metrics in both directions and average — removes
    # the "which half is newest" asymmetry. Gives an empirical estimate
    # of "what does adding 60 more gens to a 60-gen base look like" —
    # which is a tighter sample than the run-vs-run scenario but still
    # informative as a noise floor.

    def one_dir(prior: list[dict], newest: list[dict]) -> dict:
        prior_by_q = _per_q_face_counts(prior)
        newest_by_q = _per_q_face_counts(newest)
        new_faces = _new_faces(newest_by_q, prior_by_q)
        per_q_js = _per_q_js(newest_by_q, prior_by_q)
        agree_frac, agree_n, agree_total = _modal_agreement(
            prior, prior + newest
        )
        return dict(
            new_face_count=len(new_faces),
            new_face_examples=sorted(new_faces)[:10],
            per_q_js=per_q_js,
            mean_js=sum(per_q_js.values()) / len(per_q_js),
            modal_agreement=agree_frac,
            modal_agreement_count=agree_n,
            modal_agreement_total=agree_total,
        )

    ab = one_dir(a, b)
    ba = one_dir(b, a)

    print("split-half saturation metrics (avg of A→B, B→A):\n")
    print(f"  new-face count            avg = {(ab['new_face_count'] + ba['new_face_count']) / 2:.1f}  "
          f"(A→B={ab['new_face_count']}, B→A={ba['new_face_count']})")
    print(f"  mean per-quadrant JS      avg = "
          f"{(ab['mean_js'] + ba['mean_js']) / 2:.4f}  "
          f"(A→B={ab['mean_js']:.4f}, B→A={ba['mean_js']:.4f})")
    print(f"  modal-quadrant agreement  avg = "
          f"{(ab['modal_agreement'] + ba['modal_agreement']) / 2:.3f}  "
          f"(A→B={ab['modal_agreement']:.3f} on n={ab['modal_agreement_total']}, "
          f"B→A={ba['modal_agreement']:.3f} on n={ba['modal_agreement_total']})")
    print()
    print("  per-quadrant JS (A→B):")
    for q in QUADRANT_ORDER:
        print(f"    {q:<5}  {ab['per_q_js'][q]:.4f}")
    print()
    print(f"new faces introduced going A→B (≤10 shown): {ab['new_face_examples']}")
    print()
    print("=== suggested thresholds (vs configured) ===")
    avg_js = (ab["mean_js"] + ba["mean_js"]) / 2
    avg_new = (ab["new_face_count"] + ba["new_face_count"]) / 2
    avg_agree = (ab["modal_agreement"] + ba["modal_agreement"]) / 2
    print(f"  NEW_FACE_MAX       : configured={NEW_FACE_MAX}, "
          f"intra-pilot half = {avg_new:.1f}")
    print(f"  JS_MAX             : configured={JS_MAX}, "
          f"intra-pilot half = {avg_js:.4f} (suggest ~{avg_js / 2:.4f})")
    print(f"  MODAL_AGREE_MIN    : configured={MODAL_AGREE_MIN}, "
          f"intra-pilot half = {avg_agree:.3f}")

    # Hard-fail diagnostics on the full pilot to record the baseline.
    print()
    print("=== hard-fail baseline (run-0, full) ===")
    diag = _hard_fail_diagnostics(rows)
    print(f"  frame_break_rate   = {diag['frame_break_rate']:.4f}  "
          f"(threshold ≤ {FRAME_BREAK_MAX})")
    print(f"  emit_rate          = {diag['emit_rate']:.4f}  "
          f"(threshold ≥ {EMIT_RATE_MIN})")
    print(f"  output_len_median  = {diag['output_len_median']:.1f}  "
          f"(threshold ≥ {OUTPUT_LEN_MIN_MEDIAN})")
    return 0


# ---------------------------------------------------------------------------
# Compare mode: newest run vs pooled prior.
# ---------------------------------------------------------------------------


def _quadrants_with_emits(by_q: dict[str, Counter]) -> list[str]:
    """Quadrants with ≥1 emitted face in the given counts dict."""
    return [q for q in QUADRANT_ORDER if sum(by_q.get(q, Counter()).values()) > 0]


def _compare(claude_runs_dir: Path) -> int:
    runs = find_run_files(claude_runs_dir)
    if not runs:
        print(f"no runs in {claude_runs_dir}/; nothing to compare")
        return 1
    if len(runs) < 2:
        print(f"only run-{runs[0][0]} on disk; need ≥2 runs to compare. "
              f"verdict: CONTINUE (run another and re-check).")
        return 2

    newest_idx, newest_path = runs[-1]
    prior_paths = [(idx, p) for idx, p in runs[:-1]]
    print(f"newest: run-{newest_idx} ({newest_path.name})")
    print(f"prior:  {[f'run-{i}' for i, _ in prior_paths]} "
          f"({len(prior_paths)} runs)\n")

    newest_rows = load_run_rows(newest_path)
    prior_rows: list[dict] = []
    for _, p in prior_paths:
        prior_rows.extend(load_run_rows(p))
    print(f"  newest rows: {len(newest_rows)}")
    print(f"  prior rows:  {len(prior_rows)}")
    print(f"  total rows:  {len(newest_rows) + len(prior_rows)}\n")

    newest_by_q = _per_q_face_counts(newest_rows)
    prior_by_q = _per_q_face_counts(prior_rows)

    # ---- per-quadrant saturation analysis ----
    # Quadrants present in newest (have ≥1 emit). Quadrants absent from
    # newest are already-dropped (saturated in some earlier comparison).
    active_in_newest = _quadrants_with_emits(newest_by_q)
    already_dropped = [q for q in QUADRANT_ORDER if q not in active_in_newest]

    per_q_new = _per_q_new_faces(newest_by_q, prior_by_q)
    per_q_js = _per_q_js(newest_by_q, prior_by_q)

    print("=== per-quadrant saturation ===")
    print(f"  {'Q':<5} {'new':>4} {'JS':>7}  verdict")
    quadrant_verdicts: dict[str, str] = {}
    for q in QUADRANT_ORDER:
        if q in already_dropped:
            quadrant_verdicts[q] = "dropped (prior)"
            print(f"  {q:<5} {'-':>4} {'-':>7}  dropped — saturated in earlier round")
            continue
        n_new = len(per_q_new[q])
        js_q = per_q_js[q]
        sat = (n_new <= PER_Q_NEW_FACE_MAX) and (js_q <= PER_Q_JS_MAX)
        quadrant_verdicts[q] = "saturated" if sat else "active"
        tag = "SATURATED → drop" if sat else "active — keep running"
        print(f"  {q:<5} {n_new:>4} {js_q:>7.4f}  {tag}")
        if per_q_new[q]:
            sample = sorted(per_q_new[q])[:5]
            print(f"          new faces in {q} (≤5 shown): {sample}")
    print(
        f"  thresholds: PER_Q_NEW_FACE_MAX ≤ {PER_Q_NEW_FACE_MAX}, "
        f"PER_Q_JS_MAX ≤ {PER_Q_JS_MAX:.4f}\n"
    )

    # ---- global saturation (informational under research-value framing) ----
    new_faces = _new_faces(newest_by_q, prior_by_q)
    mean_js = sum(per_q_js[q] for q in active_in_newest) / max(
        len(active_in_newest), 1
    )
    agree_frac, agree_n, agree_total = _modal_agreement(
        prior_rows, prior_rows + newest_rows
    )

    print("=== global metrics (informational) ===")
    print(f"  new-face count             = {len(new_faces)}  "
          f"(threshold ≤ {NEW_FACE_MAX})")
    if new_faces:
        print(f"    new faces (≤10 shown)    : {sorted(new_faces)[:10]}")
    print(f"  mean JS over active quads  = {mean_js:.4f}  "
          f"(threshold ≤ {JS_MAX})")
    print(f"  modal-quadrant agreement   = {agree_frac:.3f}  "
          f"({agree_n}/{agree_total} faces)  "
          f"(threshold ≥ {MODAL_AGREE_MIN})\n")

    # ---- hard-fail diagnostics on newest run ----
    print("=== hard-fail diagnostics (newest run) ===")
    diag = _hard_fail_diagnostics(newest_rows)
    fb_fail = diag["frame_break_rate"] > FRAME_BREAK_MAX
    em_fail = diag["emit_rate"] < EMIT_RATE_MIN
    ol_fail = diag["output_len_median"] < OUTPUT_LEN_MIN_MEDIAN
    print(f"  frame_break_rate   = {diag['frame_break_rate']:.4f}  "
          f"(threshold ≤ {FRAME_BREAK_MAX})  "
          f"{'FAIL' if fb_fail else 'ok'}")
    print(f"  emit_rate          = {diag['emit_rate']:.4f}  "
          f"(threshold ≥ {EMIT_RATE_MIN})  "
          f"{'FAIL' if em_fail else 'ok'}")
    print(f"  output_len_median  = {diag['output_len_median']:.1f}  "
          f"(threshold ≥ {OUTPUT_LEN_MIN_MEDIAN})  "
          f"{'FAIL' if ol_fail else 'ok'}\n")

    hard_fail = fb_fail or em_fail or ol_fail

    # ---- next-run recommendation ----
    next_run_quadrants = [
        q for q in QUADRANT_ORDER
        if quadrant_verdicts[q] == "active"
    ]
    all_saturated = len(next_run_quadrants) == 0
    cap_hit = newest_idx >= RUN_CAP

    print("=== verdict ===")
    if hard_fail:
        print("  ABORT — hard-fail diagnostic exceeded threshold.")
        print(f"  Investigate before running run-{newest_idx + 1}.")
        return 1
    if all_saturated:
        print(f"  STOP — every quadrant saturated by run-{newest_idx}.")
        print("  GT corpus is sufficient; no need to run further.")
        return 0
    if cap_hit:
        print(f"  STOP — run-cap hit (run-{newest_idx} = RUN_CAP={RUN_CAP}).")
        print(f"  Active quadrants at cap: {next_run_quadrants}")
        print("  Further runs require an explicit amendment to the design doc.")
        return 0
    newly_saturated = [
        q for q in active_in_newest
        if quadrant_verdicts[q] == "saturated"
    ]
    if newly_saturated:
        print(f"  CONTINUE — {len(newly_saturated)} quadrant(s) newly "
              f"saturated this round: {newly_saturated}")
    else:
        print("  CONTINUE — no quadrants saturated this round.")
    print(f"  next run: --run-index {newest_idx + 1} "
          f"--quadrants {','.join(next_run_quadrants)}")
    return 2


# ---------------------------------------------------------------------------
# Cross-arm mode: pool naturalistic + introspection arms, per-Q diff.
# ---------------------------------------------------------------------------


def _cross_arm(
    naturalistic_dir: Path = CLAUDE_RUNS_DIR,
    introspection_dir: Path = CLAUDE_RUNS_INTROSPECTION_DIR,
) -> int:
    nat_runs = find_run_files(naturalistic_dir)
    intro_runs = find_run_files(introspection_dir)
    if not nat_runs:
        print(f"no naturalistic runs in {naturalistic_dir}/; "
              f"can't compare arms.")
        return 1
    if not intro_runs:
        print(f"no introspection runs in {introspection_dir}/; "
              f"can't compare arms.")
        return 1

    nat_rows: list[dict] = []
    for _, p in nat_runs:
        nat_rows.extend(load_run_rows(p))
    intro_rows: list[dict] = []
    for _, p in intro_runs:
        intro_rows.extend(load_run_rows(p))

    print(f"naturalistic arm: {[f'run-{i}' for i, _ in nat_runs]}  "
          f"({len(nat_rows)} rows pooled)")
    print(f"introspection arm: {[f'run-{i}' for i, _ in intro_runs]}  "
          f"({len(intro_rows)} rows pooled)\n")

    nat_by_q = _per_q_face_counts(nat_rows)
    intro_by_q = _per_q_face_counts(intro_rows)

    # Per-quadrant analysis: JS, faces unique to each arm in Q.
    per_q: dict[str, dict] = {}
    for q in QUADRANT_ORDER:
        n_q = nat_by_q.get(q, Counter())
        i_q = intro_by_q.get(q, Counter())
        vocab = sorted(set(n_q.keys()) | set(i_q.keys()))
        if not vocab:
            per_q[q] = dict(
                js=0.0,
                nat_only=set(),
                intro_only=set(),
                shared=set(),
                n_nat=0,
                n_intro=0,
                distinguishable=False,
            )
            continue
        p = _normalize(n_q, vocab)
        i = _normalize(i_q, vocab)
        js_q = _js(p, i)
        n_only = set(n_q.keys()) - set(i_q.keys())
        i_only = set(i_q.keys()) - set(n_q.keys())
        shared = set(n_q.keys()) & set(i_q.keys())
        per_q[q] = dict(
            js=js_q,
            nat_only=n_only,
            intro_only=i_only,
            shared=shared,
            n_nat=sum(n_q.values()),
            n_intro=sum(i_q.values()),
            distinguishable=js_q > PER_Q_JS_MAX,
        )

    # Console + markdown output.
    print("=== per-quadrant cross-arm comparison ===")
    print(f"  {'Q':<5} {'n_nat':>6} {'n_intr':>7} {'JS':>7} "
          f"{'shared':>7} {'nat_only':>9} {'intr_only':>9}  verdict")
    for q in QUADRANT_ORDER:
        d = per_q[q]
        verdict = (
            "DISTINGUISHABLE" if d["distinguishable"] else "indistinguishable"
        )
        print(f"  {q:<5} {d['n_nat']:>6} {d['n_intro']:>7} "
              f"{d['js']:>7.4f} {len(d['shared']):>7} "
              f"{len(d['nat_only']):>9} {len(d['intro_only']):>9}  "
              f"{verdict}")
    print(f"\n  threshold: per-Q JS > {PER_Q_JS_MAX:.4f} → DISTINGUISHABLE\n")

    # Markdown summary.
    print("--- markdown for writeup ---")
    print()
    print("| Q | n (naturalistic) | n (introspection) | JS (nats) | "
          "shared faces | naturalistic-only | introspection-only | "
          "verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for q in QUADRANT_ORDER:
        d = per_q[q]
        verdict = (
            "**distinguishable**" if d["distinguishable"]
            else "indistinguishable"
        )
        print(f"| {q} | {d['n_nat']} | {d['n_intro']} | {d['js']:.4f} | "
              f"{len(d['shared'])} | {len(d['nat_only'])} | "
              f"{len(d['intro_only'])} | {verdict} |")
    print()

    # Sample faces unique to each arm (most informative for inspection).
    print("--- introspection-only faces (≤10 per quadrant) ---")
    for q in QUADRANT_ORDER:
        d = per_q[q]
        if d["intro_only"]:
            print(f"  {q:<5} {sorted(d['intro_only'])[:10]}")

    # Headline.
    n_distinguishable = sum(
        1 for q in QUADRANT_ORDER if per_q[q]["distinguishable"]
    )
    print()
    print("=== headline ===")
    if n_distinguishable == 0:
        print(
            "  INDISTINGUISHABLE — introspection preamble does not move "
            "Claude's per-quadrant face distribution above the saturation "
            "threshold in any quadrant. Either the pool is saturated or "
            "the preamble has no measurable effect."
        )
        return 0
    print(
        f"  DISTINGUISHABLE in {n_distinguishable}/6 quadrants. "
        f"Per the decision tree (appendix step 2 → step 3): run more "
        f"naturalistic runs to saturate the naturalistic arm, then "
        f"re-compare. Persistence of the gap at naturalistic-saturation "
        f"is the genuine introspection effect."
    )
    return 2


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate thresholds against run-0 split-halves instead of "
             "comparing runs.",
    )
    parser.add_argument(
        "--cross-arm", action="store_true",
        help="Compare pooled naturalistic vs pooled introspection arms "
             "per quadrant. Informational only — does not gate runs. "
             "Emits a Markdown table for the writeup.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=CLAUDE_RUNS_DIR,
        help=f"Directory containing run-N.jsonl files. "
             f"Default: {CLAUDE_RUNS_DIR}. "
             f"For --cross-arm this is the naturalistic arm; the "
             f"introspection arm is fixed at "
             f"{CLAUDE_RUNS_INTROSPECTION_DIR.name}.",
    )
    args = parser.parse_args()

    if args.calibrate and args.cross_arm:
        parser.error("--calibrate and --cross-arm are mutually exclusive")

    if args.calibrate:
        sys.exit(_calibrate(args.runs_dir))
    if args.cross_arm:
        sys.exit(_cross_arm(
            naturalistic_dir=args.runs_dir,
            introspection_dir=CLAUDE_RUNS_INTROSPECTION_DIR,
        ))
    sys.exit(_compare(args.runs_dir))


if __name__ == "__main__":
    main()
