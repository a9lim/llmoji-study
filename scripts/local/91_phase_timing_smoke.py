"""Phase-split timing smoke for the v3 emotional-run hot path.

The v3 main loop (`scripts/local/00_emit.py`) shows ~2.7s of
constant per-row overhead on top of generation itself. This script runs
~20 v3-style generations and times each phase separately so the user
can see how that overhead splits across:

  1. session.generate(...)             — generation
  2. read_after_generate(...)          — hidden-state readout from
                                         saklas's per-layer buckets
  3. save_hidden_states(...)           — sidecar .npz write
  4. JSONL serialize + write + flush   — disk append

Inlines `run_sample`'s logic verbatim (chat-message build, sampling
config, post-generation probe-score readback) but inserts perf_counter
boundaries between phases, with a torch.cuda.synchronize() /
torch.mps.synchronize() in between to drain async GPU work before each
clock read. This is the only place phase timing lives — `capture.py` is
not modified.

Output: per-row stdout, end-of-run summary, and a TSV at
<out>/phase_timing.tsv. Pollution-free: writes sidecars under
data/local/phase_timing_smoke/<short>/local/hidden/<experiment>_phase_timing/, not
the production v3 dirs.

Usage:
    python scripts/local/91_phase_timing_smoke.py [--n N]
        [--store-full-trace | --no-store-full-trace]
        [--out DIR] [--seed S]

`LLMOJI_MODEL=qwen|ministral` to switch model the same way 03 does.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# --- module-level: argparse + tiny helpers only. No model imports. ---
# All heavy imports (saklas, llmoji_study.config, etc.) happen in main()
# so the import-test stays cheap.


def _cuda_sync() -> None:
    """Drain GPU work before reading perf_counter on a phase boundary.

    No-op on CPU. Calls torch.cuda.synchronize() if a CUDA device is
    available, torch.mps.synchronize() on Apple silicon. Imported lazily
    so the module imports without torch."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if sync is not None:
            sync()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-split timing for the v3 hot path.",
    )
    p.add_argument("--n", type=int, default=20,
                   help="number of generations (capped at len(EMOTIONAL_PROMPTS))")
    p.add_argument("--seed", type=int, default=0,
                   help="single seed used for every generation")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: data/local/phase_timing_smoke/<short>/)")
    full_trace = p.add_mutually_exclusive_group()
    full_trace.add_argument(
        "--store-full-trace", dest="store_full_trace",
        action="store_true",
        help="include the (n_tokens, hidden_dim) per-layer trace in each "
             "sidecar (matches pre-2026-05-02 default)",
    )
    full_trace.add_argument(
        "--no-store-full-trace", dest="store_full_trace",
        action="store_false",
        help="aggregates only (h_first/h_last/h_mean) — current production "
             "default; ~60x smaller sidecars",
    )
    p.set_defaults(store_full_trace=False)
    return p


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interp percentile for q in [0, 100]. Empty -> nan."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "mean": float("nan"),
                "total": float("nan")}
    return {
        "median": statistics.median(xs),
        "p10": _percentile(xs, 10),
        "p90": _percentile(xs, 90),
        "mean": statistics.fmean(xs),
        "total": sum(xs),
    }


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Heavy imports happen here so the module can be imported without
    # touching torch / saklas / loading a model.
    from saklas import SaklasSession, SamplingConfig

    from llmoji_study.config import (
        DATA_DIR,
        EMOTIONAL_CONDITION,
        KAOMOJI_INSTRUCTION,
        MAX_NEW_TOKENS,
        PROBE_CATEGORIES,
        PROBES,
        STEERED_AXIS,
        TEMPERATURE,
        current_model,
    )
    from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
    from llmoji_study.hidden_capture import read_after_generate
    from llmoji_study.hidden_state_io import (
        hidden_state_path,
        save_hidden_states,
    )
    from llmoji.taxonomy import extract

    M = current_model()

    n = max(1, min(args.n, len(EMOTIONAL_PROMPTS)))
    if n != args.n:
        print(f"WARNING: --n={args.n} clamped to {n} (len(EMOTIONAL_PROMPTS))")
    prompts = EMOTIONAL_PROMPTS[:n]

    out_dir = args.out or (DATA_DIR / "local" / "phase_timing_smoke" / M.short_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment = f"{M.experiment}_phase_timing"
    jsonl_path = out_dir / "phase_timing_rows.jsonl"
    tsv_path = out_dir / "phase_timing.tsv"

    print(f"model:     {M.short_name} ({M.model_id})")
    print(f"out dir:   {out_dir}")
    print(f"sidecars:  {out_dir / 'hidden' / experiment}/")
    print(f"jsonl:     {jsonl_path}")
    print(f"n:         {n}  seed={args.seed}  "
          f"store_full_trace={args.store_full_trace}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}  temperature: {TEMPERATURE}")

    print(f"\nloading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=PROBE_CATEGORIES,
    ) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; running {n} generations")

        per_row: list[dict[str, Any]] = []

        # Truncate the JSONL on each run — this is a smoke, not a
        # production runner with resume semantics.
        with jsonl_path.open("w") as out:
            for i, ep in enumerate(prompts, start=1):
                seed = args.seed
                content = KAOMOJI_INSTRUCTION + ep.text
                messages = [{"role": "user", "content": content}]
                sampling = SamplingConfig(
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    seed=seed,
                )

                row_uuid = uuid.uuid4().hex
                sidecar_path = hidden_state_path(out_dir, experiment, row_uuid)

                # --- PHASE 0: total-row clock starts here ---
                _cuda_sync()
                t_total_start = perf_counter()

                # --- PHASE 1: generation ---
                t0 = perf_counter()
                result = session.generate(
                    messages,
                    steering=None,
                    sampling=sampling,
                    thinking=False,
                    stateless=True,
                )
                _cuda_sync()
                t_gen = perf_counter() - t0

                # --- PHASE 2: hidden-state capture readout ---
                # Must run before any subsequent generate() — the
                # session._capture._per_layer buckets get cleared on the
                # next generate.
                t0 = perf_counter()
                capture = read_after_generate(
                    session, store_full_trace=args.store_full_trace,
                )
                _cuda_sync()
                t_capture = perf_counter() - t0

                # --- PHASE 3: sidecar npz write ---
                t0 = perf_counter()
                save_hidden_states(
                    capture, sidecar_path,
                    store_full_trace=args.store_full_trace,
                )
                t_npz = perf_counter() - t0

                # Pull probe scores + first_word the same way run_sample
                # does, so the JSONL row carries enough for a sanity
                # check. Not timed — this is bookkeeping for the row,
                # not part of the v3 hot-path overhead we're measuring.
                match = extract(result.text)
                per_token_scores: dict[str, list[float]] = (
                    getattr(session, "last_per_token_scores", None) or {}
                )
                probe_scores_t0: list[float] = []
                probe_scores_tlast: list[float] = []
                for probe in PROBES:
                    seq = per_token_scores.get(probe) or []
                    if seq:
                        probe_scores_t0.append(float(seq[0]))
                        probe_scores_tlast.append(float(seq[-1]))
                    else:
                        readings = result.readings.get(probe)
                        if readings is None or not readings.per_generation:
                            probe_scores_t0.append(float("nan"))
                            probe_scores_tlast.append(float("nan"))
                        else:
                            probe_scores_t0.append(
                                float(readings.per_generation[0])
                            )
                            probe_scores_tlast.append(
                                float(readings.per_generation[-1])
                            )

                row = {
                    "condition": EMOTIONAL_CONDITION,
                    "prompt_id": ep.id,
                    "seed": seed,
                    "text": result.text,
                    "first_word": match.first_word,
                    "token_count": result.token_count,
                    "tok_per_sec": result.tok_per_sec,
                    "row_uuid": row_uuid,
                    "probe_scores_t0": probe_scores_t0,
                    "probe_scores_tlast": probe_scores_tlast,
                    "phase_timing": {
                        "gen": t_gen,
                        "capture": t_capture,
                        "npz": t_npz,
                    },
                }

                # --- PHASE 4: JSONL serialize + write + flush ---
                t0 = perf_counter()
                out.write(json.dumps(row) + "\n")
                out.flush()
                t_jsonl = perf_counter() - t0

                t_total = perf_counter() - t_total_start

                # Note: the JSONL row's `phase_timing` field captures
                # gen/capture/npz only (t_jsonl is measured AFTER the
                # write, so it's not in the on-disk row). The in-memory
                # `per_row` list below is authoritative for the summary
                # and includes all four phases.
                tag = match.first_word if match.first_word else "(no-kaomoji)"
                print(
                    f"  [{i}/{n}] {ep.id} {tag}  "
                    f"t_total={t_total:.3f}s  "
                    f"gen={t_gen:.3f}s  "
                    f"capture={t_capture:.3f}s  "
                    f"npz={t_npz:.3f}s  "
                    f"jsonl={t_jsonl:.3f}s  "
                    f"tokens={result.token_count}  "
                    f"({result.tok_per_sec:.1f} tok/s)"
                )

                per_row.append({
                    "prompt_id": ep.id,
                    "tokens": result.token_count,
                    "t_total": t_total,
                    "t_gen": t_gen,
                    "t_capture": t_capture,
                    "t_npz": t_npz,
                    "t_jsonl": t_jsonl,
                })

    # --- Summary -----------------------------------------------------
    totals = [r["t_total"] for r in per_row]
    gens = [r["t_gen"] for r in per_row]
    captures = [r["t_capture"] for r in per_row]
    npzs = [r["t_npz"] for r in per_row]
    jsonls = [r["t_jsonl"] for r in per_row]
    tokens = [r["tokens"] for r in per_row]

    # Per-token gen cost: t_gen / tokens. Skip rows with token_count<=1
    # to avoid dividing by 1 on stop-on-first-token cases. Then the
    # prefill_estimate per row = t_gen - tokens * median_per_token_gen.
    per_tok_gens = [
        r["t_gen"] / r["tokens"] for r in per_row if r["tokens"] >= 2
    ]
    median_per_tok_gen = (
        statistics.median(per_tok_gens) if per_tok_gens else float("nan")
    )
    if math.isnan(median_per_tok_gen):
        prefills: list[float] = []
    else:
        prefills = [
            r["t_gen"] - r["tokens"] * median_per_tok_gen
            for r in per_row
        ]

    rows_for_tsv = [
        ("total", _stats(totals)),
        ("gen", _stats(gens)),
        ("capture", _stats(captures)),
        ("npz", _stats(npzs)),
        ("jsonl", _stats(jsonls)),
        ("gen_per_token", _stats(per_tok_gens)),
        ("prefill_estimate", _stats(prefills)),
    ]

    print("\n" + "=" * 72)
    print(f"PHASE TIMING SUMMARY  (n={len(per_row)} rows; "
          f"store_full_trace={args.store_full_trace})")
    print("=" * 72)
    print(f"{'phase':<20s}  {'median':>10s}  {'p10':>10s}  "
          f"{'p90':>10s}  {'mean':>10s}  {'total':>10s}")
    for name, s in rows_for_tsv:
        print(f"{name:<20s}  {s['median']:>10.4f}  {s['p10']:>10.4f}  "
              f"{s['p90']:>10.4f}  {s['mean']:>10.4f}  {s['total']:>10.4f}")
    print(f"\ntoken counts: median={statistics.median(tokens) if tokens else 0}  "
          f"min={min(tokens) if tokens else 0}  "
          f"max={max(tokens) if tokens else 0}")

    # Constant-overhead headline: median of (capture + npz + jsonl).
    # That's the per-row work that doesn't scale with token count and
    # is what 03's wall-clock-vs-tok/s gap is paying for.
    cap_med = statistics.median(captures) if captures else 0.0
    npz_med = statistics.median(npzs) if npzs else 0.0
    jsonl_med = statistics.median(jsonls) if jsonls else 0.0
    overhead = cap_med + npz_med + jsonl_med
    if overhead > 0:
        npz_pct = 100 * npz_med / overhead
        cap_pct = 100 * cap_med / overhead
        jsonl_pct = 100 * jsonl_med / overhead
    else:
        npz_pct = cap_pct = jsonl_pct = float("nan")
    print(
        f"\nconstant overhead ≈ {overhead:.3f}s median  "
        f"({npz_pct:.0f}% npz, {cap_pct:.0f}% capture, "
        f"{jsonl_pct:.0f}% jsonl)"
    )

    # TSV summary -----------------------------------------------------
    with tsv_path.open("w") as tsv:
        tsv.write("phase\tmedian\tp10\tp90\tmean\ttotal\n")
        for name, s in rows_for_tsv:
            tsv.write(
                f"{name}\t{s['median']:.6f}\t{s['p10']:.6f}\t"
                f"{s['p90']:.6f}\t{s['mean']:.6f}\t{s['total']:.6f}\n"
            )
    print(f"\nwrote summary to {tsv_path}")
    print(f"wrote per-row jsonl to {jsonl_path}")


if __name__ == "__main__":
    main()
