"""Smoke test for the hidden-state refactor.

Generates 5 samples across HP/LP/HN/LN quadrants + one neutral,
captures hidden states via the new sidecar path, then validates:

  (1) saved sidecar files exist and parse back cleanly,
  (2) post-hoc probe computation from h_last reproduces the on-the-fly
      probe_scores_tlast values (aggregate under stateless=True) within
      a reasonable tolerance — this is the probe-round-trip gate,
  (3) the hidden-state matrix has sensible shapes.

Run BEFORE any large re-run. If this fails, the refactor is broken
and re-running 1540 generations would waste time + burn through
model-welfare budget on bad data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from saklas import SaklasSession

from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    MODEL_ID,
    PROBE_CATEGORIES,
    PROBES,
    STEERED_AXIS,
)
from llmoji.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji.hidden_state_io import hidden_state_path, load_hidden_states


SMOKE_EXPERIMENT = "smoke"
SMOKE_DIR = DATA_DIR
PROBE_SCORE_TOL = 5e-3  # fp32 probe agreement tolerance


def _pick_smoke_prompts():
    """Pick one prompt per quadrant from EMOTIONAL_PROMPTS + a fake
    neutral. Keeping smoke small — five generations, not 640."""
    by_quad: dict[str, list] = {"HP": [], "LP": [], "HN": [], "LN": []}
    for p in EMOTIONAL_PROMPTS:
        q = p.id[:2].upper()
        if q in by_quad and not by_quad[q]:
            by_quad[q].append(p)
    return [by_quad["HP"][0], by_quad["LP"][0], by_quad["HN"][0], by_quad["LN"][0]]


def main() -> None:
    print(f"bootstrapping saklas session on {MODEL_ID}...")
    session = SaklasSession.from_pretrained(
        MODEL_ID, probes=PROBE_CATEGORIES,
    )

    # Register the steered profile so expressions would resolve if
    # needed. We run unsteered in the smoke test (kaomoji_prompted
    # condition takes the no-steering branch of run_sample), so no
    # need to enter/exit a steering context.
    name, profile = session.extract(STEERED_AXIS)
    session.steer(name, profile)

    prompts = _pick_smoke_prompts()
    print(f"running {len(prompts)} generations with hidden-state sidecar capture...")

    rows = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] prompt_id={prompt.id}  "
              f"text={prompt.text[:60]!r}...")
        row = run_sample(
            session=session,
            prompt=prompt,
            condition="kaomoji_prompted",
            seed=42 + i,
            hidden_dir=SMOKE_DIR,
            experiment=SMOKE_EXPERIMENT,
            store_full_trace=True,
        )
        print(f"  generated ({row.token_count} tokens): {row.text[:80]!r}")
        print(f"  row_uuid={row.row_uuid}")
        rows.append(row)

    # --- Validation gate ---
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_pass = True
    for i, row in enumerate(rows):
        print(f"\n[row {i + 1}] uuid={row.row_uuid}")
        sidecar = hidden_state_path(SMOKE_DIR, SMOKE_EXPERIMENT, row.row_uuid)

        # (1) file exists + parses
        if not sidecar.exists():
            print(f"  FAIL: sidecar missing at {sidecar}")
            all_pass = False
            continue
        try:
            capture = load_hidden_states(sidecar)
        except Exception as e:
            print(f"  FAIL: could not load sidecar — {e}")
            all_pass = False
            continue
        print(f"  sidecar ok: {sidecar.stat().st_size / 1024:.1f} KB  "
              f"seq_len={capture.seq_len}  prompt_len={capture.prompt_len}  "
              f"layers={sorted(capture.layers.keys())}")

        # (3) shapes
        for idx, lc in capture.layers.items():
            if lc.hidden_states.shape[0] != capture.seq_len:
                print(f"  FAIL: layer {idx} hidden_states seq dim "
                      f"{lc.hidden_states.shape[0]} != seq_len {capture.seq_len}")
                all_pass = False
            if lc.h_first.shape != lc.h_last.shape != lc.h_attn_weighted.shape:
                print(f"  FAIL: layer {idx} aggregate shapes inconsistent")
                all_pass = False

        # (2) Probe round-trip. On-the-fly `probe_means` is the
        # whole-generation aggregate: mean over per-token probe
        # scores. For linear probes this equals probe · mean(hidden),
        # so we reproduce by mean-pooling the per-token hidden states
        # at each probe layer and running saklas's own scorer on the
        # resulting single-vector dict.
        try:
            import torch
            mean_hidden = {
                idx: torch.from_numpy(lc.hidden_states.mean(axis=0))
                for idx, lc in capture.layers.items()
            }
            recomputed = session._monitor.score_single_token(mean_hidden)
            recomputed_aggregate = {str(k): float(v) for k, v in recomputed.items()}
        except Exception as e:
            print(f"  FAIL: probe recomputation raised — {e}")
            all_pass = False
            continue

        # Compare to row.probe_means (saklas's on-the-fly aggregate).
        probe_mismatches = []
        for probe in PROBES:
            on_the_fly = row.probe_means.get(probe, float("nan"))
            posthoc = recomputed_aggregate.get(probe, float("nan"))
            if np.isnan(on_the_fly) or np.isnan(posthoc):
                probe_mismatches.append(f"{probe}: NaN in {on_the_fly}/{posthoc}")
                continue
            diff = abs(on_the_fly - posthoc)
            tag = "OK" if diff < PROBE_SCORE_TOL else "MISMATCH"
            print(f"  {tag:9s} probe={probe:<22s}  "
                  f"on-the-fly={on_the_fly:+.6f}  posthoc={posthoc:+.6f}  "
                  f"|diff|={diff:.2e}")
            if diff >= PROBE_SCORE_TOL:
                probe_mismatches.append(f"{probe}: |diff|={diff:.2e}")

        if probe_mismatches:
            print(f"  FAIL on probe round-trip: {probe_mismatches}")
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("SMOKE TEST PASSED — refactor is ready for a pilot re-run")
    else:
        print("SMOKE TEST FAILED — fix before running any pilot")
        sys.exit(1)


if __name__ == "__main__":
    main()
