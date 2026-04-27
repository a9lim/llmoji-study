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

from llmoji_study.capture import run_sample
from llmoji_study.config import (
    DATA_DIR,
    MODEL_ID,
    PROBE_CATEGORIES,
    PROBES,
    STEERED_AXIS,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.hidden_state_io import hidden_state_path, load_hidden_states


SMOKE_EXPERIMENT = "smoke"
SMOKE_DIR = DATA_DIR
PROBE_SCORE_TOL = 5e-3  # fp32 probe agreement tolerance


def _pick_smoke_prompts():
    """Pick one prompt per quadrant from EMOTIONAL_PROMPTS — HP, LP,
    HN, LN, plus NB. Keeping smoke small: 5 generations, not 1000."""
    by_quad: dict[str, list] = {"HP": [], "LP": [], "HN": [], "LN": [], "NB": []}
    for p in EMOTIONAL_PROMPTS:
        q = p.id[:2].upper()
        if q in by_quad and not by_quad[q]:
            by_quad[q].append(p)
    return [by_quad[q][0] for q in ("HP", "LP", "HN", "LN", "NB")]


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
              f"n_tokens={capture.n_tokens}  "
              f"layers={sorted(capture.layers.keys())}")

        # (3) shapes
        for idx, lc in capture.layers.items():
            if lc.hidden_states.shape[0] != capture.n_tokens:
                print(f"  FAIL: layer {idx} hidden_states seq dim "
                      f"{lc.hidden_states.shape[0]} != n_tokens {capture.n_tokens}")
                all_pass = False
            if not (lc.h_first.shape == lc.h_last.shape == lc.h_mean.shape):
                print(f"  FAIL: layer {idx} aggregate shapes inconsistent: "
                      f"first={lc.h_first.shape}, last={lc.h_last.shape}, "
                      f"mean={lc.h_mean.shape}")
                all_pass = False

        # (2) Probe round-trip. Feed h_first and h_last through
        # saklas's own scorer; they should match probe_scores_t0 /
        # probe_scores_tlast to fp32 tolerance since the math is
        # identical (center + L2-normalize per layer, weighted-average
        # over probe layers) whether you hand in one token or a trace.
        try:
            import torch
            first_hidden = {
                idx: torch.from_numpy(lc.h_first)
                for idx, lc in capture.layers.items()
            }
            last_hidden = {
                idx: torch.from_numpy(lc.h_last)
                for idx, lc in capture.layers.items()
            }
            recomputed_first = {
                str(k): float(v) for k, v in
                session._monitor.score_single_token(first_hidden).items()
            }
            recomputed_last = {
                str(k): float(v) for k, v in
                session._monitor.score_single_token(last_hidden).items()
            }
        except Exception as e:
            print(f"  FAIL: probe recomputation raised — {e}")
            all_pass = False
            continue

        # Compare per-probe against row.probe_scores_t0 / tlast.
        probe_mismatches = []
        for label, on_the_fly_vec, posthoc in (
            ("t0", row.probe_scores_t0, recomputed_first),
            ("tlast", row.probe_scores_tlast, recomputed_last),
        ):
            for i, probe in enumerate(PROBES):
                otf = on_the_fly_vec[i] if i < len(on_the_fly_vec) else float("nan")
                ph = posthoc.get(probe, float("nan"))
                if np.isnan(otf) or np.isnan(ph):
                    probe_mismatches.append(f"{probe}[{label}]: NaN")
                    continue
                diff = abs(otf - ph)
                tag = "OK" if diff < PROBE_SCORE_TOL else "MISMATCH"
                print(f"  {tag:9s} {label:5s} probe={probe:<22s}  "
                      f"on-the-fly={otf:+.6f}  posthoc={ph:+.6f}  "
                      f"|diff|={diff:.2e}")
                if diff >= PROBE_SCORE_TOL:
                    probe_mismatches.append(f"{probe}[{label}]: |diff|={diff:.2e}")

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
