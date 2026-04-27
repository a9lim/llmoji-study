"""Pilot experiment: 6 arms × 30 prompts × 5 seeds = 900 generations.

Output streamed to data/pilot_raw.jsonl. Per-row hidden-state sidecars
under data/hidden/v1v2/<uuid>.npz. Resumable: if the JSONL exists and
has rows, already-completed (condition, prompt_id, seed) cells are
skipped. Sidecars for skipped rows are NOT re-written.

Arms:
  - baseline           : no kaomoji instruction, no steering
  - kaomoji_prompted   : kaomoji instruction, no steering
  - steered_happy      : kaomoji instruction + "0.5 happy"  steering
  - steered_sad        : kaomoji instruction + "0.5 sad"    steering
  - steered_angry      : kaomoji instruction + "0.5 angry"  steering
  - steered_calm       : kaomoji instruction + "0.5 calm"   steering

The baseline arm exists to measure probe readings in the absence of the
instruction, so we can see how much the instruction itself perturbs
activation state. It does NOT factor into the main decision rules
(those compare the kaomoji-prompted arms to each other).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.capture import run_sample
from llmoji_study.config import (
    CONDITIONS,
    DATA_DIR,
    MODEL_ID,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
    PROBE_CATEGORIES,
    SEEDS_PER_CELL,
    STEERED_AXES,
)
from llmoji_study.prompts import PROMPTS


def _already_done(path: Path) -> set[tuple[str, str, int]]:
    """Return the set of (condition, prompt_id, seed) triples with
    successful rows already in the JSONL output, so a re-run can resume
    and retry previously-errored cells.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Error rows shouldn't count as done — they'll be retried.
            if "error" in r:
                continue
            done.add((r["condition"], r["prompt_id"], int(r["seed"])))
    return done


def _drop_error_rows(path: Path) -> int:
    """Rewrite the JSONL file with error rows stripped, so a resume
    retries them. Returns number of rows dropped."""
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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(PILOT_RAW_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(PILOT_RAW_PATH)
    total = len(CONDITIONS) * len(PROMPTS) * SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {MODEL_ID} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=PROBE_CATEGORIES) as session:
        # Probes bootstrapped via `probes=` attach to the monitor but
        # aren't auto-promoted to the steering registry; explicitly
        # register every axis we intend to steer so pole-alias
        # expressions ("0.5 happy", "0.5 angry", ...) resolve against
        # live profiles. Each extract() hits the cached per-model
        # tensor rather than re-running the extraction pipeline.
        print(
            f"loaded in {time.time() - t_load:.1f}s; "
            f"registering {len(STEERED_AXES)} axes for steering ..."
        )
        for axis in STEERED_AXES:
            name, profile = session.extract(axis)
            session.steer(name, profile)
            print(f"  registered '{name}'")
        print("beginning pilot run")
        with PILOT_RAW_PATH.open("a") as out:
            i = 0
            for condition in CONDITIONS:
                for prompt in PROMPTS:
                    for seed in range(SEEDS_PER_CELL):
                        key = (condition, prompt.id, seed)
                        if key in done:
                            continue
                        i += 1
                        t0 = time.time()
                        try:
                            row = run_sample(
                                session,
                                prompt=prompt,
                                condition=condition,
                                seed=seed,
                                hidden_dir=DATA_DIR,
                                experiment=PILOT_EXPERIMENT,
                            )
                        except Exception as e:
                            # persist the failure so we don't infinite-loop
                            # on a bad cell; also so we can audit later.
                            err_row = {
                                "condition": condition,
                                "prompt_id": prompt.id,
                                "seed": seed,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row) + "\n")
                            out.flush()
                            print(f"  [{i}/{remaining}] {condition} {prompt.id} s={seed} ERR {e}")
                            continue
                        out.write(json.dumps(row.to_dict()) + "\n")
                        out.flush()
                        dt = time.time() - t0
                        tag = row.kaomoji if row.kaomoji else f"[{row.first_word!r}]"
                        print(
                            f"  [{i}/{remaining}] {condition:17s} {prompt.id} "
                            f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                        )
    print(f"\ndone. wrote rows to {PILOT_RAW_PATH}")


if __name__ == "__main__":
    main()
