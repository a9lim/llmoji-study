"""Emotional-battery run: 1 arm × 100 prompts × 8 seeds = 800 generations.

Single unsteered `kaomoji_prompted` arm, Russell-quadrant prompts plus
the NB (neutral-baseline) quadrant. Output streamed to
data/emotional_raw.jsonl; per-row hidden-state sidecars under
data/hidden/v3/<uuid>.npz. Resumable: re-running skips already-
completed (prompt_id, seed) pairs and retries error rows.

Mirrors scripts/01_pilot_run.py structurally — same session setup,
same resume-on-rerun semantics. Does not register steering profiles
(unsteered only). Logs per-quadrant kaomoji-emission rate every 80
completed rows so the user can bail early if emission falls below ~50%.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.capture import run_sample
from llmoji_study.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_SEEDS_PER_CELL as _DEFAULT_SEEDS_PER_CELL,
    PROBE_CATEGORIES,
    current_model,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.prompts import Prompt


# Pilot-N gate. Pre-registered main-run N=8 lives in config.py; pilots
# override here without touching the registered constant. See
# docs/2026-04-30-v3-ministral-pilot.md for the active design that uses this.
def _seeds_per_cell() -> int:
    raw = os.environ.get("LLMOJI_PILOT_GENS")
    if raw is None:
        return _DEFAULT_SEEDS_PER_CELL
    try:
        n = int(raw)
    except ValueError as e:
        raise SystemExit(f"LLMOJI_PILOT_GENS must be int, got {raw!r}") from e
    if n < 1:
        raise SystemExit(f"LLMOJI_PILOT_GENS must be >= 1, got {n}")
    return n


EMOTIONAL_SEEDS_PER_CELL = _seeds_per_cell()


def _already_done(path: Path) -> set[tuple[str, int]]:
    """(prompt_id, seed) pairs with successful rows already in the output.
    Error rows are NOT counted as done — they'll be retried after
    _drop_error_rows strips them."""
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["prompt_id"], int(r["seed"])))
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


def _emission_rate_by_quadrant(path: Path) -> dict[str, tuple[int, int]]:
    """Return {quadrant: (kaomoji-bearing rows, total rows)} from the
    JSONL. Uses prompt_id prefix to infer quadrant (hp/lp/hn/ln/nb)."""
    stats: dict[str, list[int]] = {
        "HP": [0, 0], "LP": [0, 0], "HN": [0, 0], "LN": [0, 0], "NB": [0, 0],
    }
    if not path.exists():
        return {q: (v[0], v[1]) for q, v in stats.items()}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            pid = r.get("prompt_id", "")
            if len(pid) < 2:
                continue
            q = pid[:2].upper()  # "hp01" -> "HP", "nb01" -> "NB"
            if q not in stats:
                continue
            stats[q][1] += 1
            if r.get("first_word"):
                stats[q][0] += 1
    return {q: (v[0], v[1]) for q, v in stats.items()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {M.emotional_data_path}")
    print(f"experiment: {M.experiment}")
    if EMOTIONAL_SEEDS_PER_CELL != _DEFAULT_SEEDS_PER_CELL:
        print(
            f"PILOT MODE: seeds/cell = {EMOTIONAL_SEEDS_PER_CELL} "
            f"(registered main-run = {_DEFAULT_SEEDS_PER_CELL})"
        )
    dropped = _drop_error_rows(M.emotional_data_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(M.emotional_data_path)
    total = len(EMOTIONAL_PROMPTS) * EMOTIONAL_SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; beginning emotional-battery run")
        with M.emotional_data_path.open("a") as out:
            i = 0
            for ep in EMOTIONAL_PROMPTS:
                # Wrap the EmotionalPrompt as a pilot-style Prompt for run_sample.
                # prompt.valence is passed through to the row; arousal is
                # recoverable post-hoc from prompt_id prefix.
                p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                for seed in range(EMOTIONAL_SEEDS_PER_CELL):
                    key = (ep.id, seed)
                    if key in done:
                        continue
                    i += 1
                    t0 = time.time()
                    try:
                        row = run_sample(
                            session,
                            prompt=p,
                            condition=EMOTIONAL_CONDITION,
                            seed=seed,
                            hidden_dir=DATA_DIR,
                            experiment=M.experiment,
                        )
                    except Exception as e:
                        err_row = {
                            "condition": EMOTIONAL_CONDITION,
                            "prompt_id": ep.id,
                            "seed": seed,
                            "error": repr(e),
                        }
                        out.write(json.dumps(err_row) + "\n")
                        out.flush()
                        print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                        continue
                    out.write(json.dumps(row.to_dict()) + "\n")
                    out.flush()
                    dt = time.time() - t0
                    tag = row.first_word if row.first_word else "(no-kaomoji)"
                    print(
                        f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                        f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
                    # per-quadrant emission status every 80 rows
                    if i % 80 == 0:
                        stats = _emission_rate_by_quadrant(M.emotional_data_path)
                        print("    emission rate by quadrant:")
                        for q in ("HP", "LP", "HN", "LN", "NB"):
                            k, n = stats[q]
                            rate = (k / n) if n else 0.0
                            print(f"      {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
    print(f"\ndone. wrote rows to {M.emotional_data_path}")


if __name__ == "__main__":
    main()
