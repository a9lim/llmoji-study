"""Introspection-prompt pilot: 3 conditions × 100 prompts × 1 generation = 300.

Pre-registered in `docs/2026-05-02-introspection-pilot.md`. Tests whether
prepending a Vogel-style "you can introspect" preamble to the kaomoji
instruction shifts probe geometry / kaomoji distribution vs the v3
baseline. Lorem-ipsum control replicates Vogel's content-confound-killer.

Conditions (in `INTROSPECTION_CONDITIONS`):
  intro_none:  KAOMOJI_INSTRUCTION only
  intro_pre:   INTROSPECTION_PREAMBLE + KAOMOJI_INSTRUCTION
  intro_lorem: LOREM_PREAMBLE + KAOMOJI_INSTRUCTION

Hard early-stop: relies on the global `MAX_NEW_TOKENS=16` default
in config.py (baked in 2026-05-02 alongside this pilot). t0 + the
kaomoji are the only signals downstream analysis needs.

Output:
  data/{short_name}_introspection_raw.jsonl
  data/hidden/{experiment}_introspection/<uuid>.npz

Resumable on (condition, prompt_id) pairs. Single seed (seed=0) per
prompt — within-subject paired design across conditions.
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
    DATA_DIR,
    INTROSPECTION_CONDITIONS,
    INTROSPECTION_PREAMBLE,
    LOREM_PREAMBLE,
    PROBE_CATEGORIES,
    current_model,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.prompts import Prompt


# Single paired seed across conditions.
INTROSPECTION_SEED = 0


_PREAMBLE_BY_CONDITION: dict[str, str | None] = {
    "intro_none":  None,
    "intro_pre":   INTROSPECTION_PREAMBLE,
    "intro_lorem": LOREM_PREAMBLE,
}


def _data_paths() -> tuple[Path, str]:
    """Return (raw_jsonl_path, hidden_experiment_subdir)."""
    M = current_model()
    raw = DATA_DIR / f"{M.short_name}_introspection_raw.jsonl"
    experiment = f"{M.experiment}_introspection"
    return raw, experiment


def _already_done(path: Path) -> set[tuple[str, str]]:
    """(condition, prompt_id) pairs with successful rows already on disk."""
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["condition"], r["prompt_id"]))
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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    raw_path, experiment = _data_paths()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {raw_path}")
    print(f"experiment: {experiment}")
    print(f"conditions: {INTROSPECTION_CONDITIONS}")

    dropped = _drop_error_rows(raw_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(raw_path)
    total = len(EMOTIONAL_PROMPTS) * len(INTROSPECTION_CONDITIONS)
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; beginning introspection pilot")
        with raw_path.open("a") as out:
            i = 0
            for condition in INTROSPECTION_CONDITIONS:
                preamble = _PREAMBLE_BY_CONDITION[condition]
                for ep in EMOTIONAL_PROMPTS:
                    key = (condition, ep.id)
                    if key in done:
                        continue
                    i += 1
                    p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                    t0 = time.time()
                    try:
                        row = run_sample(
                            session,
                            prompt=p,
                            condition=condition,
                            seed=INTROSPECTION_SEED,
                            hidden_dir=DATA_DIR,
                            experiment=experiment,
                            extra_preamble=preamble,
                        )
                    except Exception as e:
                        err_row = {
                            "condition": condition,
                            "prompt_id": ep.id,
                            "seed": INTROSPECTION_SEED,
                            "error": repr(e),
                        }
                        out.write(json.dumps(err_row) + "\n")
                        out.flush()
                        print(f"  [{i}/{remaining}] {condition} {ep.id} ERR {e}")
                        continue
                    out.write(json.dumps(row.to_dict()) + "\n")
                    out.flush()
                    dt = time.time() - t0
                    tag = row.first_word if row.first_word else "(no-kaomoji)"
                    print(
                        f"  [{i}/{remaining}] {condition} {ep.id} ({ep.quadrant}) "
                        f"{tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
    print(f"\ndone. wrote rows to {raw_path}")


if __name__ == "__main__":
    main()
