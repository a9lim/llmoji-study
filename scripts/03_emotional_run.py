"""Emotional-battery run: 1 arm × 80 prompts × 8 seeds = 640 generations.

Single unsteered `kaomoji_prompted` arm, Russell-quadrant prompts.
Output streamed to data/emotional_raw.jsonl. Resumable: re-running
skips already-completed (prompt_id, seed) pairs and retries error rows.

Mirrors scripts/01_pilot_run.py structurally — same session setup,
same resume-on-rerun semantics. Does not register steering profiles
(unsteered only). Logs per-quadrant kaomoji-emission rate every 80
completed rows so the user can bail early if emission falls below ~50%.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_SEEDS_PER_CELL,
    MODEL_ID,
    PROBE_CATEGORIES,
)
from llmoji.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji.prompts import Prompt


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
    JSONL. Uses prompt_id prefix to infer quadrant (hp/lp/hn/ln)."""
    stats: dict[str, list[int]] = {"HP": [0, 0], "LP": [0, 0], "HN": [0, 0], "LN": [0, 0]}
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
            q = pid[:2].upper()  # "hp01" -> "HP"
            if q not in stats:
                continue
            stats[q][1] += 1
            if r.get("kaomoji"):
                stats[q][0] += 1
    return {q: (v[0], v[1]) for q, v in stats.items()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(EMOTIONAL_DATA_PATH)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(EMOTIONAL_DATA_PATH)
    total = len(EMOTIONAL_PROMPTS) * EMOTIONAL_SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {MODEL_ID} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; beginning emotional-battery run")
        with EMOTIONAL_DATA_PATH.open("a") as out:
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
                    tag = row.kaomoji if row.kaomoji else f"[{row.first_word!r}]"
                    print(
                        f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                        f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
                    # per-quadrant emission status every 80 rows
                    if i % 80 == 0:
                        stats = _emission_rate_by_quadrant(EMOTIONAL_DATA_PATH)
                        print("    emission rate by quadrant:")
                        for q in ("HP", "LP", "HN", "LN"):
                            k, n = stats[q]
                            rate = (k / n) if n else 0.0
                            print(f"      {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
    print(f"\ndone. wrote rows to {EMOTIONAL_DATA_PATH}")


if __name__ == "__main__":
    main()
