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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.capture import (
    install_prefix_cache,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
    run_sample,
)
from llmoji_study.config import (
    DATA_DIR,
    INTROSPECTION_CONDITIONS,
    INTROSPECTION_PREAMBLE,
    LOREM_PREAMBLE,
    PROBE_CATEGORIES,
    current_model,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.hidden_state_io import SidecarWriter
from llmoji_study.prompts import Prompt


# JSONL flush cadence — same rationale as 03_emotional_run.py.
JSONL_FLUSH_EVERY = 20


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
        if maybe_override_ministral_chat_template(session):
            print(f"  ministral: overrode chat_template with FP8-instruct's "
                  f"({len(session.tokenizer.chat_template)} chars) so "
                  f"thinking-mode prefix doesn't eat the token budget")
        if maybe_override_gpt_oss_chat_template(session):
            print(f"  gpt_oss: pinned harmony `final` channel in chat_template "
                  f"so analysis (thinking) trace doesn't eat the token budget")
        print(f"loaded in {time.time() - t_load:.1f}s; beginning introspection pilot")
        # SidecarWriter overlaps npz writes with generation; per CLAUDE.md
        # this pilot is archive-bound, but new runs benefit from the
        # store_full_trace=False default (see capture.py) too.
        prompts = [
            Prompt(id=ep.id, valence=ep.valence, text=ep.text)
            for ep in EMOTIONAL_PROMPTS
        ]
        with raw_path.open("a") as out, SidecarWriter() as writer:
            i = 0
            try:
                for condition in INTROSPECTION_CONDITIONS:
                    preamble = _PREAMBLE_BY_CONDITION[condition]
                    # Re-cache the prefix per condition — each preamble
                    # produces a distinct chat-template head. Saklas's
                    # cache_prefix() replaces any prior entry; calling
                    # this once per outer iteration costs one extra
                    # prefill per condition (3 total) and amortizes
                    # over ~100 prompts × 1 seed = 100 generations.
                    prefix_len = install_prefix_cache(
                        session, prompts, extra_preamble=preamble,
                    )
                    print(f"[{condition}] prefix cache: {prefix_len} tokens")
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
                                sidecar_writer=writer,
                            )
                        except Exception as e:
                            err_row = {
                                "condition": condition,
                                "prompt_id": ep.id,
                                "seed": INTROSPECTION_SEED,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row) + "\n")
                            out.flush()  # always flush on error
                            print(f"  [{i}/{remaining}] {condition} {ep.id} ERR {e}")
                            continue
                        out.write(json.dumps(row.to_dict()) + "\n")
                        if i % JSONL_FLUSH_EVERY == 0:
                            out.flush()
                        dt = time.time() - t0
                        tag = row.first_word if row.first_word else "(no-kaomoji)"
                        print(
                            f"  [{i}/{remaining}] {condition} {ep.id} ({ep.quadrant}) "
                            f"{tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                        )
            finally:
                # Always flush JSONL; SidecarWriter drains via __exit__.
                out.flush()
    print(f"\ndone. wrote rows to {raw_path}")


if __name__ == "__main__":
    main()
