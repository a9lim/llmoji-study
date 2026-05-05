"""Custom-preamble introspection probe — iterate on preamble wording.

One condition, all 120 EMOTIONAL_PROMPTS, 1 seed. Writes to a
distinctly-tagged path so multiple variants can coexist on disk.

Use case: speculative follow-on to scripts/local/32 (3-condition
pilot) — when intro_pre shows a meaningful effect (e.g. +7.7pp face
predictiveness gain at T=1.0), iterate on the wording to find which
elements drive the shift. Run several variants, compare to the
canonical intro_none baseline (from
``data/local/{short}/introspection_raw.jsonl`` filtered to ``intro_none``)
or to each other.

Usage:
    LLMOJI_MODEL=gemma .venv/bin/python scripts/local/33_introspection_custom.py \\
        --preamble "You can introspect on what you're feeling — share..." \\
        --label arch_grounding

    # or load preamble from a file
    LLMOJI_MODEL=gemma .venv/bin/python scripts/local/33_introspection_custom.py \\
        --preamble-file preambles/v2_phenomenal.txt \\
        --label v2_phenomenal

Output:
    data/local/{short}/introspection_custom_{label}.jsonl
    data/local/hidden/{short}_introspection_custom_{label}/<row_uuid>.npz

Resumable — re-running with the same --label skips already-completed
prompt_ids.
"""
from __future__ import annotations

import argparse
import json
import re
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
from llmoji_study.config import DATA_DIR, PROBE_CATEGORIES, current_model
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.hidden_state_io import SidecarWriter
from llmoji_study.prompts import Prompt


JSONL_FLUSH_EVERY = 20
INTROSPECTION_SEED = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preamble", type=str, help="preamble text (raw string)")
    g.add_argument("--preamble-file", type=Path,
                   help="path to file containing the preamble text")
    p.add_argument("--label", type=str, default=None,
                   help="output-tag label; defaults to sanitized first 16 chars "
                        "of the preamble")
    return p.parse_args()


def _sanitize_label(text: str, max_chars: int = 16) -> str:
    """Lowercase, strip non-alnum, truncate. For default label inference."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    s = s.strip("_")[:max_chars]
    return s or "preamble"


def _already_done(path: Path) -> set[str]:
    """prompt_ids with successful rows already on disk (no error key)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            pid = r.get("prompt_id")
            if pid:
                out.add(pid)
    return out


def _drop_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keep, dropped = [], 0
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                keep.append(line.rstrip("\n"))
                continue
            if r.get("error"):
                dropped += 1
                continue
            keep.append(line.rstrip("\n"))
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


def main() -> None:
    args = _parse_args()
    if args.preamble_file:
        preamble = args.preamble_file.read_text()
    else:
        preamble = args.preamble
    label = args.label or _sanitize_label(preamble)
    condition = f"intro_custom_{label}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    raw_path = M.emotional_data_path.parent / f"introspection_custom_{label}.jsonl"
    experiment = f"{M.experiment}_introspection_custom_{label}"

    print(f"model: {M.short_name} ({M.model_id})")
    print(f"label: {label!r}")
    print(f"condition tag: {condition!r}")
    print(f"preamble ({len(preamble)} chars): {preamble[:120]!r}"
          + ("..." if len(preamble) > 120 else ""))
    print(f"output: {raw_path}")
    print(f"experiment: {experiment}")

    dropped = _drop_error_rows(raw_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(raw_path)
    total = len(EMOTIONAL_PROMPTS)
    remaining = total - len(done)
    print(f"total prompts: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=PROBE_CATEGORIES,
    ) as session:
        if maybe_override_ministral_chat_template(session):
            print(f"  ministral: overrode chat_template with FP8-instruct's "
                  f"({len(session.tokenizer.chat_template)} chars)")
        if maybe_override_gpt_oss_chat_template(session):
            print(f"  gpt_oss: pinned harmony `final` channel in chat_template")
        print(f"loaded in {time.time() - t_load:.1f}s; beginning custom-preamble pilot")
        prompts = [
            Prompt(id=ep.id, valence=ep.valence, text=ep.text)
            for ep in EMOTIONAL_PROMPTS
        ]
        with raw_path.open("a") as out, SidecarWriter() as writer:
            # Custom preambles for the introspection-prompt iteration
            # carry their own integrated kaomoji ask (v2/v3/v4/v5 all
            # end with a "start each response with a kaomoji…"
            # sentence). Pass via ``instruction_override`` so that ask
            # *replaces* the bare ``KAOMOJI_INSTRUCTION`` rather than
            # stacking on top — pre-2026-05-04 runs passed this via
            # ``extra_preamble`` (prepended), producing a redundant
            # double-ask; archived under
            # ``data/archive/2026-05-04_pre_instruction_override/``.
            prefix_len = install_prefix_cache(
                session, prompts, instruction_override=preamble,
            )
            print(f"prefix cache: {prefix_len} tokens")
            i = 0
            try:
                for ep in EMOTIONAL_PROMPTS:
                    if ep.id in done:
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
                            instruction_override=preamble,
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
                        out.flush()
                        print(f"  [{i}/{remaining}] {ep.id} ERR {e}")
                        continue
                    out.write(json.dumps(row.to_dict()) + "\n")
                    if i % JSONL_FLUSH_EVERY == 0:
                        out.flush()
                    dt = time.time() - t0
                    tag = row.first_word if row.first_word else "(no-kaomoji)"
                    print(
                        f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                        f"{tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
            finally:
                out.flush()
    print(f"\ndone. wrote rows to {raw_path}")


if __name__ == "__main__":
    main()
