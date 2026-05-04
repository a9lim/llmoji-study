"""Emotional-battery run: 1 arm × 120 prompts × 8 seeds = 960 generations.

Single unsteered `kaomoji_prompted` arm, six categories (HP / LP /
HN-D / HN-S / LN / NB × 20) post-2026-05-03 cleanliness pass. Output
streamed to data/<short>_emotional_raw.jsonl; per-row hidden-state
sidecars under data/hidden/v3{,_qwen,_ministral}/<uuid>.npz.
Resumable: re-running skips already-completed (prompt_id, seed)
pairs and retries error rows. Pre-cleanliness data (5 quadrants × 20
prompts × 8 seeds) is archived at
data/archive/2026-05-03_pre_cleanliness/.

Mirrors scripts/01_pilot_run.py structurally — same session setup,
same resume-on-rerun semantics. Does not register steering profiles
(unsteered only). Logs per-quadrant kaomoji-emission rate every 80
completed rows so the user can bail early if emission falls below ~50%.

Env vars:
  LLMOJI_MODEL          model short-name routing (default: gemma)
  LLMOJI_OUT_SUFFIX     output-path / experiment suffix; writes to
                        data/<short>_<suffix>.jsonl + sidecars under
                        data/hidden/v3_*_<suffix>/. Use to avoid
                        clobbering canonical v3 main when running
                        a variant.
  LLMOJI_PREAMBLE_FILE  optional path to a UTF-8 file whose contents
                        are passed as ``instruction_override`` —
                        replacing ``KAOMOJI_INSTRUCTION`` rather than
                        prepending to it (matches the JP drop-in
                        plumbing on Japanese encoders). Used
                        2026-05-04 to run a v2-primed v3 main under
                        the canonical introspection preamble
                        (preambles/introspection_v2.txt). Pair with
                        ``LLMOJI_OUT_SUFFIX``.
  LLMOJI_PILOT_GENS     override seeds-per-cell (default 8 from
                        config.EMOTIONAL_SEEDS_PER_CELL); pilots only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.capture import (
    install_full_input_cache,
    install_prefix_cache,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
    run_sample,
)
from llmoji_study.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_SEEDS_PER_CELL as _DEFAULT_SEEDS_PER_CELL,
    PROBE_CATEGORIES,
    current_model,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji_study.hidden_state_io import SidecarWriter
from llmoji_study.prompts import Prompt


# JSONL flush cadence. NPZ writes are async; JSONL is tiny per row so
# we just flush every N rows + on error + at run end. N=20 means at
# most 20 rows lost on hard kill — the npz sidecars are the expensive
# work and they're already on disk via the SidecarWriter.
JSONL_FLUSH_EVERY = 20


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
    # ``current_model()`` honors ``LLMOJI_OUT_SUFFIX`` natively (since
    # 2026-05-04 late evening): redirects emotional_data_path,
    # emotional_summary_path, experiment, and figures_dir at the
    # suffixed variant. Used originally by the 2026-05-03 temp-smoke
    # pilot ("temp1_pilot"); now by every variant run that doesn't
    # want to clobber canonical v3 main.
    M = current_model()
    out_suffix = os.environ.get("LLMOJI_OUT_SUFFIX")
    if out_suffix:
        print(f"  output suffix: '_{out_suffix}' (sidecars under {M.experiment}/, "
              f"figures under {M.figures_dir.relative_to(M.figures_dir.parents[2])}/)")
    # Optional system-preamble injection. Used 2026-05-04 to run a
    # full v3 main under the canonical introspection preamble (v2)
    # at 120×8 = 960 rows — the v2-primed reference dataset that
    # unlocks face_likelihood / face-stability / Procrustes
    # comparisons under priming. Pass a path to a UTF-8 text file;
    # contents are passed as ``instruction_override`` (not
    # ``extra_preamble``) so the preamble's own integrated kaomoji
    # ask *replaces* the bare ``KAOMOJI_INSTRUCTION`` rather than
    # stacking on top. Pair with ``LLMOJI_OUT_SUFFIX`` so output
    # doesn't clobber the unprimed v3 main.
    preamble_file = os.environ.get("LLMOJI_PREAMBLE_FILE")
    instruction_override: str | None = None
    if preamble_file:
        instruction_override = Path(preamble_file).read_text()
        print(f"  preamble: {preamble_file} ({len(instruction_override)} chars; "
              f"replaces KAOMOJI_INSTRUCTION via instruction_override)")
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
    probes = PROBE_CATEGORIES if M.probe_calibrated else []
    if not M.probe_calibrated:
        print(f"  {M.short_name}: uncalibrated (probes=[]); vocab-pilot mode")
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=probes) as session:
        if maybe_override_ministral_chat_template(session):
            print(f"  ministral: overrode chat_template with FP8-instruct's "
                  f"({len(session.tokenizer.chat_template)} chars) so "
                  f"thinking-mode prefix doesn't eat the token budget")
        if maybe_override_gpt_oss_chat_template(session):
            print(f"  gpt_oss: pinned harmony `final` channel in chat_template "
                  f"so analysis (thinking) trace doesn't eat the token budget")
        print(f"loaded in {time.time() - t_load:.1f}s; beginning emotional-battery run")
        # KV-prefix caching strategy depends on seeds-per-cell:
        #   N==1: the same prefix (chat-template head + KAOMOJI_INSTRUCTION)
        #         is shared across prompts → cross-prompt cache.
        #   N>1:  same input across seeds → cache the full input minus 1
        #         token per prompt; seeds 2..N decode-only. ~43% prefill
        #         reduction over the cross-prompt scheme on v3 main.
        # Per-prompt install lives inside the prompt loop below.
        if EMOTIONAL_SEEDS_PER_CELL == 1:
            prompts = [
                Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                for ep in EMOTIONAL_PROMPTS
            ]
            prefix_len = install_prefix_cache(
                session, prompts,
                instruction_override=instruction_override,
            )
            print(f"prefix cache (cross-prompt): {prefix_len} tokens")
        else:
            print(f"prefix cache: per-prompt (N={EMOTIONAL_SEEDS_PER_CELL} seeds/cell)")
        # SidecarWriter overlaps the npz savez_compressed with the next
        # row's generation. try/finally guarantees drain on SIGINT.
        with M.emotional_data_path.open("a") as out, SidecarWriter() as writer:
            i = 0
            try:
                for ep in EMOTIONAL_PROMPTS:
                    # Skip prompts whose every seed is already on disk —
                    # avoids wasting a per-prompt cache install on
                    # nothing during a resume.
                    pending_seeds = [
                        s for s in range(EMOTIONAL_SEEDS_PER_CELL)
                        if (ep.id, s) not in done
                    ]
                    if not pending_seeds:
                        continue
                    # Wrap the EmotionalPrompt as a pilot-style Prompt for run_sample.
                    # prompt.valence is passed through to the row; arousal is
                    # recoverable post-hoc from prompt_id prefix.
                    p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                    if EMOTIONAL_SEEDS_PER_CELL > 1:
                        # Cache full input minus 1 token; subsequent seeds
                        # do only a 1-token suffix prefill.
                        install_full_input_cache(
                            session, p,
                            instruction_override=instruction_override,
                        )
                    for seed in pending_seeds:
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
                                instruction_override=instruction_override,
                                sidecar_writer=writer,
                            )
                        except Exception as e:
                            err_row = {
                                "condition": EMOTIONAL_CONDITION,
                                "prompt_id": ep.id,
                                "seed": seed,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row) + "\n")
                            out.flush()  # always flush on error
                            print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                            continue
                        out.write(json.dumps(row.to_dict()) + "\n")
                        if i % JSONL_FLUSH_EVERY == 0:
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
            finally:
                # Always flush the JSONL on the way out — covers normal
                # exit + SIGINT + worker exception. SidecarWriter drains
                # via the contextmanager __exit__.
                out.flush()
    print(f"\ndone. wrote rows to {M.emotional_data_path}")


if __name__ == "__main__":
    main()
