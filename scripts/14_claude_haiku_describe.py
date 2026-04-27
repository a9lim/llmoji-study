"""Eriskii-replication step 1: two-stage masked-context Haiku pipeline.

Stage A: per kaomoji, sample up to INSTANCE_SAMPLE_CAP rows from
data/claude_kaomoji.jsonl (with floor — kaomoji with fewer
instances are fully sampled, deterministic via INSTANCE_SAMPLE_SEED).
For each sampled row: mask the leading kaomoji, prepend
surrounding_user (when non-empty), feed to Haiku, save per-instance
description to data/claude_haiku_descriptions.jsonl.

Stage B: per kaomoji, gather Stage-A descriptions, send to Haiku,
save synthesized one-sentence meaning to
data/claude_haiku_synthesized.jsonl.

Both stages resumable. Set ANTHROPIC_API_KEY in the environment.

Usage:
  python scripts/14_claude_haiku_describe.py [--stage A|B|both] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    CLAUDE_HAIKU_DESCRIPTIONS_PATH,
    CLAUDE_HAIKU_SYNTHESIZED_PATH,
    CLAUDE_KAOMOJI_PATH,
    DATA_DIR,
    HAIKU_MODEL_ID,
    INSTANCE_SAMPLE_CAP,
    INSTANCE_SAMPLE_SEED,
)
from llmoji_study.eriskii import call_haiku, mask_kaomoji
from llmoji.haiku_prompts import (
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    SYNTHESIZE_PROMPT,
)


def _already_described(path: Path) -> set[str]:
    """assistant_uuid set of rows already successfully described in Stage A."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add(r["assistant_uuid"])
    return done


def _already_synthesized(path: Path) -> set[str]:
    """first_word set of kaomoji already synthesized in Stage B."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add(r["first_word"])
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


def _sample_rows_per_kaomoji(
    rows: list[dict],
    *,
    cap: int,
    seed: int,
) -> list[dict]:
    """For each first_word, sample up to `cap` rows uniformly at
    random with a deterministic per-kaomoji RNG seed. Sort kaomoji
    alphabetically so iteration order is stable across reruns."""
    by_kao: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        fw = r.get("first_word", "")
        if fw:
            by_kao[fw].append(r)
    sampled: list[dict] = []
    for fw in sorted(by_kao.keys()):
        bucket = by_kao[fw]
        if len(bucket) <= cap:
            sampled.extend(bucket)
        else:
            rng = random.Random(f"{seed}:{fw}")
            sampled.extend(rng.sample(bucket, cap))
    return sampled


def stage_a(client, *, limit: int | None) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(CLAUDE_HAIKU_DESCRIPTIONS_PATH)
    if dropped:
        print(f"stage-A: dropped {dropped} prior error rows for retry")
    done = _already_described(CLAUDE_HAIKU_DESCRIPTIONS_PATH)

    with CLAUDE_KAOMOJI_PATH.open() as f:
        all_rows = [json.loads(l) for l in f.read().splitlines() if l.strip()]
    sampled = _sample_rows_per_kaomoji(
        all_rows, cap=INSTANCE_SAMPLE_CAP, seed=INSTANCE_SAMPLE_SEED,
    )
    todo = [r for r in sampled if r.get("assistant_uuid") and r["assistant_uuid"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"stage-A: sampled {len(sampled)} instances "
          f"(cap={INSTANCE_SAMPLE_CAP}, seed={INSTANCE_SAMPLE_SEED}); "
          f"already done: {len(done)}; this run: {len(todo)}")
    if not todo:
        return 0

    n_written = 0
    with CLAUDE_HAIKU_DESCRIPTIONS_PATH.open("a") as out:
        for i, r in enumerate(todo, start=1):
            t0 = time.time()
            try:
                masked = mask_kaomoji(r["assistant_text"], r["first_word"])
                user = (r.get("surrounding_user") or "").strip()
                if user:
                    prompt = DESCRIBE_PROMPT_WITH_USER.format(
                        user_text=user, masked_text=masked,
                    )
                else:
                    prompt = DESCRIBE_PROMPT_NO_USER.format(
                        masked_text=masked,
                    )
                desc = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID)
            except Exception as e:
                err_row = {"assistant_uuid": r["assistant_uuid"], "error": repr(e)}
                out.write(json.dumps(err_row) + "\n")
                out.flush()
                print(f"  [stage-A {i}/{len(todo)}] ERR {r['first_word']}: {e}")
                continue
            row = {
                "assistant_uuid": r["assistant_uuid"],
                "first_word": r["first_word"],
                "description": desc,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1
            dt = time.time() - t0
            short = desc[:70] + ("..." if len(desc) > 70 else "")
            print(f"  [stage-A {i}/{len(todo)}] {r['first_word']}  ({dt:.1f}s)  {short}")
    return n_written


def stage_b(client, *, limit: int | None) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CLAUDE_HAIKU_DESCRIPTIONS_PATH.exists():
        print("stage-B: no Stage-A output; run Stage A first")
        return 0
    dropped = _drop_error_rows(CLAUDE_HAIKU_SYNTHESIZED_PATH)
    if dropped:
        print(f"stage-B: dropped {dropped} prior error rows for retry")
    done = _already_synthesized(CLAUDE_HAIKU_SYNTHESIZED_PATH)

    descriptions_by_fw: dict[str, list[str]] = defaultdict(list)
    with CLAUDE_HAIKU_DESCRIPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            descriptions_by_fw[r["first_word"]].append(r["description"])
    todo_kaomoji = [fw for fw in sorted(descriptions_by_fw.keys()) if fw not in done]
    if limit is not None:
        todo_kaomoji = todo_kaomoji[:limit]
    print(f"stage-B: {len(descriptions_by_fw)} kaomoji with descriptions; "
          f"already synthesized: {len(done)}; this run: {len(todo_kaomoji)}")
    if not todo_kaomoji:
        return 0

    n_written = 0
    with CLAUDE_HAIKU_SYNTHESIZED_PATH.open("a") as out:
        for i, fw in enumerate(todo_kaomoji, start=1):
            descs = descriptions_by_fw[fw]
            t0 = time.time()
            try:
                listed = "\n".join(f"{j+1}. {d}" for j, d in enumerate(descs))
                prompt = SYNTHESIZE_PROMPT.format(descriptions=listed)
                synth = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID, max_tokens=200)
            except Exception as e:
                err_row = {"first_word": fw, "error": repr(e)}
                out.write(json.dumps(err_row) + "\n")
                out.flush()
                print(f"  [stage-B {i}/{len(todo_kaomoji)}] ERR {fw}: {e}")
                continue
            row = {
                "first_word": fw,
                "n_descriptions": len(descs),
                "synthesized": synth,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1
            dt = time.time() - t0
            short = synth[:70] + ("..." if len(synth) > 70 else "")
            print(f"  [stage-B {i}/{len(todo_kaomoji)}] {fw} (n={len(descs)})  ({dt:.1f}s)  {short}")
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows processed in each stage this run (smoke testing)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)
    if not CLAUDE_KAOMOJI_PATH.exists():
        print(f"no scrape at {CLAUDE_KAOMOJI_PATH}; run scripts/06_claude_scrape.py first")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    if args.stage in ("A", "both"):
        stage_a(client, limit=args.limit)
    if args.stage in ("B", "both"):
        stage_b(client, limit=args.limit)
    print("done.")


if __name__ == "__main__":
    main()
