"""Re-key data/claude_haiku_synthesized.jsonl by canonical kaomoji form,
re-synthesizing collision groups under the current
``taxonomy.canonicalize_kaomoji`` rules (A–L as of 2026-04-27).

What this resolves: post-canonicalization stale state. Stage B
synthesis was originally keyed by raw first_word; under the current
rules, multiple raw forms can canonicalize to the same key, so the
existing synthesized rows for those forms are single-variant
syntheses missing the merged-instance evidence. This script also
drops orphan rows whose canonical form has zero emissions in the
current ``claude_kaomoji.jsonl`` scrape — historical garbage from
older, less-strict hook output.

Strategy:
  - Load the raw scrape and compute the set of canonical forms with
    at least one emission. This is the "live" canonical set.
  - Group existing synthesized rows by canonical(first_word).
  - Drop groups whose canonical form is NOT in the live canonical
    set (orphans — historical garbage like
    ``(Backgrounddebugscript...)`` carried over from older hook
    output that the current ``is_kaomoji_candidate`` rejects).
  - Drop orphan rows from claude_haiku_descriptions.jsonl too —
    rows whose ``assistant_uuid`` isn't in the current scrape.
  - For each remaining canonical group with a single member:
    re-key in place (first_word := canonical), preserve description
    and n; record raw form in first_word_raw.
  - For each collision group: pool per-instance descriptions across
    all member raw first_words; re-run Stage B SYNTHESIZE_PROMPT on
    the pooled list; write a single merged row keyed by canonical
    form.

Pre/post-condition: every row in synthesized.jsonl satisfies
canonicalize_kaomoji(first_word) == first_word AND has at least one
matching emission in claude_kaomoji.jsonl.

Atomicity: writes to a temp path, only swaps the live files at the
end after all Haiku calls succeed. Originals backed up to
``{synthesized,descriptions}.jsonl.bak.{ts}`` before swap.

Cost: ~$0.01 per re-synthesis × N collision groups (typically ≤20).

Usage:
  ANTHROPIC_API_KEY=... python scripts/22_resync_haiku_canonical.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    CLAUDE_HAIKU_DESCRIPTIONS_PATH,
    CLAUDE_HAIKU_SYNTHESIZED_PATH,
    CLAUDE_KAOMOJI_PATH,
    HAIKU_MODEL_ID,
)
from llmoji_study.eriskii import call_haiku
from llmoji.haiku_prompts import SYNTHESIZE_PROMPT
from llmoji.taxonomy import canonicalize_kaomoji


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    syn_path = CLAUDE_HAIKU_SYNTHESIZED_PATH
    desc_path = CLAUDE_HAIKU_DESCRIPTIONS_PATH
    raw_path = CLAUDE_KAOMOJI_PATH
    if not syn_path.exists() or not desc_path.exists() or not raw_path.exists():
        print(f"missing inputs: need {syn_path}, {desc_path}, {raw_path}")
        sys.exit(1)

    # Load the raw scrape and compute live canonical forms. Anything in
    # synthesized that doesn't intersect this set is orphan garbage from
    # older, less-strict hook output that the current
    # is_kaomoji_candidate rejects. NOTE we do NOT use assistant_uuid
    # for descriptions cleanup — the unified-journal refactor changed
    # the uuid set, but the descriptions themselves are real Haiku
    # output describing real past kaomoji uses, valid synthesis
    # evidence regardless of whether the originating turn is still in
    # the current scrape.
    live_canonical: set[str] = set()
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fw = r.get("first_word")
            if isinstance(fw, str) and fw:
                live_canonical.add(canonicalize_kaomoji(fw))
    print(f"raw scrape: {len(live_canonical)} canonical forms")

    # Load synthesized rows.
    syn_rows = []
    with syn_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            syn_rows.append(r)
    print(f"loaded {len(syn_rows)} synthesized rows from {syn_path}")

    # Load per-instance descriptions, key raw_fw -> list[str].
    # No orphan filtering on descriptions side — see note above.
    descriptions_by_raw: dict[str, list[str]] = defaultdict(list)
    n_desc_total = 0
    with desc_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            descriptions_by_raw[r["first_word"]].append(r["description"])
            n_desc_total += 1
    n_desc_orphan = 0  # unused but kept for the printout below
    desc_rows_kept: list[dict] = []  # unused but kept for the printout below
    print(f"per-instance descriptions: {n_desc_total} across "
          f"{len(descriptions_by_raw)} raw forms")

    # Group synthesized rows by canonical form, drop groups whose
    # canonical isn't in the live raw scrape.
    by_canon_all: dict[str, list[dict]] = defaultdict(list)
    for r in syn_rows:
        by_canon_all[canonicalize_kaomoji(r["first_word"])].append(r)
    by_canon: dict[str, list[dict]] = {
        c: g for c, g in by_canon_all.items() if c in live_canonical
    }
    n_orphan_canon = len(by_canon_all) - len(by_canon)
    n_orphan_rows = sum(
        len(g) for c, g in by_canon_all.items() if c not in live_canonical
    )
    print(f"synthesized canonical groups: {len(by_canon_all)} total → "
          f"{len(by_canon)} kept ({n_orphan_canon} orphan groups / "
          f"{n_orphan_rows} orphan rows dropped)")

    n_singletons = sum(1 for g in by_canon.values() if len(g) == 1)
    n_collisions = sum(1 for g in by_canon.values() if len(g) > 1)
    print(f"  → singletons: {n_singletons}, collisions: {n_collisions}")

    # Build the new synthesized.jsonl.
    import anthropic
    client = anthropic.Anthropic()

    new_rows: list[dict] = []
    n_resynthesized = 0
    n_rekeyed = 0

    # Process collisions first so any API failure aborts before we touch
    # the cheap singleton work.
    collision_canons = sorted(
        c for c, g in by_canon.items() if len(g) > 1
    )
    print(f"\nre-synthesizing {len(collision_canons)} collision groups:")
    resynthesized: dict[str, dict] = {}
    for i, canon in enumerate(collision_canons, start=1):
        members = by_canon[canon]
        raws = sorted(m["first_word"] for m in members)
        # Pool per-instance descriptions across all member raw forms.
        pooled: list[str] = []
        for raw in raws:
            pooled.extend(descriptions_by_raw.get(raw, []))
        # Some raw forms might have synthesized rows but no per-instance
        # descriptions (e.g., the original Stage A run was capped or
        # different) — fall back to the per-variant synthesized
        # descriptions in that case so we don't lose information.
        if not pooled:
            pooled = [m["synthesized"] for m in members]
            note = "(fell back to per-variant syntheses; no per-instance descriptions found)"
        else:
            note = ""

        listed = "\n".join(f"{j+1}. {d}" for j, d in enumerate(pooled))
        prompt = SYNTHESIZE_PROMPT.format(descriptions=listed)
        t0 = time.time()
        try:
            synth = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID, max_tokens=200)
        except Exception as e:
            print(f"  [{i}/{len(collision_canons)}] ERR {canon}: {e}")
            print("aborting before any data is overwritten.")
            sys.exit(2)
        dt = time.time() - t0
        n_total = sum(int(m.get("n_descriptions", 0)) for m in members)
        resynthesized[canon] = {
            "first_word": canon,
            "first_word_raw": raws,
            "n_descriptions": n_total,
            "synthesized": synth,
        }
        n_resynthesized += 1
        short = synth[:70] + ("..." if len(synth) > 70 else "")
        print(f"  [{i}/{len(collision_canons)}] {canon}  "
              f"merged {len(raws)} variants, {len(pooled)} pooled descriptions  "
              f"({dt:.1f}s)  {short} {note}".rstrip())

    # Now build the full output preserving deterministic order:
    # alphabetical by canonical form.
    for canon in sorted(by_canon.keys()):
        members = by_canon[canon]
        if len(members) > 1:
            new_rows.append(resynthesized[canon])
        else:
            m = members[0]
            new_rows.append({
                "first_word": canon,
                "first_word_raw": [m["first_word"]],
                "n_descriptions": int(m.get("n_descriptions", 0)),
                "synthesized": m["synthesized"],
            })
            if m["first_word"] != canon:
                n_rekeyed += 1

    # Atomic swap: backup, write temp, mv. Same drill for descriptions.
    ts = time.strftime("%Y%m%d-%H%M%S")

    syn_bak = syn_path.with_suffix(f".jsonl.bak.{ts}")
    shutil.copy2(syn_path, syn_bak)
    syn_tmp = syn_path.with_suffix(".jsonl.tmp")
    with syn_tmp.open("w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    syn_tmp.replace(syn_path)
    print(f"\nbacked up {syn_path} -> {syn_bak}")
    print(f"wrote {syn_path}: {len(new_rows)} rows "
          f"({n_resynthesized} re-synthesized, {n_rekeyed} singleton-rekeyed, "
          f"{len(new_rows) - n_resynthesized - n_rekeyed} unchanged, "
          f"{n_orphan_rows} orphans dropped)")

    if n_desc_orphan:
        desc_bak = desc_path.with_suffix(f".jsonl.bak.{ts}")
        shutil.copy2(desc_path, desc_bak)
        desc_tmp = desc_path.with_suffix(".jsonl.tmp")
        with desc_tmp.open("w") as f:
            for r in desc_rows_kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        desc_tmp.replace(desc_path)
        print(f"backed up {desc_path} -> {desc_bak}")
        print(f"wrote {desc_path}: {len(desc_rows_kept)} rows "
              f"({n_desc_orphan} orphans dropped)")


if __name__ == "__main__":
    main()
