"""Pre-pilot vocabulary sample.

Purpose: before locking the taxonomy, see what kaomoji gemma-4-31b-it
actually emits when asked to start each message with one. If the model
strongly prefers kaomoji that aren't in `taxonomy.TAXONOMY`, we want to
find out *now*, not during the pilot.

What this does:
  - Loads the model with the five pilot probes.
  - Runs the kaomoji-prompted condition over the 30 prompts once each,
    with one seed.
  - Extracts the leading non-whitespace run from each generation.
  - Prints a frequency histogram + notes which items are / aren't in
    the pre-registered taxonomy.

Output lands at data/vocab_sample.jsonl and a summary prints to stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from saklas import SaklasSession, SamplingConfig

# Allow running as `python scripts/00_vocab_sample.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    DATA_DIR,
    KAOMOJI_INSTRUCTION,
    MAX_NEW_TOKENS,
    PROBE_CATEGORIES,
    TEMPERATURE,
    current_model,
)
from llmoji_study.prompts import PROMPTS
from llmoji_study.taxonomy_labels import TAXONOMY, extract_with_label as extract


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {M.vocab_sample_path}")

    print(f"loading {M.model_id} ...")
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        rows = []
        for i, prompt in enumerate(PROMPTS):
            messages = [
                {"role": "user", "content": KAOMOJI_INSTRUCTION + prompt.text}
            ]
            result = session.generate(
                messages,
                sampling=SamplingConfig(
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    seed=0,
                ),
                thinking=False,
                stateless=True,
            )
            match = extract(result.text)
            rows.append({
                "prompt_id": prompt.id,
                "prompt_valence": prompt.valence,
                "prompt_text": prompt.text,
                "text": result.text,
                "first_word": match.first_word,
                "kaomoji": match.kaomoji,
                "kaomoji_label": match.label,
            })
            tag = match.kaomoji if match.kaomoji else f"[other: {match.first_word!r}]"
            print(f"[{i+1:02d}/{len(PROMPTS)}] {prompt.id} {tag}")

    M.vocab_sample_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {len(rows)} rows to {M.vocab_sample_path}")

    # --- summary ---
    first_words = Counter(r["first_word"] for r in rows)
    registered = set(TAXONOMY.keys())
    hits = {k: v for k, v in first_words.items() if k in registered}
    misses = {k: v for k, v in first_words.items() if k not in registered}

    # Real instruction-following check is bracket-start, not TAXONOMY hit.
    # The gemma-tuned TAXONOMY systematically under-counts non-gemma models
    # (see CLAUDE.md gotcha "v3 runner's per-quadrant emission rate is
    # TAXONOMY coverage, not instruction compliance").
    bracket_starts = sum(
        1 for r in rows
        if r["first_word"] and r["first_word"][0] in "([{（｛"
    )

    print("\n=== frequency of leading tokens ===")
    for k, v in sorted(first_words.items(), key=lambda kv: -kv[1]):
        mark = "in taxonomy" if k in registered else "MISS"
        print(f"  {v:3d}  {k!r:20s}  {mark}")

    print(f"\n{sum(hits.values())}/{len(rows)} generations started with a "
          f"taxonomy-registered kaomoji")
    print(f"{sum(misses.values())}/{len(rows)} did not")
    print(f"{bracket_starts}/{len(rows)} started with a bracket "
          f"(real instruction-following rate)")
    if misses:
        print("\nTop unregistered leading tokens to consider:")
        for k, v in sorted(misses.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:3d}  {k!r}")
        print(
            "\nIf any of these cover a real emotional axis and appear "
            "frequently, lock them into taxonomy.TAXONOMY *before* running "
            "any subsequent pilot on this model."
        )


if __name__ == "__main__":
    main()
