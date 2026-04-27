"""Cursory Qwen3.6-27B kaomoji vocabulary sample.

Purpose: see what kaomoji a brand-new non-gemma model emits under the
"start each message with a kaomoji" instruction, and how much its
leading-glyph distribution overlaps with gemma-4-31b-it's. Cursory
only — no figures, no probes-driven analysis, no pre-registered
decision rule. Pairs with `21_qwen_gemma_overlap.py` for the
cross-model comparison.

Mirrors `00_vocab_sample.py`:
  - 30 v1 PROMPTS × 1 seed × KAOMOJI_INSTRUCTION
  - thinking=False (Qwen3.6 is a reasoning model; the assistant turn
    opens with `<think>\\n\\n</think>\\n\\n` when thinking is disabled,
    so the kaomoji should appear immediately after that block)
  - stateless=True
  - same TEMPERATURE / MAX_NEW_TOKENS / seed=0

Probes are bootstrapped from PROBE_CATEGORIES because saklas already
has Qwen__Qwen3.6-27B vectors cached; this is essentially free. We
don't *use* the probe scores in this script — they're recorded for
JSONL parity with `vocab_sample.jsonl` so a follow-up could fold them
in without re-running.

Output: data/qwen36_vocab_sample.jsonl + stdout summary.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from saklas import SaklasSession, SamplingConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    DATA_DIR,
    KAOMOJI_INSTRUCTION,
    MAX_NEW_TOKENS,
    PROBE_CATEGORIES,
    PROBES,
    TEMPERATURE,
)
from llmoji_study.prompts import PROMPTS
from llmoji.taxonomy import TAXONOMY, extract

# Hardcoded here, NOT in config.py — config.MODEL_ID is gemma and the v1/v2/v3
# pipelines depend on that. This script is one-off cursory exploration.
QWEN_MODEL_ID = "Qwen/Qwen3.6-27B"
OUTPUT_PATH = DATA_DIR / "qwen36_vocab_sample.jsonl"

# Defensive: strip any <think>...</think> block that leaks through.
# saklas should already do this with thinking=False, but Qwen's template
# is new to this codebase so belt-and-suspenders.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text, count=1).lstrip()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {QWEN_MODEL_ID} ...")
    with SaklasSession.from_pretrained(
        QWEN_MODEL_ID, device="auto", probes=PROBE_CATEGORIES
    ) as session:
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
            text = _strip_thinking(result.text)
            match = extract(text)

            # Collect per-prompt probe scores (whole-generation aggregate)
            # for JSONL parity with the gemma vocab_sample.jsonl. Cheap
            # since probes were bootstrapped at session init anyway.
            probe_scores = {}
            for probe in PROBES:
                reading = result.readings.get(probe)
                if reading is not None and reading.per_generation:
                    probe_scores[probe] = float(reading.per_generation[-1])
                else:
                    probe_scores[probe] = None

            rows.append({
                "prompt_id": prompt.id,
                "prompt_valence": prompt.valence,
                "prompt_text": prompt.text,
                "text": text,
                "text_raw": result.text,  # keeps any <think> for audit
                "first_word": match.first_word,
                "kaomoji": match.kaomoji,
                "kaomoji_label": match.label,
                "probe_scores": probe_scores,
            })
            tag = match.kaomoji if match.kaomoji else f"[other: {match.first_word!r}]"
            print(f"[{i+1:02d}/{len(PROMPTS)}] {prompt.id} {tag}")

    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {len(rows)} rows to {OUTPUT_PATH}")

    # --- summary (mirrors 00_vocab_sample.py) ---
    first_words = Counter(r["first_word"] for r in rows)
    registered = set(TAXONOMY.keys())
    hits = {k: v for k, v in first_words.items() if k in registered}
    misses = {k: v for k, v in first_words.items() if k not in registered}

    print("\n=== frequency of leading tokens (Qwen3.6-27B) ===")
    for k, v in sorted(first_words.items(), key=lambda kv: -kv[1]):
        mark = "in gemma taxonomy" if k in registered else "MISS"
        print(f"  {v:3d}  {k!r:20s}  {mark}")

    print(
        f"\n{sum(hits.values())}/{len(rows)} generations started with a "
        "form already in gemma's taxonomy"
    )
    print(f"{sum(misses.values())}/{len(rows)} did not")
    if misses:
        print("\nTop unregistered leading tokens (Qwen-specific candidates):")
        for k, v in sorted(misses.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:3d}  {k!r}")


if __name__ == "__main__":
    main()
