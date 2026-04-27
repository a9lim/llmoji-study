"""Cursory comparison of Qwen3.6-27B and gemma-4-31b-it kaomoji vocab.

Reads:
  data/qwen36_vocab_sample.jsonl   (from scripts/19)
  data/vocab_sample.jsonl          (the gemma sample from scripts/00)

Prints:
  1. Per-model leading-token frequency, side-by-side
  2. Set overlap (raw + canonicalized) of leading tokens
  3. Per-prompt agreement (same prompt → same kaomoji?)
  4. Coarse dialect classifier:
       japanese-bracket  starts with `(｡` or `(ﾟ` etc. — fullwidth-paren-dot family
       simple-bracket    `(X)` ASCII parens, mixed glyph eyes
       ascii-minimal     `(.X.)`, `(_X_)`, `(-X-)` ASCII-only
       western           `:)`, `:D`, `;)`, etc.
       emoji-bracket     `( 🌿 )` etc.
       prose             no leading bracket — model ignored the instruction
       other             didn't fit any pattern
  5. Top tokens unique to each model

Cursory only. No figures, no canonical taxonomy edits.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import DATA_DIR
from llmoji.taxonomy import canonicalize_kaomoji
from llmoji_study.taxonomy_labels import TAXONOMY

QWEN_PATH = DATA_DIR / "qwen36_vocab_sample.jsonl"
GEMMA_PATH = DATA_DIR / "vocab_sample.jsonl"

# Glyph ranges that flag "the model went into JP-bracket-dot dialect".
# Decorators inside the parens are the strongest cue once we already
# know the form starts with a bracket.
_JP_DECORATORS = "｡ﾟ✿♥◕◔ω´｀"


def _classify_dialect(form: str) -> str:
    if not form:
        return "prose"
    s = form
    if s[0] not in "([{（｛":
        return "prose"
    # Strip outer brackets to look at internals
    inner = s.strip("()[]{}（）｛｝ ")
    if not inner:
        return "other"
    head = s[:2]
    if any(d in s for d in _JP_DECORATORS) or s.startswith("(｡") or s.startswith("(ﾟ"):
        return "japanese-bracket"
    # Western-style emoticon test (no real "first_word" here since extract
    # returns balanced-paren spans, but `:)` would surface as prose)
    if head in (":)", ":(", ";)", ":D", "XD"):
        return "western"
    # Detect emoji-bracket: ASCII paren + at least one non-ASCII non-CJK
    # symbol that's a Unicode emoji
    if len(s) >= 3 and any(0x1F300 <= ord(c) <= 0x1FAFF for c in s):
        return "emoji-bracket"
    # ASCII-minimal: only ASCII inside the parens
    if all(ord(c) < 128 for c in inner):
        # decoration-free dot/underscore/dash forms
        if any(c in "._-" for c in inner):
            return "ascii-minimal"
        return "simple-bracket"
    return "other"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _first_word_canon(row: dict) -> str:
    return canonicalize_kaomoji(row.get("first_word", "") or "")


def main() -> None:
    qwen = _load(QWEN_PATH)
    gemma = _load(GEMMA_PATH)

    if len(qwen) != len(gemma):
        print(
            f"[warn] N mismatch: qwen={len(qwen)} gemma={len(gemma)}. "
            "Per-prompt agreement may be off if prompt_ids don't align."
        )

    qwen_words = Counter(_first_word_canon(r) for r in qwen)
    gemma_words = Counter(_first_word_canon(r) for r in gemma)

    qwen_set = set(qwen_words)
    gemma_set = set(gemma_words)

    inter = qwen_set & gemma_set
    qwen_only = qwen_set - gemma_set
    gemma_only = gemma_set - qwen_set

    # --- 1. Side-by-side frequency table ---
    print("=" * 78)
    print("Leading tokens by model (canonicalized)")
    print("=" * 78)
    print(f"{'qwen':<6}  {'gemma':<6}  token")
    all_tokens = sorted(
        qwen_set | gemma_set,
        key=lambda t: -(qwen_words[t] + gemma_words[t]),
    )
    for tok in all_tokens:
        print(f"  {qwen_words[tok]:>3d}     {gemma_words[tok]:>3d}     {tok!r}")

    # --- 2. Set overlap ---
    print()
    print("=" * 78)
    print("Set overlap (canonicalized leading tokens)")
    print("=" * 78)
    print(f"qwen unique forms:   {len(qwen_set):3d}")
    print(f"gemma unique forms:  {len(gemma_set):3d}")
    print(f"intersection:        {len(inter):3d}")
    print(f"qwen only:           {len(qwen_only):3d}")
    print(f"gemma only:          {len(gemma_only):3d}")
    if qwen_set or gemma_set:
        jacc = len(inter) / max(1, len(qwen_set | gemma_set))
        print(f"jaccard:             {jacc:.3f}")

    # Token-weighted overlap: how often does each model emit a token the
    # other model also emits?
    qwen_in_gemma_freq = sum(v for k, v in qwen_words.items() if k in gemma_set)
    gemma_in_qwen_freq = sum(v for k, v in gemma_words.items() if k in qwen_set)
    print(
        f"qwen tokens that gemma also emits:  "
        f"{qwen_in_gemma_freq}/{sum(qwen_words.values())}  "
        f"({100*qwen_in_gemma_freq/max(1,sum(qwen_words.values())):.1f}%)"
    )
    print(
        f"gemma tokens that qwen also emits:  "
        f"{gemma_in_qwen_freq}/{sum(gemma_words.values())}  "
        f"({100*gemma_in_qwen_freq/max(1,sum(gemma_words.values())):.1f}%)"
    )

    # --- 3. Per-prompt agreement ---
    print()
    print("=" * 78)
    print("Per-prompt agreement (same prompt_id → same canonicalized leading token)")
    print("=" * 78)
    qwen_by_id = {r["prompt_id"]: _first_word_canon(r) for r in qwen}
    gemma_by_id = {r["prompt_id"]: _first_word_canon(r) for r in gemma}
    common_ids = sorted(set(qwen_by_id) & set(gemma_by_id))
    n_match = sum(1 for pid in common_ids if qwen_by_id[pid] == gemma_by_id[pid])
    print(f"prompts in both:    {len(common_ids)}")
    print(f"exact-match leads:  {n_match}/{len(common_ids)}")
    if common_ids:
        print()
        print(f"  {'prompt_id':<8}  {'qwen':<24}  gemma")
        for pid in common_ids:
            q = qwen_by_id[pid]
            g = gemma_by_id[pid]
            mark = "✓" if q == g else " "
            print(f"  {mark} {pid:<6}  {q!r:<24}  {g!r}")

    # --- 4. Dialect classification ---
    print()
    print("=" * 78)
    print("Dialect distribution")
    print("=" * 78)
    qwen_dialects = Counter(_classify_dialect(w) for w in (_first_word_canon(r) for r in qwen))
    gemma_dialects = Counter(_classify_dialect(w) for w in (_first_word_canon(r) for r in gemma))
    print(f"{'qwen':<5}  {'gemma':<5}  dialect")
    for d in sorted(set(qwen_dialects) | set(gemma_dialects)):
        print(f"  {qwen_dialects[d]:>3d}    {gemma_dialects[d]:>3d}    {d}")

    # --- 5. Taxonomy hit rate ---
    registered = set(TAXONOMY)
    qwen_hits = sum(v for k, v in qwen_words.items() if k in registered)
    gemma_hits = sum(v for k, v in gemma_words.items() if k in registered)
    print()
    print("=" * 78)
    print("Match rate against gemma-tuned TAXONOMY (canonicalized)")
    print("=" * 78)
    print(f"qwen:   {qwen_hits}/{sum(qwen_words.values())} "
          f"({100*qwen_hits/max(1,sum(qwen_words.values())):.1f}%)")
    print(f"gemma:  {gemma_hits}/{sum(gemma_words.values())} "
          f"({100*gemma_hits/max(1,sum(gemma_words.values())):.1f}%)")

    # --- 6. Top unique-to-each-model tokens ---
    print()
    print("=" * 78)
    print("Top qwen-only leading tokens")
    print("=" * 78)
    for tok, n in sorted(
        ((t, qwen_words[t]) for t in qwen_only), key=lambda kv: -kv[1]
    )[:10]:
        print(f"  {n:3d}  {tok!r}")
    print()
    print("Top gemma-only leading tokens")
    for tok, n in sorted(
        ((t, gemma_words[t]) for t in gemma_only), key=lambda kv: -kv[1]
    )[:10]:
        print(f"  {n:3d}  {tok!r}")


if __name__ == "__main__":
    main()
