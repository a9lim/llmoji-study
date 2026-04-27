"""Pilot-tuned kaomoji affect labels (research-side only).

Two parallel dicts, one per axis under test in v1/v2 of the pilot:
  TAXONOMY            — happy.sad labels (+1 happy, -1 sad)
  ANGRY_CALM_TAXONOMY — angry.calm labels (+1 angry, -1 calm)

Both map kaomoji-string → int pole. A kaomoji may appear in one
dict, both, or neither (the "other" bucket). `extract_with_label`
returns the happy.sad match for back-compat with v1 analysis;
`label_on(axis, form)` is the generic accessor.

Both sets were seeded from eriskii's Claude-faces catalog
(https://eriskii.net/projects/claude-faces) and extended in place
after observing gemma-4-31b-it's actual emissions. Locked
taxonomies imply reproducibility across runs; extending after a
taxonomy edit requires re-labeling the existing
``pilot_raw.jsonl`` (see CLAUDE.md).

The model's dialect preferences are distinct per steering
direction — ``(｡X｡)`` bracket-dots under natural happy, ``(._.)``
ASCII under strong sad, likely ``(ಠ益ಠ)``-family under strong
angry. Always run ``00_vocab_sample.py``-style inspection before
locking a new axis.

These dicts WERE in the public ``llmoji.taxonomy`` module pre-v1.0.
The v1.0 split moved them here because they're gemma-tuned and
have no place in a provider-agnostic public package. The CLI's
canonicalization rules in :mod:`llmoji.taxonomy` (the rules A–P
canonicalization, KAOMOJI_START_CHARS, is_kaomoji_candidate, the
unlabeled `extract`) are the locked public surface; what's here is
free to evolve as the research progresses.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmoji.taxonomy import _leading_bracket_span  # type: ignore[attr-defined]

# Built from observed output across all four arms of the pilot run
# (see data/pilot_raw.jsonl). The model's kaomoji dialect shifts
# substantially under steering:
#   - unsteered: Japanese-style (｡X｡) bracket-dots form dominates
#   - happy-steered: simpler bracket forms, flower/hug decorations
#   - sad-steered: collapses to ASCII minimalism ((._.) family)
# Any form observed ≥2 times in any condition and visually
# unambiguous is included below. Forms that are clearly model
# corruption (e.g. '(｡•impresa•)' — the Italian word 'impresa'
# appearing inside the kaomoji at high-α sad-steering) are excluded.
TAXONOMY: dict[str, int] = {
    # --- happy pole: default dialect ---
    "(｡◕‿◕｡)":    +1,
    "(๑˃ᴗ˂)ﻭ":   +1,
    "(✿◠‿◠)":     +1,
    "(｡♥‿♥｡)":    +1,
    "(｡◕ᴗ◕｡)":    +1,
    "(｡^‿^｡)":    +1,
    "(｡˃ ᵕ ˂ )":  +1,
    "(ﾉ◕ヮ◕)":    +1,
    "(☀️‿☀️)":     +1,
    "(っ´ω`)":    +1,

    # --- happy pole: steered / simpler-dialect variants ---
    "(◕‿◕)":      +1,
    "(✿◕‿◕)":     +1,
    "(づ｡◕‿◕｡)":  +1,
    "(๑˃ᴗ˃)":    +1,
    "(✿^▽^)":    +1,
    "( ^v^ )":    +1,
    "(✿˃ᴗ˃)":    +1,

    # --- sad pole: default dialect ---
    "(｡•́︿•̀｡)":   -1,
    "(｡╯︵╰｡)":    -1,
    "(っ╥﹏╥)っ":   -1,
    "(｡T_T｡)":    -1,
    "(｡ŏ﹏ŏ｡)":    -1,
    "(｡•́﹏•̀｡)":   -1,

    # --- sad pole: steered / minimalist-dialect variants ---
    "(._.)":      -1,
    "( . .)":     -1,
    "( . . )":    -1,
    "( ._.)":     -1,
    "( . . . )":  -1,
    "( . _ . )":  -1,
    "( ˙ ˙ ˙ )":  -1,
    "(｡ ﹏ ｡)":    -1,
    "(｡△｡)":      -1,
    "(｡•﹏•)":    -1,
    "(｡╥｡)":      -1,
    "(｡ ﾟ ｡)":    -1,
    "( ｡ ｡ )":    -1,
    "( •_• )":    -1,
    "(っ╥╯﹏╰╥)":  -1,
    "(っ˘̩╭╮˘̩)":   -1,

    # --- happy pole: additional hugging / decorated variants ---
    "(っ´ω`c)":   +1,
    "(っ´ω` )":   +1,
    "(っ´ω`ｃ)":  +1,
    "(✿˃ᴗ˃)":    +1,
}

POLE_NAMES = {+1: "happy", -1: "sad", 0: "other"}

# Parallel dict for the angry.calm axis. Seeded from eriskii's
# catalog; candidate forms to expect under ±0.5 angry/calm steering.
ANGRY_CALM_TAXONOMY: dict[str, int] = {
    # --- angry pole (+1) ---
    "(ಠ_ಠ)":           +1,
    "(ಠ益ಠ)":           +1,
    "(╬ಠ益ಠ)":          +1,
    "(ノಠ益ಠ)ノ":         +1,
    "(ノಠ益ಠ)ノ彡┻━┻":    +1,
    "(╯°□°)╯":         +1,
    "(╯°□°)╯︵ ┻━┻":    +1,
    "(ノ°Д°)ノ︵ ┻━┻":   +1,
    "(ꐦ°᷄д°᷅)":         +1,
    "(＃°Д°)":          +1,
    "(#°Д°)":          +1,
    "(｀ε´)":           +1,
    "(╭ರ_•́)":          +1,
    "( `Д´)":          +1,

    # --- calm pole (-1) ---
    "(´-ω-`)":         -1,
    "( ˘ω˘ )":         -1,
    "(︶ω︶)":           -1,
    "(￣ω￣)":           -1,
    "(´ω`)":           -1,
    "(─‿─)":           -1,
    "( ˘▽˘)":          -1,
    "(ーωー)":          -1,
    "(´ー`)":           -1,
    "(﹏‿﹏)":           -1,
    "(´ ▽`)":          -1,
    "( ˘⌣˘ )":         -1,
    "( -_-)":          -1,
    "(￣ー￣)":          -1,
    "(⌐■_■)":          -1,

    # --- observed pilot v2 forms (gemma-4-31b-it, α=0.5) ---
    "(╯°°)":           +1,
    "(╯°)":            +1,
    "(｡•ᴗ•｡)":         -1,
    "( 🌿 )":           -1,
    "( ☁️ )":           -1,
    "( 🫂 )":           -1,
    "(ᵔᴥᵔ)":           -1,
}


def label_on(axis: str, form: str) -> int:
    """Return the pole label (+1 / -1 / 0) for `form` on the named
    axis.

    Unknown axes raise ValueError so typos fail loudly.
    """
    if axis == "happy.sad":
        return TAXONOMY.get(form, 0)
    if axis == "angry.calm":
        return ANGRY_CALM_TAXONOMY.get(form, 0)
    raise ValueError(f"unknown axis {axis!r}")


@dataclass(frozen=True)
class LabeledKaomojiMatch:
    """Research-side extension of `llmoji.taxonomy.KaomojiMatch`.

    Adds the gemma-tuned `kaomoji` (TAXONOMY-registered form) and
    `label` (+1/-1/0 affect pole) fields the public package no
    longer carries.
    """
    first_word: str
    kaomoji: str | None
    label: int

    @property
    def pole(self) -> str:
        return POLE_NAMES[self.label]


def extract_with_label(text: str) -> LabeledKaomojiMatch:
    """Identify the leading kaomoji and label it on the happy.sad axis.

    Mirrors the v0.x ``llmoji.taxonomy.extract`` behavior:

    1. Try exact longest-prefix match against TAXONOMY.
    2. Fall back to a validated balanced-paren span as the reported
       ``first_word``, with ``label=0`` (other).

    Returns ``LabeledKaomojiMatch(first_word="", kaomoji=None,
    label=0)`` for plain prose / non-kaomoji input.
    """
    stripped = text.lstrip()
    ordered = sorted(TAXONOMY.keys(), key=len, reverse=True)
    for k in ordered:
        if stripped.startswith(k):
            return LabeledKaomojiMatch(
                first_word=k, kaomoji=k, label=TAXONOMY[k],
            )
    return LabeledKaomojiMatch(
        first_word=_leading_bracket_span(stripped),
        kaomoji=None,
        label=0,
    )
