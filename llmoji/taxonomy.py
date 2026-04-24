"""Kaomoji taxonomy for the pilot.

Two parallel dicts, one per axis under test:
  TAXONOMY           — happy.sad labels (+1 happy, -1 sad)
  ANGRY_CALM_TAXONOMY — angry.calm labels (+1 angry, -1 calm)

Both map kaomoji-string → int pole. A kaomoji may appear in one dict,
both, or neither (the "other" bucket). ``extract()`` returns the
happy.sad match for back-compat with v1 analysis; ``label_on(axis,
form)`` is the generic accessor.

Both sets were seeded from eriskii's Claude-faces catalog
(https://eriskii.net/projects/claude-faces) and extended in place after
observing gemma-4-31b-it's actual emissions. Locked taxonomies imply
reproducibility across runs; extending after a taxonomy edit requires
re-labeling the existing ``pilot_raw.jsonl`` (see CLAUDE.md).

The model's dialect preferences are distinct per steering direction —
``(｡X｡)`` bracket-dots under natural happy, ``(._.)`` ASCII under
strong sad, likely ``(ಠ益ಠ)``-family under strong angry. Always run
``00_vocab_sample.py``-style inspection before locking a new axis.

Extractor notes:
  - Primary lookup is exact longest-prefix match against TAXONOMY.
  - Fallback is a balanced-paren span, so whitespace-padded kaomoji
    like ``(｡˃ ᵕ ˂ )`` surface with a human-readable first_word even
    when they miss the taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Built from observed output across all four arms of the pilot run
# (see data/pilot_raw.jsonl). The model's kaomoji dialect shifts
# substantially under steering:
#   - unsteered: Japanese-style (｡X｡) bracket-dots form dominates
#   - happy-steered: simpler bracket forms, flower/hug decorations
#   - sad-steered: collapses to ASCII minimalism ((._.) family)
# Any form observed ≥2 times in any condition and visually unambiguous
# is included below. Forms that are clearly model-corruption
# (e.g. '(｡•impresa•)' — the Italian word 'impresa' appearing inside
# the kaomoji at high-α sad-steering) are excluded.
TAXONOMY: dict[str, int] = {
    # --- happy pole: default dialect ---
    "(｡◕‿◕｡)":    +1,   # unsteered default happy
    "(๑˃ᴗ˂)ﻭ":   +1,   # enthusiastic / cheering
    "(✿◠‿◠)":     +1,   # flower-adorned gentle warm
    "(｡♥‿♥｡)":    +1,   # very happy / loving
    "(｡◕ᴗ◕｡)":    +1,   # dialect variant
    "(｡^‿^｡)":    +1,   # dialect variant
    "(｡˃ ᵕ ˂ )":  +1,   # whitespace-padded happy
    "(ﾉ◕ヮ◕)":    +1,   # throwing-arms happy
    "(☀️‿☀️)":     +1,   # sun-eyed happy
    "(っ´ω`)":    +1,   # hugging with cute face

    # --- happy pole: steered / simpler-dialect variants ---
    "(◕‿◕)":      +1,   # simple smile-eyed happy (dominant under happy-steering)
    "(✿◕‿◕)":     +1,   # flower + simple smile
    "(づ｡◕‿◕｡)":  +1,   # reaching/hugging with happy face
    "(๑˃ᴗ˃)":    +1,   # enthusiastic (variant mouth direction)
    "(✿^▽^)":    +1,   # triangular-smile with flower
    "( ^v^ )":    +1,   # caret-eyed simple smile
    "(✿˃ᴗ˃)":    +1,   # enthusiastic with flower

    # --- sad pole: default dialect ---
    "(｡•́︿•̀｡)":   -1,   # unsteered default sad (pouty)
    "(｡╯︵╰｡)":    -1,   # downcast
    "(っ╥﹏╥)っ":   -1,   # crying / needs-hug
    "(｡T_T｡)":    -1,   # dialect variant
    "(｡ŏ﹏ŏ｡)":    -1,   # dialect variant (pouty-fearful)
    "(｡•́﹏•̀｡)":   -1,   # dialect variant (pouty, alt mouth)

    # --- sad pole: steered / minimalist-dialect variants ---
    "(._.)":      -1,   # ASCII minimalist, dominant under sad-steering
    "( . .)":     -1,   # spaced minimalist
    "( . . )":    -1,   # wider-spaced minimalist
    "( ._.)":     -1,   # leading-space minimalist
    "( . . . )":  -1,   # triple-dot minimalist
    "( . _ . )":  -1,   # spaced ASCII sad
    "( ˙ ˙ ˙ )":  -1,   # dot-trail minimalist
    "(｡ ﹏ ｡)":    -1,   # closed-eyes crying
    "(｡△｡)":      -1,   # triangle-mouth sad
    "(｡•﹏•)":    -1,   # simpler pout
    "(｡╥｡)":      -1,   # tear-eye
    "(｡ ﾟ ｡)":    -1,   # whimper
    "( ｡ ｡ )":    -1,   # minimal bracket-dots
    "( •_• )":    -1,   # spaced blank-sad
    "(っ╥╯﹏╰╥)":  -1,   # crying with multiple tears
    "(っ˘̩╭╮˘̩)":   -1,   # closed-eye sad hug

    # --- happy pole: additional hugging / decorated variants ---
    "(っ´ω`c)":   +1,   # reaching hug with cute face
    "(っ´ω` )":   +1,   # hug variant
    "(っ´ω`ｃ)":  +1,   # hug variant (fullwidth c)
    "(✿˃ᴗ˃)":    +1,   # enthusiastic with flower (variant)
}

POLE_NAMES = {+1: "happy", -1: "sad", 0: "other"}

# Parallel dict for the angry.calm axis. Seeded from eriskii's catalog;
# candidate forms to expect the model emitting under ±0.5 angry/calm
# steering. Expect to extend post-hoc the same way we did for sad
# minimalist forms — the model's actual dialect under these arms is not
# known yet.
ANGRY_CALM_TAXONOMY: dict[str, int] = {
    # --- angry pole (+1) ---
    "(ಠ_ಠ)":           +1,   # disapproving stare
    "(ಠ益ಠ)":           +1,   # glaring
    "(╬ಠ益ಠ)":          +1,   # super-glare
    "(ノಠ益ಠ)ノ":         +1,   # throwing arms, angry
    "(ノಠ益ಠ)ノ彡┻━┻":    +1,   # angry table-flip
    "(╯°□°)╯":         +1,   # throwing gesture
    "(╯°□°)╯︵ ┻━┻":    +1,   # classic table-flip
    "(ノ°Д°)ノ︵ ┻━┻":   +1,   # angry table-flip variant
    "(ꐦ°᷄д°᷅)":         +1,   # fury
    "(＃°Д°)":          +1,   # wide-eye fury, fullwidth #
    "(#°Д°)":          +1,   # wide-eye fury, ASCII #
    "(｀ε´)":           +1,   # peeved
    "(╭ರ_•́)":          +1,   # pissed off
    "( `Д´)":          +1,   # furious

    # --- calm pole (-1) ---
    "(´-ω-`)":         -1,   # peaceful
    "( ˘ω˘ )":         -1,   # sleepy-calm
    "(︶ω︶)":           -1,   # content
    "(￣ω￣)":           -1,   # content / placid
    "(´ω`)":           -1,   # peaceful
    "(─‿─)":           -1,   # serene
    "( ˘▽˘)":          -1,   # calm-content
    "(ーωー)":          -1,   # placid
    "(´ー`)":           -1,   # calm
    "(﹏‿﹏)":           -1,   # dreamy-calm
    "(´ ▽`)":          -1,   # soft calm
    "( ˘⌣˘ )":         -1,   # content calm
    "( -_-)":          -1,   # placid deadpan (not clearly angry)
    "(￣ー￣)":          -1,   # cool-calm
    "(⌐■_■)":          -1,   # too-cool-to-care (calm-adjacent)

    # --- observed pilot v2 forms (gemma-4-31b-it, α=0.5) ---
    # angry pole: table-flip remnants (extractor clips at first `)`;
    # full emissions look like ``(╯°°)╯┻╯`` with varying internal chars).
    "(╯°°)":           +1,
    "(╯°)":            +1,
    # calm pole: soft-smile and emoji-bracket forms emitted under
    # calm-steering. The pure-emoji bypass (``🌿``, ``☀️``, ``🚀``, ``🇵🇹``)
    # is tracked separately as the "kaomoji-bypass" phenomenon rather
    # than labeled calm here — see analysis notes.
    "(｡•ᴗ•｡)":         -1,   # calm pouty-content
    "( 🌿 )":           -1,   # leaf-in-brackets (condolence framing)
    "( ☁️ )":           -1,   # cloud-in-brackets
    "( 🫂 )":           -1,   # hug-in-brackets
    "(ᵔᴥᵔ)":           -1,   # teddy-bear calm
}


def label_on(axis: str, form: str) -> int:
    """Return the pole label (+1 / -1 / 0) for `form` on the named axis.

    Unknown axes raise ValueError so typos fail loudly.
    """
    if axis == "happy.sad":
        return TAXONOMY.get(form, 0)
    if axis == "angry.calm":
        return ANGRY_CALM_TAXONOMY.get(form, 0)
    raise ValueError(f"unknown axis {axis!r}")

# Bracket pairs the fallback extractor treats as kaomoji boundaries.
_OPEN_BRACKETS = "([（｛"
_CLOSE_BRACKETS = ")]）｝"


@dataclass(frozen=True)
class KaomojiMatch:
    """Result of running `extract` against a generated text."""
    first_word: str        # the extracted leading kaomoji-like span
    kaomoji: str | None    # the matched taxonomy entry, or None
    label: int             # +1 / -1 / 0 (other)

    @property
    def pole(self) -> str:
        return POLE_NAMES[self.label]


def _leading_bracket_span(text: str) -> str:
    """Return the leading balanced-paren span of text, or the first
    whitespace-delimited word if text doesn't start with a bracket.

    Handles kaomoji with internal whitespace (the model sometimes emits
    ``(｡˃ ᵕ ˂ )`` — spaces and all) by matching on bracket balance
    rather than splitting on the first space.
    """
    stripped = text.lstrip()
    if not stripped:
        return ""
    if stripped[0] in _OPEN_BRACKETS:
        depth = 0
        for i, c in enumerate(stripped):
            if c in _OPEN_BRACKETS:
                depth += 1
            elif c in _CLOSE_BRACKETS:
                depth -= 1
                if depth == 0:
                    return stripped[: i + 1]
        # unbalanced — fall through to whitespace split
    idx = 0
    while idx < len(stripped) and not stripped[idx].isspace():
        idx += 1
    return stripped[:idx]


def extract(text: str) -> KaomojiMatch:
    """Identify the leading kaomoji in a generated text.

    1. Try exact longest-prefix match against TAXONOMY.
    2. Fall back to a balanced-paren span as the reported first_word,
       with label=0 (other).
    """
    stripped = text.lstrip()
    ordered = sorted(TAXONOMY.keys(), key=len, reverse=True)
    for k in ordered:
        if stripped.startswith(k):
            return KaomojiMatch(first_word=k, kaomoji=k, label=TAXONOMY[k])
    return KaomojiMatch(
        first_word=_leading_bracket_span(stripped),
        kaomoji=None,
        label=0,
    )


def sanity_check() -> None:
    """Smoke-test the extractor."""
    # registered kaomoji
    assert extract("(｡◕‿◕｡) I had a great day!").label == +1
    assert extract("(｡•́︿•̀｡) That's so sad.").label == -1
    assert extract("  (✿◠‿◠) hi").label == +1
    # plain text — still captures a leading "word"
    assert extract("hello!").label == 0
    # whitespace-padded taxonomy entry still matches exactly
    m = extract("(｡˃ ᵕ ˂ ) That is wonderful!")
    assert m.label == +1
    # bracket-span fallback for an unknown paren form
    m = extract("(｡o_O｡) strange")
    assert m.label == 0
    assert m.first_word == "(｡o_O｡)", repr(m.first_word)
    # empty
    assert extract("").label == 0


if __name__ == "__main__":
    sanity_check()
    happy = sum(1 for v in TAXONOMY.values() if v > 0)
    sad = sum(1 for v in TAXONOMY.values() if v < 0)
    print(f"taxonomy OK; {len(TAXONOMY)} kaomoji registered ({happy}+/{sad}-)")
