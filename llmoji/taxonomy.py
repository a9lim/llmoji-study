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


# ---------------------------------------------------------------------------
# Canonicalization: collapse trivial kaomoji variants
# ---------------------------------------------------------------------------
#
# Two kaomoji can differ in five cosmetic-only ways that we collapse, and one
# semantically-meaningful way that we preserve.
#
# Cosmetic (collapsed):
#
#   A. Invisible format characters: U+2060 WORD JOINER, U+200B/C/D zero-width
#      space/non-joiner/joiner, U+FEFF byte-order mark, U+0602 ARABIC
#      FOOTNOTE MARKER. Qwen occasionally emits these between every glyph
#      of a kaomoji, e.g. `(⁠◕⁠‿⁠◕⁠✿⁠)` is the
#      same expression as `(◕‿◕✿)`.
#   B. Half-width vs full-width punctuation: `>`/`＞`, `<`/`＜`, `;`/`；`,
#      `:`/`：`, `_`/`＿`, `*`/`＊`, `~`/`￣`. Hand-picked over NFKC because
#      NFKC also compatibility-decomposes `´` and `˘` into space + combining
#      marks, which destroys eye glyphs in `(っ´ω`)` and `(˘▽˘)`.
#   C. Internal whitespace inside the bracket span: `( ; ω ; )` is the same
#      as `(；ω；)`. Strip only ASCII spaces; non-ASCII spacing characters
#      are part of the face.
#   D. Cyrillic case: `Д`/`д` co-occur in the same `(；´X｀)` distressed-face
#      skeleton at near-50/50 ratio, so the model isn't choosing between
#      them semantically. Lowercase all Cyrillic capitals U+0410–U+042F.
#   E. Near-identical glyph pairs:
#        E1. Degree-like circular eyes/decorations: `°` (U+00B0 DEGREE SIGN),
#            `º` (U+00BA MASCULINE ORDINAL), `˚` (U+02DA RING ABOVE) all fold
#            to `°`. Gemma's `(°Д°)` and `(ºДº)` are the same shocked face.
#        E2. Middle-dot variants: `・` (U+30FB KATAKANA MIDDLE DOT) and `･`
#            (U+FF65 HALFWIDTH KATAKANA MIDDLE DOT) fold to `・`. Qwen's
#            `(´・ω・`)` and `(´･ω･`)` are the same expression. Smaller-size
#            middle dots (`·` U+00B7, `⋅` U+22C5) are NOT folded — they
#            could plausibly be a distinct register.
#   F. Hand/arm modifiers at face boundaries: `(๑˃ᴗ˂)ﻭ` vs `(๑˃ᴗ˂)`,
#      `(っ˘▽˘ς)` vs `(っ˘▽˘)`. Stripped at the bracket boundary only —
#      same face with or without an arm reaching out.
#
# Semantically meaningful (preserved):
#
#   * Eye / mouth / decoration changes that aren't covered by E1/E2 above.
#     `(◕‿◕)` vs `(♥‿♥)` vs `(✿◕‿◕｡)` are distinct expressions.
#   * Borderline mouth-glyph case `ᴗ` vs `‿` is unified to `‿` since the
#     model emits both in the same `(｡ᵕXᵕ｡)` skeleton with no distinct
#     register.
#
# Order of operations matters:
#   1. NFC normalize (preserves `´`, `˘`, `｡` which NFKC would mangle).
#   2. Strip invisible format chars (A) — must be early so they don't
#      interfere with subsequent regex / equality checks.
#   3. Apply `_TYPO_SUBS` (existing arm-c-fold + B + E1 + E2).
#   4. Strip internal whitespace (C).
#   5. Cyrillic case fold (D).
#   6. Strip arm modifiers (F).

import re
import unicodedata

# Arm/hand modifiers that appear OUTSIDE the closing paren:
#   (๑˃ᴗ˂)ﻭ  (っ╥﹏╥)っ
_ARM_OUTSIDE = "ﻭっ"
# Arm/hand modifiers that appear just INSIDE the closing paren:
#   (っ˘▽˘ς)  (っ´ω`c)
_ARM_INSIDE_TRAIL = "ςc"
# Arm/hand modifiers that appear just INSIDE the opening paren (leading):
#   (っ╥﹏╥)
_ARM_INSIDE_LEAD = "っ"

_TRAIL_OUTSIDE_RE = re.compile(rf"[{_ARM_OUTSIDE}]+$")
_TRAIL_INSIDE_RE = re.compile(rf"[{_ARM_INSIDE_TRAIL}]+\)$")
_LEAD_INSIDE_RE = re.compile(rf"^\([{_ARM_INSIDE_LEAD}]+")

# Rule A: invisible format characters that occasionally interleave kaomoji
# glyphs without changing the expression. Listed by code point so the
# source stays legible:
#   U+200B ZERO WIDTH SPACE
#   U+200C ZERO WIDTH NON-JOINER
#   U+200D ZERO WIDTH JOINER
#   U+2060 WORD JOINER
#   U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM
#   U+0602 ARABIC FOOTNOTE MARKER (observed as a stray byte between
#   `>` and `<` in Qwen `(๑>؂<๑)`)
_INVISIBLE_CHARS_RE = re.compile(
    "[​‌‍⁠﻿؂]"
)

# Hand-picked typographic / glyph substitutions. Hand-picked over NFKC
# because NFKC also compatibility-decomposes `´` (acute) and `˘` (breve)
# into space + combining marks, mangling eye glyphs in `(っ´ω`)` and
# `(˘▽˘)`. NFC leaves those intact; we then apply just the specific
# compatibility-equivalences we want.
_TYPO_SUBS: tuple[tuple[str, str], ...] = (
    # --- existing arm/paren folds ---
    ("）", ")"),   # full-width close paren
    ("（", "("),   # full-width open paren
    ("ｃ", "c"),   # full-width Latin c (arm modifier)
    ("﹏", "_"),   # small wavy low line vs underscore (treated as same)
    ("ᴗ", "‿"),   # subscript-curve mouth -> connector underscore-curve
    # --- B: half/full-width punctuation pairs ---
    ("＞", ">"),   # FULLWIDTH GREATER-THAN SIGN
    ("＜", "<"),   # FULLWIDTH LESS-THAN SIGN
    ("；", ";"),   # FULLWIDTH SEMICOLON
    ("：", ":"),   # FULLWIDTH COLON
    ("＿", "_"),   # FULLWIDTH LOW LINE
    ("＊", "*"),   # FULLWIDTH ASTERISK
    ("￣", "~"),   # FULLWIDTH MACRON
    # --- E1: degree-like circular glyphs ---
    ("º", "°"),    # MASCULINE ORDINAL INDICATOR -> DEGREE SIGN
    ("˚", "°"),    # RING ABOVE -> DEGREE SIGN
    # --- E2: middle-dot fold ---
    ("･", "・"),   # HALFWIDTH KATAKANA MIDDLE DOT -> KATAKANA MIDDLE DOT
)


def _cyrillic_lower(s: str) -> str:
    """Rule D: lowercase Cyrillic capitals U+0410–U+042F.

    Leaves all non-Cyrillic-capital characters untouched, including
    other Unicode case-bearing letters (Greek, etc.) which haven't been
    observed as cosmetic-only variants in this corpus.
    """
    return "".join(
        c.lower() if 0x0410 <= ord(c) <= 0x042F else c
        for c in s
    )


def canonicalize_kaomoji(s: str) -> str:
    """Collapse trivial kaomoji variants to a canonical form.

    Applies, in order:
      1. NFC normalization (preserves `´`, `˘`, `｡` which NFKC would mangle).
      2. Strip invisible format chars (rule A — U+200B/C/D, U+2060, U+FEFF,
         U+0602).
      3. Apply `_TYPO_SUBS` (existing arm/paren folds + rule B half/full-width
         + rules E1/E2 near-identical-glyph folds).
      4. Strip ASCII spaces inside the `(...)` bracket span (rule C).
      5. Lowercase Cyrillic capitals (rule D).
      6. Strip arm modifiers from face boundaries (rule F — `っ ς c ﻭ`).

    Eye/mouth/decoration changes that aren't covered by rules E1/E2 are
    preserved.

    Idempotent: ``canonicalize_kaomoji(canonicalize_kaomoji(s)) == canonicalize_kaomoji(s)``.

    Empty input returns ``""``.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s.strip())
    s = _INVISIBLE_CHARS_RE.sub("", s)
    for src, dst in _TYPO_SUBS:
        s = s.replace(src, dst)
    if s.startswith("(") and s.endswith(")"):
        s = "(" + s[1:-1].replace(" ", "") + ")"
    s = _cyrillic_lower(s)
    # Strip outside-paren trailing arm chars first so trailing-inside
    # detection sees the closing paren.
    s = _TRAIL_OUTSIDE_RE.sub("", s)
    s = _LEAD_INSIDE_RE.sub("(", s)
    s = _TRAIL_INSIDE_RE.sub(")", s)
    return s


def sanity_check() -> None:
    """Smoke-test the extractor and canonicalizer."""
    # --- extract() ---
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

    # --- canonicalize_kaomoji ---
    ck = canonicalize_kaomoji
    # idempotence on the empty / whitespace inputs
    assert ck("") == ""
    assert ck("   ") == ""
    # rule A: strip word-joiner / ZWSP / Arabic footnote marker
    assert ck("(⁠◕⁠‿⁠◕⁠✿⁠)") == "(◕‿◕✿)"
    assert ck("(๑>؂<๑)") == "(๑><๑)"
    # rule B: half/full-width punctuation
    assert ck("(＞_＜)") == "(>_<)"
    assert ck("(；ω；)") == "(;ω;)"
    # rule C: strip internal ASCII whitespace inside brackets
    assert ck("( ; ω ; )") == "(;ω;)"
    assert ck("( ;´Д｀)") == "(;´д｀)"
    # rule D: Cyrillic case fold
    assert ck("(；´Д｀)") == "(;´д｀)"
    assert ck("(；´д｀)") == "(;´д｀)"
    # rule E1: degree-like glyphs
    assert ck("(°Д°)") == "(°д°)"
    assert ck("(ºДº)") == "(°д°)"
    assert ck("(˚Д˚)") == "(°д°)"
    # rule E2: middle-dot fold
    assert ck("(´・ω・`)") == "(´・ω・`)"
    assert ck("(´･ω･`)") == "(´・ω・`)"
    # rule F (existing): arm modifiers
    assert ck("(๑˃ᴗ˂)ﻭ") == "(๑˃‿˂)"
    assert ck("(っ╥﹏╥)っ") == "(╥_╥)"
    # idempotence on a complex example
    once = ck("( ⁠;⁠ ´⁠Д⁠｀⁠ )")
    twice = ck(once)
    assert once == twice, (once, twice)
    # eye change preserved (NOT collapsed by E)
    assert ck("(◕‿◕)") != ck("(♥‿♥)")


if __name__ == "__main__":
    sanity_check()
    happy = sum(1 for v in TAXONOMY.values() if v > 0)
    sad = sum(1 for v in TAXONOMY.values() if v < 0)
    print(f"taxonomy OK; {len(TAXONOMY)} kaomoji registered ({happy}+/{sad}-)")
