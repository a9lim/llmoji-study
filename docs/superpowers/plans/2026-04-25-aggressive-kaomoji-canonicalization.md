# Aggressive kaomoji canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `llmoji.taxonomy.canonicalize_kaomoji` with five new merge rules (A–E) that collapse cosmetic-only variants in Qwen's vocabulary (`(；ω；)` family, U+2060 word-joiner-decorated forms, half/full-width punctuation pairs, internal-whitespace padding, Cyrillic case, and near-identical glyph pairs like `°/º`/`˚` and `・/･`). Drop Qwen v3 from 73 → 65 unique forms; gemma drops 33 → 32 (one shocked-face merge); ministral and claude-faces unchanged.

**Architecture:** Single-function change in `llmoji/taxonomy.py`. The function is already applied at load time in `load_emotional_features` (v3) and `claude_faces.load_embeddings_canonical` (claude-faces), so re-running the analysis scripts picks up the new canonicalization automatically — no model re-runs, no JSONL edits. Per CLAUDE.md "Regenerate the per-kaomoji parquets if the canonicalization rule changes," but the merge-impact survey already done in brainstorming confirms claude-faces is unaffected (12 → 12 forms), so its parquet doesn't need regeneration. Only v3 figures regenerate.

**Tech Stack:** Python stdlib (`unicodedata`, `re`). No new deps.

**Pre-registration (binding per CLAUDE.md ethics — methodology change rather than experiment, so the discipline is "rule decisions made before looking at refreshed figures"):**

- Each of the five rules is justified individually on the merge groups shown to the user during brainstorming, not on figure outcomes. The expected impact on per-quadrant separation ratios is unknown when the rule is being committed; that's the point.
- Concrete merge groups (locked in this plan, not selected post-hoc):
  - **A: invisible** strip Cf chars (U+200B–U+200D, U+2060, U+FEFF) + U+0602 ARABIC FOOTNOTE MARKER. Qwen merges: `(◕‿◕✿)` 9 + word-joiner-decorated 7 → 16; plus 4 other word-joiner forms collapse to themselves.
  - **B: half/full-width fold** `＞→>`, `＜→<`, `；→;`, `：→:`, `＿→_`, `＊→*`, `￣→~`. Qwen merges: `(>_<)` 31 + `(＞_＜)` 5 → 36.
  - **C: internal whitespace** strip ASCII spaces inside `(...)` brackets. Qwen merges: `(;ω;)` 71+11 → 82, `(;´Д｀)` 36+3 → 39, `(;_;)` 2+1 → 3, `(^ω^)` 1+1 → 2.
  - **D: Cyrillic case fold** lowercase Cyrillic capitals `А–Я` (U+0410–U+042F). Qwen merges: `(;´д｀)` adds 31 to the Д group → 70.
  - **E1: degree-like glyphs** `º→°` and `˚→°`. Gemma merges: `(°Д°)` 1 + `(ºДº)` 1 → 2 (after rule D, `(°д°)` total 2).
  - **E2: middle-dot fold** `･→・`. Qwen merges: `(´・ω・`)` 12 + `(´･ω･`)` 5 → 17.
- Total form-count impact (locked, will be verified in Task 2 against this list): gemma 33 → 32, qwen 73 → 65, ministral 9 → 9, claude-faces 12 → 12.
- Welfare: no new generations. Pure post-processing change.

**Out of scope (separate plans if pursued):**

- Eye/mouth/decoration class merges that aren't near-identical-glyph (`◕` vs `♥`, `△` vs `▽`, etc.).
- Latin-Cyrillic lookalike folds beyond what's in the data (`T/Т`, `O/О`, `K/К`). The data only contains a single `(T_T)` and no Cyrillic counterpart, so adding speculative rules is YAGNI.
- TAXONOMY dict additions for Mistral/Qwen-specific forms — orthogonal to canonicalization, separate plan if motivated.
- Re-running v1/v2 (gemma steering) analysis. v1/v2 uses `extract()` for taxonomy lookup, not `canonicalize_kaomoji`. The merge survey showed gemma's v1/v2-style vocab unaffected.

---

## File Structure

**Modified:**

- `llmoji/taxonomy.py` — extend `_TYPO_SUBS` with 7 width-fold pairs (B) + 3 glyph-fold pairs (E1+E2); add `_INVISIBLE_CHARS_RE` regex (A); add `_cyrillic_lower` helper (D); add internal-whitespace strip step inside `canonicalize_kaomoji` (C); update the docstring + comment block describing the canonicalization contract; extend `sanity_check()` with cases for each new rule.
- `CLAUDE.md` — refresh the "Kaomoji canonicalization" section to describe rules A–E and the new total-count effect; refresh the v3 gemma "Findings" subsection if PCA numbers shift; refresh the v3 Qwen "Findings" subsection (will shift more, since 8 forms collapse).

**Re-generated (re-run, content changes):**

- `data/emotional_summary.tsv` — gemma per-kaomoji summary (33 → 32 rows; the `(°Д°)` / `(ºДº)` pair merges).
- `data/qwen_emotional_summary.tsv` — qwen per-kaomoji summary (73 → 65 rows).
- `figures/fig_emo_a_kaomoji_sim.png`, `fig_emo_b_kaomoji_consistency.png`, `fig_emo_c_kaomoji_quadrant.png`, `fig_v3_pca_valence_arousal.png`, `fig_v3_face_pca_by_quadrant.png`, `fig_v3_face_probe_scatter.png`, `fig_v3_face_cosine_heatmap.png` — gemma v3 figures.
- `figures/qwen/fig_emo_a_kaomoji_sim.png`, `fig_emo_b_kaomoji_consistency.png`, `fig_emo_c_kaomoji_quadrant.png`, `fig_v3_pca_valence_arousal.png`, `fig_v3_face_pca_by_quadrant.png`, `fig_v3_face_probe_scatter.png`, `fig_v3_face_cosine_heatmap.png` — qwen v3 figures.

**Unchanged (verified during brainstorming, re-verified in Task 2):**

- `data/ministral_vocab_sample.jsonl` — Ministral pilot data, no merges available.
- `data/claude_faces_embed.parquet`, `data/claude_faces_embed_description.parquet`, all eriskii outputs — claude-faces vocab has 0 merges available under any of the 5 rules, so the parquet stays valid and eriskii descriptions don't need re-synthesis.
- `data/pilot_features.parquet`, all v1/v2 figures — v1/v2 doesn't use `canonicalize_kaomoji`.
- All scripts other than the analysis re-runs; the canonicalization is applied at load time inside loaders, not in scripts.

---

### Task 1: Implement rules A–E in `canonicalize_kaomoji`

**Files:**
- Modify: `llmoji/taxonomy.py:230–331`

- [ ] **Step 1: Replace the canonicalization comment block + helpers + `_TYPO_SUBS`**

In `llmoji/taxonomy.py`, replace lines 230–279 (the comment block from `# ---` through the closing `)` of `_TYPO_SUBS`) with:

```python
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
# glyphs without changing the expression.
_INVISIBLE_CHARS_RE = re.compile("[​‌‍⁠﻿؂]")

# Rules existing + B (half/full width) + E1 (degree-like) + E2 (middle-dot).
# Hand-picked over NFKC because NFKC also compatibility-decomposes ` ´ ` and
# ` ˘ ` into space + combining marks, mangling face glyphs.
_TYPO_SUBS: tuple[tuple[str, str], ...] = (
    # --- existing arm/paren folds ---
    ("）", ")"),   # full-width close paren
    ("（", "("),   # full-width open paren
    ("ｃ", "c"),   # full-width Latin c (arm modifier)
    ("﹏", "_"),   # small wavy low line vs underscore
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
    """Rule D: lowercase Cyrillic capitals U+0410–U+042F."""
    return "".join(
        c.lower() if 0x0410 <= ord(c) <= 0x042F else c
        for c in s
    )
```

- [ ] **Step 2: Replace `canonicalize_kaomoji` to apply all rules in order**

Replace lines 282–305 (the existing `def canonicalize_kaomoji`) with:

```python
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
```

- [ ] **Step 3: Extend `sanity_check()` with cases for each new rule**

Replace the existing `def sanity_check()` block (lines 308–324) with:

```python
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
    # rule A: strip word-joiner / ZWSP
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
```

- [ ] **Step 4: Run the sanity check to verify all cases pass**

Run:

```bash
source .venv/bin/activate && python -m llmoji.taxonomy
```

Expected output: `taxonomy OK; 42 kaomoji registered (XX+/YY-)` (the existing summary), with no `AssertionError` raised. If any assertion fails, fix the implementation before proceeding — do NOT proceed to Task 2 with broken canonicalization.

- [ ] **Step 5: Commit**

```bash
git add llmoji/taxonomy.py
git commit -m "$(cat <<'EOF'
taxonomy: aggressive canonicalization rules A-E

Adds five new merge rules to canonicalize_kaomoji that collapse
cosmetic-only kaomoji variants:

  A. Invisible format chars (U+2060 word joiner, U+0602, ZWSP/ZWNJ/ZWJ).
  B. Half/full-width punctuation (>/＞, ;/；, etc.).
  C. Internal ASCII whitespace inside (...) brackets.
  D. Cyrillic case fold (Д -> д).
  E1. Degree-like glyphs (°/º/˚ all fold to °).
  E2. Middle-dot variants (・/･ both fold to ・).

Net effect on existing corpora: gemma 33 -> 32 forms (the (°Д°)
shocked-face merge), qwen 73 -> 65 forms (mostly the (；ω；) family
merging with ASCII-padded variants). Ministral and claude-faces
unchanged (verified during brainstorming).

Function remains idempotent. Plan:
docs/superpowers/plans/2026-04-25-aggressive-kaomoji-canonicalization.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Verify pre-registered merge counts on real data

**Files:**
- Read-only: `data/emotional_raw.jsonl`, `data/qwen_emotional_raw.jsonl`, `data/ministral_vocab_sample.jsonl`, `data/claude_kaomoji.jsonl`, `data/claude_faces_embed.parquet`

- [ ] **Step 1: Run the cross-corpus merge verification**

Run:

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
from collections import Counter
from llmoji.taxonomy import canonicalize_kaomoji

def load_first_words(path):
    p = Path('/Users/a9lim/Work/llmoji') / path
    if not p.exists(): return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            fw = r.get('first_word')
            if fw:
                out.append(canonicalize_kaomoji(fw))
        except: pass
    return out

expected = {
    'gemma':     ('data/emotional_raw.jsonl',           33, 32),
    'qwen':      ('data/qwen_emotional_raw.jsonl',      73, 65),
    'ministral': ('data/ministral_vocab_sample.jsonl',   9,  9),
}

ok = True
for label, (path, before, after) in expected.items():
    forms = load_first_words(path)
    n = len(set(forms))
    status = 'OK' if n == after else 'FAIL'
    if n != after:
        ok = False
    print(f'{label:10s}  N={len(forms):4d}  unique now={n:3d}  pre-registered={after:3d}  {status}')

# Also verify claude-faces parquet (uses canonicalize_kaomoji at load time)
import pandas as pd
df = pd.read_parquet('/Users/a9lim/Work/llmoji/data/claude_faces_embed.parquet')
fws = df['first_word'].dropna().tolist()
canon = set(canonicalize_kaomoji(f) for f in fws)
# claude-faces forms are stored RAW in the parquet — apply canonicalize and count
print(f'claude    N={len(fws):4d}  unique now={len(canon):3d}  '
      f'(check matches load_embeddings_canonical output)')

if not ok:
    raise SystemExit('FAIL: post-canonicalization counts do not match pre-registration')
print('all merge counts match pre-registration.')
"
```

Expected output:

```
gemma       N=XXX  unique now= 32  pre-registered= 32  OK
qwen        N=800  unique now= 65  pre-registered= 65  OK
ministral   N= 30  unique now=  9  pre-registered=  9  OK
claude    N=XXX  unique now= XXX  (check matches load_embeddings_canonical output)
all merge counts match pre-registration.
```

If gemma/qwen/ministral counts don't match the pre-registered numbers, **stop and investigate** — the canonicalization implementation deviates from the design. Don't proceed to Task 3.

- [ ] **Step 2: Verify claude-faces parquet still matches `load_embeddings_canonical`**

Run:

```bash
source .venv/bin/activate && python -c "
from llmoji.claude_faces import load_embeddings_canonical
df = load_embeddings_canonical()
print(f'claude_faces canonical: {len(df)} rows, '
      f'{df[\"first_word\"].nunique()} unique faces')
"
```

Expected: prints a count. The number is informational — record it for the CLAUDE.md update in Task 5. If the function raises, fix the implementation; otherwise continue.

- [ ] **Step 3: No commit** (verification-only task; no files changed.)

---

### Task 3: Regenerate gemma v3 figures + summary

**Files:**
- Re-generated: `data/emotional_summary.tsv`, `figures/fig_emo_a_kaomoji_sim.png`, `figures/fig_emo_b_kaomoji_consistency.png`, `figures/fig_emo_c_kaomoji_quadrant.png`, `figures/fig_v3_pca_valence_arousal.png`, `figures/fig_v3_face_pca_by_quadrant.png`, `figures/fig_v3_face_probe_scatter.png`, `figures/fig_v3_face_cosine_heatmap.png`

- [ ] **Step 1: Run the gemma v3 analysis suite (default `LLMOJI_MODEL=gemma`)**

Run:

```bash
source .venv/bin/activate && (
  echo "=== 04 ===" &&
  python scripts/04_emotional_analysis.py &&
  echo "=== 13 ===" &&
  python scripts/13_emotional_pca_valence_arousal.py &&
  echo "=== 17 ===" &&
  python scripts/17_v3_face_scatters.py
) 2>&1 | tee logs/gemma_v3_recanon.log
```

Expected: 04 prints "model: gemma", per-quadrant summary (form counts will reflect the merge — you'll see one fewer unique form than before), writes 3 figures + summary TSV. 13 prints PCA spectrum (look for changes vs baseline 13.0% / 7.5%; minor shift expected since only 1 form merges in gemma) + separation ratios (baseline 2.02 / 2.73). 17 prints face counts + writes 3 figures.

- [ ] **Step 2: Capture headline numbers from the log**

Run:

```bash
grep -E "PC[12]|separation|kaomoji-bearing|emission|Pearson|model:" logs/gemma_v3_recanon.log | head -30
```

These are the numbers that will go into the CLAUDE.md gemma findings refresh in Task 5.

- [ ] **Step 3: Verify figures regenerated**

Run:

```bash
ls -la figures/fig_emo_a_kaomoji_sim.png figures/fig_emo_b_kaomoji_consistency.png figures/fig_emo_c_kaomoji_quadrant.png figures/fig_v3_pca_valence_arousal.png figures/fig_v3_face_pca_by_quadrant.png figures/fig_v3_face_probe_scatter.png figures/fig_v3_face_cosine_heatmap.png
```

Expected: all 7 files exist with mtimes from the current run.

- [ ] **Step 4: Commit gemma figures + summary**

```bash
git add data/emotional_summary.tsv figures/fig_emo_a_kaomoji_sim.png figures/fig_emo_b_kaomoji_consistency.png figures/fig_emo_c_kaomoji_quadrant.png figures/fig_v3_pca_valence_arousal.png figures/fig_v3_face_pca_by_quadrant.png figures/fig_v3_face_probe_scatter.png figures/fig_v3_face_cosine_heatmap.png
git commit -m "$(cat <<'EOF'
v3 gemma: regenerate figures + summary under aggressive canonicalization

(°Д°) and (ºДº) now merge into a single (°д°) shocked-face form
under rules D + E1, taking gemma v3 from 33 -> 32 unique kaomoji.
PCA / separation-ratio shifts captured in CLAUDE.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Regenerate qwen v3 figures + summary

**Files:**
- Re-generated: `data/qwen_emotional_summary.tsv`, `figures/qwen/fig_emo_a_kaomoji_sim.png`, `figures/qwen/fig_emo_b_kaomoji_consistency.png`, `figures/qwen/fig_emo_c_kaomoji_quadrant.png`, `figures/qwen/fig_v3_pca_valence_arousal.png`, `figures/qwen/fig_v3_face_pca_by_quadrant.png`, `figures/qwen/fig_v3_face_probe_scatter.png`, `figures/qwen/fig_v3_face_cosine_heatmap.png`

- [ ] **Step 1: Run the qwen v3 analysis suite**

Run:

```bash
source .venv/bin/activate && (
  echo "=== 04 ===" &&
  LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py &&
  echo "=== 13 ===" &&
  LLMOJI_MODEL=qwen python scripts/13_emotional_pca_valence_arousal.py &&
  echo "=== 17 ===" &&
  LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
) 2>&1 | tee logs/qwen_v3_recanon.log
```

Expected: 04 prints "model: qwen", per-quadrant summary showing 65 unique forms (was 73). 13 prints PCA spectrum (compare baseline 14.9% / 8.3%) + separation ratios (baseline 2.34 / 1.93). 17 prints face counts (from 73 to 65 faces; n>=3 face count changes too — record this for CLAUDE.md) + writes 3 figures.

- [ ] **Step 2: Capture headline numbers from the log**

Run:

```bash
grep -E "PC[12]|separation|kaomoji-bearing|emission|Pearson|model:|faces by dominant" logs/qwen_v3_recanon.log | head -40
```

These are the comparable numbers — record explained variance, separation ratios, face-by-quadrant counts, and the cross-model probe-pair Pearson r for the CLAUDE.md refresh.

- [ ] **Step 3: Verify figures regenerated**

Run:

```bash
ls -la figures/qwen/
```

Expected: 7 PNG files with mtimes from the current run.

- [ ] **Step 4: Commit qwen figures + summary**

```bash
git add data/qwen_emotional_summary.tsv figures/qwen/
git commit -m "$(cat <<'EOF'
v3 qwen: regenerate figures + summary under aggressive canonicalization

73 -> 65 unique kaomoji. Major merges: (；ω；) family +14% (now n=82,
absorbed ( ; ω ; ) ASCII-padded variant), (；´д｀) family absorbed
both (；´Д｀) Cyrillic-case variant and ASCII-padded variant
(now n=70), (>_<) absorbed (＞_＜) (n=36), (´・ω・`) absorbed
(´･ω･`) (n=17), (◕‿◕✿) absorbed word-joiner-decorated variant
(n=16). PCA / separation-ratio shifts captured in CLAUDE.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update CLAUDE.md (canonicalization section + v3 findings)

**Files:**
- Modify: `CLAUDE.md` — three sections.

- [ ] **Step 1: Refresh the "Kaomoji canonicalization" section**

In `CLAUDE.md`, locate the `## Kaomoji canonicalization` heading (use `grep -n "Kaomoji canonicalization" CLAUDE.md` to find the exact line). Replace its body (the three-rule description through the `Effect: ...` paragraph) with:

```markdown
## Kaomoji canonicalization

`llmoji.taxonomy.canonicalize_kaomoji(s)` collapses cosmetic-only
kaomoji variants. Applied at load time in `load_emotional_features`
(v3) and `claude_faces.load_embeddings_canonical` (claude-faces).
Six rules:

1. **NFC normalize** (NOT NFKC — NFKC compatibility-decomposes `´` and
   `˘` into space + combining marks, mangling face glyphs).
2. **Strip invisible format characters**: ZWSP/ZWNJ/ZWJ (U+200B/C/D),
   WORD JOINER (U+2060), BOM (U+FEFF), and the U+0602 ARABIC FOOTNOTE
   MARKER that Qwen occasionally emits as a stray byte. The model
   sometimes interleaves U+2060 between every glyph of a kaomoji;
   `(⁠◕⁠‿⁠◕⁠✿⁠)` collapses to `(◕‿◕✿)`.
3. **Whitelisted typographic substitutions**:
   - Existing arm folds: `）`→`)`, `（`→`(`, `ｃ`→`c`, `﹏`→`_`, `ᴗ`→`‿`.
   - Half/full-width punctuation: `＞`→`>`, `＜`→`<`, `；`→`;`, `：`→`:`,
     `＿`→`_`, `＊`→`*`, `￣`→`~`.
   - Near-identical glyph folds (E1 + E2): `º`→`°`, `˚`→`°` (degree-like
     circular eyes), `･`→`・` (middle-dot fold). NOT `·`/`⋅` — those
     are smaller and could plausibly be a distinct register.
4. **Strip ASCII spaces inside the bracket span**: `( ; ω ; )` becomes
   `(;ω;)`. Only ASCII spaces; non-ASCII spacing characters are part
   of the face. Applied only when the form starts with `(` and ends
   with `)`.
5. **Lowercase Cyrillic capitals (U+0410–U+042F)**: `Д` → `д`. The two
   forms co-occur in the same `(；´X｀)` distressed-face skeleton at
   near-50/50 ratio in Qwen output, so the model isn't choosing
   between them semantically.
6. **Strip arm-modifier characters** from face boundaries: leading `っ`
   inside `(`, trailing `[ςc]` inside `)`, trailing `[ﻭっ]` outside `)`.
   Eye/mouth/decoration changes that aren't covered by rule 3 are
   preserved.

Effect: gemma v3 33 → 32 (the `(°Д°)` / `(ºДº)` shocked-face merge
under rules 5 + E1). Qwen v3 73 → 65 (the `(；ω；)` family
absorbed ASCII-padded variants and a Cyrillic-case `(；´Д｀)` ↔
`(；´д｀)` merger; the `(>_<)` ↔ `(＞_＜)` half/full-width pair
merged; the word-joiner-decorated `(◕‿◕✿)` variant merged).
Ministral pilot 9 → 9 (no merges available). Claude-faces 12 → 12
(no merges available; eriskii descriptions stay valid without
re-synthesis).

JSONL keeps raw `first_word`; `first_word_raw` column exists for
audit. Regenerate the per-kaomoji parquets / figures if the
canonicalization rule changes — claude-faces is a no-op as of
2026-04-25 since its vocabulary doesn't trigger any of the new
rules, but v3 figures regenerate.
```

- [ ] **Step 2: Refresh the gemma v3 findings**

Locate the `### Pilot v3 — naturalistic emotional disclosure (gemma)` subsection and its **Findings (post-refactor, hidden-state space)** block. Update the affected bullets to reflect the new numbers from `logs/gemma_v3_recanon.log` (Task 3 Step 2). Specifically:

- The line `**Findings (post-refactor, hidden-state space):**` stays.
- The "Hidden-state PCA on 800 row-level vectors: PC1 13.0%, PC2 7.5%" line — replace 13.0 / 7.5 with the new numbers from Task 3.
- The "Russell quadrants separate cleanly … Separation ratio 2.02 / 2.73" line — replace 2.02 / 2.73 with the new numbers.
- The "Within-kaomoji consistency to mean (h_mean, hidden-state space)" line — the `(°Д°)` and `(ºДº)` rows merge into a single `(°д°)` row, so the n=2 entry replaces the two n=1 entries. If consistency stats shift materially, update; otherwise leave the qualitative statement unchanged.
- Keep all other findings (HP/LP discrimination, kaomoji emission rates, the cross-quadrant `(｡•́︿•̀｡)` discussion) verbatim — those don't change with one form merging.

If the changes are minimal (numbers within ±0.1), add a one-line note at the end of the Findings block: `Re-run 2026-04-25 under aggressive canonicalization (rules A–E): 33 → 32 forms; PCA / separation numbers above reflect post-merge.`

If the changes are NOT minimal, write a fresh Findings block with the new numbers and note "(pre-aggressive-canonicalization numbers archived in commit dbbf676)" so the historical record stays linkable.

- [ ] **Step 3: Refresh the qwen v3 findings**

Locate the `### Pilot v3 — Qwen3.6-27B replication` subsection. Same approach as Step 2, but the merges are bigger so most numbers will shift. Update from `logs/qwen_v3_recanon.log` (Task 4 Step 2):

- "73 unique kaomoji forms" → "65 unique kaomoji forms"
- "2.2× broader vocabulary at the same N=800" → "2.0× broader vocabulary" (gemma 32 vs Qwen 65)
- "Faces by dominant quadrant HP 10 / LP 21 / HN 11 / LN 14 / NB 17" — re-derive from the Task 4 log (numbers will shift).
- "Russell-quadrant PCA: PC1 14.9%, PC2 8.3%" / "Separation ratios PC1 2.34 / PC2 1.93" — replace from Task 4.
- "Per-quadrant centroids in PC1/PC2: HP (-22.5, -30.3), LP (-15.4, -2.7), HN (+30.6, +21.1), LN (+33.9, -4.9), NB (-23.7, +29.4)" — re-derive (these will shift since underlying form set changes).
- "Cross-quadrant emitters analogous to gemma's `(｡•́︿•̀｡)`" block — the `(；ω；)` family is now n=82 (was 71); update the count and recompute the quadrant-membership split from the Task 4 face counts. The `(｡•́︿•̀｡)` n=22 entry stays. The `(；´д｀)` entry that merged with `(；´Д｀)` is now `(;´д｀)` with new count (≈n=70 across quadrants).
- "Default / cross-context form `(≧◡≦)` n=106" — likely unchanged; verify from Task 4 log.
- "Probe geometry diverges sharply" Pearson r block — the r-value will change slightly since the per-face means recompute over a slightly different face set. Update from Task 4 log.
- The "Procedural note" about TAXONOMY coverage — leave unchanged; it's about the runner's emission-rate log, not canonicalization.

If the changes invalidate any qualitative claim ("HP and LP discriminate cleanly," "PC1 separates valence more cleanly than activation," "Qwen is closer to a true 2D Russell circumplex"), re-read the figures (`figures/qwen/fig_v3_pca_valence_arousal.png`, `fig_v3_face_pca_by_quadrant.png`) and update the qualitative statements accordingly. If the qualitative story stays intact (most likely outcome — these are robust 1.5×+ separation ratios), keep the prose and just update the numbers.

- [ ] **Step 4: Commit CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
claude.md: refresh canonicalization section + v3 findings

Documents the five new merge rules (A-E) and updates the gemma + qwen
v3 findings sections with the post-merge PCA / separation numbers.
Claude-faces is unaffected; eriskii stays valid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

- ✓ Rule A (invisible format chars) — Task 1 Step 1 (`_INVISIBLE_CHARS_RE`).
- ✓ Rule B (half/full-width fold) — Task 1 Step 1 (`_TYPO_SUBS` extension).
- ✓ Rule C (internal whitespace) — Task 1 Step 2 (the `s[1:-1].replace(" ", "")` line).
- ✓ Rule D (Cyrillic case fold) — Task 1 Step 1 (`_cyrillic_lower`) + Step 2 (call site).
- ✓ Rule E1 (degree-like) — Task 1 Step 1 (`_TYPO_SUBS` extension).
- ✓ Rule E2 (middle-dot) — Task 1 Step 1 (`_TYPO_SUBS` extension).
- ✓ Pre-registered merge counts verified — Task 2.
- ✓ Gemma v3 figures regenerated — Task 3.
- ✓ Qwen v3 figures regenerated — Task 4.
- ✓ CLAUDE.md updated (canonicalization + gemma findings + qwen findings) — Task 5.
- ✓ Welfare framing in pre-registration block (no new generations).
- ✓ Out-of-scope items called out (eye/mouth class merges; Latin-Cyrillic lookalikes beyond the data; v1/v2 re-run).

**Placeholder scan:**

- No "TBD", "TODO", "implement later", "appropriate error handling".
- Every code step contains complete code; every command step has expected output or a clear pass/fail criterion.
- Task 5 Step 2/3 give explicit instructions for each line that needs updating, with the source numbers (Task 3/4 logs) to pull from.

**Type consistency:**

- `_INVISIBLE_CHARS_RE`, `_TYPO_SUBS`, `_cyrillic_lower`, `canonicalize_kaomoji` — all defined in Task 1 and used identically.
- `LLMOJI_MODEL=qwen` env var (Task 4) matches the existing project convention from the v3 Qwen plan.

**Adaptations from rigid TDD pattern:**

- Project has no test suite per CLAUDE.md ("No public API, no pypi release, no tests"). TDD-style "write failing test first" is replaced with the existing project pattern: assertions inside `sanity_check()` (Task 1 Step 3), invoked via `python -m llmoji.taxonomy` (Task 1 Step 4). This is the same pattern the existing code uses.
- The Task 2 cross-corpus verification IS the pre-registration check — it asserts the merge counts match the locked design rather than letting figure outcomes drive the rule selection post-hoc.
- Frequent commits preserved: each task ends in a commit; code, data, and docs are committed separately so a bad analysis result doesn't taint the canonicalization commit.
