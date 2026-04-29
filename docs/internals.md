# Internals

Two technical-infrastructure surfaces — the hidden-state sidecar pipeline
and the kaomoji canonicalization rules. Treat both as load-bearing for
every v3 analysis script.

## Hidden-state pipeline

After `session.generate()`,
`llmoji.hidden_capture.read_after_generate(session)` reads saklas's per-token
last-position buckets and writes `(h_first, h_last, h_mean, per_token)` per
probe layer to `data/hidden/<experiment>/<row_uuid>.npz`. ~20–70 MB per row;
gitignored; regenerable from the runners. JSONL keeps probe scores for
back-compat and audit.

`llmoji.hidden_state_analysis.load_hidden_features(...)` returns
`(metadata df, (n_rows, hidden_dim) feature matrix)`. Defaults: `which="h_mean"`
(whole-generation aggregate; smoother and more probative than `h_last`),
`layer=None` (deepest probe layer). All v3 figures use `h_mean`.

## Kaomoji canonicalization

`llmoji.taxonomy.canonicalize_kaomoji(s)` collapses cosmetic-only variants.
Applied at load time in `load_emotional_features` (v3) and
`claude_faces.load_embeddings_canonical`. Six rules (extended 2026-04-25 from
three to six after Qwen revealed substantial cosmetic variation):

1. **NFC normalize** (NOT NFKC — NFKC compatibility-decomposes `´` and `˘`
   into space + combining marks, mangling face glyphs).
2. **Strip invisible format characters**: ZWSP/ZWNJ/ZWJ (U+200B/C/D), WORD
   JOINER (U+2060), BOM (U+FEFF), and the U+0602 ARABIC FOOTNOTE MARKER Qwen
   occasionally emits as a stray byte. Model sometimes interleaves U+2060
   between every glyph; `(⁠◕⁠‿⁠◕⁠✿⁠)` collapses to `(◕‿◕✿)`.
3. **Whitelisted typographic substitutions**: arm folds (`）`→`)`, `（`→`(`,
   `ｃ`→`c`, `﹏`→`_`, `ᴗ`→`‿`); half/full-width punctuation (`＞`→`>`,
   `＜`→`<`, `；`→`;`, `：`→`:`, `＿`→`_`, `＊`→`*`, `￣`→`~`); near-identical
   glyph folds (`º`→`°`, `˚`→`°`, `･`→`・`). NOT `·`/`⋅` — those are smaller
   and could plausibly be a distinct register.
4. **Strip ASCII spaces inside the bracket span**: `( ; ω ; )` → `(;ω;)`.
   ASCII spaces only; non-ASCII spacing is part of the face. Applied only
   when the form starts with `(` and ends with `)`.
5. **Lowercase Cyrillic capitals** (U+0410–U+042F): `Д` → `д`. Two forms
   co-occur in the same `(；´X｀)` distressed-face skeleton at near-50/50 in
   Qwen, so the model isn't choosing semantically.
6. **Strip arm-modifier characters** from face boundaries: leading `っ`
   inside `(`, trailing `[ςc]` inside `)`, trailing `[ﻭっ]` outside `)`.
   Eye/mouth/decoration changes not covered by rule 3 are preserved.

Effect on form counts:
- Gemma v3: 42 raw → **32** canonical (the `(°Д°)` / `(ºДº)` shocked-face
  pair merged under rule 5 + glyph-fold). Single-form merge doesn't move
  the 800-row PCA materially.
- Qwen v3: 73 raw → **65** canonical. Big merges: `(；ω；)` family absorbed
  ASCII-padded variants → n=82, `(;´д｀)` group merged Cyrillic-case +
  ASCII-pad variants → n=70, `(>_<)` ↔ `(＞_＜)` → n=36, `(◕‿◕✿)` ↔
  word-joiner-decorated → n=16, `(´・ω・`)` ↔ `(´･ω･`)` → n=17.
- Ministral pilot: 9 → 9 (no merges available at this N).
- Claude-faces: contributor-side canonicalization in `llmoji analyze`
  before upload; `06_claude_hf_pull.py` re-canonicalizes on the way in
  (in case bundles were produced under different package versions). The
  pre-refactor 160 → 144 row collapse no longer applies — corpus arrives
  canonical.

JSONL keeps raw `first_word`; `first_word_raw` column exists for audit on
v1/v2/v3 data. Regenerate per-kaomoji parquets and figures if the rule
changes.
