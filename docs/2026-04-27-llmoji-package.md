# llmoji package + repo split

**Status:** plan, pre-registered. Not yet executing — gated on
canonicalization re-synthesis (see § v1.0 prereqs).

**Date:** 2026-04-27.

## Goal

Ship a provider-agnostic CLI (`llmoji`, on PyPI) that lets anyone
running a coding harness with a bash-shaped stop-event hook
(Claude Code, Codex, Hermes — and any other harness whose hook
contract is "run a script with the event payload on stdin") install
our kaomoji journal, run the locked Haiku description pipeline
locally, and submit a privacy-preserving aggregate (counts +
per-kaomoji synthesized descriptions, nothing else) back to us for
cross-corpus comparison with our own corpus and eriskii's published
numbers.

Non-bash hook architectures (notably OpenClaw, whose hooks are
TS/JS handlers receiving the payload as a function argument) are
deferred to post-v1.0. The generic JSONL-append contract still
covers them — an OpenClaw user can wire their own TS handler that
writes to `~/.llmoji/journals/openclaw.jsonl` against the published
6-field schema and `analyze` will pick it up — we just don't ship
a first-class adapter or `install` flow for them in v1.0.

In doing so, split this repo (currently `llmoji`, soon-to-be
`llmoji-study`) into two:

- **`llmoji-study`** (this repo, renamed) — research side. Local
  model steering / probes / hidden state, v1/v2/v3 pilots, MiniLM
  embedding, eriskii axis projection, all cross-pilot analyses and
  figures. Consumes both our own and submitted user bundles as
  input data. Heavy deps (saklas, torch, sentence-transformers,
  matplotlib).
- **`llmoji`** (new sibling repo at `../llmoji`, PyPI-published) —
  collection + canonicalization + Haiku synthesis + CLI. The
  end-user side. Provider-agnostic. Zero dependency on saklas, torch,
  sentence-transformers, or any analysis machinery; only `anthropic`
  (for Haiku) and `huggingface_hub` (for the upload target). The
  user's only local artifact is the bundle they'd ship.

The split is the right shape for a separate reason from the CLI: it
makes the public-facing data-layer code installable for non-research
users, while keeping the probe-vector / hidden-state machinery (which
needs saklas, GPU-class hardware, and is not particularly portable)
in research-only territory.

## Why split

Current `llmoji-study/llmoji/` is ~14 modules. The end-user side
needs only a handful of them (`taxonomy`, the source adapters,
`backfill_journals`, plus the Haiku prompts), and shouldn't drag in
saklas, torch, sentence-transformers, or matplotlib. Today the
package mixes both because everything was prototyped together.

Concrete consequences of mixing:

- Anyone who wants to install just the data-collection bits has to
  drag in saklas + torch + sentence-transformers + matplotlib.
- Locked artifacts (taxonomy, KAOMOJI_START_CHARS, Haiku prompts,
  canonicalization rules, system-injection prefix lists) live
  inside an unpackaged research repo that nobody outside this
  machine can import.
- Versioning is awkward — research-side iterations bump version
  numbers that the data-collection side doesn't care about.

After the split:

- `llmoji` ships the locked v1.0 public API. Bumping any of those
  artifacts is a major version bump and explicitly invalidates
  cross-corpus aggregation. Runtime deps: `anthropic`,
  `huggingface_hub`. That's it.
- `llmoji-study` `pip install -e ../llmoji` during dev; `pip install
  llmoji>=1.0,<2` at freeze. Imports `taxonomy`, `haiku_prompts`,
  the source adapters etc. from the package. All analysis
  primitives (MiniLM embedding, eriskii axis projection, the 21
  anchors, figures) live here, NOT in `llmoji`.
- Future users `pip install llmoji` and never touch our research
  repo. Their only output is the bundle. If they want figures or
  axes on their own data, they clone `llmoji-study` separately.

## Code migration map

Stays in `llmoji-study/llmoji/`:

- `config.py` — local-model registry, probe categories, paths
- `prompts.py`, `emotional_prompts.py` — pilot prompt sets
- `capture.py`, `hidden_capture.py`, `hidden_state_io.py`,
  `hidden_state_analysis.py`
- `analysis.py` (v1/v2 figures), `emotional_analysis.py` (v3
  figures), `cross_pilot_analysis.py`
- `claude_faces.py` — MiniLM-based per-kaomoji embedding (analysis
  primitive; research-side only)
- `eriskii.py` — axis projection + masking (analysis primitive;
  research-side only)
- The 21 anchored axes (currently in `eriskii_prompts.py`; split
  out to a new `eriskii_anchors.py` here when the Haiku prompts
  move to `llmoji`)
- All `scripts/00-04`, `10-13`, `17`, `99` — pilot drivers and
  analysis scripts that depend on saklas + local probes
- `scripts/05`, `06`, `07`, `08`, `09`, `14`, `15`, `16`, `18`, `21`
  stay as pipeline drivers; refactored to import scrape /
  taxonomy / Haiku-prompts from `llmoji` and continue using local
  embedding + axis-projection primitives from this repo

Moves to `../llmoji/llmoji/`:

- `taxonomy.py` — canonical home; KAOMOJI_START_CHARS lives here
  (resolves the "five copies in sync" gotcha; both shell hooks read
  from a generated config file the CLI writes)
- `claude_scrape.py` → `scrape.py`
- `claude_hook_source.py` → `sources/claude_code.py`
- `claude_export_source.py` → `sources/claude_export.py`
- `backfill_journals.py` → `backfill.py`
- The Haiku per-instance + synthesis prompts from
  `eriskii_prompts.py` → `haiku_prompts.py` (the 21-anchor list
  stays in `llmoji-study`; only the prompts move because only the
  prompts shape the description text that ships in the bundle)

Explicitly NOT moved (stays research-side):

- MiniLM-based `embed.py` (was `claude_faces.py`)
- `eriskii.py` axis projection
- 21-anchor list

The end-user pipeline produces *only* counts + Haiku-synthesized
descriptions per canonical kaomoji. Embedding and axis projection
happen exclusively on the research side, against either our own
corpus or submitted user bundles.

New in `../llmoji/llmoji/`:

- `providers.py` — `Provider` interface and concrete impls for
  `claude_code`, `codex`, `hermes` (see § providers)
- `cli.py` — entry point bound to `[project.scripts]` `llmoji`
- `bundle.py` — produces the share-bundle directory
- `upload.py` — HF + email targets
- `haiku.py` — thin wrapper around `anthropic` for the two-stage
  description / synthesis calls
- `_hooks/` — bash hook templates: `claude_code.sh.tmpl`,
  `codex.sh.tmpl`, `hermes.sh.tmpl`. Each contains the
  provider-specific `jq` extraction inline plus the shared
  kaomoji-check + journal-append body. `${KAOMOJI_START_CHARS}` and
  `${SYSTEM_INJECTED_PREFIXES}` are filled in at install time from
  `taxonomy.py` + the Provider definition.

## CLI surface

```
llmoji install <provider>      # writes hook + settings entry, idempotent
llmoji uninstall <provider>    # removes them, idempotent
llmoji status                  # installed providers, journal sizes, paths
llmoji parse --provider <name> <path>
                               # ingests static export dump (e.g.
                               # claude.ai data export) into the
                               # canonical journal as a per-source file
llmoji analyze                 # scrape + canonicalize + Haiku
                               # synthesize, write bundle to
                               # ~/.llmoji/bundle/ (loose files, NOT
                               # yet tarred). Per-instance Haiku
                               # descriptions cached at
                               # ~/.llmoji/cache/per_instance.jsonl
                               # so re-runs only hit the API on new
                               # rows. NO local embedding, NO axes,
                               # NO figures — those are research-side
                               # concerns and live in `llmoji-study`.
llmoji upload --target {hf,email}
                               # tarballs ~/.llmoji/bundle/, ships to
                               # the chosen target
```

`analyze` is the one-shot the user actually runs day-to-day; the
bundle landing on disk between `analyze` and `upload` is the
deliberate consent gap (user `ls`s / `cat`s the bundle dir before
deciding to ship).

`upload` requires `--target` (no default). For `--target email` the
CLI builds a `mailto:mx@a9l.im` URI with an attach hint in the body
and prints stdout instructions; we don't ship SMTP. For `--target
hf`, the CLI uses `huggingface_hub` to commit a row to a public HF
dataset (see § submission target).

## Provider abstraction

`Provider` interface:

```python
class Provider(Protocol):
    name: str                              # "claude_code" | "codex" | "hermes"
    hooks_dir: Path                        # ~/.claude/hooks etc.
    settings_path: Path                    # ~/.claude/settings.json etc.
    settings_format: Literal["json", "yaml"]
    journal_path: Path                     # ~/.<harness>/kaomoji-journal.jsonl
    kaomoji_position: Literal["first", "last", "single"]
    sidechain_strategy: Literal["none", "field_flag", "session_correlation"]
    sidechain_config: dict                 # strategy-specific (field name,
                                           # correlation event name, etc.)
    system_injected_prefixes: list[str]
    install(self) -> None
    uninstall(self) -> None
    status(self) -> ProviderStatus         # installed?, journal row count
```

First-class providers (built-in to v1.0, all bash-shaped):

1. **claude_code** — existing.
   - hook: `~/.claude/hooks/kaomoji-log.sh`, registered as Stop hook
     in `~/.claude/settings.json` (JSON)
   - kaomoji_position: `"first"` (interleaved content blocks; first
     text block is the kaomoji-bearing reply)
   - sidechain_strategy: `"field_flag"` with field `isSidechain`
   - system_injected_prefixes: `["Base directory for this skill:"]`

2. **codex** — existing.
   - hook: `~/.codex/hooks/kaomoji-log.sh`
   - kaomoji_position: `"last"` (per-message events; kaomoji lands
     on `task_complete.last_agent_message`, not first agent_message)
   - sidechain_strategy: `"none"` (no subagent equivalent;
     `collaboration_mode` is `"default"` for every observed turn)
   - system_injected_prefixes: `["# AGENTS.md",
     "<environment_context>", "<INSTRUCTIONS>"]`

3. **hermes** (NousResearch) — researched 2026-04-27 against
   hermes-agent v0.11.0.
   - hook: `~/.hermes/agent-hooks/kaomoji-log.sh`, registered under
     `hooks: { post_llm_call: [...] }` in `~/.hermes/config.yaml`
     (YAML)
   - event: `post_llm_call`, fires once per turn after the
     tool-calling loop. payload (stdin JSON): `hook_event_name`,
     `session_id`, `cwd`, and `extra` dict containing
     `user_message` (pre-injection original), `assistant_response`
     (final text), `model`, `platform`, `conversation_history`.
     all six llmoji fields land cleanly in one event.
   - kaomoji_position: `"single"` — single final-text field, no
     first/last ambiguity
   - sidechain_strategy: `"session_correlation"` — `post_llm_call`
     fires for both parent and child sessions; child sessions are
     identified by correlating `session_id` against `delegate_task`
     events. Concrete config: track child session_ids from a
     companion `subagent_stop` registration; drop `post_llm_call`
     events whose session_id matches a known child.
   - system_injected_prefixes: `[]` (empty — hermes delivers
     `user_message` pre-injection)
   - return contract: stdout JSON, `{}` is no-op, fail-open
     semantics (matches Claude). Default timeout 60s.
   - hook fail-open semantics: malformed JSON / non-zero exit /
     timeout never aborts the agent loop.

For everyone else (including OpenClaw): a published "generic JSONL
append" contract. The 6-field unified row schema (`ts, model, cwd,
kaomoji, user_text, assistant_text`), the leading-prefix filter
rule (≥2 bytes, first char ∈ KAOMOJI_START_CHARS, kaomoji is the
leading non-letter prefix), and the system-injection filter pattern.
Anyone whose harness can fire a stop event with assistant text can
write directly to `~/.llmoji/journals/<name>.jsonl` and `analyze`
picks it up. For bash hooks this is ~30 lines of shell; for TS-shaped
harnesses (OpenClaw) it's a similarly short handler that builds the
canonical row and appends. Documented in the README with a worked
OpenClaw example so the post-v1.0 path is concrete.

### Hook architecture: template-generated, taxonomy-sourced

Each provider's bash hook is generated at `install` time from a
template at `_hooks/<provider>.sh.tmpl` with `${KAOMOJI_START_CHARS}`
and `${SYSTEM_INJECTED_PREFIXES}` placeholders interpolated from
`taxonomy.py` and the Provider definition. Single source of truth
in Python; no hand-maintained shell. Resolves the existing
"KAOMOJI_START_CHARS lives in five places" gotcha.

The per-provider extraction (`jq` calls against the stdin payload)
is in the template itself — different for each harness because the
payload schemas are different. The shared kaomoji-check + journal-
append logic is the same template body across all three providers.

## Bundle format

`~/.llmoji/bundle/` after `analyze`:

```
manifest.json
  llmoji_version: "1.0.0"
  generated_at: "2026-..."
  providers_seen: ["claude_code", "codex"]
  total_rows_scraped: 1234
  total_kaomoji_unique_canonical: 87
  notes: free-form user opt-in field

descriptions.jsonl
  one row per canonical kaomoji (no count floor):
  {kaomoji, count, haiku_synthesis_description, llmoji_version}
```

Explicitly NOT in the bundle:

- raw `user_text` / `assistant_text`
- per-instance Haiku descriptions (only the synthesis ships)
- MiniLM embedding vectors (analysis-side; we re-embed on our end)
- per-axis projections (analysis-side; we re-project on our end)
- timestamps, cwds, model names per row

No count floor. Singletons ship. Filtering by `count` is an
analysis-time concern on our end (apply n≥3 / n≥5 / whatever during
clustering), and shipping the raw distribution preserves more
information for cross-corpus comparison than pre-filtering would.

Privacy note for the README: a singleton synthesis is a paraphrase
of a single user turn, and a topic-narrow corpus can leak through
it. The mitigation is the inspection gap, not a floor:

- `analyze` prints a per-face preview at the end (count + first ~80
  chars of synthesis per face) so the user sees what's about to be
  shippable
- `upload` re-prompts with the bundle path and asks for explicit
  confirmation before committing

The bundle landing as loose files in `~/.llmoji/bundle/` between
`analyze` and `upload` exists specifically so the user can `cat
descriptions.jsonl` and review.

## Submission targets

**HF dataset.** Public dataset at `a9lim/llmoji` (matching the
package name). Each submission becomes a row keyed by a salted hash
of (the user's submission token + llmoji_version). `upload --target
hf` uses `huggingface_hub.HfApi.upload_file` to write into a
contributor-named subfolder; the dataset README documents the
bundle schema and the v1.0 invariants (locked taxonomy, Haiku
prompts, journal row schema, etc.).

**Email.** `mailto:mx@a9l.im` with the bundle tarball as an
attach hint. CLI prints "tarball at <path>; opening your mail
client now; please attach manually." For users who don't want to
make their submission public.

Submitter chooses; no default.

## Privacy

Levels:

1. **Local journal** (`~/.claude/kaomoji-journal.jsonl` etc.) —
   contains real user prompts and assistant text. NEVER leaves the
   machine. Live hook writes here on every kaomoji-bearing turn.
2. **Per-instance description cache**
   (`~/.llmoji/cache/per_instance.jsonl`) — Haiku-generated
   paraphrases of single turns, keyed by content-hash of
   (canonical_kaomoji + user_text + assistant_text). Written to
   disk so `analyze` re-runs don't re-hit the API on unchanged
   rows. NEVER bundled; NEVER shipped. Higher leak surface than the
   bundle since each row IS one user turn paraphrased.
3. **Bundle** (`~/.llmoji/bundle/`) — per-canonical-kaomoji counts
   + Haiku syntheses, no count floor. The artifact `analyze` writes
   for `upload` to consume. Designed for the user to review before
   sharing. Inspected by user between `analyze` and `upload`.

The cache directory is the leakier-but-non-shippable disk tier. The
package documents it clearly, includes a `~/.llmoji/.gitignore` on
first run, and `llmoji status` prints the cache size + entry count
so the user is aware it exists. `llmoji uninstall <provider>` does
not touch the cache; a separate `llmoji cache clear` is worth
considering (cheap to add, gives users a one-shot wipe).

Threat model: the per-face Haiku synthesis is the highest-leakage
artifact in the bundle. For frequent forms (n large), the synthesis
abstracts across many contexts and shouldn't single out any
particular turn. For singletons, the synthesis IS a paraphrase of
one user turn, which can leak in topic-narrow corpora. Mitigations:
(a) the synthesis prompt is locked and abstracts to
functional/affective descriptors rather than verbatim quote, (b)
per-instance descriptions are NOT bundled (only the synthesized
one), (c) `analyze` prints a per-face preview (count + first ~80
chars of synthesis) so the user sees what's about to be shippable,
(d) `upload` re-prompts with the bundle path before committing.

We deliberately do NOT impose a count floor. Filtering is an
analysis-time concern (we apply n≥3 etc. on our end during
clustering); pre-filtering at submission time would lose vocabulary
breadth without meaningfully changing the privacy properties of any
*specific* description that ships.

Future hardening if leakage becomes a real concern: synthesize
per-face descriptions over instances drawn across multiple
contributors (server-side), so individual contributions never
appear as their own face-row. Out of scope for v1.0.

## v1.0 prereqs (gating items)

Before any code in `../llmoji` is cut:

1. **Re-Haiku-synthesize the 16 merge groups in
   `data/claude_haiku_synthesized.jsonl`.** Currently stale w.r.t.
   the 2026-04-25 aggressive canonicalization (rules A–E now in
   `taxonomy.canonicalize_kaomoji`). The eriskii outputs
   (`eriskii_axes.tsv`, `eriskii_clusters.tsv`,
   `eriskii_per_*.tsv`, `eriskii_user_kaomoji_axis_corr.tsv`,
   `figures/eriskii_*`, `figures/claude_faces_interactive.html`)
   are also stale.
2. **Regenerate the eriskii pipeline end-to-end** so we have a
   canonicalization-correct reference corpus.
3. **Lock the public API surface in v1.0:**
   - `taxonomy.py` (KAOMOJI_TAXONOMY, KAOMOJI_START_CHARS,
     `canonicalize_kaomoji` rules — A–P as of 2026-04-27,
     equivalence-class structure with eye/mouth/decoration folds)
   - `is_kaomoji_candidate` validator contract
   - Haiku per-instance description prompt + synthesis prompt
     (these shape what description text ships in the bundle)
   - System-injection prefix lists per provider
   - The 6-field unified journal row schema
   - The `Provider` interface
   - The bundle schema

   Bumping any of these is a major version bump (`llmoji` 2.0.0)
   and the dataset README declares "v1 corpus only" aggregation
   rules.

   Not in the freeze (free to change without bumping the public
   API): MiniLM id, 21-axis anchors, any count floor, cache key
   derivation.

4. **Hermes API empirical validation.** The 2026-04-27 doc-read
   resolved most questions (live `post_llm_call` hook, payload
   schema, sidechain strategy). Three items still need an
   empirical pass before the hermes provider ships:
   - Run an echo-the-stdin hook against `post_llm_call` and
     confirm the exact `extra.*` keys (docs example block was for
     `pre_tool_call`).
   - Confirm the session-correlation sidechain strategy works
     against actual `delegate_task` traffic.
   - Verify `user_message` arrives clean (no system-injection
     prefixes that need filtering).

After (1)–(4) are done, the actual code work begins.

## Sequencing

In rough order, after gating items clear:

1. `git mv` rename this repo to `llmoji-study` (local + GitHub).
   Update `pyproject.toml` package name; update `CLAUDE.md` "What
   this is" line.
2. Create `../llmoji/` repo skeleton (pyproject, README, MIT
   license matching this repo, src layout, GitHub remote).
3. Move modules per the migration map. In `llmoji-study`, replace
   the moved imports with `from llmoji import …`. Verify the v3
   pipeline still runs on gemma + qwen end-to-end.
4. Implement `Provider` interface; refactor existing claude + codex
   hooks behind it. Generate the shell hook scripts from templates
   so KAOMOJI_START_CHARS / system-injected-prefixes come from
   `taxonomy.py` rather than being hand-maintained in shell.
5. Implement `install` / `uninstall` / `status` for claude_code +
   codex. End-to-end smoke test: clean macOS user, no existing
   hooks, run install, generate a CC turn with kaomoji, see journal
   row.
6. Implement `parse --provider claude.ai` (start here because it's
   pure file-IO, easy to test against existing exports).
7. Implement `analyze`. Two-stage Haiku pipeline (per-instance
   description cached at `~/.llmoji/cache/per_instance.jsonl` keyed
   by content-hash of canonical_kaomoji + user_text + assistant_text,
   then per-kaomoji synthesis written to the bundle) plus the
   canonicalization + count pass. The MiniLM / axis-projection
   stages from `scripts/15` and `scripts/16` are NOT part of
   `analyze` — they stay research-side. Reuse
   `scripts/14_claude_haiku_describe.py` for the prompt template
   and the API-call pattern; the canonicalization comes from the
   moved `taxonomy.py`.
8. Implement `bundle`-write step inside `analyze`. Verify bundle is
   exactly the documented schema.
9. Implement `upload --target email` first (cheap, no HF auth).
10. Implement `upload --target hf` against a private test dataset,
    then promote to `a9lim/llmoji`.
11. Hermes provider implementation, including the empirical
    validation items in the v1.0 prereqs section.
12. **Write `../llmoji/CLAUDE.md`** (new repo) and **update
    `llmoji-study/CLAUDE.md`** (this repo) to reflect the split.
    Comes near the end of sequencing because it documents the
    *implemented* surface (interface signatures, command flags,
    failure modes that emerged during smoke-testing) rather than
    the design intent.

    `../llmoji/CLAUDE.md` covers:
    - what the package is — provider-agnostic CLI for kaomoji
      journal collection + canonical Haiku synthesis +
      privacy-preserving bundle/upload; runtime deps `anthropic` +
      `huggingface_hub`, nothing else
    - architecture overview — `Provider` interface, hook templates,
      `taxonomy.canonicalize_kaomoji` equivalence-class structure,
      `is_kaomoji_candidate` validator, two-stage Haiku
      description→synthesis pipeline (with per-instance cache),
      bundle + upload
    - the v1.0 frozen public surface (taxonomy + Haiku prompts +
      system-injection prefixes + journal schema + Provider
      interface + bundle schema) and the "bumping any of this is
      2.0" policy
    - gotchas migrated from this repo's CLAUDE.md:
      KAOMOJI_START_CHARS sync (resolved via templating, but
      document the resolution); per-provider quirks (Claude
      first-block vs Codex last-message vs Hermes single-event
      payload); sidechain-strategy interface; cache-directory
      privacy (leakier-than-bundle disk tier)
    - commands: `install <provider>`, `uninstall <provider>`,
      `status`, `parse --provider <name> <path>`, `analyze`,
      `upload --target {hf,email}`, `cache clear`
    - layout: src tree, hook templates, tests, pyproject

    `llmoji-study/CLAUDE.md` updates:
    - top-of-file pointer: "for the data-collection + canonical
      Haiku side, see `../llmoji/CLAUDE.md`"
    - remove the now-migrated sections (taxonomy.py details, hook
      setup, scrape pipeline, Haiku prompts, eriskii_prompts) —
      replace with one-line summaries pointing at the package
    - update Layout to show `from llmoji import …` for the moved
      modules and explicit listing of what's still here (probes,
      hidden state, MiniLM embedding, eriskii axis projection,
      figures, all pilot scripts)
    - update Commands to reflect `pip install -e ../llmoji` during
      dev / `pip install llmoji>=1.0,<2` at freeze
    - keep all probe / hidden-state / pilot / figure conventions
      and gotchas — those are research-side and stay here
13. v1.0 freeze + PyPI publish + HF dataset launch.

Post-v1.0 / future work: OpenClaw first-class support (TS adapter
+ session-state lookup against the `sessionKey` SDK API). Until
then, OpenClaw users follow the generic JSONL contract path.

## Open questions

- Should `parse` accept multiple paths in one invocation? Probably
  yes — symmetry with how `06_claude_scrape.py` already handles
  multiple Claude.ai export dirs.
- Should `status` print kaomoji emission rate (computed from journal
  size) as a sanity check that the hook is actually firing? Probably
  yes — cheap, helps debugging.
- v1.0 README: how much of the experimental motivation do we explain
  vs. just the "here's how to install + submit" tutorial? The
  research framing belongs in `llmoji-study`'s README; the package
  README should be tutorial-first.
- Worked OpenClaw example for the generic-JSONL-contract README
  section: write a 30-line TS handler that registers on
  `message:sent`, looks up session state, and appends a canonical
  row to `~/.llmoji/journals/openclaw.jsonl`. Lets motivated users
  participate without us shipping a first-class adapter.
