"""llmoji-study — research-side companion to the ``llmoji`` PyPI package.

Renamed from ``llmoji`` after the v1.0 split (see
``docs/2026-04-27-llmoji-package.md``). The package namespace was
moved to ``llmoji_study`` so this repo can ``pip install llmoji>=1.0,<2``
for the data-collection / canonical-Haiku / bundle-and-upload side
without colliding with our own ``llmoji.*`` namespace.

Modules here are everything that wasn't migrated to the
provider-agnostic CLI:

  - :mod:`llmoji_study.config` — local-model registry, probe
    categories, dataset paths.
  - :mod:`llmoji_study.prompts` / :mod:`llmoji_study.emotional_prompts` —
    pilot prompt sets.
  - :mod:`llmoji_study.capture` / :mod:`llmoji_study.hidden_capture` /
    :mod:`llmoji_study.hidden_state_io` /
    :mod:`llmoji_study.hidden_state_analysis` — saklas + hidden-
    state pipeline.
  - :mod:`llmoji_study.analysis` / :mod:`llmoji_study.emotional_analysis` /
    :mod:`llmoji_study.cross_pilot_analysis` — pilot analyses + figures.
  - :mod:`llmoji_study.claude_faces` — MiniLM-based per-kaomoji
    embedding (research-side analysis primitive).
  - :mod:`llmoji_study.eriskii` — axis projection + masking + Haiku
    primitives.
  - :mod:`llmoji_study.eriskii_anchors` — 21 anchored axes (positive
    and negative anchor strings).

Modules previously here that moved to ``llmoji`` (the package):
``taxonomy`` (canonical kaomoji rules + start-chars + extractor),
``claude_scrape``, ``claude_hook_source``, ``claude_export_source``,
``backfill_journals`` (now ``llmoji.scrape`` / ``llmoji.sources.*`` /
``llmoji.backfill``), and the Haiku prompt strings (now
``llmoji.haiku_prompts``).
"""

__version__ = "0.1.0"
