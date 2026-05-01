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

The gemma-tuned ``TAXONOMY`` / ``ANGRY_CALM_TAXONOMY`` happy-sad
labels lived at :mod:`llmoji_study.taxonomy_labels` between the v1.0
package split and 2026-04-30. Dropped in the post-pilot cleanup —
v3 analyses key off ``first_word`` (canonicalized) and v1/v2 pole
assignment is now data-driven from per-face mean ``t0_<axis>`` probe
score sign (see ``llmoji_study.analysis._add_axis_label_column``).
The change generalizes pole labeling across models that didn't
share gemma's vocabulary.

Modules previously here that moved to ``llmoji`` (the public
package): ``taxonomy`` (the canonicalization rules + KAOMOJI_START_CHARS
+ is_kaomoji_candidate + slim ``extract``), ``claude_scrape``,
``claude_hook_source``, ``claude_export_source``,
``backfill_journals`` (now ``llmoji.scrape`` / ``llmoji.sources.*`` /
``llmoji.backfill``), and the description / synthesis prompt
strings (now ``llmoji.synth_prompts``; renamed from
``llmoji.haiku_prompts`` in the 1.1 split).
"""

__version__ = "0.1.0"
