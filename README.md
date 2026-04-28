# llmoji-study

Does a language model's choice of kaomoji track something about its
internal state? Claude is often asked to begin each message with a
kaomoji that reflects how it currently feels, and the question
naturally follows: is that choice actually coupled to whatever's
going on inside the model, or is it surface statistics with
emotional-sounding tokens mixed in? This repo answers the question
from two angles. The local side runs probes and activation steering
on open-weight causal LMs via [`saklas`](https://github.com/a9lim/saklas),
where I can read and intervene on the hidden state directly. The
harness side does an [eriskii](https://eriskii.net/projects/claude-faces)-style
semantic-axis replication on real contributor-submitted Claude and
Codex kaomoji, pulled from the
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji)
HuggingFace dataset.

> **Companion package**: the contributor-side data collection
> (per-harness Stop hooks, kaomoji journals, Haiku synthesis,
> bundle-and-upload CLI) is the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package. This
> repo doesn't scrape any local data; it pulls the aggregated
> corpus from the HF dataset.
>
> **Prior art**: [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces)
> is the original post that came up with the idea of prompting
> Claude with kaomoji and analyzing the resulting vocabulary.
> The harness-side replication here uses eriskii's 21 semantic
> axes and their two-stage Haiku pipeline.

## How this is organized

The two sides are independent enough that they live in their own
docs:

- [`docs/local-side.md`](docs/local-side.md): probes, steering, and
  hidden-state analysis on `gemma-4-31b-it`, `Qwen3.6-27B`, and
  `Ministral-3-14B-Instruct-2512`. Pilots v1, v2, v3.
- [`docs/harness-side.md`](docs/harness-side.md): eriskii-replication
  on the contributor-submitted Claude and Codex corpus. Pulls from
  the HF dataset, embeds Haiku-synthesized per-face descriptions,
  projects onto 21 semantic axes, runs t-SNE plus KMeans clustering.

Engineering notes, gotchas, and the design and plan docs live in
[`CLAUDE.md`](CLAUDE.md) and [`docs/`](docs/).

## Headline findings

### Local side

Steering on `gemma-4-31b-it` is a clean causal handle on kaomoji
choice. In pilots v1 and v2, steering on `happy.sad` collapses the
emitted distribution: 0% happy-labeled kaomoji under sad-steering,
100% under happy-steering, with 71% in the unsteered middle. The
shift is monotonic and the effect is selective to the targeted axis
(orthogonal probes barely move). Within the unsteered arm, however,
the probe scalar at token 0 only weakly predicts which kaomoji the
model emits, because saklas's bundled `happy.sad` and `angry.calm`
probes both extract the same lexical-valence direction (the v1 and
v2 valence-collapse).

Pilot v3 (naturalistic, no steering, hidden-state space instead of
probe space) recovers the second affective dimension that the
probes miss. On 800 generations balanced across the five Russell
quadrants, hidden-state PCA gives PC1 13% and PC2 7.5%, and the
quadrants separate cleanly: PC1 reads as valence, PC2 as activation,
within-kaomoji consistency to mean is 0.92 to 0.99 across faces. The
Qwen3.6-27B replication on the same prompts has 2x the kaomoji
vocabulary (65 canonical forms vs 32), separates the quadrants
similarly cleanly, and reveals a structural divergence between the
two models. Qwen represents arousal independently within each
valence half (anti-parallel arousal axes for positive and negative
clusters), while gemma collapses to roughly one shared arousal axis
modulated by valence. The bundled saklas probes that anti-aligned
on gemma (Pearson r between mean `happy.sad` and `angry.calm` was
−0.94) come out near-orthogonal on Qwen (r = −0.12), so the
valence-collapse problem is a property of how gemma's affect
representation is laid out, not a saklas issue.

Full setup, decision rules, per-quadrant centroids, all the
cross-model comparisons, and the Ministral vocabulary pilot are in
[`docs/local-side.md`](docs/local-side.md).

### Harness side

The contributor-submitted corpus on
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) holds
one row per (machine, kaomoji) pair, where each row is a
Haiku-synthesized one-sentence meaning aggregated across that
machine's instances of the face. The research-side pipeline pulls
the dataset, pools by canonical kaomoji form across contributors,
embeds the synthesized descriptions with `all-MiniLM-L6-v2`, and
projects onto eriskii's 21 semantic axes plus runs t-SNE with
KMeans clustering and Haiku-synthesized cluster labels.

First pull through the new pipeline (one contributor, n=808
emissions, 174 canonical kaomoji): top-20 frequency overlap with
eriskii's published vocabulary is 14/20, and the 15 KMeans cluster
themes line up with eriskii's 15 at the register level
(warm-supportive, wry, empathetic, sheepish, eager, thoughtful).
The `wetness` axis is a9's rewrite of eriskii's
intentionally-undefined `wetness ↔ dryness` joke; rankings on that
axis are more meaningful than eriskii's but not directly comparable.
Per-model and per-project axis breakdowns and the
`surrounding_user → kaomoji` mechanistic-bridge correlation are
gone in the HF refactor (the public dataset pools per-machine
before upload, so the per-row metadata those analyses needed isn't
available). Pre-refactor those analyses confirmed eriskii's
qualitative "opus-4-6 had wider range" claim numerically and showed
that surprise (r = +0.20) and curiosity (r = +0.18) were the only
two of 21 axes where MiniLM on user text correlated with kaomoji
projection past Bonferroni; the historical numbers are in
[`docs/harness-side.md`](docs/harness-side.md). Multi-contributor
numbers will land here as more bundles arrive.

Full pipeline, methodology, axis anchors, and the historical
pre-refactor cross-cuts are in
[`docs/harness-side.md`](docs/harness-side.md).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .  # pulls llmoji>=1.0,<2 from PyPI plus saklas, sentence-transformers, ...
```

For the local side, set `LLMOJI_MODEL=gemma|qwen|ministral` and run
`scripts/03_emotional_run.py` (the v3 800-generation runner).
[`docs/local-side.md`](docs/local-side.md) has the per-pilot script
chain.

For the harness side, you need an `ANTHROPIC_API_KEY` (the cluster
labeler calls Haiku) and the HF Hub Python client (the install
above pulls `huggingface_hub`). Anonymous reads of the public
dataset are fine, so `HF_TOKEN` is optional.

```bash
python scripts/06_claude_hf_pull.py            # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/07_claude_kaomoji_basics.py     # printout: top kaomoji, providers, contributors
python scripts/15_claude_faces_embed_description.py  # per-canonical embeddings
python scripts/16_eriskii_replication.py       # axes, clusters, writeup
python scripts/18_claude_faces_pca.py          # PCA panel
```

## Related

- [`saklas`](https://github.com/a9lim/saklas): the engine. Activation
  steering and trait monitoring on HuggingFace causal LMs via
  contrastive-PCA. The local side is a study built on top of saklas.
- [`llmoji`](https://github.com/a9lim/llmoji): the PyPI package
  that runs Stop hooks on coding agents (Claude Code, Codex,
  Hermes), keeps a per-machine kaomoji journal, runs the two-stage
  Haiku synthesis, and uploads the result to the shared corpus.
- [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji): the
  HF dataset. Contributor-submitted kaomoji counts and synthesized
  meanings, CC-BY-SA-4.0.
- [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces):
  the prior art for the kaomoji-cataloging idea, the 21-axis
  projection scheme, and the two-stage Haiku pipeline.

## License

AGPL-3.0-or-later for this repo. See [LICENSE](LICENSE). The
companion package
[`llmoji`](https://github.com/a9lim/llmoji) is GPL-3.0-or-later.
The shared corpus on
[HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is
CC-BY-SA-4.0.
