# Ensemble per-face distributions

**Encoders:** gemma_intro_v7_primed, opus  (sources: {'gemma_intro_v7_primed': 'full', 'opus': 'full'})
**Faces predicted:** 684
**Faces with GT (for evaluation):** 203
**GT mode:** pooled v3+Claude+wild (total ≥ 3)

## Methodology

For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes). GT is Claude's (or pooled) empirical per-quadrant emission distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report **distribution similarity** = `1 − JSD/ln 2` ∈ [0, 1] (1.0 = identical; max JSD ≈ 0.6931 nats). The deployable output is the *full distribution per face* — "this face is 56% HP, 23% LP, ..." — not a single hard label.

## Headline

- **Face-uniform mean similarity:** 0.728  (each face counts equally; characterizes Claude's *vocabulary*)
- **Emit-weighted mean similarity:** 0.843  (faces weighted by how often Claude emits them; characterizes Claude's *emission distribution* — closer to deployment-relevance)
  - n_faces evaluated: 203
  - mean JSD: 0.1882 (face-uniform), 0.1088 (emit-weighted) nats

## Per-GT-modal-quadrant breakdown

| GT modal | n | similarity (face-uniform) | similarity (emit-weighted) |
|---|---:|---:|---:|
| HP | 35 | 0.747 | 0.829 |
| LP | 46 | 0.755 | 0.828 |
| HN-D | 22 | 0.680 | 0.858 |
| HN-S | 34 | 0.700 | 0.842 |
| LN | 34 | 0.683 | 0.863 |
| NB | 32 | 0.782 | 0.842 |

## Output schema (per-face TSV)

Each row carries:

- `ensemble_p_<q>` for q in {HP, LP, HN-D, HN-S, LN, NB} — **the headline output**, the full ensemble distribution.
- `gt_p_<q>` (when GT exists) — Claude's empirical distribution for the same face.
- `jsd_vs_gt`, `similarity` — per-face evaluation.
- `<encoder>_pred`, `<encoder>_conf` — per-encoder argmax + top-1 mass (for transparency about individual contributors).
- Supplementary: `ensemble_pred` (argmax of distribution), `ensemble_conf` (top-1 mass), `argmax_matches_gt_modal` (boolean). These are *derived* from the distribution; they're the production-shaped reading but not the primary output.

## Supplementary metrics (argmax-shaped reading)

- Hard accuracy (argmax matches GT modal): 54.2% (110/203)
- Cohen's κ on argmax: 0.445

These characterize a *deployed plugin that emits a single quadrant call*. They lose information at GT-tie boundaries and aren't the headline.

