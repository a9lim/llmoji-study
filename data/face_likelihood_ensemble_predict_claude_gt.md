# Ensemble per-face distributions

**Encoders:** gemma_v7primed, haiku  (sources: {'gemma_v7primed': 'full', 'haiku': 'full'})
**Faces predicted:** 652
**Faces with GT (for evaluation):** 128
**GT mode:** Claude empirical (total ≥ 1)

## Methodology

For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes). GT is Claude's (or pooled) empirical per-quadrant emission distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report **distribution similarity** = `1 − JSD/ln 2` ∈ [0, 1] (1.0 = identical; max JSD ≈ 0.6931 nats). The deployable output is the *full distribution per face* — "this face is 56% HP, 23% LP, ..." — not a single hard label.

## Headline

- **Face-uniform mean similarity:** 0.652  (each face counts equally; characterizes Claude's *vocabulary*)
- **Emit-weighted mean similarity:** 0.801  (faces weighted by how often Claude emits them; characterizes Claude's *emission distribution* — closer to deployment-relevance)
  - n_faces evaluated: 128
  - mean JSD: 0.2414 (face-uniform), 0.1381 (emit-weighted) nats

## Per-GT-modal-quadrant breakdown

| GT modal | n | similarity (face-uniform) | similarity (emit-weighted) |
|---|---:|---:|---:|
| HP | 28 | 0.690 | 0.834 |
| LP | 35 | 0.662 | 0.780 |
| HN-D | 10 | 0.638 | 0.843 |
| HN-S | 22 | 0.622 | 0.685 |
| LN | 12 | 0.732 | 0.893 |
| NB | 21 | 0.577 | 0.740 |

## Output schema (per-face TSV)

Each row carries:

- `ensemble_p_<q>` for q in {HP, LP, HN-D, HN-S, LN, NB} — **the headline output**, the full ensemble distribution.
- `gt_p_<q>` (when GT exists) — Claude's empirical distribution for the same face.
- `jsd_vs_gt`, `similarity` — per-face evaluation.
- `<encoder>_pred`, `<encoder>_conf` — per-encoder argmax + top-1 mass (for transparency about individual contributors).
- Supplementary: `ensemble_pred` (argmax of distribution), `ensemble_conf` (top-1 mass), `argmax_matches_gt_modal` (boolean). These are *derived* from the distribution; they're the production-shaped reading but not the primary output.

## Supplementary metrics (argmax-shaped reading)

- Hard accuracy (argmax matches GT modal): 57.0% (73/128)
- Cohen's κ on argmax: 0.480

These characterize a *deployed plugin that emits a single quadrant call*. They lose information at GT-tie boundaries and aren't the headline.

