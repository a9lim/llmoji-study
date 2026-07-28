# Figures

The public figure tree is tracked:

- `local/<model>/` — hidden-state geometry, prompt-grouped predictiveness, and
  face-likelihood views for each open-weight model.
- `local/gemma_self_event*` and `local/gemma_true_self/` — explicitly
  single-model self-event and prefill pilots.
- `harness/` — Claude-GT, BoL, face-union, and ensemble diagnostics.

Privacy-sensitive per-project and local deployment-telemetry plots remain
explicitly gitignored. Their absence is intentional and documented in
`.gitignore`; aggregate public findings do not depend on exposing them.

Headline values and taxonomy scope are in
[`data/summary/results.json`](../data/summary/results.json). Detailed provenance
and caveats are in [`docs/findings.md`](../docs/findings.md).
