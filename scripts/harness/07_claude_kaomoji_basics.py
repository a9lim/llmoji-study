"""Basic descriptive stats on the contributor-submitted HF corpus.

Reads ``data/claude_descriptions.jsonl`` (the flat per-canonical
output of ``scripts/06_claude_hf_pull.py``) and prints:

  - Total canonical kaomoji, total emissions across contributors,
    n_contributors, n_bundles, n_source_models.
  - Top-N kaomoji by total count.
  - Provider mix (claude_code-hook / codex-hook / hermes-hook /
    static export / mixed bundles).
  - Source-model breakdown (which underlying agent model wrote the
    kaomoji-bearing turns) — this is the new 1.1 research signal.
  - Synthesis-backend breakdown (which API wrote the prose:
    anthropic / openai / local).
  - Llmoji-package-version distribution.

Purely informational; doesn't write files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.config import CLAUDE_DESCRIPTIONS_PATH


def main() -> None:
    if not CLAUDE_DESCRIPTIONS_PATH.exists():
        print(
            f"no corpus at {CLAUDE_DESCRIPTIONS_PATH}; "
            "run scripts/06_claude_hf_pull.py first"
        )
        return

    rows: list[dict] = []
    with CLAUDE_DESCRIPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n_kaomoji = len(rows)
    n_emissions = sum(int(r["count_total"]) for r in rows)
    contributors: set[str] = set()
    bundles_seen: set[tuple[str, str]] = set()        # (contributor, bundle)
    source_model_emissions: Counter[str] = Counter()  # weighted by row count
    source_model_faces: Counter[str] = Counter()      # distinct faces per src
    backend_bundles: Counter[str] = Counter()         # 1 per (contrib,bundle)
    provider_bundles: Counter[str] = Counter()        # 1 per (contrib,bundle)
    version_bundles: Counter[str] = Counter()         # 1 per (contrib,bundle)

    # Per-(contributor,bundle) metadata: take it from the first
    # description we encounter for that bundle. Manifest-level fields
    # (synthesis_backend, providers, llmoji_version) are uniform across
    # rows in the same bundle, so the first encounter is fine.
    bundle_meta: dict[tuple[str, str], dict] = {}
    for r in rows:
        # Track distinct (face, source_model) so the same canonical
        # face that's been written by N source models contributes N to
        # the source-model "faces" count (this is the cross-model
        # signal — the same face read differently by different agents).
        face_models_seen: set[str] = set()
        for d in r["descriptions"]:
            contrib = d["contributor"]
            bundle = d.get("bundle", "")
            contributors.add(contrib)
            key = (contrib, bundle)
            bundles_seen.add(key)
            bundle_meta.setdefault(key, {
                "providers": d.get("providers", []) or [],
                "synthesis_backend": d.get("synthesis_backend", "") or "(unknown)",
                "llmoji_version": d.get("llmoji_version", "") or "(unknown)",
            })
            sm = d.get("source_model", "") or "(unknown)"
            source_model_emissions[sm] += int(d.get("count", 0))
            face_models_seen.add(sm)
        for sm in face_models_seen:
            source_model_faces[sm] += 1

    for meta in bundle_meta.values():
        provider_key = "+".join(sorted(meta["providers"] or ["(none)"]))
        provider_bundles[provider_key] += 1
        backend_bundles[meta["synthesis_backend"]] += 1
        version_bundles[meta["llmoji_version"]] += 1

    print(f"canonical kaomoji:    {n_kaomoji}")
    print(f"total emissions:      {n_emissions}")
    print(f"contributors:         {len(contributors)}")
    print(f"bundles:              {len(bundles_seen)}")
    print(f"source models seen:   {len(source_model_emissions)}")

    print("\ntop 25 by total count:")
    for r in rows[:25]:
        print(
            f"  {r['count_total']:5d}  {r['kaomoji']:<14}  "
            f"({r['n_contributors']} contrib, {r['n_bundles']} bundles, "
            f"{r.get('n_source_models', 0)} models)"
        )

    print("\nemissions by source model:")
    for sm, c in source_model_emissions.most_common():
        faces = source_model_faces[sm]
        print(f"  {c:5d} emissions / {faces:3d} faces  {sm}")

    print("\nbundles by synthesis backend:")
    for k, v in backend_bundles.most_common():
        print(f"  {v:4d}  {k}")

    print("\nbundles by provider mix:")
    for k, v in provider_bundles.most_common():
        print(f"  {v:4d}  {k}")

    print("\nbundles by llmoji version:")
    for k, v in version_bundles.most_common():
        print(f"  {v:4d}  {k}")

    print("\ncoverage histogram (n_contributors per kaomoji):")
    cov: Counter[int] = Counter(r["n_contributors"] for r in rows)
    for k in sorted(cov):
        print(f"  {k} contributor(s): {cov[k]} kaomoji")

    print("\ncross-model histogram (n_source_models per kaomoji):")
    cross: Counter[int] = Counter(r.get("n_source_models", 0) for r in rows)
    for k in sorted(cross):
        print(f"  {k} source model(s): {cross[k]} kaomoji")


if __name__ == "__main__":
    main()
