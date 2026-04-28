"""Basic descriptive stats on the contributor-submitted HF corpus.

Reads ``data/claude_descriptions.jsonl`` (the flat per-canonical
output of ``scripts/06_claude_hf_pull.py``) and prints:

  - Total canonical kaomoji, total emissions across contributors,
    n_contributors, n_bundles.
  - Top-N kaomoji by total count.
  - Provider mix (claude_code / codex / hermes / mixed bundles).
  - Llmoji-package-version distribution.

Purely informational; doesn't write files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    bundle_seen: set[tuple] = set()
    provider_bundle_counts: Counter[str] = Counter()
    version_bundle_counts: Counter[str] = Counter()
    for r in rows:
        for d in r["descriptions"]:
            contributors.add(d["contributor"])
            # The per-canonical row doesn't expose bundle id directly,
            # so use (contributor, providers tuple, llmoji_version) as
            # a bundle fingerprint. Same machine running multiple
            # `analyze` passes against different package versions
            # counts as distinct bundles, which matches reality.
            key = (
                d["contributor"],
                tuple(d.get("providers", [])),
                d.get("llmoji_version", ""),
            )
            if key not in bundle_seen:
                bundle_seen.add(key)
                provider_key = "+".join(
                    sorted(d.get("providers", []) or ["(none)"])
                )
                provider_bundle_counts[provider_key] += 1
                version_bundle_counts[d.get("llmoji_version", "(unknown)")] += 1

    print(f"canonical kaomoji: {n_kaomoji}")
    print(f"total emissions:   {n_emissions}")
    print(f"contributors:      {len(contributors)}")
    print(f"bundles:           {len(bundle_seen)}")

    print("\ntop 25 by total count:")
    for r in rows[:25]:
        print(
            f"  {r['count_total']:5d}  {r['kaomoji']:<14}  "
            f"({r['n_contributors']} contrib, {r['n_bundles']} bundles)"
        )

    print("\nbundles by provider mix:")
    for k, v in provider_bundle_counts.most_common():
        print(f"  {v:4d}  {k}")

    print("\nbundles by llmoji version:")
    for k, v in version_bundle_counts.most_common():
        print(f"  {v:4d}  {k}")

    print("\ncoverage histogram (n_contributors per kaomoji):")
    cov: Counter[int] = Counter(r["n_contributors"] for r in rows)
    for k in sorted(cov):
        print(f"  {k} contributor(s): {cov[k]} kaomoji")


if __name__ == "__main__":
    main()
