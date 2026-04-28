"""Pull the contributor-submitted kaomoji corpus from the
``a9lim/llmoji`` HuggingFace dataset and flatten it into a single
per-canonical-kaomoji JSONL.

The HF dataset layout is one folder per submission:

    contributors/<32-hex>/bundle-<UTC>/
        manifest.json        # llmoji_version, generated_at, providers_seen, ...
        descriptions.jsonl   # one row per kaomoji from THAT machine

Each ``descriptions.jsonl`` row carries
``{kaomoji, count, haiku_synthesis_description, llmoji_version}`` where
``count`` is per-machine and the description is Haiku's synthesized
meaning across that machine's instances of the face. Per-machine
pooling already happened on the contributor side; cross-contributor
pooling happens here.

We snapshot-download the dataset, walk every bundle, canonicalize each
kaomoji form via ``llmoji.taxonomy.canonicalize_kaomoji`` (so
near-duplicate forms across contributors merge correctly), and write a
flat ``data/claude_descriptions.jsonl`` with one row per canonical form
plus all its per-bundle descriptions. Downstream scripts (07/15/16/18)
read this file and don't need to know the dataset layout.

Output schema, one JSON object per line:

    {
      "kaomoji": "(◕‿◕)",                  # canonical form
      "count_total": 192,                   # sum across contributors
      "n_contributors": 4,                  # distinct submitter ids
      "n_bundles": 5,                       # distinct submissions
      "providers": ["claude_code", "codex"],# union of providers_seen
      "descriptions": [
        {"description": "...", "count": 47,
         "contributor": "abc...", "providers": ["claude_code"],
         "llmoji_version": "1.0.0"},
        ...
      ]
    }

Usage:
    python scripts/06_claude_hf_pull.py [--repo a9lim/llmoji] [--revision main]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji
from llmoji_study.config import (
    CLAUDE_DATASET_DIR,
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_HF_REPO,
    DATA_DIR,
)


def _snapshot(repo: str, revision: str) -> Path:
    """Download (or update) the dataset snapshot to ``CLAUDE_DATASET_DIR``.

    Returns the local path to the snapshot root. Anonymous reads are
    fine — the dataset is public — so HF_TOKEN is not required.
    """
    from huggingface_hub import snapshot_download
    CLAUDE_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(CLAUDE_DATASET_DIR),
        # Skip the dataset README + cards; we only need contributor data.
        allow_patterns=["contributors/**"],
    )
    return Path(local)


def _iter_bundles(snapshot_root: Path):
    """Yield ``(contributor_id, bundle_name, manifest_dict, descriptions_rows)``
    for every bundle that has both files. Bundles missing one or the
    other are skipped with a warning."""
    contrib_root = snapshot_root / "contributors"
    if not contrib_root.exists():
        return
    for contrib_dir in sorted(contrib_root.iterdir()):
        if not contrib_dir.is_dir():
            continue
        for bundle_dir in sorted(contrib_dir.iterdir()):
            if not bundle_dir.is_dir() or not bundle_dir.name.startswith("bundle-"):
                continue
            manifest_path = bundle_dir / "manifest.json"
            desc_path = bundle_dir / "descriptions.jsonl"
            if not (manifest_path.exists() and desc_path.exists()):
                print(
                    f"  skipping {contrib_dir.name}/{bundle_dir.name}: "
                    f"missing manifest or descriptions"
                )
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  skipping {bundle_dir}: bad manifest ({e})")
                continue
            rows: list[dict] = []
            for line in desc_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            yield contrib_dir.name, bundle_dir.name, manifest, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo", default=CLAUDE_HF_REPO,
        help=f"HF dataset repo id (default: {CLAUDE_HF_REPO})",
    )
    ap.add_argument(
        "--revision", default="main",
        help="dataset revision / branch (default: main)",
    )
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"snapshot: {args.repo} @ {args.revision} → {CLAUDE_DATASET_DIR}")
    snapshot_root = _snapshot(args.repo, args.revision)

    # Aggregate across all bundles, keyed by canonical kaomoji form.
    by_canon: dict[str, dict] = defaultdict(lambda: {
        "count_total": 0,
        "contributors": set(),
        "bundles": set(),
        "providers": set(),
        "descriptions": [],
    })

    n_bundles = 0
    n_rows = 0
    n_contributors = set()
    for contrib, bundle, manifest, rows in _iter_bundles(snapshot_root):
        n_bundles += 1
        n_contributors.add(contrib)
        bundle_providers = list(manifest.get("providers_seen") or [])
        for r in rows:
            kao = r.get("kaomoji")
            if not isinstance(kao, str) or not kao:
                continue
            canon = canonicalize_kaomoji(kao)
            count = int(r.get("count", 0))
            description = (r.get("haiku_synthesis_description") or "").strip()
            if not description:
                continue
            agg = by_canon[canon]
            agg["count_total"] += count
            agg["contributors"].add(contrib)
            agg["bundles"].add(f"{contrib}/{bundle}")
            agg["providers"].update(bundle_providers)
            agg["descriptions"].append({
                "description": description,
                "count": count,
                "contributor": contrib,
                "providers": bundle_providers,
                "llmoji_version": r.get("llmoji_version", ""),
            })
            n_rows += 1

    print(
        f"  scanned {n_bundles} bundles from {len(n_contributors)} contributors; "
        f"{n_rows} rows → {len(by_canon)} canonical kaomoji"
    )

    # Write the flat per-canonical-form JSONL, sorted by total count
    # descending so downstream scripts can `head` it for a quick
    # eyeball.
    out_rows = []
    for canon, agg in by_canon.items():
        out_rows.append({
            "kaomoji": canon,
            "count_total": agg["count_total"],
            "n_contributors": len(agg["contributors"]),
            "n_bundles": len(agg["bundles"]),
            "providers": sorted(agg["providers"]),
            "descriptions": sorted(
                agg["descriptions"],
                key=lambda d: (-d["count"], d["contributor"]),
            ),
        })
    out_rows.sort(key=lambda r: (-r["count_total"], r["kaomoji"]))

    with CLAUDE_DESCRIPTIONS_PATH.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {CLAUDE_DESCRIPTIONS_PATH}: {len(out_rows)} canonical kaomoji")

    # Quick sanity printout
    print("\ntop 10 by total count:")
    for r in out_rows[:10]:
        print(
            f"  {r['count_total']:5d}  {r['kaomoji']:<14}  "
            f"({r['n_contributors']} contrib, {r['n_bundles']} bundles)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
