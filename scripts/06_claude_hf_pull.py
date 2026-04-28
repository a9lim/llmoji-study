"""Pull the contributor-submitted kaomoji corpus from the
``a9lim/llmoji`` HuggingFace dataset and flatten it into a single
per-canonical-kaomoji JSONL.

The dataset is in 1.1 layout, with backwards-compat for 1.0 bundles
that haven't been re-analyzed yet:

    contributors/<32-hex>/bundle-<UTC>/
        manifest.json                       # 1.0 + 1.1
        <sanitized-source-model>.jsonl ...  # 1.1, one per source model
        descriptions.jsonl                  # 1.0, single pooled file

In 1.1, each `<source-model>.jsonl` carries one row per canonical
kaomoji as that specific source model wrote it:

    {"kaomoji": "(◕‿◕)", "count": 47,
     "synthesis_description": "..."}

The filename stem is the sanitized source-model id
(``llmoji._util.sanitize_model_id_for_path``: lowercase, ``/`` →
``__``, ``:`` → ``-``). Per-machine pooling already happened on the
contributor side; cross-source-model and cross-contributor pooling
happens here.

The 1.0 ``descriptions.jsonl`` rows used the field name
``haiku_synthesis_description`` and weren't split by source model;
we read those as-is and tag them with ``source_model = "_pre_1_1"``
so downstream can opt into / out of legacy bundles.

We snapshot-download the dataset, walk every bundle, canonicalize each
kaomoji form via ``llmoji.taxonomy.canonicalize_kaomoji`` (so
near-duplicate forms across contributors merge correctly), and write a
flat ``data/claude_descriptions.jsonl`` with one row per canonical form
plus all its per-bundle / per-source-model descriptions. Downstream
scripts (07/15/16/18) read this file and don't need to know the
dataset layout.

Output schema, one JSON object per line:

    {
      "kaomoji": "(◕‿◕)",                    # canonical form
      "count_total": 192,                     # sum across all entries
      "n_contributors": 4,                    # distinct submitter ids
      "n_bundles": 5,                         # distinct submissions
      "n_source_models": 3,                   # distinct source models
      "providers": ["claude_code-hook", ...], # union of providers_seen
      "source_models": ["claude-sonnet-...","gpt-5.4-..."],
      "synthesis_backends": ["anthropic"],    # union across descs
      "descriptions": [
        {"description": "...",
         "count": 47,
         "contributor": "abc...",
         "source_model": "claude-sonnet-4-5-20250929",
         "synthesis_model_id": "claude-haiku-4-5-20251001",
         "synthesis_backend": "anthropic",
         "providers": ["claude_code-hook"],
         "llmoji_version": "1.1.0",
         "bundle": "bundle-..."},
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


# Sentinel used for legacy 1.0 ``descriptions.jsonl`` rows that weren't
# split per source model. Distinct from ``"unknown"`` (which the 1.1
# package uses for rows whose ScrapeRow.model was empty) so downstream
# can tell "we don't know because the bundle predates the split" apart
# from "we know the contributor's harness didn't stamp a model id".
LEGACY_SOURCE_MODEL = "_pre_1_1"


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
    """Yield ``(contributor_id, bundle_name, manifest_dict, jsonl_paths)``
    for every bundle that has a manifest.json plus at least one
    ``*.jsonl``. Bundles missing one or the other are skipped with a
    warning.

    ``jsonl_paths`` is the list of every ``*.jsonl`` at the bundle
    root — one per source model in 1.1, or a single
    ``descriptions.jsonl`` in 1.0.
    """
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
            jsonl_paths = sorted(bundle_dir.glob("*.jsonl"))
            if not manifest_path.exists() or not jsonl_paths:
                print(
                    f"  skipping {contrib_dir.name}/{bundle_dir.name}: "
                    f"missing manifest or no *.jsonl"
                )
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  skipping {bundle_dir}: bad manifest ({e})")
                continue
            yield contrib_dir.name, bundle_dir.name, manifest, jsonl_paths


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _resolve_source_model(jsonl_path: Path) -> str:
    """Filename stem is the sanitized source model id in 1.1; the
    one 1.0 file is named ``descriptions.jsonl``. We don't reverse the
    sanitization (it's lossy on case and punctuation) — the stem is
    stable enough to group by, and downstream comparisons key on it
    rather than on the raw model id."""
    stem = jsonl_path.stem
    if stem == "descriptions":
        return LEGACY_SOURCE_MODEL
    return stem


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
        "source_models": set(),
        "synthesis_backends": set(),
        "descriptions": [],
    })

    n_bundles = 0
    n_rows = 0
    n_contributors: set[str] = set()
    n_legacy_bundles = 0
    n_source_model_files = 0
    for contrib, bundle, manifest, jsonl_paths in _iter_bundles(snapshot_root):
        n_bundles += 1
        n_contributors.add(contrib)
        bundle_providers = list(manifest.get("providers_seen") or [])
        # 1.1 manifest fields, with 1.0 fallbacks. ``haiku_model_id``
        # was the 1.0 name for what's now ``synthesis_model_id``;
        # 1.0 only ever ran the anthropic backend so we can hard-code
        # the fallback safely.
        synthesis_model_id = (
            manifest.get("synthesis_model_id")
            or manifest.get("haiku_model_id")
            or ""
        )
        synthesis_backend = manifest.get("synthesis_backend") or (
            "anthropic" if manifest.get("haiku_model_id") else ""
        )
        llmoji_version = manifest.get("llmoji_version", "")

        for jsonl_path in jsonl_paths:
            source_model = _resolve_source_model(jsonl_path)
            if source_model == LEGACY_SOURCE_MODEL:
                n_legacy_bundles += 1
            else:
                n_source_model_files += 1
            for r in _read_jsonl(jsonl_path):
                kao = r.get("kaomoji")
                if not isinstance(kao, str) or not kao:
                    continue
                canon = canonicalize_kaomoji(kao)
                count = int(r.get("count", 0))
                # 1.1 renamed haiku_synthesis_description →
                # synthesis_description. Accept either; prefer the new
                # name when both are present.
                description = (
                    r.get("synthesis_description")
                    or r.get("haiku_synthesis_description")
                    or ""
                ).strip()
                if not description:
                    continue
                agg = by_canon[canon]
                agg["count_total"] += count
                agg["contributors"].add(contrib)
                agg["bundles"].add(f"{contrib}/{bundle}")
                agg["providers"].update(bundle_providers)
                agg["source_models"].add(source_model)
                if synthesis_backend:
                    agg["synthesis_backends"].add(synthesis_backend)
                agg["descriptions"].append({
                    "description": description,
                    "count": count,
                    "contributor": contrib,
                    "bundle": bundle,
                    "source_model": source_model,
                    "synthesis_model_id": synthesis_model_id,
                    "synthesis_backend": synthesis_backend,
                    "providers": bundle_providers,
                    # In 1.1 llmoji_version isn't on the row anymore;
                    # we promote the manifest-level value so the per-
                    # description record stays self-describing.
                    "llmoji_version": r.get("llmoji_version") or llmoji_version,
                })
                n_rows += 1

    print(
        f"  scanned {n_bundles} bundles from {len(n_contributors)} contributors; "
        f"{n_source_model_files} per-source-model files + "
        f"{n_legacy_bundles} legacy descriptions.jsonl; "
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
            "n_source_models": len(agg["source_models"]),
            "providers": sorted(agg["providers"]),
            "source_models": sorted(agg["source_models"]),
            "synthesis_backends": sorted(agg["synthesis_backends"]),
            "descriptions": sorted(
                agg["descriptions"],
                key=lambda d: (-d["count"], d["source_model"], d["contributor"]),
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
        models_str = ",".join(r["source_models"][:3])
        if len(r["source_models"]) > 3:
            models_str += f"+{len(r['source_models']) - 3}"
        print(
            f"  {r['count_total']:5d}  {r['kaomoji']:<14}  "
            f"({r['n_contributors']} contrib, {r['n_bundles']} bundles, "
            f"{r['n_source_models']} models: {models_str})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
