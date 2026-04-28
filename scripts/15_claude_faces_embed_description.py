"""Per-canonical-kaomoji description embeddings.

Reads ``data/claude_descriptions.jsonl`` (the flat HF-corpus output of
``scripts/06_claude_hf_pull.py``). For each canonical kaomoji, embeds
every per-bundle synthesized description with
``sentence-transformers/all-MiniLM-L6-v2``, weighted-means by per-bundle
count, L2-normalizes, and writes a parquet keyed by canonical form.

Pre-refactor this script consumed
``data/claude_haiku_synthesized.jsonl`` (the local Stage-B output of the
old ``scripts/14_claude_haiku_describe.py``). That whole pipeline now
runs contributor-side via the ``llmoji`` package and ships in the HF
bundles. We just embed and pool here.

Usage:
    python scripts/15_claude_faces_embed_description.py [--device mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.claude_faces import (
    embed_descriptions, load_descriptions, save_embeddings,
)
from llmoji_study.config import (
    CLAUDE_DESCRIPTIONS_PATH,
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    DATA_DIR,
)


def _default_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=_default_device())
    args = ap.parse_args()

    if not CLAUDE_DESCRIPTIONS_PATH.exists():
        print(
            f"no corpus at {CLAUDE_DESCRIPTIONS_PATH}; "
            "run scripts/06_claude_hf_pull.py first"
        )
        sys.exit(1)

    print(f"loading corpus from {CLAUDE_DESCRIPTIONS_PATH}...")
    rows = load_descriptions(CLAUDE_DESCRIPTIONS_PATH)
    n_descs = sum(len(r.get("descriptions", [])) for r in rows)
    print(f"  {len(rows)} canonical kaomoji, {n_descs} per-bundle descriptions")
    if not rows:
        print("nothing to embed.")
        return

    print(f"embedding (device={args.device})...")
    fw, n, E = embed_descriptions(rows, device=args.device)
    print(f"  {len(fw)} kaomoji embedded, dim={E.shape[1] if len(E) else 0}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(fw, n, E, CLAUDE_FACES_EMBED_DESCRIPTION_PATH)
    print(f"wrote {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}")


if __name__ == "__main__":
    main()
