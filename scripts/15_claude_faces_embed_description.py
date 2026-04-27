"""Eriskii-replication step 2: synthesized-description per-kaomoji embeddings.

Reads data/claude_haiku_synthesized.jsonl (one row per kaomoji
with the Stage-B synthesized meaning), embeds each synthesized
string with sentence-transformers/all-MiniLM-L6-v2, L2-normalizes,
and saves a parquet keyed by first_word.

Usage:
  python scripts/15_claude_faces_embed_description.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.claude_faces import EMBED_DIM, EMBED_MODEL
from llmoji_study.config import (
    CLAUDE_FACES_EMBED_DESCRIPTION_PATH,
    CLAUDE_HAIKU_SYNTHESIZED_PATH,
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

    if not CLAUDE_HAIKU_SYNTHESIZED_PATH.exists():
        print(f"no syntheses at {CLAUDE_HAIKU_SYNTHESIZED_PATH}; "
              "run scripts/14 (both stages) first")
        sys.exit(1)

    print(f"loading syntheses from {CLAUDE_HAIKU_SYNTHESIZED_PATH}...")
    rows: list[dict] = []
    with CLAUDE_HAIKU_SYNTHESIZED_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    print(f"  {len(rows)} synthesized kaomoji")
    if not rows:
        print("nothing to embed.")
        return

    from sentence_transformers import SentenceTransformer
    print(f"embedding (device={args.device})...")
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    texts = [r["synthesized"] for r in rows]
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=float)
    assert embs.shape == (len(rows), EMBED_DIM), embs.shape

    out_rows = []
    for r, vec in zip(rows, embs):
        row = {"first_word": r["first_word"], "n": int(r["n_descriptions"])}
        for i, v in enumerate(vec.tolist()):
            row[f"e{i:03d}"] = v
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLAUDE_FACES_EMBED_DESCRIPTION_PATH, index=False)
    print(f"wrote {CLAUDE_FACES_EMBED_DESCRIPTION_PATH}")


if __name__ == "__main__":
    main()
