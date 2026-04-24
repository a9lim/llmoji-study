"""Driver: compute per-kaomoji response-based embeddings from the
scrape, persist to parquet.

Usage:
  python scripts/08_claude_faces_embed.py [--min-count N] [--device mps]

Defaults: min_count=5 (aggressive; eriskii needed 519 kaomoji from a
larger corpus, we'll likely see fewer), device="mps" if on Apple
Silicon else CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.claude_faces import compute_embeddings, load_rows, save_embeddings
from llmoji.config import CLAUDE_FACES_EMBED_PATH, CLAUDE_KAOMOJI_PATH, DATA_DIR


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
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--device", type=str, default=_default_device())
    args = ap.parse_args()

    if not CLAUDE_KAOMOJI_PATH.exists():
        print(f"no scrape at {CLAUDE_KAOMOJI_PATH}; run scripts/06_claude_scrape.py first")
        return
    print(f"loading {CLAUDE_KAOMOJI_PATH}")
    df = load_rows(CLAUDE_KAOMOJI_PATH)
    print(f"  {len(df)} rows; {df['first_word'].nunique()} distinct first_words")

    print(f"computing embeddings (device={args.device}, min_count={args.min_count})...")
    embeds = compute_embeddings(df, min_count=args.min_count, device=args.device)
    print(f"  {len(embeds)} kaomoji embedded")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeds, CLAUDE_FACES_EMBED_PATH)
    print(f"wrote {CLAUDE_FACES_EMBED_PATH}")


if __name__ == "__main__":
    main()
