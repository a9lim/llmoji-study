# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportPrivateImportUsage=false
"""HF-corpus loaders + per-kaomoji description embeddings.

Pre-refactor this module also did response-based embeddings on raw
``data/claude_kaomoji.jsonl`` rows (whole assistant_text minus the
leading kaomoji). Post-refactor that path is gone — the corpus is
the HF dataset ``a9lim/llmoji``, where every row is already
pre-aggregated per-machine and (in 1.1) per-source-model to
``(kaomoji, count, synthesis_description)``, so the only thing left
to embed on the research side is the synthesized description string.

Public surface:

  - :data:`EMBED_MODEL` / :data:`EMBED_DIM` — sentence-transformers
    backbone, kept identical to the eriskii pipeline so axis vectors
    line up.
  - :func:`load_descriptions` — read
    ``data/harness/claude_descriptions.jsonl`` (the flat per-canonical output
    of ``scripts/60_corpus_pull.py``).
  - :func:`embed_descriptions` — for each kaomoji, embed every
    per-bundle / per-source-model description, then weighted-mean by
    per-bundle count and L2-renormalize. Returns
    ``(canonical_kaomoji, count_total, E)``. The same canonical face
    written by N source models contributes N description rows to the
    weighted mean — that's the cross-model pooling the dataset's 1.1
    per-source-model split makes possible.
  - :func:`save_embeddings` / :func:`load_embeddings` — parquet round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def load_descriptions(path: Path) -> list[dict]:
    """Read the flat per-canonical-form JSONL emitted by
    ``scripts/60_corpus_pull.py``.

    Each row carries ``kaomoji`` (canonical), ``count_total``,
    ``n_contributors``, ``n_bundles``, ``n_source_models``,
    ``providers`` (list), ``source_models`` (list),
    ``synthesis_backends`` (list), and ``descriptions`` (list of
    per-bundle / per-source-model dicts with ``description``,
    ``count``, ``contributor``, ``bundle``, ``source_model``,
    ``synthesis_model_id``, ``synthesis_backend``, ``providers``,
    ``llmoji_version``).
    """
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def embed_descriptions(
    rows: list[dict],
    *,
    device: str | None = None,
    progress: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """For each canonical kaomoji, embed every per-bundle description
    string, then weighted-mean by per-bundle count and L2-normalize.

    Returns ``(canonical_kaomoji, count_total, E)`` where ``E`` has
    shape ``(n_kaomoji, EMBED_DIM)``.
    """
    from sentence_transformers import SentenceTransformer

    # Flatten to one row per (kaomoji, bundle-description) so we can
    # batch a single encode call.
    flat: list[tuple[int, str, int]] = []  # (parent_idx, text, count)
    for i, r in enumerate(rows):
        for d in r.get("descriptions", []):
            text = (d.get("description") or "").strip()
            if not text:
                continue
            count = int(d.get("count", 0))
            flat.append((i, text, max(count, 1)))
    if not flat:
        return [], np.array([], dtype=int), np.empty((0, EMBED_DIM), dtype=float)

    model = SentenceTransformer(EMBED_MODEL, device=device)
    texts = [t for _, t, _ in flat]
    embs = np.asarray(
        model.encode(
            texts,
            batch_size=64,
            show_progress_bar=progress,
            normalize_embeddings=True,
        ),
        dtype=float,
    )

    out_fw: list[str] = []
    out_n: list[int] = []
    out_E: list[np.ndarray] = []
    for i, r in enumerate(rows):
        idxs = [j for j, (pi, _, _) in enumerate(flat) if pi == i]
        if not idxs:
            continue
        weights = np.array([flat[j][2] for j in idxs], dtype=float)
        weights = weights / weights.sum()
        E_avg = (embs[idxs] * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(E_avg))
        if norm > 0:
            E_avg = E_avg / norm
        out_fw.append(r["kaomoji"])
        out_n.append(int(r.get("count_total", 0)))
        out_E.append(E_avg)
    return out_fw, np.asarray(out_n, dtype=int), np.asarray(out_E)


def save_embeddings(
    fw: list[str], n: np.ndarray, E: np.ndarray, path: Path,
) -> None:
    """Persist per-kaomoji embeddings to parquet."""
    rows = []
    for fwi, ni, vec in zip(fw, n.tolist(), E.tolist()):
        row = {"first_word": fwi, "n": int(ni)}
        for k, v in enumerate(vec):
            row[f"e{k:03d}"] = float(v)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``(first_words, counts, embedding_matrix)``."""
    df: pd.DataFrame = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    E = df[emb_cols].to_numpy(dtype=float)
    return df["first_word"].tolist(), df["n"].to_numpy(dtype=int), E
