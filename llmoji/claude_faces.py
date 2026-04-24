# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportPrivateImportUsage=false
"""Per-kaomoji response-based embeddings.

For each unique first_word with count >= min_count, embed each
occurrence's assistant_text (with the leading kaomoji stripped) using
sentence-transformers/all-MiniLM-L6-v2, then mean-pool to a single
384-dim vector per kaomoji. Eriskii-style analysis consumes these.

Rationale for response-based (vs user-based): the user message is
short and varied; the assistant's own text around the kaomoji is
longer and carries the tonal context that drives which face was
chosen. This matches option (B) in the design sketch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass
class KaomojiEmbed:
    first_word: str
    n: int
    mean_embedding: np.ndarray  # shape (EMBED_DIM,)


def _strip_leading(text: str, kaomoji: str) -> str:
    """Remove a leading kaomoji occurrence from text so embeddings don't
    collapse on the literal face string."""
    stripped = text.lstrip()
    if stripped.startswith(kaomoji):
        return stripped[len(kaomoji):].lstrip()
    return stripped


def load_rows(path: Path) -> pd.DataFrame:
    """Load data/claude_kaomoji.jsonl into a DataFrame."""
    df: pd.DataFrame = pd.read_json(path, lines=True)
    return df


def compute_embeddings(
    df: pd.DataFrame,
    *,
    min_count: int = 3,
    batch_size: int = 64,
    device: str | None = None,
    progress: bool = True,
) -> list[KaomojiEmbed]:
    """For each first_word with >= min_count rows, mean-pool an
    embedding over its assistant_text occurrences (kaomoji stripped).

    Uses sentence-transformers on CPU by default. Pass device="mps"
    on Apple Silicon for speed.
    """
    from sentence_transformers import SentenceTransformer

    counts = df["first_word"].value_counts()
    keep = counts[counts >= min_count].index.tolist()
    if not keep:
        return []

    sub = df[df["first_word"].isin(keep)].copy()
    # strip leading kaomoji in place
    sub["stripped"] = [
        _strip_leading(str(t), str(fw))
        for t, fw in zip(sub["assistant_text"], sub["first_word"])
    ]

    model = SentenceTransformer(EMBED_MODEL, device=device)
    texts: list[str] = sub["stripped"].tolist()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        normalize_embeddings=True,
    )

    sub = sub.assign(emb_idx=range(len(sub)))
    out: list[KaomojiEmbed] = []
    for fw in keep:
        idx = sub.loc[sub["first_word"] == fw, "emb_idx"].to_numpy()
        if len(idx) == 0:
            continue
        mean = np.asarray(embs)[idx].mean(axis=0)
        # renormalize the mean so cosine comparisons are well-behaved
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        out.append(KaomojiEmbed(first_word=str(fw), n=int(len(idx)), mean_embedding=mean))
    return out


def save_embeddings(embeds: Iterable[KaomojiEmbed], path: Path) -> None:
    """Persist per-kaomoji embeddings to parquet."""
    rows = []
    for e in embeds:
        row = {"first_word": e.first_word, "n": e.n}
        for i, v in enumerate(e.mean_embedding.tolist()):
            row[f"e{i:03d}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (first_words, counts, embedding matrix)."""
    df: pd.DataFrame = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    E = df[emb_cols].to_numpy(dtype=float)
    return df["first_word"].tolist(), df["n"].to_numpy(dtype=int), E
