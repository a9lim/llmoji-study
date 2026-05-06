# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false, reportPrivateImportUsage=false
"""HF-corpus loaders + per-kaomoji bag-of-lexicon (BoL) representation.

History:

  - Pre-2026-04-27: response-based MiniLM embeddings on raw
    ``data/claude_kaomoji.jsonl`` rows (whole assistant_text minus the
    leading kaomoji).
  - 2026-04-27 → 2026-05-06: description-based MiniLM embeddings on
    contributor-side synthesized prose
    (``synthesis_description: str`` from llmoji v1.1).
  - 2026-05-06+ (this module): structured **bag-of-lexicon** vectors
    pulled directly from llmoji v2's ``synthesis: {primary_affect,
    stance_modality_function}`` adjective bag. No encoder, no MiniLM,
    no prose round-trip — the synthesizer's pick over the locked
    48-word LEXICON is the vector. See :mod:`llmoji_study.lexicon`
    for the LEXICON index + Russell-quadrant tags.

Public surface:

  - :func:`load_descriptions` — read
    ``data/harness/claude_descriptions.jsonl`` (the flat per-canonical
    output of ``scripts/harness/60_corpus_pull.py``).
  - :func:`embed_lexicon_bags` — for each kaomoji, build a 48-d BoL
    by count-weighting the per-bundle synthesis picks. Returns
    ``(canonical_kaomoji, count_total, n_v2_descs, BoL_matrix)``.
    Faces with zero v2 descriptions are dropped (BoL is undefined
    for v1-only legacy bundles).
  - :func:`save_bol_parquet` / :func:`load_bol_parquet` — parquet
    round-trip with ``lexicon_version`` stamped via column constant
    so :func:`llmoji_study.lexicon.assert_lexicon_v1` can refuse
    cross-version mixes on read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmoji_study.lexicon import (
    LEXICON_VERSION,
    LEXICON_WORDS,
    N_LEXICON,
    assert_lexicon_v1,
    bol_from_synthesis,
    pool_bol,
)


# Column prefix for lexicon-word weights in the parquet. Prefix avoids
# collisions with metadata columns and makes downstream column
# selection (e.g. via DataFrame.filter) trivial.
LEX_COL_PREFIX = "lex__"
LEX_COLUMNS = [f"{LEX_COL_PREFIX}{w}" for w in LEXICON_WORDS]


def load_descriptions(path: Path) -> list[dict]:
    """Read the flat per-canonical-form JSONL emitted by
    ``scripts/harness/60_corpus_pull.py``.

    Each row carries ``kaomoji`` (canonical), ``count_total``,
    ``n_contributors``, ``n_bundles``, ``n_source_models``,
    ``providers`` (list), ``source_models`` (list),
    ``synthesis_backends`` (list), and ``descriptions`` (list of
    per-bundle / per-source-model dicts). v2 description rows carry
    a structured ``synthesis: {primary_affect,
    stance_modality_function}`` object plus ``lexicon_version``;
    v1.x rows carry only the legacy free-form ``description`` string.
    """
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def embed_lexicon_bags(
    rows: list[dict],
    *,
    primary_weight: float = 1.0,
    extension_weight: float = 0.5,
    require_lexicon_version: int = LEXICON_VERSION,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """For each canonical kaomoji, build a 48-d count-weighted BoL.

    Per per-bundle row: convert ``synthesis`` → 48-d weighted indicator
    via :func:`bol_from_synthesis`. Per canonical face: count-weighted
    pool across bundles via :func:`pool_bol` (L1-normalized so the
    output reads as a soft distribution over the lexicon).

    Faces with zero v2 descriptions (legacy v1.x bundles only) are
    dropped — BoL is undefined for those. Rows tagged with a
    ``lexicon_version`` other than ``require_lexicon_version`` are
    skipped per-row with a warning printed once at the end (mixing
    v1+v2 lexicon picks would silently corrupt the bag).

    Returns ``(canonical_kaomoji, count_total, n_v2_descs, BoL)``
    where ``BoL`` has shape ``(n_kept, 48)`` and rows are L1-normalized.
    """
    out_fw: list[str] = []
    out_count_total: list[int] = []
    out_n_v2_descs: list[int] = []
    out_bol: list[np.ndarray] = []

    n_skipped_no_v2 = 0
    n_skipped_bad_lex = 0

    for r in rows:
        per_bundle: list[np.ndarray] = []
        per_bundle_weights: list[float] = []
        for d in r.get("descriptions", []):
            syn = d.get("synthesis")
            if not isinstance(syn, dict):
                # v1.x row — no structured pick, drop silently for
                # this face's BoL pool.
                continue
            row_lex = d.get("lexicon_version")
            if row_lex != require_lexicon_version:
                n_skipped_bad_lex += 1
                continue
            bag = bol_from_synthesis(
                syn,
                primary_weight=primary_weight,
                extension_weight=extension_weight,
            )
            if not np.any(bag):
                # Synthesizer drift: row's picks were entirely outside
                # the locked lexicon. Skip; pool would be a no-op.
                continue
            count = max(int(d.get("count", 0)), 1)
            per_bundle.append(bag)
            per_bundle_weights.append(float(count))
        if not per_bundle:
            n_skipped_no_v2 += 1
            continue
        pooled = pool_bol(per_bundle, weights=per_bundle_weights, l1_normalize=True)
        out_fw.append(r["kaomoji"])
        out_count_total.append(int(r.get("count_total", 0)))
        out_n_v2_descs.append(len(per_bundle))
        out_bol.append(pooled)

    if n_skipped_no_v2:
        print(
            f"  embed_lexicon_bags: skipped {n_skipped_no_v2} faces with "
            "no v2 description (legacy v1.x only)"
        )
    if n_skipped_bad_lex:
        print(
            f"  embed_lexicon_bags: skipped {n_skipped_bad_lex} v2 rows "
            f"with mismatched lexicon_version (require={require_lexicon_version})"
        )

    if not out_bol:
        return [], np.array([], dtype=int), np.array([], dtype=int), np.empty((0, N_LEXICON), dtype=float)
    return (
        out_fw,
        np.asarray(out_count_total, dtype=int),
        np.asarray(out_n_v2_descs, dtype=int),
        np.asarray(out_bol, dtype=float),
    )


def embed_lexicon_bags_per_source(
    rows: list[dict],
    *,
    primary_weight: float = 1.0,
    extension_weight: float = 0.5,
    require_lexicon_version: int = LEXICON_VERSION,
    min_count: int = 1,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """For each (canonical kaomoji, source_model) cell, build a 48-d BoL.

    Long-format counterpart to :func:`embed_lexicon_bags`. Each row of
    the returned matrix is a per-(face, source_model) BoL — one synthesis
    per source per face per bundle, pooled across bundles within a source
    when a face was synthesized from the same source by multiple
    contributors / submissions. Drops cells whose total emit count is
    below ``min_count`` (1 by default — the v2 corpus is small enough
    that we keep singletons).

    Returns ``(faces, source_models, counts, n_descs, B)`` where:
      - ``faces[i]`` is the canonical kaomoji for row i
      - ``source_models[i]`` is the source model id for row i
      - ``counts[i]`` is the summed per-bundle emit count for that
        (face, source_model) cell
      - ``n_descs[i]`` is the number of v2 description rows pooled
      - ``B[i]`` is the L1-normalized 48-d BoL

    Useful for cross-source drift analysis (does claude-opus-4-7's
    deployment use of `(╯°□°)` synthesize differently from
    codex-hook's?). The same canonical face will appear once per
    source-model that synthesized it.
    """
    out_faces: list[str] = []
    out_sms: list[str] = []
    out_counts: list[int] = []
    out_n_descs: list[int] = []
    out_bol: list[np.ndarray] = []

    n_skipped_bad_lex = 0

    for r in rows:
        # Group v2 descriptions by source_model within this face.
        per_source: dict[str, list[tuple[np.ndarray, float]]] = {}
        for d in r.get("descriptions", []):
            syn = d.get("synthesis")
            if not isinstance(syn, dict):
                continue
            row_lex = d.get("lexicon_version")
            if row_lex != require_lexicon_version:
                n_skipped_bad_lex += 1
                continue
            bag = bol_from_synthesis(
                syn,
                primary_weight=primary_weight,
                extension_weight=extension_weight,
            )
            if not np.any(bag):
                continue
            sm = d.get("source_model", "?")
            count = max(int(d.get("count", 0)), 1)
            per_source.setdefault(sm, []).append((bag, float(count)))

        for sm, items in per_source.items():
            total_count = int(sum(c for _, c in items))
            if total_count < min_count:
                continue
            bols = [b for b, _ in items]
            weights = [c for _, c in items]
            pooled = pool_bol(bols, weights=weights, l1_normalize=True)
            out_faces.append(r["kaomoji"])
            out_sms.append(sm)
            out_counts.append(total_count)
            out_n_descs.append(len(items))
            out_bol.append(pooled)

    if n_skipped_bad_lex:
        print(
            f"  embed_lexicon_bags_per_source: skipped {n_skipped_bad_lex} "
            f"v2 rows with mismatched lexicon_version "
            f"(require={require_lexicon_version})"
        )

    if not out_bol:
        return (
            [], [],
            np.array([], dtype=int), np.array([], dtype=int),
            np.empty((0, N_LEXICON), dtype=float),
        )
    return (
        out_faces,
        out_sms,
        np.asarray(out_counts, dtype=int),
        np.asarray(out_n_descs, dtype=int),
        np.asarray(out_bol, dtype=float),
    )


def save_bol_parquet_per_source(
    faces: list[str],
    source_models: list[str],
    counts: np.ndarray,
    n_descs: np.ndarray,
    B: np.ndarray,
    path: Path,
    *,
    lexicon_version: int = LEXICON_VERSION,
) -> None:
    """Persist per-(face, source_model) BoL to parquet (long format).

    Columns: ``first_word``, ``source_model``, ``count``, ``n_v2_descs``,
    ``lexicon_version`` (constant), then 48 ``lex__<word>`` columns.
    """
    if B.shape[1] != N_LEXICON:
        raise ValueError(
            f"BoL matrix has {B.shape[1]} columns, expected {N_LEXICON}"
        )
    df = pd.DataFrame({
        "first_word": faces,
        "source_model": source_models,
        "count": np.asarray(counts, dtype=int),
        "n_v2_descs": np.asarray(n_descs, dtype=int),
        "lexicon_version": np.full(len(faces), int(lexicon_version), dtype=int),
    })
    for j, col in enumerate(LEX_COLUMNS):
        df[col] = B[:, j].astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_bol_parquet_per_source(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Round-trip of :func:`save_bol_parquet_per_source`. Hard-fails on
    lexicon version mismatch.

    Returns ``(faces, source_models, counts, n_v2_descs, BoL)``.
    """
    df: pd.DataFrame = pd.read_parquet(path)
    if "lexicon_version" not in df.columns:
        raise ValueError(
            f"per-source BoL parquet at {path} is missing 'lexicon_version' column"
        )
    versions = df["lexicon_version"].unique().tolist()
    if len(versions) != 1:
        raise ValueError(
            f"per-source BoL parquet at {path} mixes lexicon versions: {versions}"
        )
    assert_lexicon_v1(int(versions[0]))
    missing = [c for c in LEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"per-source BoL parquet at {path} missing {len(missing)} lexicon columns"
        )
    faces = df["first_word"].tolist()
    sms = df["source_model"].tolist()
    counts = df["count"].to_numpy(dtype=int)
    n_descs = df["n_v2_descs"].to_numpy(dtype=int)
    B = df[LEX_COLUMNS].to_numpy(dtype=float)
    return faces, sms, counts, n_descs, B


def save_bol_parquet(
    fw: list[str],
    n: np.ndarray,
    n_v2_descs: np.ndarray,
    B: np.ndarray,
    path: Path,
    *,
    lexicon_version: int = LEXICON_VERSION,
) -> None:
    """Persist per-face BoL to parquet.

    Columns: ``first_word``, ``n``, ``n_v2_descs``, ``lexicon_version``
    (constant), then 48 ``lex__<word>`` columns in
    :data:`LEXICON_WORDS` order.
    """
    if B.shape[1] != N_LEXICON:
        raise ValueError(
            f"BoL matrix has {B.shape[1]} columns, expected {N_LEXICON}"
        )
    df = pd.DataFrame({
        "first_word": fw,
        "n": np.asarray(n, dtype=int),
        "n_v2_descs": np.asarray(n_v2_descs, dtype=int),
        "lexicon_version": np.full(len(fw), int(lexicon_version), dtype=int),
    })
    for j, col in enumerate(LEX_COLUMNS):
        df[col] = B[:, j].astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_bol_parquet(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Round-trip of :func:`save_bol_parquet`. Hard-fails on lexicon
    version mismatch via :func:`assert_lexicon_v1`.

    Returns ``(first_words, counts, n_v2_descs, BoL)``.
    """
    df: pd.DataFrame = pd.read_parquet(path)
    if "lexicon_version" not in df.columns:
        raise ValueError(
            f"BoL parquet at {path} is missing 'lexicon_version' column"
        )
    versions = df["lexicon_version"].unique().tolist()
    if len(versions) != 1:
        raise ValueError(
            f"BoL parquet at {path} mixes lexicon versions: {versions}"
        )
    assert_lexicon_v1(int(versions[0]))
    missing = [c for c in LEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"BoL parquet at {path} missing {len(missing)} lexicon columns "
            f"(first 3: {missing[:3]}) — stale schema"
        )
    fw = df["first_word"].tolist()
    n = df["n"].to_numpy(dtype=int)
    n_v2 = df["n_v2_descs"].to_numpy(dtype=int)
    B = df[LEX_COLUMNS].to_numpy(dtype=float)
    return fw, n, n_v2, B
