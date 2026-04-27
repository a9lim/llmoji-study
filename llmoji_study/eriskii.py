"""Pipeline primitives for the eriskii-replication experiment.

Three functional layers:

  - mask_kaomoji(text, first_word) — replace the leading kaomoji
    span with the literal token [FACE].
  - call_haiku(client, prompt, *, model_id, max_tokens) — single
    Haiku call returning the assistant text, stripped.
  - (later tasks) project_axes / label_clusters / weighted_group_stats /
    user_kaomoji_axis_correlation — analysis primitives consumed by
    scripts/16.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MASK_TOKEN = "[FACE]"


def mask_kaomoji(text: str, first_word: str) -> str:
    """Replace the leading kaomoji span with MASK_TOKEN.

    The leading kaomoji is identified by `first_word` (the value
    captured by llmoji.taxonomy.extract at scrape time). We strip
    leading whitespace, verify the text starts with first_word, and
    swap it. If the leading text doesn't match (e.g. the row had a
    kaomoji mid-line), we don't mutate — return the original text.
    """
    stripped = text.lstrip()
    if not first_word or not stripped.startswith(first_word):
        return text
    return MASK_TOKEN + stripped[len(first_word):]


def call_haiku(
    client: Any,
    prompt: str,
    *,
    model_id: str,
    max_tokens: int = 200,
) -> str:
    """Single Haiku call with a pre-formatted prompt. Returns the
    assistant's first text-block content, stripped. Raises on API
    error (caller's resume loop handles).

    `client` is an anthropic.Anthropic instance. We don't import the
    SDK here so this module is importable without anthropic being
    installed (matters for the smoke test in Step 2, which doesn't
    call Haiku)."""
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""


def compute_axis_vectors(
    embedder: Any,
    anchors: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    """For each axis name → (positive_anchor, negative_anchor),
    embed both, return the L2-normalized difference (positive − negative).

    `embedder` is a sentence_transformers.SentenceTransformer instance.
    """
    pos_texts = [pos for pos, _ in anchors.values()]
    neg_texts = [neg for _, neg in anchors.values()]
    # one batch call for all anchors at once
    pos_emb = embedder.encode(
        pos_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    neg_emb = embedder.encode(
        neg_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(anchors.keys()):
        diff = np.asarray(pos_emb[i]) - np.asarray(neg_emb[i])
        norm = float(np.linalg.norm(diff))
        if norm > 0:
            diff = diff / norm
        out[name] = diff
    return out


def project_onto_axes(
    E: np.ndarray,
    axis_vectors: dict[str, np.ndarray],
    axis_order: list[str],
) -> np.ndarray:
    """Return (n_kaomoji, n_axes) projection matrix.

    Rows of E are assumed already L2-normalized (matches what
    save_embeddings/load_embeddings produce). Axis vectors are
    L2-normalized by compute_axis_vectors. Cosine similarity collapses
    to dot product under that normalization, so result[i, j] is the
    cosine of kaomoji i's description-embedding with axis j.
    """
    A = np.stack([axis_vectors[name] for name in axis_order], axis=1)
    return E @ A


def label_cluster_via_haiku(
    client: Any,
    members: list[tuple[str, str]],
    *,
    model_id: str,
    prompt_template: str,
    max_tokens: int = 60,
) -> str:
    """Given member [(first_word, description), ...], ask Haiku for
    a 3-5 word eriskii-style cluster label. Returns the stripped
    response text. Caller's resume loop handles errors."""
    members_str = "\n".join(
        f"- {fw}: {desc}" for fw, desc in members
    )
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": prompt_template.format(members=members_str),
        }],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""


def weighted_group_axis_stats(
    rows: "pd.DataFrame",
    axes_df: "pd.DataFrame",
    *,
    group_col: str,
    axis_names: list[str],
    min_emissions: int = 10,
) -> "pd.DataFrame":
    """For each group g and axis a, compute emission-weighted mean and
    std of axis-scores.

    `rows` is the full claude_kaomoji.jsonl DataFrame (one row per
    emission). `axes_df` is the eriskii_axes table (one row per
    kaomoji × 21 axis columns). Group is taken from rows[group_col]
    (e.g. 'model' or 'project_slug'). Groups with fewer than
    min_emissions total rows are dropped.

    Returns long-form DataFrame with columns
    [group_col, 'axis', 'mean', 'std', 'n'].
    """
    # left-join axes onto rows by first_word
    merged = rows.merge(
        axes_df.set_index("first_word")[axis_names],
        left_on="first_word", right_index=True, how="inner",
    )
    out_rows = []
    for g, sub in merged.groupby(group_col, sort=False):
        if len(sub) < min_emissions:
            continue
        for a in axis_names:
            vals = sub[a].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            out_rows.append({
                group_col: g,
                "axis": a,
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n":    int(len(vals)),
            })
    return pd.DataFrame(out_rows)


def user_kaomoji_axis_correlation(
    rows: "pd.DataFrame",
    axes_df: "pd.DataFrame",
    *,
    embedder: Any,
    axis_anchors: dict[str, tuple[str, str]],
    axis_order: list[str],
) -> "pd.DataFrame":
    """For rows with non-empty surrounding_user, correlate
    user-text axis-projection with kaomoji axis-projection.

    Embeds each surrounding_user with `embedder`, projects onto
    the same 21 axes the kaomoji embeddings were projected onto,
    then for each axis computes Pearson r between user-text and
    kaomoji axis scores. p-values are Bonferroni-corrected across
    all axes (`p_bonf = min(1, p * len(axis_order))`).

    Returns long-form DataFrame [axis, r, p, p_bonf, n].
    """
    from scipy.stats import pearsonr

    sub = rows.copy()
    sub["surrounding_user"] = sub["surrounding_user"].fillna("")
    sub = sub[sub["surrounding_user"].str.strip() != ""]
    sub = sub.merge(
        axes_df.set_index("first_word")[axis_order],
        left_on="first_word", right_index=True, how="inner",
        suffixes=("", "_kao"),
    )
    if len(sub) == 0:
        return pd.DataFrame(columns=["axis", "r", "p", "p_bonf", "n"])

    print(f"  embedding {len(sub)} user messages...")
    user_emb = embedder.encode(
        sub["surrounding_user"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    user_emb = np.asarray(user_emb)

    axis_vecs = compute_axis_vectors(embedder, axis_anchors)
    A = np.stack([axis_vecs[name] for name in axis_order], axis=1)
    user_proj = user_emb @ A  # (n_rows, n_axes)

    out = []
    n_axes = len(axis_order)
    for j, name in enumerate(axis_order):
        u = user_proj[:, j]
        k = sub[name].to_numpy(dtype=float)
        r, p = pearsonr(u, k)
        p_bonf = float(min(1.0, p * n_axes))
        out.append({"axis": name, "r": float(r), "p": float(p),
                    "p_bonf": p_bonf, "n": int(len(u))})
    return pd.DataFrame(out)
