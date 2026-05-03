# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""claude-faces encoded through qwen's own forward pass.

Alternative to script 43's MiniLM-based assignment. The hypothesis
script 43 surfaced: face descriptions and v3 prompts are both
descriptive prose, so MiniLM matches them on register rather than
affect content — collapses argmax onto LP regardless of corpus
quality (GPT or Haiku).

This script bypasses MiniLM entirely. For each canonical face's
description, we run qwen forward (kaomoji-instruction system prompt
+ description as user message, override_max_tokens=1), capture
h_first @ L59 — the same state the v3 emission rows live in.
By construction, descriptions and v3 prompts share embedding space.

Method:

  1. Boot qwen via saklas, same setup as v3 main run
  2. For each v3 prompt, h_first @ L59 is already captured per-row
     in qwen v3 sidecars (and is prompt-deterministic per script 36).
     Group-by prompt_id, take any row's h_first → 120 prompt vectors.
     Mean per quadrant → 6 quadrant centroids in qwen h_first space.
  3. For each claude-face: concatenate all per-bundle descriptions
     into one text block, run through qwen with kaomoji instruction,
     capture h_first @ L59, extract from sidecar.
  4. Per face: cosine vs each quadrant centroid → 6-D profile in
     qwen h_first space. Argmax = display label, margin = top1−top2.
  5. Validate against qwen v3 emission with mean-centered profile
     cosine + face-permutation null (same as script 43).

Output:
  data/claude_faces_quadrant_assignment_qwen_descriptions.tsv
    — per face: 6-D profile + argmax label + margin + qwen profile cos
  data/hidden/desc_qwen_emb/*.npz (intermediate, ~228 sidecars)

Compare to script 46 (qwen forward pass on the *face string itself*
— no description corpus needed) for the cleaner version that
sidesteps the description-text register issue entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from saklas import SaklasSession

from llmoji_study.capture import run_sample
from llmoji_study.claude_faces import load_descriptions
from llmoji_study.config import (
    DATA_DIR, MODEL_REGISTRY, PROBE_CATEGORIES, STEERED_AXIS,
)
from llmoji_study.emotional_analysis import apply_hn_split, load_emotional_features
from llmoji_study.hidden_state_io import hidden_state_path, load_hidden_states
from llmoji_study.prompts import Prompt


CANONICAL_MODEL = "qwen"
EXPERIMENT_TAG = "desc_qwen_emb"
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
N_PERM = 1000


def _profile_cosine(p_desc: np.ndarray, p_emit: np.ndarray) -> float:
    pd_c = p_desc - p_desc.mean()
    pe_c = p_emit - p_emit.mean()
    denom = float(np.linalg.norm(pd_c) * np.linalg.norm(pe_c))
    return float("nan") if denom <= 0 else float(pd_c @ pe_c) / denom


def _qwen_quadrant_centroids() -> tuple[dict[str, np.ndarray], int]:
    """Mean h_first @ L59 across v3 prompts grouped by quadrant.
    Returns (centroids by HP/LP/HN-D/HN-S/LN/NB, hidden_dim)."""
    M = MODEL_REGISTRY[CANONICAL_MODEL]
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_first",
        layer=M.preferred_layer, split_hn=True,
    )
    df = df[df["quadrant"].isin(QUADRANT_ORDER)]
    seen: set[str] = set()
    h_per_prompt: dict[str, np.ndarray] = {}
    q_per_prompt: dict[str, str] = {}
    df = df.reset_index(drop=True)
    for i, row in enumerate(df.itertuples(index=False)):
        pid = str(row.prompt_id)
        if pid in seen:
            continue
        seen.add(pid)
        h_per_prompt[pid] = X[i].astype(np.float64)
        q_per_prompt[pid] = str(row.quadrant)
    centroids: dict[str, np.ndarray] = {}
    for q in QUADRANT_ORDER:
        h_list = [h_per_prompt[p] for p, qq in q_per_prompt.items() if qq == q]
        if not h_list:
            raise RuntimeError(f"no v3 prompts for quadrant {q}")
        centroids[q] = np.mean(h_list, axis=0)
    return centroids, X.shape[1]


def _qwen_emission_profiles() -> dict[str, np.ndarray]:
    """Per-face 6-D v3-emission count profile (same as script 43)."""
    M = MODEL_REGISTRY[CANONICAL_MODEL]
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_first",
        layer=M.preferred_layer,
    )
    df, _ = apply_hn_split(df, X)
    df = df[df["quadrant"].isin(QUADRANT_ORDER)]
    out: dict[str, np.ndarray] = {}
    for fw, sub in df.groupby("first_word"):
        prof = np.zeros(6, dtype=np.float64)
        for q, n in sub["quadrant"].value_counts().items():
            prof[QUADRANT_ORDER.index(str(q))] = float(n)
        out[str(fw)] = prof
    return out


def _build_description_text(row: dict) -> str:
    """Concatenate all per-bundle descriptions for a canonical face."""
    parts: list[str] = []
    for d in row.get("descriptions", []):
        text = (d.get("description") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def main() -> None:
    M = MODEL_REGISTRY[CANONICAL_MODEL]

    # Step 1: per-quadrant centroids in qwen h_first space (cheap, sidecar-cached)
    print(f"loading qwen v3 prompt centroids @ L{M.preferred_layer}")
    centroids, hidden_dim = _qwen_quadrant_centroids()
    centroid_mat = np.stack([centroids[q] for q in QUADRANT_ORDER])  # (6, hidden_dim)
    centroid_norms = np.linalg.norm(centroid_mat, axis=1, keepdims=True)
    centroid_mat_n = centroid_mat / np.where(centroid_norms > 0, centroid_norms, 1.0)
    print(f"  centroids built; hidden_dim={hidden_dim}")

    # Step 2: load all face descriptions
    desc_rows = load_descriptions(DATA_DIR / "claude_descriptions.jsonl")
    print(f"loading qwen for description forward passes (n={len(desc_rows)})")
    print(f"  this is the slow step — ~3-10s/face × {len(desc_rows)} = "
          f"~{len(desc_rows) * 5 / 60:.0f}–{len(desc_rows) * 10 / 60:.0f} min")

    # Step 3: boot qwen
    session = SaklasSession.from_pretrained(M.model_id, probes=PROBE_CATEGORIES)
    name, profile = session.extract(STEERED_AXIS)
    session.steer(name, profile)

    # Step 4: per face, run forward pass → sidecar → read h_first @ L59
    h_first_per_face: dict[str, np.ndarray] = {}
    for i, row in enumerate(desc_rows):
        fw = row["kaomoji"]
        text = _build_description_text(row)
        if not text:
            print(f"  [{i+1}/{len(desc_rows)}] {fw}: no description text, skipping")
            continue
        # Truncate very long concatenations to keep generation fast.
        # MiniLM-side weighted-mean already collapses descriptions; here we
        # let qwen process the lot but cap to ~600 chars to avoid context blowup.
        if len(text) > 600:
            text = text[:600]
        synthetic = Prompt(id=f"desc_{i:04d}", valence=0, text=text)
        try:
            sample = run_sample(
                session=session, prompt=synthetic,
                condition="kaomoji_prompted", seed=0,
                hidden_dir=DATA_DIR, experiment=EXPERIMENT_TAG,
                override_max_tokens=1, store_full_trace=False,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(desc_rows)}] {fw}: forward failed — {e}")
            continue
        sidecar = hidden_state_path(DATA_DIR, EXPERIMENT_TAG, sample.row_uuid)
        cap = load_hidden_states(sidecar)
        if M.preferred_layer not in cap.layers:
            print(f"  [{i+1}/{len(desc_rows)}] {fw}: no L{M.preferred_layer} in capture, skipping")
            continue
        h_first_per_face[fw] = cap.layers[M.preferred_layer].h_first.astype(np.float64)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(desc_rows)}] captured")

    print(f"captured h_first for {len(h_first_per_face)}/{len(desc_rows)} faces")

    # Step 5: per face, cosine vs each centroid → 6-D profile
    rows_out: list[dict] = []
    for fw, h in h_first_per_face.items():
        h_n = h / max(float(np.linalg.norm(h)), 1e-12)
        prof = centroid_mat_n @ h_n   # (6,)
        order = np.argsort(-prof)
        top_q = QUADRANT_ORDER[order[0]]
        margin = float(prof[order[0]] - prof[order[1]])
        row: dict = {"first_word": fw, "soft_assignment": top_q, "soft_margin": margin}
        for k, q in enumerate(QUADRANT_ORDER):
            row[f"proto_sim_{q}"] = float(prof[k])
        rows_out.append(row)
    df_out = pd.DataFrame(rows_out)

    print(f"argmax distribution: " +
          ", ".join(f"{q}={int((df_out.soft_assignment == q).sum())}" for q in QUADRANT_ORDER))

    # Step 6: validate against qwen v3 emission
    qwen_emission = _qwen_emission_profiles()
    shared = [fw for fw in h_first_per_face if fw in qwen_emission]
    print(f"shared faces (qwen-emission ∩ described): {len(shared)}")

    obs_per_face: list[float] = []
    fw_to_idx = {r["first_word"]: i for i, r in enumerate(rows_out)}
    proto_cols = [f"proto_sim_{q}" for q in QUADRANT_ORDER]
    for fw in shared:
        i = fw_to_idx[fw]
        prof = np.asarray([rows_out[i][c] for c in proto_cols])
        obs_per_face.append(_profile_cosine(prof, qwen_emission[fw]))
    obs_arr = np.asarray(obs_per_face, dtype=np.float64)
    obs_mean = float(np.nanmean(obs_arr))

    rng = np.random.default_rng(20260502)
    null_means = np.empty(N_PERM)
    shared_arr = np.asarray(shared)
    for t in range(N_PERM):
        shuffled = rng.permutation(shared_arr)
        cs: list[float] = []
        for k_fw, fw in enumerate(shared_arr):
            i = fw_to_idx[fw]
            prof = np.asarray([rows_out[i][c] for c in proto_cols])
            cs.append(_profile_cosine(prof, qwen_emission[shuffled[k_fw]]))
        null_means[t] = float(np.nanmean(cs))
    perm_p = float((null_means >= obs_mean).sum() + 1) / (N_PERM + 1)
    print(
        f"qwen aggregate profile cosine = {obs_mean:+.3f}  "
        f"(perm null mean = {float(np.mean(null_means)):+.3f}, perm-p = {perm_p:.3f})"
    )

    df_out["qwen_profile_cos"] = df_out["first_word"].map(
        {fw: float(c) for fw, c in zip(shared, obs_arr)},
    )
    for k, q in enumerate(QUADRANT_ORDER):
        df_out[f"qwen_emission_{q}"] = df_out["first_word"].map(
            {fw: float(qwen_emission[fw][k]) for fw in shared},
        )

    out_tsv = DATA_DIR / "claude_faces_quadrant_assignment_qwen_descriptions.tsv"
    df_out.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
