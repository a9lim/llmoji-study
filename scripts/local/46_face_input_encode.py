# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Encode union of all-models faces through `--model` via face-input forward pass.

Faces come from 4 sources (face union):
  - gemma v3 emission (~52 unique kaomoji)
  - qwen v3 emission (~89 unique kaomoji)
  - ministral v3 emission (~231 unique kaomoji, ~204 are emoji-in-parens noise)
  - claude-faces corpus (228 from a9lim/llmoji)

Each face encoded once through the chosen encoder model with the same
forward-pass mode: face string as user message, kaomoji-instruction
system prompt, capture h_first @ M.preferred_layer.

Per-face quadrant ground truth is the **summed** emission distribution
across all 3 v3 models (`total_emit_*` columns). e.g. (⊙_⊙) gets
HN-S=99+0+1=100, NB=0+1+0=1, total=101 — the quadrant blend reflects
how often it appeared in each prompt category aggregated across all
models. Faces only in claude-faces have total_emit_count == 0 and
get NN-classified downstream by `44_face_input_pc_space.py`.

Usage:
  python scripts/local/46_face_input_encode.py --model qwen
  python scripts/local/46_face_input_encode.py --model gemma
  python scripts/local/46_face_input_encode.py --model ministral

Output:
  data/face_h_first_<model>.parquet
    — one row per face in the union, columns:
      first_word, is_{gemma,qwen,ministral}_emitted, is_claude,
      total_emit_count, total_emit_<HP|LP|HN-D|HN-S|LN|NB>,
      {gemma,qwen,ministral}_emit_count + per-quadrant breakdowns,
      h0000..h<hidden_dim-1>
  data/hidden/face_<model>_emb/*.npz (intermediate sidecars)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.claude_faces import load_embeddings as load_claude_embeddings
from llmoji_study.config import (
    DATA_DIR, KAOMOJI_INSTRUCTION, MODEL_REGISTRY, PROBE_CATEGORIES,
    STEERED_AXIS,
)
from llmoji_study.emotional_analysis import load_emotional_features


V3_MODELS = ["gemma", "qwen", "ministral"]
QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--model", required=True, choices=list(MODEL_REGISTRY.keys()),
        help="encoder model (key into MODEL_REGISTRY)",
    )
    return p.parse_args()


def _all_models_face_metadata() -> dict[str, dict]:
    """Per-face emission counts across all 3 v3 models."""
    out: dict[str, dict] = {}
    for m in V3_MODELS:
        M = MODEL_REGISTRY[m]
        df, _ = load_emotional_features(
            str(M.emotional_data_path), DATA_DIR,
            experiment=M.experiment, which="h_first",
            layer=M.preferred_layer, split_hn=True,
        )
        df = df[df["quadrant"].isin(QUADRANT_ORDER)]
        df = df[df["first_word"].astype(str).str.startswith("(")]
        for fw, sub in df.groupby("first_word"):
            fw_s = str(fw)
            if fw_s not in out:
                rec: dict = {"total_emit_count": 0}
                for q in QUADRANT_ORDER:
                    rec[f"total_emit_{q}"] = 0
                for mm in V3_MODELS:
                    rec[f"{mm}_emit_count"] = 0
                    for q in QUADRANT_ORDER:
                        rec[f"{mm}_emit_{q}"] = 0
                out[fw_s] = rec
            counts = {q: 0 for q in QUADRANT_ORDER}
            for q, n in sub["quadrant"].value_counts().items():
                counts[str(q)] = int(n)
            total = sum(counts.values())
            out[fw_s][f"{m}_emit_count"] = total
            out[fw_s]["total_emit_count"] += total
            for q in QUADRANT_ORDER:
                out[fw_s][f"{m}_emit_{q}"] = counts[q]
                out[fw_s][f"total_emit_{q}"] += counts[q]
    return out


def _encode_via_saklas(M, union: list[str], experiment_tag: str) -> dict[str, np.ndarray]:
    """Saklas-mode forward pass: probes + steering + sidecar capture.
    Used for gemma / qwen / ministral (probe-calibrated)."""
    from saklas import SaklasSession
    from llmoji_study.capture import run_sample
    from llmoji_study.hidden_state_io import hidden_state_path, load_hidden_states
    from llmoji_study.prompts import Prompt

    session = SaklasSession.from_pretrained(M.model_id, probes=PROBE_CATEGORIES)
    name, profile = session.extract(STEERED_AXIS)
    session.steer(name, profile)

    h_first_per_face: dict[str, np.ndarray] = {}
    for i, fw in enumerate(union):
        synthetic = Prompt(id=f"face_{i:04d}", valence=0, text=fw)
        try:
            sample = run_sample(
                session=session, prompt=synthetic,
                condition="kaomoji_prompted", seed=0,
                hidden_dir=DATA_DIR, experiment=experiment_tag,
                override_max_tokens=1, store_full_trace=False,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(union)}] {fw}: forward failed — {e}")
            continue
        sidecar = hidden_state_path(DATA_DIR, experiment_tag, sample.row_uuid)
        cap = load_hidden_states(sidecar)
        if M.preferred_layer not in cap.layers:
            print(f"  [{i+1}/{len(union)}] {fw}: no L{M.preferred_layer} in capture, skipping")
            continue
        h_first_per_face[fw] = cap.layers[M.preferred_layer].h_first.astype(np.float64)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(union)}] captured")
    return h_first_per_face


def _encode_via_raw_hf(M, union: list[str]) -> dict[str, np.ndarray]:
    """Raw HF forward pass: AutoModelForCausalLM + output_hidden_states.
    No probes, no steering. Captures last-input-position residual at
    M.preferred_layer (index into transformers' hidden_states tuple,
    where 0=embedding output and N=output of layer N).

    Used for models without saklas probe calibration or with
    architectures saklas can't load (e.g. nemotron_h Mamba/hybrid)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"  device={device}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        M.model_id, trust_remote_code=M.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        M.model_id, trust_remote_code=M.trust_remote_code,
        torch_dtype=dtype, device_map=device,
    )
    model.eval()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    print(f"  chat_template available: {has_chat_template}")

    h_first_per_face: dict[str, np.ndarray] = {}
    for i, fw in enumerate(union):
        # Mirror the saklas-side prompt structure: kaomoji-instruction
        # injected into the user message, no system role.
        user_msg = KAOMOJI_INSTRUCTION + fw
        if has_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = user_msg
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
        except Exception as e:
            print(f"  [{i+1}/{len(union)}] {fw}: forward failed — {e}")
            continue
        hs_tuple = out.hidden_states
        layer_idx = M.preferred_layer
        if layer_idx is None or layer_idx < 0 or layer_idx >= len(hs_tuple):
            layer_idx = len(hs_tuple) - 1
        h = hs_tuple[layer_idx][0, -1, :].float().cpu().numpy()
        h_first_per_face[fw] = h.astype(np.float64)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(union)}] captured")
    return h_first_per_face


def main() -> None:
    args = _parse_args()
    encoder = args.model
    M = MODEL_REGISTRY[encoder]
    experiment_tag = f"face_{encoder}_emb"

    all_meta = _all_models_face_metadata()
    print("v3 emission vocab sizes:")
    for m in V3_MODELS:
        n = sum(1 for v in all_meta.values() if v[f"{m}_emit_count"] > 0)
        print(f"  {m}: {n}")
    print(f"  union of all 3 v3 vocabs: {len(all_meta)}")

    claude_fw, _, _ = load_claude_embeddings(
        DATA_DIR / "claude_faces_embed_description.parquet"
    )
    claude_set = set(claude_fw)
    print(f"claude-faces from corpus: {len(claude_set)}")

    all_emitted = set(all_meta.keys())
    union = sorted(all_emitted | claude_set)
    print(f"face union (any-source): {len(union)}")
    only_claude = claude_set - all_emitted
    only_emitted = all_emitted - claude_set
    overlap = all_emitted & claude_set
    print(f"  v3-emitted only: {len(only_emitted)}  "
          f"claude-only: {len(only_claude)}  "
          f"overlap: {len(overlap)}")

    # Drop ministral-only-not-claude faces — most are emoji-in-parens noise
    # rather than canonical kaomoji. Filter at the encoder so they never
    # reach the parquet (vs. filtering downstream).
    to_drop: set[str] = set()
    for fw, meta in all_meta.items():
        if (meta["ministral_emit_count"] > 0
                and meta["gemma_emit_count"] == 0
                and meta["qwen_emit_count"] == 0
                and fw not in claude_set):
            to_drop.add(fw)
    print(f"  dropping ministral-only-not-claude: {len(to_drop)}")
    union = [fw for fw in union if fw not in to_drop]
    print(f"  filtered union: {len(union)}")

    print(f"loading {encoder} for face forward passes (n={len(union)})  "
          f"[{'saklas' if M.use_saklas else 'raw HF'}]")
    if M.use_saklas:
        h_first_per_face = _encode_via_saklas(M, union, experiment_tag)
    else:
        h_first_per_face = _encode_via_raw_hf(M, union)

    print(f"captured h_first for {len(h_first_per_face)}/{len(union)} faces")

    rows: list[dict] = []
    for fw in union:
        if fw not in h_first_per_face:
            continue
        meta = all_meta.get(fw, {})
        rec: dict = {
            "first_word": fw,
            "is_gemma_emitted": int(meta.get("gemma_emit_count", 0)) > 0,
            "is_qwen_emitted": int(meta.get("qwen_emit_count", 0)) > 0,
            "is_ministral_emitted": int(meta.get("ministral_emit_count", 0)) > 0,
            "is_claude": fw in claude_set,
            "total_emit_count": int(meta.get("total_emit_count", 0)),
        }
        for q in QUADRANT_ORDER:
            rec[f"total_emit_{q}"] = int(meta.get(f"total_emit_{q}", 0))
        for m in V3_MODELS:
            rec[f"{m}_emit_count"] = int(meta.get(f"{m}_emit_count", 0))
            for q in QUADRANT_ORDER:
                rec[f"{m}_emit_{q}"] = int(meta.get(f"{m}_emit_{q}", 0))
        for k, v in enumerate(h_first_per_face[fw].tolist()):
            rec[f"h{k:04d}"] = float(v)
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    print(
        f"output stats: total={len(df_out)}, "
        f"gemma_emitted={int(df_out.is_gemma_emitted.sum())}, "
        f"qwen_emitted={int(df_out.is_qwen_emitted.sum())}, "
        f"ministral_emitted={int(df_out.is_ministral_emitted.sum())}, "
        f"claude={int(df_out.is_claude.sum())}, "
        f"any_v3={int((df_out.total_emit_count > 0).sum())}"
    )

    out_path = DATA_DIR / f"face_h_first_{encoder}.parquet"
    df_out.to_parquet(out_path, index=False)
    h_dim = sum(1 for c in df_out.columns if c.startswith("h") and c[1:].isdigit())
    print(f"wrote {out_path}  (n_faces={len(df_out)}, hidden_dim={h_dim})")


if __name__ == "__main__":
    main()
