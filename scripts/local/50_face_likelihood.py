# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Bayesian-inversion quadrant classifier — log P(face | prompt) under the local LM.

For each (model, face f, v3 emotional prompt p), build the same chat-
templated input v3 generation uses (KAOMOJI_INSTRUCTION + p.text via
the user message, then add_generation_prompt=True), append the
tokenized face string, and teacher-force forward to compute
``log P(face_tokens | prefix)``. Aggregate per-prompt log-probs into
per-face × per-quadrant means → 6-D log-prob distribution; argmax
gives the predicted quadrant. Length cancels under within-face
softmax over quadrants, so raw sum-log-prob is a valid score.

The score for a face f under quadrant q is
    score(f, q) = mean_{p in q} log P(f | p)
predicted_quadrant(f) = argmax_q score(f, q).
softmax over q gives a confidence distribution per face.

This is approach (1) from the 2026-05-03 brainstorm: use the local
model as a likelihood evaluator rather than a neighbor-lookup. Skips
joint-PCA + cosine-NN entirely — every claude face gets a quadrant
distribution regardless of whether it has neighbors in v3-emitted
space.

Validation: for v3-emitted faces (total_emit_count > 0) we have
ground-truth empirical-emission distributions over quadrants. Under
self-consistency, the predicted argmax should usually match the
empirical-emission majority.

Reads the canonical face union from ``data/v3_face_union.parquet``
(built by ``45_build_face_union.py``). Must be regenerated whenever
the v3 main lineup changes; encoder-invariant per face.

Always runs the full 120 prompts × 573 faces — batching makes that
fast enough that the old --pilot gate is no longer worth the
operational complexity.

Usage:
  python scripts/local/50_face_likelihood.py --model gemma
  python scripts/local/50_face_likelihood.py --model qwen
  python scripts/local/50_face_likelihood.py --model rinna_jp_3_6b --prompt-lang jp
  python scripts/local/50_face_likelihood.py --model rinna_bilingual_4b --prompt-lang en

Outputs:
  data/face_likelihood_<model>{,_jp}.parquet
    — one row per (face, prompt_id), columns:
      first_word, prompt_id, quadrant, log_prob, n_face_tokens,
      log_prob_per_token
  data/face_likelihood_<model>{,_jp}_summary.tsv
    — one row per face, columns:
      first_word, n_prompts_<q> per quadrant,
      mean_log_prob_<q> per quadrant,
      softmax_<q> per quadrant,
      predicted_quadrant (argmax),
      max_softmax (confidence), n_face_tokens,
      total_emit_<q> per quadrant (from v3_face_union parquet),
      empirical_majority_quadrant (None if total_emit_count == 0),
      argmax_matches_empirical (None if no ground truth).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch

from llmoji_study.capture import (
    build_messages,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
    maybe_override_rinna_chat_template,
)
from llmoji_study.config import (
    DATA_DIR,
    KAOMOJI_INSTRUCTION,
    KAOMOJI_INSTRUCTION_JP,
    MODEL_REGISTRY,
    PROBE_CATEGORIES,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS, EmotionalPrompt
from llmoji_study.emotional_prompts_jp import EMOTIONAL_PROMPTS_JP
from llmoji_study.prompts import Prompt


QUADRANT_ORDER = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
# Per-face evaluation batch size (faces stacked into a single forward
# pass per prompt). 31B model on M5 Max bf16 — keep small enough to
# fit; 64 chosen so that prefix(~50)+face(~15) padded to ~70 tokens
# x 64 rows ~= 4500 tokens per fwd. Tune up if memory headroom allows.
DEFAULT_FACE_BATCH = 64


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--model", required=True,
        choices=["gemma", "qwen", "ministral",
                 "llama32_3b", "glm47_flash", "gpt_oss_20b",
                 "deepseek_v2_lite", "qwen35_27b", "gemma3_27b",
                 "phi4_mini", "granite",
                 "rinna_jp_3_6b", "rinna_bilingual_4b"],
        help="encoder model. gemma/qwen/ministral are probe-calibrated v3 "
             "models; the rest are uncalibrated (voting only)",
    )
    p.add_argument("--prompt-lang", default="en", choices=["en", "jp"],
                   help="language of the kaomoji-ask portion of the prompt. "
                        "'jp' swaps in KAOMOJI_INSTRUCTION_JP — useful for "
                        "Japanese-trained encoders (rinna_*) where the EN ask "
                        "underperforms. Output suffix '_jp' when set.")
    p.add_argument("--prompt-body", default="en", choices=["en", "jp"],
                   help="language of the prompt body. 'en' uses the canonical "
                        "120-prompt EMOTIONAL_PROMPTS; 'jp' uses the 120-prompt "
                        "JP-translated set from EMOTIONAL_PROMPTS_JP (paired "
                        "1:1 by ID). Combined with --prompt-lang jp this gives "
                        "a full-Japanese run; output suffix becomes '_jpfull' "
                        "(or '_jpbody' if only --prompt-body jp).")
    p.add_argument("--face-batch", type=int, default=DEFAULT_FACE_BATCH,
                   help=f"faces per forward pass (default {DEFAULT_FACE_BATCH})")
    p.add_argument("--summary-topk", type=int, default=None,
                   help="if set, the per-(face, quadrant) score is the mean "
                        "of the top-k highest-log-prob prompts only, instead "
                        "of mean-over-all. Empirically (Claude-GT, May 2026) "
                        "k=5 is a robust one-size-fits-all on 20-prompts-per-"
                        "quadrant runs. Set 0 or omit to use mean-over-all.")
    p.add_argument("--no-incremental", action="store_true",
                   help="Re-score every (face, prompt) cell from scratch. "
                        "Default behavior is incremental: load the existing "
                        "per-cell parquet (if any), identify cells not yet "
                        "scored, score only those, and concat-write back to "
                        "the same parquet. Useful after the union grows "
                        "(e.g. new Claude runs land) to avoid re-scoring "
                        "the unchanged ~91%% of cells.")
    return p.parse_args()


def _quadrant_of(prompt: EmotionalPrompt) -> str:
    """Map EmotionalPrompt to the 6-quadrant code (HN bisected on pad_dominance)."""
    q = prompt.quadrant
    if q == "HN":
        return "HN-D" if prompt.pad_dominance > 0 else "HN-S"
    return q


def _load_face_union(model_short: str) -> pd.DataFrame:
    """Load face union + total_emit_* ground truth from the canonical
    `data/v3_face_union.parquet`.

    The face list and total_emit_* columns are encoder-invariant — they
    come from v3 emission distributions, not from any one model's
    encoded vectors. The canonical union is built by
    ``45_build_face_union.py``. ``model_short`` is unused here; kept for
    call-site compatibility.
    """
    del model_short  # no longer used (was per-encoder before refactor)
    p = DATA_DIR / "v3_face_union.parquet"
    if not p.exists():
        raise SystemExit(
            f"missing {p} — run "
            f"`python scripts/local/45_build_face_union.py` first"
        )
    df = pd.read_parquet(p)
    keep = ["first_word", "is_claude", "total_emit_count"] + [
        f"total_emit_{q}" for q in QUADRANT_ORDER
    ]
    return df[keep].copy()


def _build_prefix_ids(
    tokenizer,
    prompt: EmotionalPrompt,
    *,
    instruction: str = KAOMOJI_INSTRUCTION,
) -> torch.Tensor:
    """Chat-template prefix ending right before the assistant turn starts.
    Routes through ``saklas.build_chat_input`` with ``thinking=False`` so
    the prefix mirrors what v3 generation feeds the model — including the
    ``enable_thinking=False`` flag for templates that support it (gemma 4,
    qwen 3.6). Without this the prefix would land on the thinking-channel
    header instead of the response-channel header and likelihoods would
    score against the wrong distribution.

    ``instruction`` swaps in a non-default kaomoji ask (e.g.
    ``KAOMOJI_INSTRUCTION_JP`` for Japanese-trained encoders).
    """
    from saklas.core.generation import build_chat_input
    p = Prompt(id=prompt.id, valence=prompt.valence, text=prompt.text)
    msgs = build_messages(p, kaomoji_instructed=True, instruction_override=instruction)
    ids = build_chat_input(tokenizer, msgs, thinking=False, add_generation_prompt=True)
    return ids[0]


def _expand_kv_cache(cache, batch_size: int):
    """Tile a batch=1 transformer KV cache to batch_size, returning a
    fresh cache so the model's in-place ``cache.update`` during the
    suffix forward doesn't mutate the source prefix cache.

    Uses ``copy.deepcopy`` + ``batch_repeat_interleave`` for modern
    ``DynamicCache`` (transformers ≥4.40), per-layer key/value clone
    for the older tuple-of-tuples format. The deepcopy approach is
    structure-agnostic across cache layer types (DynamicLayer,
    DynamicSlidingWindowLayer, etc.) which the per-attribute approach
    is not."""
    # Modern DynamicCache (and subclasses) implement batch_repeat_interleave.
    if hasattr(cache, "batch_repeat_interleave"):
        import copy
        expanded = copy.deepcopy(cache)
        expanded.batch_repeat_interleave(batch_size)
        return expanded
    # Legacy tuple-of-tuples format.
    if isinstance(cache, tuple):
        return tuple(
            (k.expand(batch_size, -1, -1, -1).contiguous(),
             v.expand(batch_size, -1, -1, -1).contiguous())
            for k, v in cache
        )
    raise TypeError(f"unsupported KV cache type: {type(cache).__name__}")


def _score_faces_for_prompt(
    model: torch.nn.Module,
    tokenizer,
    prefix_ids: torch.Tensor,
    faces: list[str],
    *,
    face_batch: int,
    device: torch.device,
) -> list[tuple[float, int]]:
    """Return [(sum_log_prob, n_face_tokens), ...] for each face under this prompt.

    Optimization: prefix is forwarded ONCE per prompt (not once per face
    batch). The resulting KV cache is tiled to the face-batch dimension
    and reused across all faces; only the face-token suffix is forwarded
    per batch. For long prefixes (e.g. ministral's 565-token chat
    template) this is the difference between O(n_faces / face_batch) full
    prefix forwards and exactly one — ~30x speedup on ministral, ~3-4x
    on gemma/qwen.
    """
    face_id_lists: list[list[int]] = []
    for f in faces:
        ids = tokenizer(f, add_special_tokens=False)["input_ids"]
        face_id_lists.append(list(ids) if ids else [])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    prefix_len = prefix_ids.shape[0]
    results: list[tuple[float, int]] = [(0.0, 0)] * len(faces)

    # Step 1: prefix forward (batch=1) → cache + last-token logits.
    # Last-prefix-token logits predict face[0] for every face; cache
    # gets reused as the shared past for the face-suffix forward.
    prefix_in = prefix_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_in, use_cache=True)
    prefix_kv = prefix_out.past_key_values
    # Keep last-token logits as fp32 on-device for downstream concat.
    prefix_last_logits = prefix_out.logits[:, -1, :].float()  # [1, vocab]
    del prefix_out

    # Step 2: process face batches with the cached prefix.
    for start in range(0, len(faces), face_batch):
        batch_idx = list(range(start, min(start + face_batch, len(faces))))
        batch_face_ids = [face_id_lists[i] for i in batch_idx]
        valid_mask = [len(ids) > 0 for ids in batch_face_ids]
        if not any(valid_mask):
            continue
        max_face_len = max(
            len(ids) for ids, ok in zip(batch_face_ids, valid_mask) if ok
        )
        n = len(batch_idx)

        # Build face-only inputs (no prefix in the input tensor — it's in
        # the cache). For ok=False rows we still pass a 1-token pad so
        # the batched forward shape is uniform; their results stay (0, 0).
        face_inputs = torch.full((n, max_face_len), pad_id, dtype=torch.long)
        face_attn_local = torch.zeros((n, max_face_len), dtype=torch.long)
        for i, (ids, ok) in enumerate(zip(batch_face_ids, valid_mask)):
            if ok:
                fl = len(ids)
                face_inputs[i, :fl] = torch.tensor(ids, dtype=torch.long)
                face_attn_local[i, :fl] = 1
        face_inputs = face_inputs.to(device)
        face_attn_local = face_attn_local.to(device)

        # Attention mask must span (prefix + face) so the attention layer
        # knows the full causal context length includes the cached prefix.
        full_attn = torch.cat([
            torch.ones((n, prefix_len), dtype=torch.long, device=device),
            face_attn_local,
        ], dim=1)

        # Tile cache to batch=n. .contiguous() copy guarantees the model's
        # in-place cache.update during this forward doesn't bleed into the
        # source prefix_kv (which we reuse for the next batch).
        expanded_kv = _expand_kv_cache(prefix_kv, n)

        with torch.no_grad():
            out = model(
                input_ids=face_inputs,
                attention_mask=full_attn,
                past_key_values=expanded_kv,
                use_cache=False,
            )
        # out.logits shape: [n, max_face_len, vocab]. With cached prefix
        # of length P, out.logits[i, t] predicts the token at absolute
        # position P+t+1 in row i. So:
        #   face[0]   ← prefix_last_logits[0]  (broadcast, since prefix is shared)
        #   face[t≥1] ← out.logits[i, t-1]
        for i, (ids, ok) in enumerate(zip(batch_face_ids, valid_mask)):
            if not ok:
                results[batch_idx[i]] = (0.0, 0)
                continue
            fl = len(ids)
            if fl == 1:
                face_logits = prefix_last_logits[0:1, :]
            else:
                face_logits = torch.cat([
                    prefix_last_logits[0:1, :],
                    out.logits[i, :fl - 1, :].float(),
                ], dim=0)
            log_probs = torch.log_softmax(face_logits, dim=-1)
            targets = torch.tensor(ids, dtype=torch.long, device=device)
            face_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
            results[batch_idx[i]] = (float(face_lp), int(fl))
        del out, expanded_kv

    return results


def _empirical_majority(row: pd.Series) -> str | None:
    """Quadrant with the most v3 emissions for this face, or None if total == 0."""
    total = int(row.get("total_emit_count", 0))
    if total <= 0:
        return None
    counts = {q: int(row.get(f"total_emit_{q}", 0)) for q in QUADRANT_ORDER}
    best_q = max(counts, key=lambda k: counts[k])
    if counts[best_q] == 0:
        return None
    return best_q


def _summarize(
    rows_df: pd.DataFrame,
    faces_df: pd.DataFrame,
    *,
    summary_topk: int | None = None,
) -> pd.DataFrame:
    """Per (face, quadrant) -> mean log_prob; argmax + softmax per face.

    ``summary_topk`` (default ``None`` = mean over all prompts) takes
    the mean of the top-k highest-log-prob prompts per quadrant only —
    a noise-reducing aggregation. Empirically (Claude-GT, May 2026):
    different encoders prefer different k (gemma k=3, granite k=5,
    qwen k=2, ministral k=3, gpt_oss k=all). k=5 is a reasonable
    one-size-fits-all default for 20-prompts-per-quadrant runs.
    """
    if summary_topk is None:
        agg = rows_df.groupby(["first_word", "quadrant"], as_index=False).agg(
            mean_log_prob=("log_prob", "mean"),
            n_prompts=("log_prob", "count"),
            n_face_tokens=("n_face_tokens", "first"),
        )
    else:
        # Top-k per (face, quadrant) — sort prompts by log_prob desc, take
        # mean of top-k. n_prompts == k_effective per quadrant.
        def _topk_mean(s: pd.Series) -> float:
            return float(s.nlargest(summary_topk).mean())
        topk = (rows_df.groupby(["first_word", "quadrant"])["log_prob"]
                       .apply(_topk_mean).rename("mean_log_prob").reset_index())
        n_per_q = (rows_df.groupby(["first_word", "quadrant"])["log_prob"]
                          .count().rename("n_prompts_total").reset_index())
        topk = topk.merge(n_per_q, on=["first_word", "quadrant"], how="left")
        topk["n_prompts"] = topk["n_prompts_total"].clip(upper=summary_topk)
        n_tok = (rows_df.groupby(["first_word", "quadrant"])["n_face_tokens"]
                        .first().rename("n_face_tokens").reset_index())
        agg = topk.merge(n_tok, on=["first_word", "quadrant"], how="left")
    grouped = agg
    pivot_lp = grouped.pivot(index="first_word", columns="quadrant", values="mean_log_prob")
    pivot_n = grouped.pivot(index="first_word", columns="quadrant", values="n_prompts")
    n_tok = grouped.groupby("first_word")["n_face_tokens"].first()

    out_rows: list[dict] = []
    for fw, lp_row in pivot_lp.iterrows():
        rec: dict = {"first_word": fw}
        present_q = [
            q for q in QUADRANT_ORDER
            if q in pivot_lp.columns and not bool(pd.isna(lp_row[q]))
        ]
        for q in QUADRANT_ORDER:
            n_q = (
                int(pivot_n.loc[fw, q])
                if q in pivot_n.columns and not bool(pd.isna(pivot_n.loc[fw, q]))
                else 0
            )
            rec[f"n_prompts_{q}"] = n_q
            mlp = (
                float(lp_row[q])
                if q in pivot_lp.columns and not bool(pd.isna(lp_row[q]))
                else float("nan")
            )
            rec[f"mean_log_prob_{q}"] = mlp
        if present_q:
            lps = np.array([rec[f"mean_log_prob_{q}"] for q in present_q], dtype=np.float64)
            shifted = lps - lps.max()
            ex = np.exp(shifted)
            probs = ex / ex.sum()
            sm: dict[str, float] = {q: 0.0 for q in QUADRANT_ORDER}
            for q, pr in zip(present_q, probs):
                sm[q] = float(pr)
            for q in QUADRANT_ORDER:
                rec[f"softmax_{q}"] = sm[q]
            argmax_q = present_q[int(np.argmax(probs))]
            rec["predicted_quadrant"] = argmax_q
            rec["max_softmax"] = float(np.max(probs))
        else:
            for q in QUADRANT_ORDER:
                rec[f"softmax_{q}"] = float("nan")
            rec["predicted_quadrant"] = ""
            rec["max_softmax"] = float("nan")
        rec["n_face_tokens"] = int(n_tok.loc[fw]) if fw in n_tok.index else 0
        out_rows.append(rec)

    summary = pd.DataFrame(out_rows)
    meta = faces_df.copy()
    meta["empirical_majority_quadrant"] = meta.apply(_empirical_majority, axis=1)
    keep = ["first_word", "is_claude", "total_emit_count", "empirical_majority_quadrant"] + [
        f"total_emit_{q}" for q in QUADRANT_ORDER
    ]
    summary = summary.merge(meta[keep], on="first_word", how="left")

    def _match(row: pd.Series) -> object:
        emp = row.get("empirical_majority_quadrant")
        if not isinstance(emp, str) or not emp:
            return None
        pred = row.get("predicted_quadrant", "")
        return bool(pred == emp)

    summary["argmax_matches_empirical"] = summary.apply(_match, axis=1)
    return summary


def main() -> None:
    args = _parse_args()
    M = MODEL_REGISTRY[args.model]
    print(f"model: {M.short_name} ({M.model_id})")
    instruction = KAOMOJI_INSTRUCTION_JP if args.prompt_lang == "jp" else KAOMOJI_INSTRUCTION
    # Optional: replace the bare KAOMOJI ask with an introspection
    # preamble (canonical v7 in INTROSPECTION_PREAMBLE since 2026-05-04).
    # Mirrors script 03's plumbing — the preamble's own integrated
    # kaomoji ask becomes the sole instruction via the
    # ``instruction_override`` codepath in build_messages. Used to test
    # whether priming the LM head changes the face/state coupling
    # measured by face_likelihood.
    import os as _os
    _preamble_file = _os.environ.get("LLMOJI_PREAMBLE_FILE")
    if _preamble_file:
        instruction = Path(_preamble_file).read_text()
        print(f"  preamble override: {_preamble_file} ({len(instruction)} chars; "
              f"replaces KAOMOJI ask)")
    print(f"prompt-lang: {args.prompt_lang}  (kaomoji ask: {instruction!r:.100})")

    faces_df = _load_face_union(M.short_name)
    print(f"face union from data/v3_face_union.parquet: n={len(faces_df)}")
    n_emitted = int((faces_df["total_emit_count"] > 0).sum())
    n_claude = int(faces_df["is_claude"].sum())
    print(f"  v3-emitted: {n_emitted}  is_claude: {n_claude}")

    if args.prompt_body == "jp":
        prompts = list(EMOTIONAL_PROMPTS_JP)
        print(f"  prompt body: jp (claude-translated, paired 1:1 with EN by id)")
    else:
        prompts = list(EMOTIONAL_PROMPTS)
    faces = faces_df["first_word"].astype(str).tolist()
    print(f"  using {len(prompts)} prompts x {len(faces)} faces  "
          f"= {len(prompts) * len(faces)} (face, prompt) cells")

    # Compute output paths upfront so we can check existing scoring.
    if args.prompt_lang == "jp" and args.prompt_body == "jp":
        suffix = "_jpfull"
    elif args.prompt_body == "jp":
        suffix = "_jpbody"
    elif args.prompt_lang == "jp":
        suffix = "_jp"
    else:
        suffix = ""
    # If a preamble override was supplied, tag the output so it doesn't
    # clobber the canonical face_likelihood TSV. Reads the file's stem
    # (e.g. "introspection_v7" → suffix "_v7primed").
    if _preamble_file:
        stem = Path(_preamble_file).stem.replace("introspection_", "")
        suffix = f"{suffix}_{stem}primed"
    rows_path = DATA_DIR / f"face_likelihood_{M.short_name}{suffix}.parquet"

    # Incremental: load existing scoring (if any) and skip already-scored
    # (face, prompt_id) cells. Default behavior; --no-incremental forces
    # full rescore.
    incremental = not args.no_incremental
    existing_rows: pd.DataFrame | None = None
    already_scored: set[tuple[str, str]] = set()
    if incremental and rows_path.exists():
        existing_rows = pd.read_parquet(rows_path)
        already_scored = set(zip(
            existing_rows["first_word"].astype(str),
            existing_rows["prompt_id"].astype(str),
        ))
        print(f"  incremental: loaded {len(existing_rows)} existing rows "
              f"from {rows_path.name}")

    # Plan the work per prompt — what (face, prompt) cells need scoring?
    faces_per_prompt: dict[str, list[str]] = {}
    n_to_score = 0
    for prompt in prompts:
        ftodo = [f for f in faces if (f, prompt.id) not in already_scored]
        faces_per_prompt[prompt.id] = ftodo
        n_to_score += len(ftodo)
    total_cells = len(faces) * len(prompts)
    print(f"  cells to score: {n_to_score} / {total_cells}  "
          f"(skipping {total_cells - n_to_score} already-scored)")

    new_rows: list[dict] = []
    if n_to_score > 0:
        print(f"loading {M.model_id} via SaklasSession ...")
        t_load = time.time()
        from saklas import SaklasSession  # noqa: E402
        probes = PROBE_CATEGORIES if M.probe_calibrated else []
        if not M.probe_calibrated:
            print(f"  {M.short_name}: uncalibrated (probes=[]); face_likelihood "
                  f"only reads LM-head logits, so this is fine")
        with SaklasSession.from_pretrained(
            M.model_id, device="auto", probes=probes,
        ) as session:
            if maybe_override_ministral_chat_template(session):
                print(f"  ministral: overrode chat_template with FP8-instruct's "
                      f"({len(session.tokenizer.chat_template)} chars) so "
                      f"prefix matches v3 generation")
            if maybe_override_gpt_oss_chat_template(session):
                print(f"  gpt_oss: pinned harmony `final` channel in chat_template")
            if maybe_override_rinna_chat_template(session):
                print(f"  rinna: installed native ユーザー:/システム: chat_template "
                      f"({len(session.tokenizer.chat_template)} chars)")
            print(f"  loaded in {time.time() - t_load:.1f}s")
            model = session.model
            tokenizer = session.tokenizer
            device = session.device
            model.train(False)  # disable dropout etc — inference mode

            t0 = time.time()
            for pi, prompt in enumerate(prompts):
                ftodo = faces_per_prompt[prompt.id]
                if not ftodo:
                    continue  # all faces already scored for this prompt
                q = _quadrant_of(prompt)
                prefix_ids = _build_prefix_ids(tokenizer, prompt, instruction=instruction)
                t_p = time.time()
                scored = _score_faces_for_prompt(
                    model, tokenizer, prefix_ids, ftodo,
                    face_batch=args.face_batch, device=device,
                )
                for face, (lp, n_tok) in zip(ftodo, scored):
                    new_rows.append({
                        "first_word": face,
                        "prompt_id": prompt.id,
                        "quadrant": q,
                        "log_prob": float(lp),
                        "n_face_tokens": int(n_tok),
                        "log_prob_per_token": float(lp / n_tok) if n_tok > 0 else float("nan"),
                    })
                print(f"  [{pi+1}/{len(prompts)}] {prompt.id} ({q}) "
                      f"prefix_len={prefix_ids.shape[0]} scored {len(ftodo)} "
                      f"({time.time() - t_p:.1f}s)")
            print(f"\nscored {len(new_rows)} new cells in {time.time() - t0:.1f}s")
    else:
        print("  nothing to score — every cell already in the existing parquet.")

    # Concat new rows with existing (if any) and write back to the same path.
    new_rows_df = pd.DataFrame(new_rows) if new_rows else None
    if existing_rows is not None and new_rows_df is not None:
        rows_df = pd.concat([existing_rows, new_rows_df], ignore_index=True)
    elif existing_rows is not None:
        rows_df = existing_rows
    elif new_rows_df is not None:
        rows_df = new_rows_df
    else:
        # No existing, no new — shouldn't happen (faces × prompts > 0), but
        # guard for empty union edge case.
        raise SystemExit(
            "no rows to write: face union or prompt set is empty"
        )
    rows_df.to_parquet(rows_path, index=False)
    print(f"wrote per-cell rows to {rows_path}  "
          f"(total n={len(rows_df)}; +{len(new_rows)} new)")

    topk_arg = args.summary_topk if args.summary_topk and args.summary_topk > 0 else None
    summary = _summarize(rows_df, faces_df, summary_topk=topk_arg)
    if topk_arg is not None:
        print(f"  summary aggregation: top-{topk_arg} mean per (face, quadrant)")
    summary_path = DATA_DIR / f"face_likelihood_{M.short_name}{suffix}_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"wrote per-face summary to {summary_path}")

    val = summary[summary["total_emit_count"].fillna(0) >= 3].copy()
    val = val[val["argmax_matches_empirical"].notna()]
    n_val = len(val)
    if n_val > 0:
        n_match = int(val["argmax_matches_empirical"].astype(bool).sum())
        rate = n_match / n_val
        print(f"\nvalidation (faces with >=3 v3 emissions, n={n_val}):")
        print(f"  argmax matches empirical majority: {n_match}/{n_val}  ({rate:.1%})")
        print(f"  per-empirical-quadrant breakdown:")
        for q in QUADRANT_ORDER:
            sub = val[val["empirical_majority_quadrant"] == q]
            if len(sub) == 0:
                continue
            m = int(sub["argmax_matches_empirical"].astype(bool).sum())
            print(f"    {q}: {m}/{len(sub)}")
    else:
        print("\nno faces with >=3 v3 emissions in summary — cannot validate")


if __name__ == "__main__":
    main()
