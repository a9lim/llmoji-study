# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""All v3 + claude faces in `--model` face-input PCA space.

Consumes `data/face_h_first_<model>.parquet` (script 46): the chosen
encoder's h_first for the union of (gemma+qwen+ministral)-emitted ∪
claude-faces. Per-face quadrant blend = SUMMED emission distribution
across all 3 v3 models (`total_emit_*` columns).

Categorization: faces with no v3 emission (claude-only / non-emitted)
get nearest-neighbor against v3-emitted anchors by cosine in raw
hidden-dim space. Inherit the NN anchor's summed quadrant blend.

Visualization (interactive 3D HTML):
  - **v3-emitted faces** (circles): position in face-input PCA(3),
    size = log(total_emit_count), color = summed-emission blend
  - **non-emitted faces** (diamonds): position in face-input PCA(3),
    size = constant, color = inherited from cosine-NN anchor

Usage:
  python scripts/local/44_face_input_pc_space.py --model qwen
  python scripts/local/44_face_input_pc_space.py --model gemma
  python scripts/local/44_face_input_pc_space.py --model ministral

Outputs:
  figures/local/<model>/fig_v3_claude_faces_in_<model>_pc_space.html
  data/claude_faces_quadrant_assignment_<model>_nn.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    QUADRANT_ORDER_SPLIT, mix_quadrant_color,
)


QUADRANT_ORDER = QUADRANT_ORDER_SPLIT
V3_MODELS = ["gemma", "qwen", "ministral"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--model", required=True, choices=list(MODEL_REGISTRY.keys()),
        help="encoder model whose face_h_first parquet to consume",
    )
    return p.parse_args()


def _face_size(n: int) -> float:
    """Mirror script 29's size scaling for emitted-face anchors."""
    return float(np.clip(15.0 + 30.0 * math.log1p(n), 15.0, 250.0))


def _per_model_breakdown(row: pd.Series) -> str:
    parts: list[str] = []
    for m in V3_MODELS:
        n = int(row[f"{m}_emit_count"])
        if n == 0:
            continue
        modes = [(q, int(row[f"{m}_emit_{q}"])) for q in QUADRANT_ORDER]
        modes = [(q, c) for q, c in modes if c > 0]
        modes.sort(key=lambda kv: -kv[1])
        breakdown = "/".join(f"{q}={c}" for q, c in modes)
        parts.append(f"{m}={n} ({breakdown})")
    return "; ".join(parts) or "—"


def _hover_emit(row: pd.Series, weights: dict[str, float],
                xyz: tuple[float, float, float],
                ax_labels: tuple[str, str, str]) -> str:
    by_w = sorted(((q, w) for q, w in weights.items() if w > 0), key=lambda kv: -kv[1])
    blend = ", ".join(f"{q}={w:.0%}" for q, w in by_w) or "—"
    lines = [
        f"<b>{row.first_word}</b>  [v3-emitted, total n={int(row.total_emit_count)}]",
        f"summed blend: {blend}",
        f"by model: {_per_model_breakdown(row)}",
    ]
    for lab, v in zip(ax_labels, xyz):
        lines.append(f"{lab} = {v:+.3f}")
    return "<br>".join(lines)


def _hover_non_emit(fw: str, nn_fw: str, nn_cos: float, nn_weights: dict[str, float],
                    argmax: str, margin: float,
                    xyz: tuple[float, float, float],
                    ax_labels: tuple[str, str, str]) -> str:
    by_w = sorted(((q, w) for q, w in nn_weights.items() if w > 0), key=lambda kv: -kv[1])
    blend = ", ".join(f"{q}={w:.0%}" for q, w in by_w) or "—"
    lines = [
        f"<b>{fw}</b>  [non-emitted in v3]",
        f"nearest v3-emitted: <b>{nn_fw}</b>  (cos {nn_cos:+.3f})",
        f"inherited blend: {blend}",
        f"soft argmax: {argmax}  (margin {margin:.2f})",
    ]
    for lab, v in zip(ax_labels, xyz):
        lines.append(f"{lab} = {v:+.3f}")
    return "<br>".join(lines)


def _weights_from_total(row: pd.Series) -> dict[str, float]:
    counts = {q: float(row[f"total_emit_{q}"]) for q in QUADRANT_ORDER}
    total = sum(counts.values())
    if total <= 0:
        return {q: 0.0 for q in QUADRANT_ORDER}
    return {q: counts[q] / total for q in QUADRANT_ORDER}


def main() -> None:
    args = _parse_args()
    encoder = args.model
    M = MODEL_REGISTRY[encoder]

    parquet_path = DATA_DIR / f"face_h_first_{encoder}.parquet"
    if not parquet_path.exists():
        raise RuntimeError(
            f"missing {parquet_path}; run scripts/local/46_face_input_encode.py "
            f"--model {encoder} first"
        )
    df: pd.DataFrame = pd.read_parquet(parquet_path)
    h_cols = sorted(
        (c for c in df.columns if c.startswith("h") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    H = df[h_cols].to_numpy(dtype=np.float64)
    print(f"loaded {len(df)} faces × hidden_dim={H.shape[1]}")
    for m in V3_MODELS:
        n = int(df[f"is_{m}_emitted"].sum())
        print(f"  {m}-emitted: {n}")
    print(f"  any-v3-emitted: {int((df.total_emit_count > 0).sum())}")
    print(f"  claude: {int(df.is_claude.sum())}")
    print(f"  non-emitted (NN target): "
          f"{int((df.total_emit_count == 0).sum())}")

    pca = PCA(n_components=3).fit(H)
    Y = pca.transform(H)
    evr = pca.explained_variance_ratio_
    print(f"  PCA explained: PC1={evr[0]:.3f}  PC2={evr[1]:.3f}  PC3={evr[2]:.3f}  "
          f"(sum {evr.sum():.3f})")

    emit_mask = (df["total_emit_count"] > 0).to_numpy()
    non_emit_mask = ~emit_mask

    H_emit = H[emit_mask]
    H_emit_n = H_emit / np.linalg.norm(H_emit, axis=1, keepdims=True).clip(min=1e-12)
    H_norm = H / np.linalg.norm(H, axis=1, keepdims=True).clip(min=1e-12)
    cos_to_emit = H_norm @ H_emit_n.T

    emit_fw = df.loc[emit_mask, "first_word"].tolist()
    emit_weights_per_face: dict[str, dict[str, float]] = {}
    for _, row in df[emit_mask].iterrows():
        emit_weights_per_face[str(row["first_word"])] = _weights_from_total(row)

    nn_records: list[dict] = []
    for i, row in df.iterrows():
        fw = str(row["first_word"])
        sims = cos_to_emit[i].copy()
        if fw in emit_weights_per_face:
            j_self = emit_fw.index(fw)
            sims[j_self] = -np.inf
        j = int(np.argmax(sims))
        nn_face = emit_fw[j]
        nn_cos = float(sims[j])
        w = emit_weights_per_face[nn_face]
        sw = sorted(w.items(), key=lambda kv: -kv[1])
        argmax_q = sw[0][0]
        margin = float(sw[0][1] - sw[1][1]) if len(sw) > 1 else float(sw[0][1])
        nn_records.append(dict(
            first_word=fw,
            is_gemma_emitted=bool(row["is_gemma_emitted"]),
            is_qwen_emitted=bool(row["is_qwen_emitted"]),
            is_ministral_emitted=bool(row["is_ministral_emitted"]),
            is_claude=bool(row["is_claude"]),
            total_emit_count=int(row["total_emit_count"]),
            pc1=float(Y[i, 0]), pc2=float(Y[i, 1]), pc3=float(Y[i, 2]),
            nn_first_word=nn_face,
            nn_cosine=nn_cos,
            soft_assignment=argmax_q,
            soft_margin=margin,
            **{f"weight_{q}": float(w[q]) for q in QUADRANT_ORDER},
        ))
    nn_df = pd.DataFrame(nn_records)

    non_emit_nn = nn_df[nn_df.total_emit_count == 0]
    print(
        f"NN soft-argmax distribution (non-emitted, n={len(non_emit_nn)}): " +
        ", ".join(f"{q}={int((non_emit_nn.soft_assignment == q).sum())}"
                  for q in QUADRANT_ORDER)
    )

    out_tsv = DATA_DIR / f"claude_faces_quadrant_assignment_{encoder}_nn.tsv"
    nn_df[nn_df.is_claude].to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}")

    AX_LABELS = (
        f"PC1 ({evr[0]*100:.1f}%)",
        f"PC2 ({evr[1]*100:.1f}%)",
        f"PC3 ({evr[2]*100:.1f}%)",
    )

    emit_idx = np.where(emit_mask)[0].tolist()
    emit_colors: list[str] = []
    emit_sizes: list[float] = []
    emit_hovers: list[str] = []
    emit_text: list[str] = []
    for i in emit_idx:
        row = df.iloc[i]
        fw = str(row["first_word"])
        n = int(row["total_emit_count"])
        w = emit_weights_per_face[fw]
        rgb = mix_quadrant_color(w)
        emit_colors.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")
        emit_sizes.append(_face_size(n) / 8.0)
        emit_hovers.append(_hover_emit(
            row, w, (float(Y[i, 0]), float(Y[i, 1]), float(Y[i, 2])), AX_LABELS,
        ))
        emit_text.append(fw)

    trace_emit = go.Scatter3d(
        x=Y[emit_idx, 0], y=Y[emit_idx, 1], z=Y[emit_idx, 2],
        mode="markers+text",
        marker=dict(symbol="circle", size=emit_sizes, color=emit_colors,
                    opacity=0.85, line=dict(width=0.5, color="white")),
        text=emit_text, textposition="top center", textfont=dict(size=9, color="#333"),
        name=f"v3-emitted (n={len(emit_idx)})",
        hovertext=emit_hovers, hovertemplate="%{hovertext}<extra></extra>",
        legendgroup="emit",
    )

    non_idx = np.where(non_emit_mask)[0].tolist()
    non_colors: list[str] = []
    non_hovers: list[str] = []
    for i in non_idx:
        rec = nn_records[i]
        w = {q: rec[f"weight_{q}"] for q in QUADRANT_ORDER}
        rgb = mix_quadrant_color(w)
        non_colors.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")
        non_hovers.append(_hover_non_emit(
            rec["first_word"], rec["nn_first_word"], rec["nn_cosine"], w,
            rec["soft_assignment"], rec["soft_margin"],
            (rec["pc1"], rec["pc2"], rec["pc3"]), AX_LABELS,
        ))

    trace_non = go.Scatter3d(
        x=Y[non_idx, 0], y=Y[non_idx, 1], z=Y[non_idx, 2],
        mode="markers",
        marker=dict(symbol="diamond", size=6, color=non_colors, opacity=0.85,
                    line=dict(width=0.6, color="black")),
        name=f"non-emitted in v3 (n={len(non_idx)})",
        hovertext=non_hovers, hovertemplate="%{hovertext}<extra></extra>",
        legendgroup="non",
    )

    fig = go.Figure(data=[trace_emit, trace_non])
    fig.update_layout(
        title=(
            f"{encoder} face-input geometry: v3-emitted ◯ + non-emitted ◇ "
            f"in joint PC1×PC2×PC3 of h_first @ L{M.preferred_layer}<br>"
            f"<span style='font-size:11px'>"
            f"v3-emitted color = summed-emission blend across gemma+qwen+ministral; "
            f"non-emitted color = inherited from cosine-NN v3-emitted anchor; "
            f"PC1+2+3 explain {evr.sum()*100:.1f}% of joint face-input variance; "
            f"size of circles = log(total emission count)"
            f"</span>"
        ),
        scene=dict(
            xaxis=dict(title=AX_LABELS[0]),
            yaxis=dict(title=AX_LABELS[1]),
            zaxis=dict(title=AX_LABELS[2]),
            aspectmode="cube",
        ),
        legend=dict(font=dict(size=10)),
        margin=dict(l=10, r=10, b=10, t=70),
    )
    out_dir = M.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig_v3_claude_faces_in_{encoder}_pc_space.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
