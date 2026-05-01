"""Interactive 3D plots for the v3 extension-probe rescore (HTML output).

Four figures, each three-panel (gemma | qwen | ministral), each colored
by Russell quadrant with the rule-3-redesign HN split (HN-D / HN-S
broken out as separate categories). Per-row plots use one-marker-per-
generation; per-face plots are aggregated (mean over each canonical
kaomoji) with marker size log-scaled to emission count and color = RGB
blend of per-quadrant emission distribution.

  fig_v3_extension_3d_probes.html              — per-row, all rows
  fig_v3_extension_3d_probes_per_face.html     — per-face aggregate
      x = fearful.unflinching (h_first, score_single_token)
      y = happy.sad           (probe_scores_t0)
      z = angry.calm          (probe_scores_t0)

  fig_v3_extension_3d_pca.html                 — per-row, all rows
  fig_v3_extension_3d_pca_per_face.html        — per-face aggregate
      x, y, z = PC1, PC2, PC3 of h_mean at each model's preferred
      layer (gemma L31, qwen L61, ministral L21), fit independently
      per model from the cached
      `data/cache/v3_<short>_h_mean_all_layers.npz`. Per-face: PCA fit
      on the per-face mean h_mean matrix (one row per canonical
      kaomoji), so the PC axes describe variance ACROSS faces rather
      than across individual generations.

Hover: prompt_id / kaomoji / prompt-text preview / axis values for
per-row; canonical face / total emissions / per-quadrant emission
counts / axis values for per-face.

All saved as standalone HTMLs (plotly's default include_plotlyjs).
No model time, no new generations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY, PROBES
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    _hn_split_map,
    mix_quadrant_color,
    per_face_quadrant_weights,
)
from llmoji_study.hidden_state_analysis import load_hidden_features_all_layers

# Use the rule-3-redesign split ordering (HN→HN-D/HN-S) throughout.
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT
_HN_SPLIT = _hn_split_map()
MODELS = ("gemma", "qwen", "ministral")


def _quad(pid: str) -> str:
    """Return the split-mode quadrant label: HP/LP/HN-D/HN-S/LN/NB.
    Untagged-HN prompts (hn06/hn15/hn17) return ``"??"`` so they fall
    through every QUADRANT_ORDER iterator."""
    if len(pid) < 2:
        return "??"
    base = pid[:2].upper()
    if base == "HN":
        return _HN_SPLIT.get(pid, "??")
    return base


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _hover(r: dict, ax_labels: tuple[str, str, str], ax_vals: tuple[float, float, float]) -> str:
    """Per-point hover text. <br>-separated; HTML-safe enough for
    plotly's default rendering (no embedded < or > tags)."""
    fw = r.get("first_word") or "—"
    text = (r.get("prompt_text") or "").replace("\n", " ").strip()
    if len(text) > 110:
        text = text[:107] + "…"
    pid = r.get("prompt_id", "?")
    q = _quad(pid)
    lines = [
        f"<b>{pid}</b> ({q}) seed={r.get('seed','?')}",
        f"kaomoji: {fw}",
        f"prompt: {text}",
    ]
    for lab, v in zip(ax_labels, ax_vals):
        lines.append(f"{lab} = {v:+.3f}")
    return "<br>".join(lines)


def _hover_face(face: str, n_total: int, q_counts: dict[str, int],
                ax_labels: tuple[str, str, str],
                ax_vals: tuple[float, float, float]) -> str:
    """Per-face hover. Shows the canonical face, total emissions, and
    the per-quadrant breakdown (only quadrants with non-zero count
    appear; sorted by count descending)."""
    by_count = sorted(
        ((q, n) for q, n in q_counts.items() if n > 0),
        key=lambda kv: -kv[1],
    )
    breakdown = ", ".join(f"{q}={n}" for q, n in by_count) or "—"
    lines = [
        f"<b>{face}</b>  n={n_total}",
        f"by quadrant: {breakdown}",
    ]
    for lab, v in zip(ax_labels, ax_vals):
        lines.append(f"{lab} = {v:+.3f}")
    return "<br>".join(lines)


def _face_size(n: int) -> float:
    """Match scripts/17 size scaling: clip(15 + 30*log1p(n), 15, 250)."""
    import math
    return float(np.clip(15.0 + 30.0 * math.log1p(n), 15.0, 250.0))


# ---------------------------------------------------------------------------
# Figure 1: 3D probe scatter (fearful, happy, angry)
# ---------------------------------------------------------------------------


def _probe_axes(rows: list[dict]) -> tuple[list[float], list[float], list[float], list[int]]:
    """Returns parallel xs/ys/zs/keep-indices; only rows that have all
    three axis values populated are kept."""
    happy_idx = PROBES.index("happy.sad")
    angry_idx = PROBES.index("angry.calm")
    xs: list[float] = []; ys: list[float] = []; zs: list[float] = []
    keep: list[int] = []
    for i, r in enumerate(rows):
        if "error" in r: continue
        ext = r.get("extension_probe_scores_t0") or {}
        fearful = ext.get("fearful.unflinching")
        if fearful is None: continue
        tlast = r.get("probe_scores_t0") or []
        if len(tlast) <= max(happy_idx, angry_idx): continue
        xs.append(float(fearful))
        ys.append(float(tlast[happy_idx]))
        zs.append(float(tlast[angry_idx]))
        keep.append(i)
    return xs, ys, zs, keep


def fig_3d_probes(by_model: dict[str, list[dict]], out: Path) -> None:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("gemma", "qwen", "ministral"),
        horizontal_spacing=0.05,
    )

    AX_LABELS = ("fearful.unflinching (h_first)",
                 "happy.sad (probe_scores_t0)",
                 "angry.calm (probe_scores_t0)")

    for ci, short in enumerate(MODELS, start=1):
        rows = by_model[short]
        xs, ys, zs, keep = _probe_axes(rows)
        # One trace per quadrant for legend grouping + correct
        # quadrant-color rendering.
        for q in QUADRANT_ORDER:
            qx, qy, qz, qhover = [], [], [], []
            for x, y, z, i in zip(xs, ys, zs, keep):
                if _quad(rows[i]["prompt_id"]) != q: continue
                qx.append(x); qy.append(y); qz.append(z)
                qhover.append(_hover(rows[i], AX_LABELS, (x, y, z)))
            fig.add_trace(
                go.Scatter3d(
                    x=qx, y=qy, z=qz,
                    mode="markers",
                    marker=dict(size=3.5, color=QUADRANT_COLORS[q],
                                opacity=0.78,
                                line=dict(width=0)),
                    name=q,
                    legendgroup=q,
                    showlegend=(ci == 1),  # one legend, on the gemma panel
                    text=qhover,
                    hoverinfo="text",
                ),
                row=1, col=ci,
            )
        scene_id = "scene" if ci == 1 else f"scene{ci}"
        fig.layout[scene_id].xaxis.title = "fearful"
        fig.layout[scene_id].yaxis.title = "happy"
        fig.layout[scene_id].zaxis.title = "angry"
        fig.layout[scene_id].aspectmode = "cube"

    fig.update_layout(
        title=("3D probe scatter: fearful (h_first) × happy × angry "
               "(probe_scores_t0)  —  colored by Russell quadrant"),
        height=720, width=2000,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: 3D PCA scatter (PC1, PC2, PC3 of h_mean at preferred layer)
# ---------------------------------------------------------------------------


def _pca3_at_preferred_layer(short: str) -> tuple[np.ndarray, list[dict], int, np.ndarray]:
    """Returns (X3, kept_rows, layer_idx, explained_variance_ratio[:3]).

    Reads from the v3 multi-layer cache; falls back to the slow path
    (loading sidecars one by one) if the cache hasn't been built.
    """
    M = MODEL_REGISTRY[short]
    cache = DATA_DIR / "cache" / f"v3_{short}_h_mean_all_layers.npz"
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which="h_first", cache_path=cache,
    )
    layer = M.preferred_layer if M.preferred_layer is not None else max(layer_idxs)
    li = layer_idxs.index(layer)
    X = X3[:, li, :]                          # (n, hidden_dim)
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)                  # (n, 3)
    rows = df.to_dict(orient="records")
    return Y, rows, layer, pca.explained_variance_ratio_


def fig_3d_pca(out: Path) -> None:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("gemma  (loading…)", "qwen  (loading…)", "ministral  (loading…)"),
        horizontal_spacing=0.05,
    )

    summaries: list[tuple[str, int, np.ndarray]] = []

    for ci, short in enumerate(MODELS, start=1):
        Y, rows, layer, evr = _pca3_at_preferred_layer(short)
        summaries.append((short, layer, evr))
        AX_LABELS = (
            f"PC1 ({100*evr[0]:.1f}%)",
            f"PC2 ({100*evr[1]:.1f}%)",
            f"PC3 ({100*evr[2]:.1f}%)",
        )
        for q in QUADRANT_ORDER:
            qx, qy, qz, qhover = [], [], [], []
            for i, r in enumerate(rows):
                if _quad(r.get("prompt_id", "")) != q: continue
                if "error" in r: continue
                x, y, z = float(Y[i, 0]), float(Y[i, 1]), float(Y[i, 2])
                qx.append(x); qy.append(y); qz.append(z)
                qhover.append(_hover(r, AX_LABELS, (x, y, z)))
            fig.add_trace(
                go.Scatter3d(
                    x=qx, y=qy, z=qz,
                    mode="markers",
                    marker=dict(size=3.5, color=QUADRANT_COLORS[q],
                                opacity=0.78, line=dict(width=0)),
                    name=q,
                    legendgroup=q,
                    showlegend=(ci == 1),
                    text=qhover,
                    hoverinfo="text",
                ),
                row=1, col=ci,
            )
        scene_id = "scene" if ci == 1 else f"scene{ci}"
        fig.layout[scene_id].xaxis.title = AX_LABELS[0]
        fig.layout[scene_id].yaxis.title = AX_LABELS[1]
        fig.layout[scene_id].zaxis.title = AX_LABELS[2]
        fig.layout[scene_id].aspectmode = "cube"

    # Overwrite the placeholder subplot titles with real layer info.
    for ci, (short, layer, evr) in enumerate(summaries, start=1):
        cumulative = float(evr.sum())
        fig.layout.annotations[ci - 1].text = (
            f"{short}  —  L{layer},  PC1+2+3 = {100*cumulative:.1f}%"
        )

    fig.update_layout(
        title=("3D PCA scatter of h_mean at each model's preferred layer  "
               "(gemma L31, qwen L61)  —  colored by Russell quadrant"),
        height=720, width=2000,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Per-face aggregates
# ---------------------------------------------------------------------------


def _face_quadrant_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """{face -> {quadrant -> count}} from a list of v3 JSONL rows.
    Faces are first_word values starting with '('. Mirrors the
    `per_face_quadrant_weights` shape but in raw counts (the weighted
    version normalizes per face)."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if "error" in r: continue
        fw = r.get("first_word", "")
        if not fw.startswith("("): continue
        q = _quad(r.get("prompt_id", ""))
        if q not in QUADRANT_ORDER: continue
        d = out.setdefault(fw, {q: 0 for q in QUADRANT_ORDER})
        d[q] = d.get(q, 0) + 1
    return out


def _face_color(weights: dict[str, float]) -> str:
    """Normalize weights then call mix_quadrant_color (which expects
    a {q -> [0,1]} weight dict)."""
    total = sum(weights.values()) or 1.0
    norm = {q: weights.get(q, 0) / total for q in QUADRANT_ORDER}
    return mix_quadrant_color(norm)


def _per_face_probe_centroids(rows: list[dict]) -> dict[str, tuple[float, float, float, int, dict[str, int]]]:
    """{face -> (mean_fearful, mean_happy, mean_angry, n_total, q_counts)}.
    Means computed only over rows where ALL three values are present."""
    happy_idx = PROBES.index("happy.sad")
    angry_idx = PROBES.index("angry.calm")
    accum: dict[str, list[tuple[float, float, float, str]]] = {}
    for r in rows:
        if "error" in r: continue
        fw = r.get("first_word", "")
        if not fw.startswith("("): continue
        ext = r.get("extension_probe_scores_t0") or {}
        fearful = ext.get("fearful.unflinching")
        if fearful is None: continue
        tlast = r.get("probe_scores_t0") or []
        if len(tlast) <= max(happy_idx, angry_idx): continue
        q = _quad(r.get("prompt_id", ""))
        accum.setdefault(fw, []).append(
            (float(fearful), float(tlast[happy_idx]), float(tlast[angry_idx]), q),
        )
    out: dict[str, tuple[float, float, float, int, dict[str, int]]] = {}
    for fw, points in accum.items():
        n = len(points)
        if n == 0: continue
        f = sum(p[0] for p in points) / n
        h = sum(p[1] for p in points) / n
        a = sum(p[2] for p in points) / n
        q_counts = {q: 0 for q in QUADRANT_ORDER}
        for *_, q in points:
            if q in q_counts: q_counts[q] += 1
        out[fw] = (f, h, a, n, q_counts)
    return out


def fig_3d_probes_per_face(by_model: dict[str, list[dict]], out: Path) -> None:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("gemma  (loading…)", "qwen  (loading…)", "ministral  (loading…)"),
        horizontal_spacing=0.05,
    )

    AX_LABELS = ("fearful.unflinching (h_first)",
                 "happy.sad (probe_scores_t0)",
                 "angry.calm (probe_scores_t0)")

    summaries: list[tuple[str, int]] = []
    for ci, short in enumerate(MODELS, start=1):
        rows = by_model[short]
        face_data = _per_face_probe_centroids(rows)
        n_faces = len(face_data)
        summaries.append((short, n_faces))

        xs = [v[0] for v in face_data.values()]
        ys = [v[1] for v in face_data.values()]
        zs = [v[2] for v in face_data.values()]
        ns = [v[3] for v in face_data.values()]
        q_counts_list = [v[4] for v in face_data.values()]
        faces = list(face_data.keys())

        sizes = [_face_size(n) / 8.0 for n in ns]   # plotly's marker size is in pixels
        colors = [_face_color({q: c for q, c in qc.items()}) for qc in q_counts_list]
        hovers = [
            _hover_face(f, n, qc, AX_LABELS, (x, y, z))
            for f, n, qc, x, y, z in zip(faces, ns, q_counts_list, xs, ys, zs)
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=sizes, color=colors,
                            opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=short,
                showlegend=False,
                text=hovers,
                hoverinfo="text",
            ),
            row=1, col=ci,
        )
        scene_id = "scene" if ci == 1 else f"scene{ci}"
        fig.layout[scene_id].xaxis.title = "fearful"
        fig.layout[scene_id].yaxis.title = "happy"
        fig.layout[scene_id].zaxis.title = "angry"
        fig.layout[scene_id].aspectmode = "cube"

    for ci, (short, n_faces) in enumerate(summaries, start=1):
        fig.layout.annotations[ci - 1].text = (
            f"{short}  —  {n_faces} faces, mean per face"
        )

    fig.update_layout(
        title=("3D probe scatter (per-face): fearful × happy × angry  "
               "—  size = log(emissions), color = per-face quadrant blend"),
        height=720, width=2000,
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out}")


def fig_3d_pca_per_face(out: Path) -> None:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("gemma  (loading…)", "qwen  (loading…)", "ministral  (loading…)"),
        horizontal_spacing=0.05,
    )

    summaries: list[tuple[str, int, int, np.ndarray]] = []

    for ci, short in enumerate(MODELS, start=1):
        M = MODEL_REGISTRY[short]
        cache = DATA_DIR / "cache" / f"v3_{short}_h_mean_all_layers.npz"
        df, X3, layer_idxs = load_hidden_features_all_layers(
            M.emotional_data_path, DATA_DIR, M.experiment,
            which="h_first", cache_path=cache,
        )
        layer = M.preferred_layer if M.preferred_layer is not None else max(layer_idxs)
        li = layer_idxs.index(layer)
        X = X3[:, li, :]                          # (n_rows, hidden_dim)

        # Aggregate per face: mean h_mean per canonical first_word.
        rows = df.to_dict(orient="records")
        face_to_idxs: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            fw = r.get("first_word", "")
            if not fw.startswith("("): continue
            face_to_idxs.setdefault(fw, []).append(i)

        faces = sorted(face_to_idxs.keys(),
                       key=lambda f: -len(face_to_idxs[f]))
        face_means = np.stack([X[face_to_idxs[f]].mean(axis=0) for f in faces])
        face_ns = [len(face_to_idxs[f]) for f in faces]

        # PCA on the (n_faces, hidden_dim) per-face matrix.
        pca = PCA(n_components=3)
        Y = pca.fit_transform(face_means)         # (n_faces, 3)
        evr = pca.explained_variance_ratio_
        summaries.append((short, layer, len(faces), evr))

        AX_LABELS = (
            f"PC1 ({100*evr[0]:.1f}%)",
            f"PC2 ({100*evr[1]:.1f}%)",
            f"PC3 ({100*evr[2]:.1f}%)",
        )

        # Per-face quadrant counts + colors
        q_counts_per_face: list[dict[str, int]] = []
        colors: list[str] = []
        hovers: list[str] = []
        for i, fw in enumerate(faces):
            qc = {q: 0 for q in QUADRANT_ORDER}
            for j in face_to_idxs[fw]:
                q = _quad(rows[j].get("prompt_id", ""))
                if q in qc: qc[q] += 1
            q_counts_per_face.append(qc)
            colors.append(_face_color(qc))
            x, y, z = float(Y[i, 0]), float(Y[i, 1]), float(Y[i, 2])
            hovers.append(_hover_face(fw, face_ns[i], qc, AX_LABELS, (x, y, z)))

        sizes = [_face_size(n) / 8.0 for n in face_ns]
        fig.add_trace(
            go.Scatter3d(
                x=Y[:, 0], y=Y[:, 1], z=Y[:, 2],
                mode="markers",
                marker=dict(size=sizes, color=colors,
                            opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=short,
                showlegend=False,
                text=hovers,
                hoverinfo="text",
            ),
            row=1, col=ci,
        )
        scene_id = "scene" if ci == 1 else f"scene{ci}"
        fig.layout[scene_id].xaxis.title = AX_LABELS[0]
        fig.layout[scene_id].yaxis.title = AX_LABELS[1]
        fig.layout[scene_id].zaxis.title = AX_LABELS[2]
        fig.layout[scene_id].aspectmode = "cube"

    for ci, (short, layer, n_faces, evr) in enumerate(summaries, start=1):
        cum = float(evr.sum())
        fig.layout.annotations[ci - 1].text = (
            f"{short}  —  L{layer}, {n_faces} faces, PC1+2+3 = {100*cum:.1f}%"
        )

    fig.update_layout(
        title=("3D PCA scatter (per-face): PC1 × PC2 × PC3 of per-face mean h_mean  "
               "—  size = log(emissions), color = per-face quadrant blend"),
        height=720, width=2000,
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    by_model: dict[str, list[dict]] = {}
    for short in MODELS:
        path = MODEL_REGISTRY[short].emotional_data_path
        rows = _load_jsonl(path)
        with_ext = sum(1 for r in rows if "extension_probe_scores_t0" in r)
        print(f"{short}: {len(rows)} rows ({with_ext} with extension scores)")
        by_model[short] = rows

    out_dir = (Path(__file__).resolve().parent.parent
               / "figures" / "local" / "cross_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_3d_probes(by_model, out_dir / "fig_v3_extension_3d_probes.html")
    fig_3d_pca(out_dir / "fig_v3_extension_3d_pca.html")
    fig_3d_probes_per_face(by_model, out_dir / "fig_v3_extension_3d_probes_per_face.html")
    fig_3d_pca_per_face(out_dir / "fig_v3_extension_3d_pca_per_face.html")


if __name__ == "__main__":
    main()
