# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
"""v3 PC × probe rotation in 3D, interactive HTML.

In 2D the PCs don't visually map cleanly onto the canonical probes
(happy.sad / angry.calm / fearful.unflinching) — that mismatch is what
prompted this whole face-stability thread. This script builds the 3D
analogue: top-3 PCs of the h_first layer-stack, then rotated so the
3 probes align as cleanly as possible with the canonical x/y/z axes
of the plot.

Algorithm:

  1. PCA(3) on centered h_first.
  2. Project each probe direction into the PC subspace via row-level
     correlations: D[k, j] = corr(probe_scores[k], pc_scores[:, j]).
     Each row of D is a unit-normalized probe direction in 3D PC
     space.
  3. Orthogonal Procrustes: find R such that R @ D[k] ≈ e_k for each
     probe k. Maximize tr(R · D^T) over orthogonal R.
     SVD of D^T = U S V^T gives R = V U^T.
  4. Apply R to PC scores → rotated coordinates where x, y, z
     approximate the 3 probe directions.

What you see in the resulting HTML:

  • One marker per row, colored by Russell quadrant (HN bisected).
  • Probe arrows from origin: should land on or near the canonical
    axes (modulo the fraction of the probe direction that's *outside*
    the PC subspace — angle with the axis tells you that).
  • PC unit vectors as a second arrow set: rotated alongside the
    rows, so wherever they land = where PC structure points.
    If PCs disagree with probes, PC arrows visibly miss the axes.

Per-model HTMLs at:
  figures/local/<short>/fig_v3_pc_probe_rotation_3d.html

TSV summary at:
  figures/local/cross_model/v3_pc_probe_rotation.tsv
    columns: model, probe, axis, residual_norm (1 = no PC alignment,
             0 = probe entirely captured by PC subspace),
             angle_deg (after rotation, to its target axis)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from llmoji_study.config import FIGURES_DIR, MODEL_REGISTRY, PROBES, resolve_model
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    load_emotional_features_stack,
    load_rows,
)


def _attach_probe_scores(
    df: pd.DataFrame, jsonl_path: Path,
) -> pd.DataFrame:
    """Merge the t0_<probe> columns from the JSONL into df by row_uuid.
    df comes from load_emotional_features which doesn't unpack probes.
    """
    raw = load_rows(str(jsonl_path))
    keep_cols = ["row_uuid"] + [f"t0_{p}" for p in PROBES if f"t0_{p}" in raw.columns]
    raw = raw[keep_cols]
    return df.merge(raw, on="row_uuid", how="left")


def _probe_directions_in_pc_subspace(
    pc_scores: np.ndarray, probe_scores: np.ndarray,
) -> np.ndarray:
    """For each probe, its unit-normalized direction in 3D PC space —
    estimated as the per-PC Pearson correlation with the probe score.
    pc_scores: (n, 3); probe_scores: (n, n_probes). Returns (n_probes, 3).

    Correlation is the geometrically-meaningful quantity here: it's
    cos(angle) between the probe-score signal and each PC axis when
    both are zero-meaned and unit-variance scaled — i.e. the probe
    direction's projection onto each PC, normalized."""
    n_probes = probe_scores.shape[1]
    D = np.zeros((n_probes, 3), dtype=np.float64)
    for k in range(n_probes):
        for j in range(3):
            x = probe_scores[:, k]
            y = pc_scores[:, j]
            mask = ~np.isnan(x)
            if mask.sum() < 3:
                continue
            xm = x[mask] - x[mask].mean()
            ym = y[mask] - y[mask].mean()
            denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
            if denom <= 0:
                continue
            D[k, j] = float(xm @ ym) / denom
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return D / norms


def _orthogonal_procrustes(D: np.ndarray) -> np.ndarray:
    """Find R ∈ O(3) maximizing Σₖ (R @ D[k])·eₖ — i.e. R @ D[k] ≈ eₖ
    for all k. SVD of D^T (probes as columns of P): P = U S V^T,
    R = V U^T. Reflection allowed; for an interactive viz the parity
    doesn't matter.

    D is (n_probes, 3). If n_probes != 3, we still solve the same
    least-squares: R minimizes ||R D^T − I||_F over the relevant
    submatrix; with 3 probes and 3 axes the solution is exact when D
    is non-singular."""
    P = D.T   # columns = probe directions
    U, _, Vt = np.linalg.svd(P)
    return Vt.T @ U.T


def _build_arrow(
    origin: tuple[float, float, float],
    end: tuple[float, float, float],
    *, color: str, name: str, width: int = 4,
) -> list[go.Scatter3d]:
    """Two traces: a line from origin to end + a cone-style arrowhead
    at end. Plotly doesn't have a native 3D arrow primitive, so we use
    a small Cone trace pointed along (end - origin)."""
    line = go.Scatter3d(
        x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
        mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=True, legendgroup=name, hoverinfo="name",
    )
    direction = np.array(end) - np.array(origin)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return [line]
    head = go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=norm * 0.18,
        anchor="tip", showscale=False,
        colorscale=[[0, color], [1, color]],
        name=name, showlegend=False, legendgroup=name, hoverinfo="skip",
    )
    return [line, head]


def _plot_one_model(
    short: str,
    pc_scores: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    explained_var: np.ndarray,
    quadrants: np.ndarray,
    out_path: Path,
    *,
    arrow_scale: float = 1.0,
) -> None:
    """3D scatter in the rotated frame (probes ≈ canonical axes).

    Rows colored by HN-split quadrant; probe arrows for happy.sad /
    angry.calm / fearful.unflinching land on x / y / z; rotated PC
    unit vectors shown as a second arrow set."""
    rotated = pc_scores @ R.T   # (n, 3)
    span = float(np.percentile(np.abs(rotated), 99)) * 1.05
    if span <= 0:
        span = 1.0

    traces: list = []
    for q in QUADRANT_ORDER_SPLIT:
        mask = quadrants == q
        if not mask.any():
            continue
        traces.append(go.Scatter3d(
            x=rotated[mask, 0],
            y=rotated[mask, 1],
            z=rotated[mask, 2],
            mode="markers",
            marker=dict(
                size=4, opacity=0.7,
                color=QUADRANT_COLORS.get(q, "#666"),
                line=dict(width=0),
            ),
            name=f"{q} (n={int(mask.sum())})",
            legendgroup=f"q_{q}",
            hovertemplate=f"{q}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>",
        ))

    arrow_len = span * 0.85 * arrow_scale
    probe_axis_color = "#000000"
    probe_names = ["happy.sad", "angry.calm", "fearful.unflinching"]
    rotated_probes = D @ R.T   # (n_probes, 3) — should ≈ identity
    for k, p_name in enumerate(probe_names):
        end = tuple((rotated_probes[k] * arrow_len).tolist())
        traces.extend(_build_arrow(
            (0.0, 0.0, 0.0), end,  # type: ignore[arg-type]
            color=probe_axis_color, name=f"probe: {p_name}", width=6,
        ))

    pc_color = "#cc4477"
    pc_unit = R.T  # rotated PC basis vectors are columns of R^T → rows after transpose
    for j in range(3):
        end_pc = tuple((pc_unit[j] * arrow_len * 0.7).tolist())
        traces.extend(_build_arrow(
            (0.0, 0.0, 0.0), end_pc,  # type: ignore[arg-type]
            color=pc_color,
            name=f"PC{j+1} ({explained_var[j]*100:.1f}%)",
            width=4,
        ))

    title = (
        f"{short}: h_first layer-stack, top-3 PCs rotated so probes "
        f"≈ canonical axes<br>"
        f"<span style='font-size:11px'>"
        f"x = happy.sad direction, y = angry.calm, z = fearful.unflinching; "
        f"pink arrows = rotated PCs (where the variance actually points)"
        f"</span>"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="probe: happy.sad", range=[-span, span]),
            yaxis=dict(title="probe: angry.calm", range=[-span, span]),
            zaxis=dict(title="probe: fearful.unflinching", range=[-span, span]),
            aspectmode="cube",
        ),
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, b=10, t=60),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _per_model(short: str) -> tuple[list[dict], None] | tuple[None, None]:
    M = resolve_model(short)  # honors LLMOJI_OUT_SUFFIX for active model
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no v3 data; skipping")
        return None, None

    print(f"\n{short}  (h_first, layer-stack)")
    df, X = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
    )
    if len(df) == 0:
        print(f"  [{short}] no kaomoji-bearing rows")
        return None, None

    df = _attach_probe_scores(df, M.emotional_data_path)
    probe_cols = [f"t0_{p}" for p in PROBES if f"t0_{p}" in df.columns]
    if len(probe_cols) < 3:
        print(f"  [{short}] missing probe columns: {set(PROBES) - {c[3:] for c in probe_cols}}")
        return None, None

    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=3)
    pc_scores = pca.fit_transform(Xc)
    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance ratio: "
          f"PC1={explained_var[0]:.3f}  PC2={explained_var[1]:.3f}  "
          f"PC3={explained_var[2]:.3f}  (sum={explained_var.sum():.3f})")

    probe_scores = df[probe_cols].to_numpy(dtype=float)
    D = _probe_directions_in_pc_subspace(pc_scores, probe_scores)
    pc_capture = np.linalg.norm(
        np.array([
            [
                np.corrcoef(probe_scores[:, k], pc_scores[:, j])[0, 1]
                for j in range(3)
            ]
            for k in range(probe_scores.shape[1])
        ]),
        axis=1,
    )
    print(f"  PC-subspace capture per probe (||correlation vector||): "
          + ", ".join(f"{p}={c:.3f}" for p, c in zip(PROBES, pc_capture)))

    R = _orthogonal_procrustes(D)
    rotated_probes = D @ R.T
    rows: list[dict] = []
    for k, p in enumerate(PROBES):
        target = np.zeros(3)
        target[k] = 1.0
        cos_to_target = float(rotated_probes[k] @ target /
                              max(np.linalg.norm(rotated_probes[k]), 1e-12))
        cos_to_target = max(min(cos_to_target, 1.0), -1.0)
        angle_deg = float(np.degrees(np.arccos(cos_to_target)))
        rows.append({
            "model": short,
            "probe": p,
            "axis": ["x", "y", "z"][k],
            "pc_capture_norm": float(pc_capture[k]),
            "rotated_x": float(rotated_probes[k, 0]),
            "rotated_y": float(rotated_probes[k, 1]),
            "rotated_z": float(rotated_probes[k, 2]),
            "angle_deg_to_target": angle_deg,
        })
        print(f"    {p:24s} → axis {['x','y','z'][k]}  angle = {angle_deg:5.2f}°  "
              f"(capture = {pc_capture[k]:.3f})")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = M.figures_dir / "fig_v3_pc_probe_rotation_3d.html"
    quadrants = df["quadrant"].astype(str).to_numpy()
    _plot_one_model(
        short, pc_scores, R, D, explained_var, quadrants, out_path,
    )
    print(f"  wrote {out_path}")

    return rows, None


def main() -> None:
    out_dir = FIGURES_DIR / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for short in MODEL_REGISTRY:
        rows, _ = _per_model(short)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("no models produced output")
        return

    summary = pd.DataFrame(all_rows)
    tsv_path = out_dir / "v3_pc_probe_rotation.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {tsv_path}")


if __name__ == "__main__":
    main()
