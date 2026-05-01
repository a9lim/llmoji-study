"""Introspection-pilot analysis: PCA(2) + kaomoji distribution + rule-3b recompute.

Reads `data/{short_name}_introspection_raw.jsonl` (written by script 32).

Three outputs:

1. **Rule I (gating, qualitative)** —
   `figures/local/{short_name}/fig_introspection_pca_pair.png`
   Side-by-side PCA(2) of probe-vector-at-t0 per condition. Same axes
   across the three panels (joint PCA fit on the pooled data, then
   plotted per condition) so visual comparison is fair. Color by
   Russell quadrant + HN-D/HN-S split.

2. **Rule II (descriptive)** —
   `figures/local/{short_name}/fig_introspection_kaomoji_dist.png`
   Per-quadrant top-5 face frequencies × condition. Plus a TSV of
   per-quadrant KL(introspection || baseline) and KL(lorem || baseline).

3. **Rule III (sanity)** —
   `data/{short_name}_introspection_summary.tsv`
   Rule-3b recompute (HN-S minus HN-D on `t0_fearful.unflinching`)
   per condition with bootstrap 95% CI.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import (
    DATA_DIR,
    INTROSPECTION_CONDITIONS,
    PROBES,
    current_model,
)
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    _hn_split_map,
    _use_cjk_font,
    available_extension_probes,
    load_emotional_features,
    load_rows,
)


_use_cjk_font()


_HN_SPLIT = _hn_split_map()
_RNG = np.random.default_rng(20260502)


def _quadrant_with_split(prompt_id: str) -> str:
    """Map prompt_id to HP/LP/HN-D/HN-S/LN/NB. Mirrors the rule-3
    redesign split — borderline-untagged HN prompts (hn06/hn15/hn17)
    map to the bare "HN" code so they're filterable from HN-only views.
    """
    base = prompt_id[:2].upper()
    if base != "HN":
        return base
    return _HN_SPLIT.get(prompt_id, "HN")


def _all_probe_columns(df: pd.DataFrame) -> list[str]:
    """t0_<probe> columns for core PROBES + every extension probe present."""
    core_cols = [f"t0_{p}" for p in PROBES if f"t0_{p}" in df.columns]
    ext_cols = [f"t0_{p}" for p in available_extension_probes(df)
                if f"t0_{p}" in df.columns]
    return core_cols + ext_cols


def _bootstrap_ci_diff(
    a: np.ndarray, b: np.ndarray, *, n: int = 2000, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean(a) - Mean(b), bootstrap CI on the difference.

    Returns (point, lo, hi).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    point = float(a.mean() - b.mean())
    if len(a) == 0 or len(b) == 0:
        return point, float("nan"), float("nan")
    diffs = np.empty(n, dtype=float)
    for i in range(n):
        ia = _RNG.integers(0, len(a), size=len(a))
        ib = _RNG.integers(0, len(b), size=len(b))
        diffs[i] = a[ia].mean() - b[ib].mean()
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return point, lo, hi


def _kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    """KL(P || Q) over the union of keys, with epsilon-smoothing."""
    keys = set(p) | set(q)
    pp = np.array([p.get(k, 0.0) for k in keys]) + eps
    qq = np.array([q.get(k, 0.0) for k in keys]) + eps
    pp /= pp.sum()
    qq /= qq.sum()
    return float(np.sum(pp * np.log(pp / qq)))


def _face_dist(faces: pd.Series) -> dict[str, float]:
    """Normalized frequency over a Series of canonical face strings."""
    counts = Counter(faces)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _color_for(quadrant: str) -> str:
    return QUADRANT_COLORS.get(quadrant, "#888888")


def main() -> None:
    M = current_model()
    raw_path = DATA_DIR / f"{M.short_name}_introspection_raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path} — run scripts/32 first")
    figdir = M.figures_dir
    figdir.mkdir(parents=True, exist_ok=True)

    df = load_rows(str(raw_path))
    df = df[df["first_word"].astype(bool)].copy()
    df["quadrant_split"] = df["prompt_id"].map(_quadrant_with_split)

    print(f"model: {M.short_name}; rows: {len(df)}")
    for c in INTROSPECTION_CONDITIONS:
        n = (df["condition"] == c).sum()
        n_emit = ((df["condition"] == c) & df["first_word"].astype(bool)).sum()
        print(f"  {c}: {n} rows ({n_emit} kaomoji-bearing)")

    # ---------- Rule I: paired PCA(2) on probe-vector-at-t0 ----------
    probe_cols = _all_probe_columns(df)
    print(f"probe columns ({len(probe_cols)}): {probe_cols}")
    X_all = df[probe_cols].to_numpy(dtype=float)
    # Joint PCA fit so the three panels share axes — fair visual comparison.
    pca = PCA(n_components=2, random_state=0).fit(X_all)
    df["pc1"] = pca.transform(X_all)[:, 0]
    df["pc2"] = pca.transform(X_all)[:, 1]
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, condition in zip(axes, INTROSPECTION_CONDITIONS):
        sub = df[df["condition"] == condition]
        for q, gd in sub.groupby("quadrant_split"):
            ax.scatter(
                gd["pc1"], gd["pc2"],
                s=42, c=_color_for(q), label=q,
                edgecolors="white", linewidths=0.5, alpha=0.85,
            )
        # centroids
        for q, gd in sub.groupby("quadrant_split"):
            cx, cy = gd["pc1"].mean(), gd["pc2"].mean()
            ax.scatter([cx], [cy], s=200, c=_color_for(q),
                       edgecolors="black", linewidths=1.4, marker="X", zorder=5)
        ax.set_title(f"{condition} (n={len(sub)})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.axhline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center",
               ncol=len(by_label), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        f"{M.short_name}: introspection-pilot probe-PCA(2) at t0 — "
        "gate is qualitative shift introspection ≠ baseline AND lorem ≈ baseline",
        y=-0.02,
    )
    fig.tight_layout()
    out = figdir / "fig_introspection_pca_pair.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---------- Rule I-bis: paired PCA(2) on hidden states (h_first @ L_pref) ----------
    # The probe-PCA above lives in 17-D probe-score space; this one lives
    # in the actual hidden-state space at gemma's preferred layer (L31).
    # h_first is the state producing the kaomoji-emission token (= t0).
    pref_layer = M.preferred_layer  # L31 for gemma; falls back to deepest if None
    print(f"\nloading hidden states at L{pref_layer} (h_first) ...")
    df_h, X_h = load_emotional_features(
        str(raw_path), DATA_DIR,
        experiment=f"{M.experiment}_introspection",
        which="h_first", layer=pref_layer,
        split_hn=True,
    )
    print(f"  hidden-state rows: {len(df_h)}; dim: {X_h.shape[1]}")
    pca_h = PCA(n_components=2, random_state=0).fit(X_h)
    coords_h = pca_h.transform(X_h)
    df_h = df_h.assign(pc1=coords_h[:, 0], pc2=coords_h[:, 1])
    print(f"  PCA explained variance: {pca_h.explained_variance_ratio_}")

    fig_h, axes_h = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, condition in zip(axes_h, INTROSPECTION_CONDITIONS):
        sub = df_h[df_h["condition"] == condition]
        for q, gd in sub.groupby("quadrant"):
            ax.scatter(
                gd["pc1"], gd["pc2"],
                s=42, c=_color_for(q), label=q,
                edgecolors="white", linewidths=0.5, alpha=0.85,
            )
        for q, gd in sub.groupby("quadrant"):
            cx, cy = gd["pc1"].mean(), gd["pc2"].mean()
            ax.scatter([cx], [cy], s=200, c=_color_for(q),
                       edgecolors="black", linewidths=1.4, marker="X", zorder=5)
        ax.set_title(f"{condition} (n={len(sub)})")
        ax.set_xlabel(f"PC1 ({pca_h.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca_h.explained_variance_ratio_[1]:.2%})")
        ax.axhline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.grid(True, alpha=0.2)
    handles, labels = axes_h[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_h.legend(by_label.values(), by_label.keys(), loc="upper center",
                 ncol=len(by_label), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig_h.suptitle(
        f"{M.short_name}: introspection-pilot HIDDEN-STATE PCA(2) at "
        f"L{pref_layer} (h_first) — joint-fit, same axes across conditions",
        y=-0.02,
    )
    fig_h.tight_layout()
    out_h = figdir / "fig_introspection_hidden_pca_pair.png"
    fig_h.savefig(out_h, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    print(f"wrote {out_h}")

    # ---------- Rule I-bis: per-condition hidden-state PCA ----------
    # Joint-fit PCA above showed the dominant axis is "preamble presence/
    # type." This pass fits PCA *within each condition*, factoring out
    # that nuisance axis so we can see whether quadrant structure
    # survives independent of preamble.
    fig_pc, axes_pc = plt.subplots(1, 3, figsize=(18, 6))
    for ax, condition in zip(axes_pc, INTROSPECTION_CONDITIONS):
        sub_mask = (df_h["condition"] == condition).to_numpy()
        Xs = X_h[sub_mask]
        sub_meta = df_h[sub_mask].reset_index(drop=True)
        pca_s = PCA(n_components=2, random_state=0).fit(Xs)
        coords_s = pca_s.transform(Xs)
        sub_meta = sub_meta.assign(pc1=coords_s[:, 0], pc2=coords_s[:, 1])
        for q, gd in sub_meta.groupby("quadrant"):
            ax.scatter(
                gd["pc1"], gd["pc2"],
                s=42, c=_color_for(q), label=q,
                edgecolors="white", linewidths=0.5, alpha=0.85,
            )
        for q, gd in sub_meta.groupby("quadrant"):
            cx, cy = gd["pc1"].mean(), gd["pc2"].mean()
            ax.scatter([cx], [cy], s=200, c=_color_for(q),
                       edgecolors="black", linewidths=1.4, marker="X", zorder=5)
        ax.set_title(
            f"{condition} (n={len(sub_meta)}) — "
            f"PC1 {pca_s.explained_variance_ratio_[0]:.2%}, "
            f"PC2 {pca_s.explained_variance_ratio_[1]:.2%}"
        )
        ax.set_xlabel("PC1 (within-condition fit)")
        ax.set_ylabel("PC2 (within-condition fit)")
        ax.axhline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.grid(True, alpha=0.2)
    handles, labels = axes_pc[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_pc.legend(by_label.values(), by_label.keys(), loc="upper center",
                  ncol=len(by_label), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig_pc.suptitle(
        f"{M.short_name}: introspection-pilot HIDDEN-STATE PCA(2) per-condition "
        f"fit at L{pref_layer} — factors out the preamble nuisance axis",
        y=-0.02,
    )
    fig_pc.tight_layout()
    out_pc = figdir / "fig_introspection_hidden_pca_per_condition.png"
    fig_pc.savefig(out_pc, dpi=150, bbox_inches="tight")
    plt.close(fig_pc)
    print(f"wrote {out_pc}")

    # ---------- Rule II: per-quadrant kaomoji distribution + KL ----------
    quadrants = ["HP", "LP", "HN-D", "HN-S", "LN", "NB"]
    rows: list[dict] = []
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9))
    for ax, q in zip(axes2.flat, quadrants):
        sub_q = df[df["quadrant_split"] == q]
        # union top-5 faces across conditions
        all_faces = Counter(sub_q["first_word"])
        top5 = [f for f, _ in all_faces.most_common(5)]
        x = np.arange(len(top5))
        width = 0.27
        for i, c in enumerate(INTROSPECTION_CONDITIONS):
            sub_c = sub_q[sub_q["condition"] == c]
            dist = _face_dist(sub_c["first_word"])
            heights = [dist.get(f, 0.0) for f in top5]
            ax.bar(x + (i - 1) * width, heights, width, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(top5, rotation=0, fontsize=9)
        ax.set_title(f"{q} (n={len(sub_q)})")
        ax.set_ylabel("frequency")
        ax.grid(True, alpha=0.2, axis="y")
        # KL (introspection || baseline) and (lorem || baseline)
        d_base = _face_dist(sub_q[sub_q["condition"] == "intro_none"]["first_word"])
        d_pre = _face_dist(sub_q[sub_q["condition"] == "intro_pre"]["first_word"])
        d_lor = _face_dist(sub_q[sub_q["condition"] == "intro_lorem"]["first_word"])
        kl_pre = _kl(d_pre, d_base)
        kl_lor = _kl(d_lor, d_base)
        rows.append({
            "quadrant": q,
            "n_baseline": int((sub_q["condition"] == "intro_none").sum()),
            "n_introspection": int((sub_q["condition"] == "intro_pre").sum()),
            "n_lorem": int((sub_q["condition"] == "intro_lorem").sum()),
            "kl_intro_vs_base": kl_pre,
            "kl_lorem_vs_base": kl_lor,
        })
    axes2[0, 0].legend(loc="upper right", fontsize=9)
    fig2.suptitle(f"{M.short_name}: per-quadrant top-5 kaomoji × condition")
    fig2.tight_layout()
    out2 = figdir / "fig_introspection_kaomoji_dist.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")
    kl_df = pd.DataFrame(rows)
    print("\nper-quadrant KL divergences:")
    print(kl_df.to_string(index=False))

    # ---------- Rule III: rule-3b recompute per condition ----------
    fearful_col = "t0_fearful.unflinching"
    if fearful_col not in df.columns:
        print(f"skipping Rule III — {fearful_col} not in data")
        rule3b_rows: list[dict] = []
    else:
        rule3b_rows = []
        for c in INTROSPECTION_CONDITIONS:
            sub_c = df[df["condition"] == c]
            d = sub_c[sub_c["quadrant_split"] == "HN-D"][fearful_col].to_numpy()
            s = sub_c[sub_c["quadrant_split"] == "HN-S"][fearful_col].to_numpy()
            point, lo, hi = _bootstrap_ci_diff(s, d)
            rule3b_rows.append({
                "condition": c,
                "n_HN_D": int(len(d)),
                "n_HN_S": int(len(s)),
                "rule3b_t0_diff": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_excludes_zero": (lo > 0) or (hi < 0),
            })
        print("\nRule III — rule-3b (HN-S - HN-D on fearful.unflinching at t0):")
        for r in rule3b_rows:
            sig = "*" if r["ci_excludes_zero"] else " "
            print(
                f"  {r['condition']:14s} n=({r['n_HN_D']:>2d}D,{r['n_HN_S']:>2d}S) "
                f"diff={r['rule3b_t0_diff']:+.4f} "
                f"CI[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] {sig}"
            )

    # ---------- Summary TSV ----------
    summary_path = DATA_DIR / f"{M.short_name}_introspection_summary.tsv"
    summary = {
        "model": M.short_name,
        "n_rows": int(len(df)),
        "pca_var_ratio": pca.explained_variance_ratio_.tolist(),
        "kl_per_quadrant": rows,
        "rule3b_per_condition": rule3b_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
