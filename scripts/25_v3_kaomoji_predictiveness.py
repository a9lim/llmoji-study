# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""v3 kaomoji predictiveness — how well does each face pin down state?

Two complementary directions, both on h_mean at each model's
preferred_layer (gemma L31, qwen L61):

(a) hidden → face. Multi-class logistic regression on PCA(50)-reduced
    h_mean predicts which canonical face was emitted. 5-fold stratified
    CV; report top-1 accuracy, macro-F1, and per-face recall. Compare
    to a parallel hidden → quadrant classifier (5-class) — if face
    accuracy is much higher than quadrant accuracy, the kaomoji
    carries more state-information than the design's Russell labels.

(b) face → hidden. η² (eta-squared) of face identity across each of
    the top-k PC components: between-face variance / total variance.
    Summed weighted by explained variance gives a single
    "fraction of top-k variance explained by face identity" headline.
    Plus per-face distinctiveness: 1 − mean cosine(this face mean,
    other face means) in centered hidden-state space — faces that
    occupy unique regions score high, generic-default faces score low.

Filter: faces with n ≥ ``min_n`` (default 5). Below that, classifier
recall is dominated by sample-size noise.

Outputs (per model, into ``figures/local/<short>/``):
  fig_v3_kaomoji_predictiveness.png      bar grid: per-face metrics
  v3_kaomoji_predictiveness.tsv          per-face metric table
  v3_kaomoji_predictiveness_summary.json single-model summary numbers

Plus a cross-model summary at
``figures/local/cross_model/v3_kaomoji_predictiveness_compare.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llmoji_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY, current_model
from llmoji_study.emotional_analysis import (
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
    per_face_dominant_quadrant,
    per_face_quadrant_weights,
    mix_quadrant_color,
)


# ---------------------------------------------------------------------------
# Hidden → label classifier
# ---------------------------------------------------------------------------


def _multiclass_classifier_metrics(
    X: np.ndarray, y: np.ndarray, *,
    seed: int = 0, pca_dim: int = 50,
) -> dict[str, Any]:
    """5-fold stratified CV on PCA(pca_dim) → l2-logistic. Returns
    dict with overall accuracy, macro-F1, per-class recall, and the
    cross-validated predictions for confusion-matrix work."""
    counts = pd.Series(y).value_counts()
    majority_acc = float(counts.max() / len(y))
    # n_splits limited by smallest class size (StratifiedKFold needs
    # at least n_splits members per class).
    n_splits = max(2, min(5, int(counts.min())))
    # PCA cap: smallest train fold is ~n*(n_splits-1)/n_splits, and
    # PCA needs n_components < min(n_samples, n_features) on every fold.
    pca_cap = max(2, min(pca_dim, len(y) // 3))

    pipe = Pipeline([
        ("pca", PCA(n_components=pca_cap, random_state=seed)),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            C=0.1, max_iter=4000,
            solver="lbfgs", random_state=seed,
        )),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    accuracy = float((y_pred == y).mean())
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    per_class_recall = {
        str(label): float(report[str(label)]["recall"])
        for label in counts.index if str(label) in report
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_baseline": majority_acc,
        "uniform_baseline": 1.0 / len(counts),
        "n_classes": int(len(counts)),
        "per_class_recall": per_class_recall,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Face → hidden: η² and distinctiveness
# ---------------------------------------------------------------------------


def _eta_squared_per_pc(
    X: np.ndarray, y: np.ndarray, *, k: int = 5,
) -> dict[str, Any]:
    """For the top-k PCs of X, compute η² of label y on each PC.

    Returns:
      pcs: list of (pc_index, explained_var_ratio, eta2)
      weighted: sum_k explained_var_ratio_k * eta2_k — single number,
        "fraction of top-k PC variance explained by face identity".
    """
    pca = PCA(n_components=k)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    pcs = []
    for i in range(k):
        z = coords[:, i]
        grand = z.mean()
        ss_total = float(((z - grand) ** 2).sum())
        # SS_between = sum over groups of n_g * (mean_g - grand)^2.
        ss_between = 0.0
        for label in pd.unique(y):
            zi = z[y == label]
            if len(zi) == 0:
                continue
            ss_between += len(zi) * (float(zi.mean()) - grand) ** 2
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        pcs.append({
            "pc": i + 1,
            "explained_variance_ratio": float(var[i]),
            "eta_squared": float(eta2),
        })
    weighted = float(sum(p["explained_variance_ratio"] * p["eta_squared"]
                         for p in pcs))
    total_var_in_topk = float(var.sum())
    return {
        "pcs": pcs,
        "weighted_eta2_top_k": weighted,
        "topk_explained_variance": total_var_in_topk,
        "weighted_eta2_normalized": (
            weighted / total_var_in_topk if total_var_in_topk > 0 else 0.0
        ),
    }


def _face_centroid_approximation_quality(
    df: pd.DataFrame, X: np.ndarray, *,
    quadrant_baseline: bool = True,
) -> dict[str, Any]:
    """Quantify "if you only know the face, how close do you get to the
    actual hidden state?" Three complementary numbers:

    * **multivariate R²** in full hidden-state space
      ``1 − sum_i ||x_i − face_mean(x_i)||² / sum_i ||x_i − grand||²``.
      Equivalently: fraction of total variance explained by knowing
      face identity. 0 = useless, 1 = perfect reconstruction.
    * **mean centered cosine** between each row's centered vector
      ``x_i − grand`` and its face centroid ``face_mean(x_i) − grand``.
      Bounded [−1, 1]; 1 means the face centroid points exactly along
      the row's deviation direction.
    * **mean error/baseline ratio** ``mean ||x_i − face_mean|| /
      mean ||x_i − grand||``. Bounded [0, 1] in practice; 0 = perfect,
      1 = no improvement over predicting the grand mean.

    Computed at the row level using FULL hidden-state vectors, not
    PCA-reduced. With ``quadrant_baseline=True``, also returns the
    same three numbers using quadrant centroids instead of face
    centroids — face improvement over quadrant tells you how much
    extra structure the kaomoji vocabulary captures over the
    five-class design label.
    """
    grand = X.mean(axis=0)
    Xc = X - grand
    ss_total = float((Xc * Xc).sum())
    base_norms = np.linalg.norm(Xc, axis=1)

    def per_label(labels: np.ndarray) -> dict[str, float]:
        ss_within = 0.0
        cos_terms = []
        ratio_terms = []
        unique = pd.unique(labels)
        for label in unique:
            mask = labels == label
            if mask.sum() == 0:
                continue
            xs = X[mask]
            mu = xs.mean(axis=0)  # face centroid
            mu_c = mu - grand
            mu_norm = float(np.linalg.norm(mu_c))
            resid = xs - mu
            ss_within += float((resid * resid).sum())
            # Per-row cosine + error ratio.
            for i in np.where(mask)[0]:
                row_c = Xc[i]
                row_norm = base_norms[i]
                if row_norm > 0 and mu_norm > 0:
                    cos_terms.append(float(np.dot(row_c, mu_c) / (row_norm * mu_norm)))
                err = float(np.linalg.norm(X[i] - mu))
                if row_norm > 0:
                    ratio_terms.append(err / row_norm)
        r2 = 1.0 - ss_within / ss_total if ss_total > 0 else 0.0
        return {
            "r2": float(r2),
            "mean_centered_cosine": float(np.mean(cos_terms)) if cos_terms else 0.0,
            "median_centered_cosine": float(np.median(cos_terms)) if cos_terms else 0.0,
            "mean_error_over_baseline": float(np.mean(ratio_terms)) if ratio_terms else 0.0,
        }

    out = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "face": per_label(df["first_word"].to_numpy()),
    }
    if quadrant_baseline:
        out["quadrant"] = per_label(df["quadrant"].to_numpy())
    return out


def _per_face_distinctiveness(
    df: pd.DataFrame, X: np.ndarray, *,
    min_n: int = 5,
) -> pd.DataFrame:
    """For each face with n ≥ min_n, compute distinctiveness =
    1 − mean cosine(face_mean − grand_mean, other_face_mean − grand_mean).

    High distinctiveness = face's mean direction is unlike other faces'
    means; the model commits unique state to this face. Low =
    near-centroid-of-all-faces; the face is generic / shared register.
    """
    rows = []
    grand = X.mean(axis=0)
    face_means: dict[str, np.ndarray] = {}
    face_n: dict[str, int] = {}
    for face, sub in df.groupby("first_word"):
        if len(sub) < min_n:
            continue
        idxs = sub.index.to_numpy()
        face_means[str(face)] = X[idxs].mean(axis=0) - grand
        face_n[str(face)] = int(len(sub))

    if len(face_means) < 2:
        return pd.DataFrame()

    faces = list(face_means)
    M = np.asarray([face_means[f] for f in faces])
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Mn = M / norms
    sim = Mn @ Mn.T  # (n_faces, n_faces) cosine matrix
    np.fill_diagonal(sim, np.nan)
    mean_off_diag_sim = np.nanmean(sim, axis=1)
    for face, sim_to_others in zip(faces, mean_off_diag_sim):
        rows.append({
            "first_word": face,
            "n": face_n[face],
            "mean_cos_to_other_face_means": float(sim_to_others),
            "distinctiveness": float(1.0 - sim_to_others),
        })
    return pd.DataFrame(rows)


def _within_face_consistency(
    df: pd.DataFrame, X: np.ndarray, *,
    min_n: int = 5,
) -> pd.DataFrame:
    """Per-face cosine-to-mean consistency. Same metric used in
    Figure B / `summary_table` from `emotional_analysis.py`, reproduced
    here so the predictiveness summary stands alone."""
    from llmoji_study.hidden_state_analysis import cosine_to_mean
    rows = []
    for face, sub in df.groupby("first_word"):
        if len(sub) < min_n:
            continue
        idxs = sub.index.to_numpy()
        sims = cosine_to_mean(X[idxs])
        rows.append({
            "first_word": face,
            "median_within_consistency": float(np.median(sims)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------


def run_model(short_name: str, *, min_n: int = 5) -> dict[str, Any]:
    M = MODEL_REGISTRY[short_name]
    if not M.emotional_data_path.exists():
        print(f"[{short_name}] no v3 data at {M.emotional_data_path}; skipping")
        return {}

    layer_label = "max" if M.preferred_layer is None else f"L{M.preferred_layer}"
    print(f"\n=== {short_name} (h_mean, {layer_label}) ===")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
        layer=M.preferred_layer,
    )
    print(f"  {len(df)} rows, X {X.shape}, "
          f"{df['first_word'].nunique()} unique faces")

    # --- (a) hidden → label classifiers
    face_counts = df["first_word"].value_counts()
    keep_faces = face_counts[face_counts >= min_n].index
    df_face = df[df["first_word"].isin(keep_faces)].reset_index(drop=True)
    X_face = X[df["first_word"].isin(keep_faces).to_numpy()]
    print(f"  filtering to faces with n ≥ {min_n}: "
          f"{len(keep_faces)}/{df['first_word'].nunique()} faces, "
          f"{len(df_face)} rows kept")

    print("  hidden → face classifier (multi-class, PCA(50)→l2-LR, 5-fold CV)...")
    face_metrics = _multiclass_classifier_metrics(
        X_face, df_face["first_word"].to_numpy(),
    )
    print(f"    n_classes={face_metrics['n_classes']}  "
          f"acc={face_metrics['accuracy']:.3f}  "
          f"macro-F1={face_metrics['macro_f1']:.3f}  "
          f"majority={face_metrics['majority_baseline']:.3f}  "
          f"uniform={face_metrics['uniform_baseline']:.3f}")

    print("  hidden → quadrant classifier (5-class)...")
    quad_metrics = _multiclass_classifier_metrics(
        X, df["quadrant"].to_numpy(),
    )
    print(f"    n_classes={quad_metrics['n_classes']}  "
          f"acc={quad_metrics['accuracy']:.3f}  "
          f"macro-F1={quad_metrics['macro_f1']:.3f}  "
          f"majority={quad_metrics['majority_baseline']:.3f}")

    # --- (b) face → hidden
    print("  η² of face identity across top-5 PCs...")
    eta = _eta_squared_per_pc(X_face, df_face["first_word"].to_numpy(), k=5)
    for p in eta["pcs"]:
        print(f"    PC{p['pc']}  explained_var={p['explained_variance_ratio']*100:5.2f}%  "
              f"η²={p['eta_squared']:.3f}")
    print(f"    top-5 weighted η² = {eta['weighted_eta2_top_k']:.4f}  "
          f"(top-5 explains {eta['topk_explained_variance']*100:.1f}% of total var; "
          f"face identity recovers "
          f"{eta['weighted_eta2_normalized']*100:.1f}% of that subspace)")

    # --- (b') face-centroid approximation quality (FULL hidden space)
    print("  approximation quality: predict h_mean = face_centroid")
    approx = _face_centroid_approximation_quality(df_face, X_face)
    f = approx["face"]
    q = approx["quadrant"]
    print(f"    face-centroid:       R²={f['r2']:.3f}  "
          f"mean cos(row, centroid)={f['mean_centered_cosine']:+.3f}  "
          f"median={f['median_centered_cosine']:+.3f}  "
          f"||error||/||deviation||={f['mean_error_over_baseline']:.3f}")
    print(f"    quadrant-centroid:   R²={q['r2']:.3f}  "
          f"mean cos(row, centroid)={q['mean_centered_cosine']:+.3f}  "
          f"median={q['median_centered_cosine']:+.3f}  "
          f"||error||/||deviation||={q['mean_error_over_baseline']:.3f}")
    print(f"    face improvement over quadrant (R² gain): "
          f"+{(f['r2'] - q['r2'])*100:.1f} percentage points")

    # --- per-face metrics
    distinct = _per_face_distinctiveness(df_face, X_face, min_n=min_n)
    consist = _within_face_consistency(df_face, X_face, min_n=min_n)
    per_face = (
        distinct
        .merge(consist, on="first_word", how="outer")
        .merge(
            pd.DataFrame({
                "first_word": list(face_metrics["per_class_recall"].keys()),
                "classifier_recall": list(face_metrics["per_class_recall"].values()),
            }),
            on="first_word", how="left",
        )
    )
    quad_for = per_face_dominant_quadrant(df_face)
    per_face = per_face.assign(
        dominant_quadrant=per_face["first_word"].map(quad_for),
    ).sort_values("n", ascending=False).reset_index(drop=True)

    # --- write outputs
    M.figures_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = M.figures_dir / "v3_kaomoji_predictiveness.tsv"
    per_face.to_csv(tsv_path, sep="\t", index=False)
    print(f"  wrote {tsv_path}")

    summary = {
        "model": short_name,
        "layer": M.preferred_layer,
        "n_rows_total": int(len(df)),
        "n_rows_kept": int(len(df_face)),
        "min_n": min_n,
        "n_faces_total": int(df["first_word"].nunique()),
        "n_faces_kept": int(len(keep_faces)),
        "hidden_to_face": {k: v for k, v in face_metrics.items() if k != "y_pred"},
        "hidden_to_quadrant": {k: v for k, v in quad_metrics.items() if k != "y_pred"},
        "face_to_hidden_eta2": eta,
        "centroid_approximation_quality": approx,
    }
    summary_path = M.figures_dir / "v3_kaomoji_predictiveness_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  wrote {summary_path}")

    fig_path = M.figures_dir / "fig_v3_kaomoji_predictiveness.png"
    _plot_per_face(per_face, df_face, short_name, M.preferred_layer, fig_path,
                   face_metrics, quad_metrics, eta)
    print(f"  wrote {fig_path}")

    return summary


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_per_face(
    per_face: pd.DataFrame,
    df_face: pd.DataFrame,
    short_name: str,
    layer: int | None,
    out_path: Path,
    face_metrics: dict,
    quad_metrics: dict,
    eta: dict,
) -> None:
    """Three-panel: per-face classifier recall, distinctiveness,
    within-consistency. Bars colored by per-face quadrant blend so the
    reader sees which faces are HP/LP/HN/LN/NB at a glance."""
    _use_cjk_font()
    if len(per_face) == 0:
        return
    weights = per_face_quadrant_weights(df_face)
    sorted_pf = per_face.sort_values("classifier_recall", ascending=True).reset_index(drop=True)
    n = len(sorted_pf)
    colors = [
        mix_quadrant_color(weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER}))
        for fw in sorted_pf["first_word"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.3 * n + 2)), sharey=True)

    # Panel A: classifier recall
    ax = axes[0]
    y = np.arange(n)
    ax.barh(y, sorted_pf["classifier_recall"].fillna(0).to_numpy(),
            color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['first_word']}  n={int(r['n'])}" for _, r in sorted_pf.iterrows()],
        fontsize=8,
    )
    ax.set_xlabel("classifier recall  (multi-class h_mean → face)")
    ax.set_xlim(0, 1.05)
    ax.axvline(face_metrics["uniform_baseline"], color="#888", linestyle="--",
               linewidth=0.8, label=f"uniform 1/K={face_metrics['uniform_baseline']:.3f}")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("(a) hidden → face")

    # Panel B: distinctiveness
    ax = axes[1]
    ax.barh(y, sorted_pf["distinctiveness"].to_numpy(),
            color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="#888", linewidth=0.6)
    ax.set_xlabel("distinctiveness\n(1 − mean cos to other face means)")
    ax.set_title("(b) face → hidden, per-face")

    # Panel C: within-face consistency
    ax = axes[2]
    ax.barh(y, sorted_pf["median_within_consistency"].to_numpy(),
            color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("median cos(row, face mean)")
    ax.set_xlim(0, 1.05)
    ax.set_title("(c) within-face consistency")

    layer_str = "max" if layer is None else f"L{layer}"
    fig.suptitle(
        f"v3 kaomoji predictiveness — {short_name} ({layer_str})\n"
        f"hidden→face acc {face_metrics['accuracy']:.2f} (vs majority "
        f"{face_metrics['majority_baseline']:.2f}, uniform "
        f"{face_metrics['uniform_baseline']:.2f}); "
        f"hidden→quadrant acc {quad_metrics['accuracy']:.2f}; "
        f"top-5 weighted η² of face identity = {eta['weighted_eta2_top_k']:.2f} "
        f"({eta['weighted_eta2_normalized']*100:.0f}% of top-5 subspace)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    import os
    if "LLMOJI_MODEL" in os.environ:
        # Single-model mode — respect the env var.
        candidates = [current_model().short_name]
    else:
        candidates = [
            name for name, M in MODEL_REGISTRY.items()
            if M.emotional_data_path.exists()
        ]

    print(f"models: {candidates}")
    summaries: dict[str, dict] = {}
    for short in candidates:
        s = run_model(short)
        if s:
            summaries[short] = s

    if len(summaries) >= 2:
        cross = {
            short: {
                "layer": s["layer"],
                "n_faces_kept": s["n_faces_kept"],
                "hidden_to_face_accuracy": s["hidden_to_face"]["accuracy"],
                "hidden_to_face_macro_f1": s["hidden_to_face"]["macro_f1"],
                "hidden_to_face_majority_baseline": s["hidden_to_face"]["majority_baseline"],
                "hidden_to_face_uniform_baseline": s["hidden_to_face"]["uniform_baseline"],
                "hidden_to_quadrant_accuracy": s["hidden_to_quadrant"]["accuracy"],
                "face_to_hidden_top5_weighted_eta2": s["face_to_hidden_eta2"]["weighted_eta2_top_k"],
                "face_to_hidden_top5_eta2_normalized": s["face_to_hidden_eta2"]["weighted_eta2_normalized"],
                "face_centroid_r2_full_hidden": s["centroid_approximation_quality"]["face"]["r2"],
                "face_centroid_mean_cosine": s["centroid_approximation_quality"]["face"]["mean_centered_cosine"],
                "face_centroid_error_ratio": s["centroid_approximation_quality"]["face"]["mean_error_over_baseline"],
                "quadrant_centroid_r2_full_hidden": s["centroid_approximation_quality"]["quadrant"]["r2"],
            }
            for short, s in summaries.items()
        }
        out_dir = FIGURES_DIR / "local" / "cross_model"
        out_dir.mkdir(parents=True, exist_ok=True)
        cross_path = out_dir / "v3_kaomoji_predictiveness_compare.json"
        with cross_path.open("w") as f:
            json.dump(cross, f, indent=2)
        print(f"\nwrote cross-model summary {cross_path}")
        print("\nCROSS-MODEL HEADLINE:")
        print(f"  {'model':<10}  {'h→face':>8}  {'h→quad':>8}  "
              f"{'face η²':>10}  {'norm η²':>10}")
        for short, c in cross.items():
            print(f"  {short:<10}  "
                  f"{c['hidden_to_face_accuracy']:>8.3f}  "
                  f"{c['hidden_to_quadrant_accuracy']:>8.3f}  "
                  f"{c['face_to_hidden_top5_weighted_eta2']:>10.4f}  "
                  f"{c['face_to_hidden_top5_eta2_normalized']:>10.3f}")


if __name__ == "__main__":
    main()
