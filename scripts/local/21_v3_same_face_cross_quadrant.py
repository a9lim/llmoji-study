"""v3 cross-quadrant emitter natural experiment.

Some kaomoji appear across multiple Russell quadrants — gemma's
``(｡•́︿•̀｡)`` (n=171, LN+HN), qwen's ``(;ω;)``, ``(;´д｀)``,
``(｡•́︿•̀｡)``. The model is emitting the *same face* in design conditions
the user-side schema says are different. Two outcomes are possible
and the answer constrains every later interpretation:

* If hidden states *also* don't separate the LN-instance from the
  HN-instance of one face, the model genuinely doesn't distinguish
  the quadrants for that prompt cluster. The Russell labelling is
  finer-grained than the model's internal representation.
* If hidden states *do* separate them, the kaomoji vocabulary is the
  bottleneck — the model knows the difference but doesn't have a face
  for it. Kaomoji is partial readout, not state itself.

Per qualifying face we train a logistic regression on h_mean to
predict quadrant from that face's rows alone, with stratified 5-fold
CV. The null is the majority-class accuracy plus a label-shuffle
distribution (200 shuffles) — accuracy above the upper null quantile
counts as separable.

Outputs:
  figures/local/<short>/fig_v3_same_face_cross_quadrant_<face>.png
    one PCA scatter per qualifying face, points colored by quadrant
  figures/local/<short>/v3_same_face_cross_quadrant.tsv
    summary table: face, n, quadrants, classifier_acc, null_q95,
    separates, n_per_quadrant
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from llmoji_study.config import current_model
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT as QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features_stack,
)


def _sanitize_face_for_path(face: str) -> str:
    """Filename-safe slug from a kaomoji string. We want stable, short
    labels so figures are easy to grep — fall back to a hex hash if
    the printable representation is empty after stripping."""
    import hashlib
    safe = "".join(c for c in face if c.isalnum())
    h = hashlib.md5(face.encode("utf-8")).hexdigest()[:8]
    if safe:
        return f"{safe[:16]}_{h}"
    return f"face_{h}"


def _qualifying_faces(
    df: pd.DataFrame,
    *,
    min_per_quadrant: int = 3,
    min_quadrants: int = 2,
) -> list[tuple[str, dict[str, int]]]:
    """Faces that emit in at least ``min_quadrants`` distinct quadrants
    with at least ``min_per_quadrant`` rows in each."""
    out = []
    for face, sub in df.groupby("first_word"):
        per_q = sub["quadrant"].value_counts().to_dict()
        ok = {q: int(n) for q, n in per_q.items() if n >= min_per_quadrant}
        if len(ok) >= min_quadrants:
            out.append((str(face), ok))
    # Order by total emissions desc so the most informative faces come first.
    out.sort(key=lambda t: -sum(t[1].values()))
    return out


def _classifier_score(
    X: np.ndarray, y: np.ndarray, *,
    seed: int = 0, n_perm: int = 30,
) -> tuple[float, float, float, float]:
    """5-fold stratified CV mean accuracy on a logistic regression
    pipeline. Returns (cv_mean, cv_std, majority_acc, null_q95).

    Null: ``n_perm`` label-shuffles, take the 95th percentile of CV
    mean accuracy as the upper bound of "what chance permutation can
    produce". Above that = separable. 30 shuffles is enough for a
    one-sided 0.05 threshold (the 95th percentile of 30 draws has a
    standard error of ~1 rank); larger n_perm only smooths it.

    Min-fold = 2 per class for stratified k-fold. We pre-screen so
    every quadrant has ``min_per_quadrant`` rows.

    Dim-reduction: hidden states are 5376-dim, classes have 3-30
    examples per quadrant. PCA-prefix projects to a small fixed
    component count (cap at 20, no more than 4n/5 - 2 to leave room
    for the CV split). Without it lbfgs takes ages; with it the
    classifier still has more degrees of freedom than the data.
    """
    from sklearn.decomposition import PCA as _PCA
    n = len(y)
    counts = pd.Series(y).value_counts()
    majority_acc = float(counts.max() / n)

    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return float("nan"), float("nan"), majority_acc, float("nan")

    # Conservative PCA cap. PCA needs n_components < min(n_samples,
    # n_features) on EVERY CV training fold, and stratified splits can
    # leave the smallest fold uncomfortably small when one class has
    # exactly min_per_quadrant rows. n // 3 is enough headroom in every
    # observed configuration.
    pca_cap = max(2, min(20, n // 3))

    pipe = Pipeline([
        ("pca", _PCA(n_components=pca_cap, random_state=seed)),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            C=0.1, max_iter=2000,
            solver="lbfgs", random_state=seed,
        )),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    cv_mean = float(scores.mean())
    cv_std = float(scores.std(ddof=0))

    rng = np.random.default_rng(seed)
    null_means = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            s = cross_val_score(pipe, X, y_perm, cv=cv,
                                scoring="accuracy", n_jobs=1)
            null_means.append(float(s.mean()))
        except ValueError:
            continue
    null_q95 = float(np.quantile(null_means, 0.95)) if null_means else float("nan")

    return cv_mean, cv_std, majority_acc, null_q95


def _plot_per_face_pca(
    face: str,
    sub_df: pd.DataFrame,
    sub_X: np.ndarray,
    cv_mean: float,
    null_q95: float,
    out_path: Path,
) -> None:
    """One PCA scatter for one face's rows, colored by quadrant."""
    _use_cjk_font()
    pca = PCA(n_components=2)
    Y = pca.fit_transform(sub_X)
    quadrants = sub_df["quadrant"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for q in QUADRANT_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        ax.scatter(
            Y[mask, 0], Y[mask, 1],
            c=QUADRANT_COLORS[q], s=60, alpha=0.78,
            edgecolor="black", linewidth=0.4, label=f"{q}  n={int(mask.sum())}",
        )
    # Per-quadrant centroid stars.
    for q in QUADRANT_ORDER:
        mask = quadrants == q
        if mask.sum() < 2:
            continue
        cent = Y[mask].mean(axis=0)
        ax.plot(
            cent[0], cent[1], marker="*", markersize=22,
            color=QUADRANT_COLORS[q],
            markeredgecolor="black", markeredgewidth=1.0, zorder=5,
        )

    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    sep = "yes" if cv_mean > null_q95 else "no"
    ax.set_title(
        f"{face}  ({len(sub_df)} rows)\n"
        f"PCA on this face's rows; quadrant classifier acc {cv_mean:.2f} "
        f"vs null q95 {null_q95:.2f} → separates? {sep}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8, frameon=False, title="quadrant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_grid(
    summary: pd.DataFrame,
    short_name: str,
    out_path: Path,
) -> None:
    """Bar-strip: classifier accuracy vs null_q95 vs majority for every
    qualifying face. Threshold line at null_q95 makes the
    "separable / not" reading immediate."""
    _use_cjk_font()
    if len(summary) == 0:
        return
    s = summary.sort_values("classifier_acc", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(s) + 2)))
    y = np.arange(len(s))
    ax.barh(y, s["classifier_acc"], color="#1f77b4", alpha=0.78,
            label="CV accuracy")
    ax.scatter(s["null_q95"], y, color="#d62728", marker="|", s=140,
               linewidths=2.0, label="null q95 (perm shuffles)", zorder=5)
    ax.scatter(s["majority_acc"], y, color="#7f7f7f", marker="x", s=60,
               linewidths=1.5, label="majority-class baseline", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['face']}  n={int(r['n_total'])}"
                        for _, r in s.iterrows()], fontsize=8)
    ax.set_xlabel("classifier accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f"{short_name} cross-quadrant emitters: per-face "
        f"quadrant separability\n"
        f"L2-logistic on h_mean, 5-fold stratified CV"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-face", action="store_true",
        help="Also write per-face PCA panels. Default: only summary "
             "PNG + TSV, since the per-face panels duplicate what the "
             "summary already shows.",
    )
    args = ap.parse_args()

    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no v3 data at {M.emotional_data_path}")
        sys.exit(1)

    print(f"model: {M.short_name}")
    print("loading v3 hidden-state features (h_first, layer-stack)...")
    df, X = load_emotional_features_stack(
        M.short_name, which="h_first", split_hn=True,
    )
    print(f"  {len(df)} kaomoji-bearing rows, X {X.shape} (layer-stack)")

    qualifying = _qualifying_faces(df, min_per_quadrant=3, min_quadrants=2)
    print(f"  {len(qualifying)} cross-quadrant emitters "
          f"(≥2 quadrants × ≥3 rows each)")
    if not qualifying:
        print("nothing to test; rerun once vocab gets richer")
        return

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for face, per_q in qualifying:
        face_mask = df["first_word"].to_numpy() == face
        sub_df = df.loc[face_mask].reset_index(drop=True)
        sub_X = X[face_mask]
        # Drop any quadrant with < 3 rows so the classifier doesn't see them.
        keep_quads = list(per_q.keys())
        keep_mask = np.isin(sub_df["quadrant"].to_numpy(), keep_quads)
        sub_df = sub_df.loc[keep_mask].reset_index(drop=True)
        sub_X = sub_X[keep_mask]

        y = sub_df["quadrant"].to_numpy()
        cv_mean, cv_std, majority_acc, null_q95 = _classifier_score(sub_X, y)

        separates = (
            (not np.isnan(cv_mean))
            and (not np.isnan(null_q95))
            and cv_mean > null_q95
        )

        face_slug = _sanitize_face_for_path(face)
        out_path = M.figures_dir / f"fig_v3_same_face_cross_quadrant_{face_slug}.png"
        if args.per_face:
            _plot_per_face_pca(face, sub_df, sub_X, cv_mean, null_q95, out_path)

        rows.append({
            "face": face,
            "n_total": int(len(sub_df)),
            "n_quadrants": int(len(per_q)),
            "quadrants": ",".join(sorted(per_q.keys())),
            "per_quadrant_n": "; ".join(
                f"{q}:{n}" for q, n in sorted(per_q.items())
            ),
            "classifier_acc": cv_mean,
            "classifier_std": cv_std,
            "majority_acc": majority_acc,
            "null_q95": null_q95,
            "separates": separates,
            "fig": out_path.name if args.per_face else "",
        })
        sep_str = "✓" if separates else "·"
        print(f"  {sep_str} {face}  n={len(sub_df):3d}  "
              f"acc={cv_mean:.2f}±{cv_std:.2f}  "
              f"null_q95={null_q95:.2f}  majority={majority_acc:.2f}")

    summary = pd.DataFrame(rows)
    tsv_path = M.figures_dir / "v3_same_face_cross_quadrant.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nwrote {tsv_path}")

    grid_path = M.figures_dir / "fig_v3_same_face_cross_quadrant_summary.png"
    _plot_summary_grid(summary, M.short_name, grid_path)
    print(f"wrote {grid_path}")

    n_separates = int(summary["separates"].sum())
    print(f"\nseparable: {n_separates}/{len(summary)} cross-quadrant emitters")
    if n_separates > 0:
        print("→ kaomoji vocabulary IS a bottleneck for those faces — model "
              "discriminates internally but has no distinct face")
    if n_separates < len(summary):
        print(f"→ {len(summary) - n_separates} face(s) genuinely don't "
              "separate; quadrant labels finer than internal representation")


if __name__ == "__main__":
    main()
