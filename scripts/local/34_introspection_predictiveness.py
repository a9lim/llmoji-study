# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Per-condition kaomoji predictiveness on the introspection-pilot data.

Companion to the introspection pilot (`docs/2026-05-02-introspection-pilot.md`).
Runs the same metrics as `scripts/25_v3_kaomoji_predictiveness.py` but
splits by `condition` (intro_none / intro_pre / intro_lorem) instead of
by model. The headline question:

  Does the introspection preamble change how predictive the kaomoji is
  of the underlying hidden state? If introspection makes kaomoji a
  better predictor of internal state, that's evidence "introspection
  helps self-report." If it makes kaomoji a *worse* predictor, the
  preamble decouples kaomoji choice from state — surface register
  shift, not improved alignment.

Three metrics per condition:
  * **hidden → face** classifier accuracy (multiclass logistic on
    PCA(50)-reduced hidden state). High accuracy = kaomoji is
    recoverable from the state, i.e. state and kaomoji are tightly
    coupled.
  * **hidden → quadrant** classifier accuracy. Same machinery, on
    Russell labels. Reference for "how legible is affect from state."
  * **face-centroid R² in full hidden state**. "If you only knew the
    kaomoji, how much hidden-state variance would you recover?"

CLI:
  --which h_first | h_mean   default h_first. h_first is the
    methodology-invariant aggregate — same meaning before and after
    the 2026-05-02 MAX_NEW_TOKENS=16 cutover, so it gives clean
    cross-comparison between v3 main runs and post-cutover pilots.
    h_mean is available for back-compat / artifact-of-window analyses.
    See docs/gotchas.md for the cross-cutover semantics.
  --main                     read main-run data (M.emotional_data_path
    + M.experiment) instead of the introspection pilot. With main data
    there's only one condition (kaomoji_prompted), so the script
    reports a single row.

Outputs:
  data/{short}_{introspection|main}_predictiveness_{which}.tsv
  figures/local/{short}/fig_{introspection|main}_predictiveness_{which}.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.config import DATA_DIR, INTROSPECTION_CONDITIONS, current_model
from llmoji_study.emotional_analysis import _use_cjk_font, load_emotional_features_stack_at


# Import script 25's helpers via importlib (filename starts with a
# digit, so a normal `import` won't work). Treat 25 as the source of
# truth for the metric definitions.
_S25_PATH = Path(__file__).parent / "25_v3_kaomoji_predictiveness.py"
_spec = importlib.util.spec_from_file_location("_s25", _S25_PATH)
assert _spec is not None and _spec.loader is not None
_s25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s25)


_use_cjk_font()


def _per_condition(
    df, X, condition: str, *, min_n: int = 5,
) -> dict[str, Any]:
    sub_mask = (df["condition"] == condition).to_numpy()
    sub_df = df[sub_mask].reset_index(drop=True)
    sub_X = X[sub_mask]
    if len(sub_df) == 0:
        return {"condition": condition, "n_rows": 0}

    face_counts = sub_df["first_word"].value_counts()
    keep_faces = face_counts[face_counts >= min_n].index
    df_face = sub_df[sub_df["first_word"].isin(keep_faces)].reset_index(drop=True)
    X_face = sub_X[sub_df["first_word"].isin(keep_faces).to_numpy()]

    print(f"\n=== {condition} ===")
    print(f"  rows: {len(sub_df)} → kept {len(df_face)} "
          f"({len(keep_faces)}/{sub_df['first_word'].nunique()} faces with n≥{min_n})")

    if len(df_face) < 10 or len(keep_faces) < 2:
        print("  too few faces with sufficient n; skipping classifier")
        face_metrics: dict[str, Any] = {}
        eta: dict[str, Any] = {}
        approx: dict[str, Any] = {}
    else:
        face_metrics = _s25._multiclass_classifier_metrics(
            X_face, df_face["first_word"].to_numpy(),
        )
        print(f"  hidden → face:     n_classes={face_metrics['n_classes']}  "
              f"acc={face_metrics['accuracy']:.3f}  "
              f"macro-F1={face_metrics['macro_f1']:.3f}  "
              f"(majority {face_metrics['majority_baseline']:.3f}, "
              f"uniform {face_metrics['uniform_baseline']:.3f})")
        eta = _s25._eta_squared_per_pc(
            X_face, df_face["first_word"].to_numpy(), k=5,
        )
        print(f"  face → hidden:     top-5 weighted η² = "
              f"{eta['weighted_eta2_top_k']:.4f}  "
              f"(normalized {eta['weighted_eta2_normalized']*100:.1f}% of top-5 subspace)")
        approx = _s25._face_centroid_approximation_quality(df_face, X_face)
        print(f"  face-centroid R² in full hidden: "
              f"{approx['face']['r2']:.3f}  "
              f"(quadrant-centroid R² {approx['quadrant']['r2']:.3f}; "
              f"face gain +{(approx['face']['r2']-approx['quadrant']['r2'])*100:.1f}pp)")

    quad_metrics = _s25._multiclass_classifier_metrics(
        sub_X, sub_df["quadrant"].to_numpy(),
    )
    print(f"  hidden → quadrant: n_classes={quad_metrics['n_classes']}  "
          f"acc={quad_metrics['accuracy']:.3f}  "
          f"macro-F1={quad_metrics['macro_f1']:.3f}  "
          f"(majority {quad_metrics['majority_baseline']:.3f})")

    return {
        "condition": condition,
        "n_rows": int(len(sub_df)),
        "n_faces_kept": int(len(keep_faces)),
        "n_rows_kept": int(len(df_face)),
        "hidden_to_face": {k: v for k, v in face_metrics.items() if k != "y_pred"},
        "hidden_to_quadrant": {k: v for k, v in quad_metrics.items() if k != "y_pred"},
        "face_to_hidden_eta2": eta,
        "centroid_approximation_quality": approx,
    }


def _parse_args(argv: list[str]) -> tuple[str, bool, str | None]:
    which = "h_first"
    use_main = False
    custom_label: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--which" and i + 1 < len(argv):
            which = argv[i + 1]
            if which not in ("h_first", "h_mean", "h_last"):
                raise SystemExit(f"--which must be h_first|h_mean|h_last, got {which!r}")
            i += 2
            continue
        if a == "--main":
            use_main = True
            i += 1
            continue
        if a == "--custom-label" and i + 1 < len(argv):
            custom_label = argv[i + 1]
            i += 2
            continue
        i += 1
    return which, use_main, custom_label


def main() -> None:
    which, use_main, custom_label = _parse_args(sys.argv[1:])

    M = current_model()
    figdir = M.figures_dir
    figdir.mkdir(parents=True, exist_ok=True)

    if use_main:
        raw_path = M.emotional_data_path
        experiment = M.experiment
        # Main-run JSONL has condition="kaomoji_prompted" (1 condition).
        conditions = ["kaomoji_prompted"]
        tag = "main"
    else:
        raw_path = DATA_DIR / f"{M.short_name}_introspection_raw.jsonl"
        experiment = f"{M.experiment}_introspection"
        conditions = list(INTROSPECTION_CONDITIONS)
        tag = "introspection"
        if custom_label:
            conditions.append("intro_custom")
            tag = f"introspection_custom_{custom_label}"

    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path}; nothing to analyze")

    print(f"model: {M.short_name} ({tag} mode); aggregate: {which}")
    print(f"jsonl: {raw_path}")
    print(f"experiment: {experiment}")
    print(f"loading {which} (layer-stack) ...")
    df, X = load_emotional_features_stack_at(
        str(raw_path), DATA_DIR,
        experiment=experiment,
        which=which,
        split_hn=False,  # use the 5-class quadrant view for the quadrant classifier
    )

    if custom_label and not use_main:
        custom_path = (
            DATA_DIR / f"{M.short_name}_introspection_custom_{custom_label}.jsonl"
        )
        if not custom_path.exists():
            raise SystemExit(
                f"missing {custom_path} — run scripts/43 with "
                f"--label {custom_label} first"
            )
        custom_experiment = f"{M.experiment}_introspection_custom_{custom_label}"
        df_c, X_c = load_emotional_features_stack_at(
            str(custom_path), DATA_DIR,
            experiment=custom_experiment,
            which=which,
            split_hn=False,
        )
        df_c["condition"] = "intro_custom"
        import numpy as _np
        import pandas as _pd
        df = _pd.concat([df, df_c], ignore_index=True)
        X = _np.concatenate([X, X_c], axis=0)
        print(f"  merged {len(df_c)} custom-{custom_label} rows")

    print(f"  rows: {len(df)}, hidden dim: {X.shape[1]}")

    summaries = []
    for c in conditions:
        s = _per_condition(df, X, c)
        summaries.append(s)

    # --- write summary TSV
    import pandas as pd
    rows = []
    for s in summaries:
        if not s.get("hidden_to_face"):
            continue
        rows.append({
            "condition": s["condition"],
            "n_rows_kept": s["n_rows_kept"],
            "n_faces_kept": s["n_faces_kept"],
            "hidden_to_face_acc": s["hidden_to_face"]["accuracy"],
            "hidden_to_face_macroF1": s["hidden_to_face"]["macro_f1"],
            "majority_baseline": s["hidden_to_face"]["majority_baseline"],
            "uniform_baseline": s["hidden_to_face"]["uniform_baseline"],
            "hidden_to_quadrant_acc": s["hidden_to_quadrant"]["accuracy"],
            "face_to_hidden_top5_weighted_eta2": s["face_to_hidden_eta2"]["weighted_eta2_top_k"],
            "face_to_hidden_top5_eta2_norm": s["face_to_hidden_eta2"]["weighted_eta2_normalized"],
            "face_centroid_r2_full_hidden": s["centroid_approximation_quality"]["face"]["r2"],
            "quadrant_centroid_r2_full_hidden": s["centroid_approximation_quality"]["quadrant"]["r2"],
            "face_gain_over_quadrant_pp": (
                s["centroid_approximation_quality"]["face"]["r2"]
                - s["centroid_approximation_quality"]["quadrant"]["r2"]
            ) * 100,
        })
    summary_df = pd.DataFrame(rows)
    tsv_path = DATA_DIR / f"{M.short_name}_{tag}_predictiveness_{which}.tsv"
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nwrote {tsv_path}")
    print(summary_df.to_string(index=False))

    # --- bar comparison plot
    if len(summary_df) > 0:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        x = np.arange(len(summary_df))
        labels = summary_df["condition"].tolist()

        # Panel A: hidden → face accuracy (with baselines)
        ax = axes[0]
        ax.bar(x, summary_df["hidden_to_face_acc"], color="#3a86ff",
               edgecolor="black", linewidth=0.6)
        # majority baseline as a red dashed line per bar
        for i, (_, r) in enumerate(summary_df.iterrows()):
            ax.plot([i - 0.4, i + 0.4],
                    [r["majority_baseline"], r["majority_baseline"]],
                    color="#d44a4a", linestyle="--", linewidth=1.0)
            ax.plot([i - 0.4, i + 0.4],
                    [r["uniform_baseline"], r["uniform_baseline"]],
                    color="#888", linestyle=":", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("accuracy")
        ax.set_title("(A) hidden → face\n(red dashed = majority; gray dotted = uniform)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

        # Panel B: hidden → quadrant accuracy
        ax = axes[1]
        ax.bar(x, summary_df["hidden_to_quadrant_acc"], color="#9d4ad4",
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("accuracy")
        ax.set_title("(B) hidden → quadrant (5-class)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

        # Panel C: top-5 weighted η²
        ax = axes[2]
        ax.bar(x, summary_df["face_to_hidden_top5_weighted_eta2"], color="#ffb84a",
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("weighted η²")
        ax.set_title("(C) face → top-5 PCs of hidden\n(face-identity variance / total)")
        ax.grid(True, alpha=0.2, axis="y")

        # Panel D: full-hidden R² for face vs quadrant centroid
        ax = axes[3]
        w = 0.4
        ax.bar(x - w/2, summary_df["face_centroid_r2_full_hidden"],
               width=w, color="#3a86ff", edgecolor="black", linewidth=0.6,
               label="face centroid")
        ax.bar(x + w/2, summary_df["quadrant_centroid_r2_full_hidden"],
               width=w, color="#9d4ad4", edgecolor="black", linewidth=0.6,
               label="quadrant centroid")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("R² (full hidden)")
        ax.set_title("(D) face vs quadrant centroid R²\nfull hidden-state space")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2, axis="y")

        fig.suptitle(
            f"{M.short_name}: {tag} kaomoji predictiveness × condition\n"
            f"({which}, layer-stack; min_n=5 per face for face classifier)"
        )
        fig.tight_layout()
        fig_path = figdir / f"fig_{tag}_predictiveness_{which}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
