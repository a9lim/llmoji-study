"""Figures for the v3 extension-probe rescore (2026-04-29).

Reads the JSONLs (which now carry `extension_probe_scores_t0/_tlast`
and `extension_probe_means` from script 27) and produces four
cross-model figures:

  fig_v3_extension_quadrant_means.png
      Per-quadrant mean probe score at h_last for the five most
      relevant extension probes, gemma | qwen side-by-side. Shows
      gemma's clean PAD-aligned dominance separation and qwen's
      flatness at h_last.

  fig_v3_extension_dominance_scatter.png
      Per-row scatter of fearful.unflinching (h_last) vs
      powerful.powerless (h_last), colored by Russell quadrant, one
      panel per model. Annotates per-row Pearson r. Shows gemma's
      r=-0.94 axis collapse vs qwen's r≈0 flatness.

  fig_v3_extension_hn_dominance_split.png
      HN-only natural experiment: for each model split HN rows into
      bottom-third / middle / top-third by powerful.powerless and
      stack-bar the kaomoji register (shocked / sad-teary / other).
      Shows that gemma's kaomoji vocabulary actually marks the
      within-HN dominance split (high-fear → shocked register;
      low-fear → sad-teary).

  fig_v3_extension_probe_correlations.png
      Per-row Pearson correlation matrix over the 5 core probes
      (probe_means) + extension probes (extension_probe_scores_tlast),
      gemma | qwen. Shows gemma's affect probes collapsing into a
      single 1D direction (huge red+blue blocks) vs qwen's more
      distributed structure.

No model time, no new generations — strictly from the JSONLs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import MODEL_REGISTRY
from llmoji_study.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
)


# Probes to highlight in the per-quadrant figure (subset of the 12
# extension probes, ordered by theoretical relevance to the V-A
# circumplex extension story).
HIGHLIGHT_PROBES = [
    "powerful.powerless",      # PAD dominance — headline
    "fearful.unflinching",     # direct fear probe (auto-discovered)
    "disgusted.accepting",     # Plutchik disgust
    "surprised.unsurprised",   # Plutchik surprise
    "curious.disinterested",   # auto-discovered, register-y
]

# Kaomoji register categories for the HN dominance-split natural
# experiment. Lists chosen from the gemma + qwen v3 vocabulary
# (CLAUDE.md "shocked/distress register" notes plus inspection of
# the JSONLs).
SHOCKED_FORMS = {
    # gemma + qwen overlap on the table-flip / wide-eye / pained set
    "(╯°□°)", "(╯°□°）", "(°Д°)", "(ºДº)",
    "(⊙_⊙)", "(⊙﹏⊙)", "(⊙_⊙;)",
    "(>_<)", "(>﹏<)", "(＞_＜)", "(＞﹏＜)",
    "(；´д｀)", "(;´д｀)", "(；´Д｀)", "(;´Д｀)",
    "(；′⌒`)", "(；′⌒\\`)",
    "(っ°Д°)",
}
SAD_FORMS = {
    "(｡•́︿•̀｡)", "(･́︿･̀)", "(｡╯︵╰｡)", "(╯︵╰)",
    "(╥_╥)", "(╥﹏╥)", "(っ╥﹏╥)", "(っ╥﹏╥)っ",
    "(；ω；)", "(；_；)", "(;_;)", "(;ω;)",
    "(っ´ω`)", "(っ´ω`ｃ)", "(っ´ω`c)",
    "(っ﹏ò)", "(っò_ó)",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _quad(pid: str) -> str:
    return pid[:2].upper() if len(pid) >= 2 else "??"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    x = np.asarray(xs); y = np.asarray(ys)
    x = x - x.mean(); y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


def _register_of(form: str) -> str:
    if form in SHOCKED_FORMS:
        return "shocked"
    if form in SAD_FORMS:
        return "sad-teary"
    return "other"


# ---------------------------------------------------------------------------
# Figure A: per-quadrant mean probe score at h_last
# ---------------------------------------------------------------------------


def fig_quadrant_means(by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, short in zip(axes, ("gemma", "qwen")):
        rows = by_model[short]
        n_probes = len(HIGHLIGHT_PROBES)
        n_quads = len(QUADRANT_ORDER)
        bar_w = 0.16
        x = np.arange(n_probes)
        for qi, q in enumerate(QUADRANT_ORDER):
            scores: list[float] = []
            for probe in HIGHLIGHT_PROBES:
                vals = [
                    r["extension_probe_scores_tlast"][probe]
                    for r in rows
                    if "extension_probe_scores_tlast" in r
                    and probe in r["extension_probe_scores_tlast"]
                    and _quad(r["prompt_id"]) == q
                    and "error" not in r
                ]
                scores.append(float(np.mean(vals)) if vals else 0.0)
            offset = (qi - (n_quads - 1) / 2) * bar_w
            ax.bar(
                x + offset, scores, width=bar_w,
                color=QUADRANT_COLORS[q], edgecolor="black", linewidth=0.4,
                label=q if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [p.split(".")[0] for p in HIGHLIGHT_PROBES],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_title(f"{short}", fontsize=12)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("mean probe score at h_last")
    axes[0].legend(
        title="quadrant", loc="upper left", fontsize=8, ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "v3 extension probes: per-quadrant mean at h_last",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure B: per-row scatter fearful vs powerful, colored by quadrant
# ---------------------------------------------------------------------------


def fig_dominance_scatter(by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, short in zip(axes, ("gemma", "qwen")):
        rows = [r for r in by_model[short]
                if "extension_probe_scores_tlast" in r
                and "fearful.unflinching" in r["extension_probe_scores_tlast"]
                and "powerful.powerless" in r["extension_probe_scores_tlast"]]
        xs_all: list[float] = []
        ys_all: list[float] = []
        for q in QUADRANT_ORDER:
            sub = [r for r in rows if _quad(r["prompt_id"]) == q]
            xs = [r["extension_probe_scores_tlast"]["fearful.unflinching"]
                  for r in sub]
            ys = [r["extension_probe_scores_tlast"]["powerful.powerless"]
                  for r in sub]
            ax.scatter(
                xs, ys, c=QUADRANT_COLORS[q], s=14, alpha=0.65,
                edgecolors="none", label=f"{q} (n={len(sub)})",
            )
            xs_all.extend(xs); ys_all.extend(ys)
        r = _pearson(xs_all, ys_all)
        ax.set_xlabel("fearful.unflinching  (h_last)")
        ax.set_ylabel("powerful.powerless  (h_last)")
        ax.set_title(f"{short}  —  per-row Pearson r = {r:+.3f}")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        ax.legend(loc="best", fontsize=7, ncols=2, framealpha=0.85)
    fig.suptitle(
        "fearful vs powerful at h_last  —  PAD's dominance × fear plane",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure C: HN dominance-split kaomoji register stack
# ---------------------------------------------------------------------------


def fig_hn_dominance_split(by_model: dict[str, list[dict]], out: Path) -> None:
    REGISTER_COLORS = {
        "shocked":   "#d44a4a",  # red — fear-y register
        "sad-teary": "#4a7ed4",  # blue — sad register
        "other":     "#a0a0a0",  # gray
    }
    REGISTERS = ["shocked", "sad-teary", "other"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, short in zip(axes, ("gemma", "qwen")):
        rows = [r for r in by_model[short]
                if _quad(r.get("prompt_id", "")) == "HN"
                and "extension_probe_scores_tlast" in r
                and r.get("first_word", "").startswith("(")]
        rows.sort(key=lambda r: r["extension_probe_scores_tlast"]["powerful.powerless"])
        n = len(rows)
        thirds = [
            ("low",  rows[: n // 3]),
            ("mid",  rows[n // 3: 2 * n // 3]),
            ("high", rows[-(n // 3):]),
        ]
        x = np.arange(len(thirds))
        bottoms = np.zeros(len(thirds))
        for reg in REGISTERS:
            counts = []
            for _, sub in thirds:
                c = sum(1 for r in sub if _register_of(r["first_word"]) == reg)
                counts.append(c)
            counts = np.asarray(counts, dtype=float)
            ax.bar(x, counts, bottom=bottoms, color=REGISTER_COLORS[reg],
                   edgecolor="black", linewidth=0.4, label=reg)
            bottoms += counts
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lab}\n(n={len(sub)})" for lab, sub in thirds], fontsize=9,
        )
        ax.set_title(f"{short}  —  HN rows split by powerful.powerless")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("kaomoji count")
    axes[0].legend(
        loc="upper left", fontsize=8, frameon=False, title="register",
    )
    fig.suptitle(
        "HN dominance-split natural experiment  "
        "—  kaomoji register vs powerful.powerless tertile",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure D: per-row Pearson correlation matrix across all probes
# ---------------------------------------------------------------------------


CORE_PROBES = [
    "happy.sad", "angry.calm", "confident.uncertain",
    "warm.clinical", "humorous.serious",
]


def _build_probe_matrix(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """Build (probe_names, n_rows × n_probes matrix). Core probes use
    `probe_means`; extension probes use `extension_probe_scores_tlast`.
    Drops rows missing any probe."""
    if not rows:
        return [], np.zeros((0, 0))

    sample = next((r for r in rows if "extension_probe_scores_tlast" in r), None)
    if sample is None:
        return [], np.zeros((0, 0))
    ext_names = sorted(sample["extension_probe_scores_tlast"].keys())
    names = CORE_PROBES + ext_names

    mat: list[list[float]] = []
    for r in rows:
        if "error" in r: continue
        try:
            row_vals = [r["probe_means"][p] for p in CORE_PROBES] + \
                       [r["extension_probe_scores_tlast"][p] for p in ext_names]
        except KeyError:
            continue
        mat.append(row_vals)
    return names, np.asarray(mat, dtype=np.float32)


def fig_probe_correlations(by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    im = None
    for ax, short in zip(axes, ("gemma", "qwen")):
        names, X = _build_probe_matrix(by_model[short])
        if X.size == 0:
            ax.set_title(f"{short}  —  no data")
            continue
        # Mean-center then divide by stdev for each probe before computing
        # the correlation matrix; equivalent to Pearson per pair.
        Xc = X - X.mean(axis=0, keepdims=True)
        Xn = Xc / (Xc.std(axis=0, keepdims=True) + 1e-12)
        corr = (Xn.T @ Xn) / Xn.shape[0]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=7.5)
        # Mark the boundary between core and extension probes.
        sep = len(CORE_PROBES) - 0.5
        ax.axhline(sep, color="black", linewidth=0.6)
        ax.axvline(sep, color="black", linewidth=0.6)
        ax.set_title(f"{short}  (n={X.shape[0]} rows)")
        for i in range(len(names)):
            for j in range(len(names)):
                v = corr[i, j]
                if abs(v) >= 0.5 and i != j:
                    ax.text(j, i, f"{v:+.2f}",
                            ha="center", va="center",
                            color="white" if abs(v) > 0.7 else "black",
                            fontsize=5.5)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    fig.suptitle(
        "per-row Pearson correlations across probes  "
        "(core: probe_means; extension: h_last)",
        fontsize=13, y=1.02,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    _use_cjk_font()

    by_model: dict[str, list[dict]] = {}
    for short in ("gemma", "qwen"):
        path = MODEL_REGISTRY[short].emotional_data_path
        rows = _load(path)
        with_ext = sum(1 for r in rows if "extension_probe_scores_tlast" in r)
        print(f"{short}: {len(rows)} rows ({with_ext} with extension scores)")
        if with_ext == 0:
            print(f"  WARN: no extension scores on {short}; run scripts/27 first")
        by_model[short] = rows

    out_dir = Path(__file__).resolve().parent.parent / "figures" / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_quadrant_means(by_model, out_dir / "fig_v3_extension_quadrant_means.png")
    fig_dominance_scatter(by_model, out_dir / "fig_v3_extension_dominance_scatter.png")
    fig_hn_dominance_split(by_model, out_dir / "fig_v3_extension_hn_dominance_split.png")
    fig_probe_correlations(by_model, out_dir / "fig_v3_extension_probe_correlations.png")


if __name__ == "__main__":
    main()
