"""Figures for the v3 extension-probe rescore (2026-04-29; rule-3
redesign 2026-05-01).

Reads the JSONLs (which now carry `extension_probe_scores_t0/_tlast`
and `extension_probe_means` from script 27) and produces three
cross-model figures (gemma | qwen | ministral):

  fig_v3_extension_quadrant_means.png
      Per-quadrant mean probe score at h_first for the five most
      relevant extension probes. With HN split into HN-D / HN-S
      (rule-3 redesign 2026-05-01), the per-quadrant bars show
      whether HN-D vs HN-S sit at meaningfully different probe
      values — directly answering rule 3a / 3b.

  fig_v3_extension_hn_dominance_split.png
      HN-D vs HN-S kaomoji register stack: for each model, count
      how often HN-D and HN-S rows reach for the shocked register
      vs the sad-teary register. Tests whether kaomoji vocabulary
      itself reflects the dominance split that the registry tags
      were designed to capture. (Pre-2026-05-01 this used a
      powerful.powerless tertile as a workaround for not having
      labels; with explicit HN-D/HN-S tags from
      `EmotionalPrompt.pad_dominance`, the cleaner version splits
      on the tags directly.)

  fig_v3_extension_probe_correlations.png
      Per-row Pearson correlation matrix over the 5 core probes
      (probe_means) + extension probes (extension_probe_scores_t0).
      Descriptive — shows which probes load on the same direction
      across each model.

Dropped 2026-05-01:
  fig_v3_extension_dominance_scatter.png — per-row fear×dominance
  scatter assumed `powerful.powerless` reads PAD dominance; the rule
  3 redesign showed it doesn't track HN-D / HN-S in the predicted
  direction across any of the three models. The relationship the
  scatter plotted was real but its theoretical interpretation was
  wrong, and we now read the dominance signal from the registry
  tags instead.

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
    QUADRANT_ORDER_SPLIT,
    _hn_split_map,
    _use_cjk_font,
)
# Use the HN-split ordering (HN→HN-D/HN-S) for all panels of script 28.
# Untagged-HN rows (hn06/hn15/hn17) drop out of the per-quadrant means
# automatically — the registry split returns None for them and _quad
# below filters those out.
QUADRANT_ORDER = QUADRANT_ORDER_SPLIT
_HN_SPLIT = _hn_split_map()
MODELS = ("gemma", "qwen", "ministral")


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
    """Return the split-mode quadrant label: HP/LP/HN-D/HN-S/LN/NB.
    Untagged-HN prompts (hn06/hn15/hn17) return ``"??"`` so they fall
    through every QUADRANT_ORDER iterator."""
    if len(pid) < 2:
        return "??"
    base = pid[:2].upper()
    if base == "HN":
        return _HN_SPLIT.get(pid, "??")
    return base


def _register_of(form: str) -> str:
    if form in SHOCKED_FORMS:
        return "shocked"
    if form in SAD_FORMS:
        return "sad-teary"
    return "other"


# ---------------------------------------------------------------------------
# Figure A: per-quadrant mean probe score at h_first
# ---------------------------------------------------------------------------


def fig_quadrant_means(by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    for ax, short in zip(axes, MODELS):
        rows = by_model[short]
        n_probes = len(HIGHLIGHT_PROBES)
        n_quads = len(QUADRANT_ORDER)
        bar_w = 0.16
        x = np.arange(n_probes)
        for qi, q in enumerate(QUADRANT_ORDER):
            scores: list[float] = []
            for probe in HIGHLIGHT_PROBES:
                vals = [
                    r["extension_probe_scores_t0"][probe]
                    for r in rows
                    if "extension_probe_scores_t0" in r
                    and probe in r["extension_probe_scores_t0"]
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
    axes[0].set_ylabel("mean probe score at h_first")
    axes[0].legend(
        title="quadrant", loc="upper left", fontsize=8, ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "v3 extension probes: per-quadrant mean at h_first",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure B: HN-D vs HN-S kaomoji register stack
# ---------------------------------------------------------------------------


def fig_hn_dominance_split(by_model: dict[str, list[dict]], out: Path) -> None:
    """For each model, count how often HN-D vs HN-S rows reach for the
    shocked register vs the sad-teary register. Tests whether the
    kaomoji vocabulary itself reflects the dominance split that the
    `pad_dominance` registry tags were designed to capture.

    Pre-2026-05-01 this used a powerful.powerless tertile as a
    workaround for not having labels — that probe turned out not to
    track HN-D vs HN-S, and we now have explicit registry tags. The
    cleaner version splits on the tags directly."""
    REGISTER_COLORS = {
        "shocked":   "#d44a4a",  # red — anger / fear-shocked register
        "sad-teary": "#4a7ed4",  # blue — sadness register
        "other":     "#a0a0a0",  # gray
    }
    REGISTERS = ["shocked", "sad-teary", "other"]
    GROUPS = ["HN-D", "HN-S"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    for ax, short in zip(axes, MODELS):
        rows = [r for r in by_model[short]
                if _quad(r.get("prompt_id", "")) in GROUPS
                and r.get("first_word", "").startswith("(")
                and "error" not in r]
        x = np.arange(len(GROUPS))
        bottoms = np.zeros(len(GROUPS))
        per_group_n: list[int] = []
        for g in GROUPS:
            per_group_n.append(sum(1 for r in rows if _quad(r["prompt_id"]) == g))
        for reg in REGISTERS:
            counts = []
            for g in GROUPS:
                c = sum(
                    1 for r in rows
                    if _quad(r["prompt_id"]) == g
                    and _register_of(r["first_word"]) == reg
                )
                counts.append(c)
            counts = np.asarray(counts, dtype=float)
            ax.bar(x, counts, bottom=bottoms, color=REGISTER_COLORS[reg],
                   edgecolor="black", linewidth=0.4, label=reg)
            bottoms += counts
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{g}\n(n={n})" for g, n in zip(GROUPS, per_group_n)],
            fontsize=10,
        )
        ax.set_title(f"{short}  —  HN-D vs HN-S register stack")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("kaomoji count")
    axes[0].legend(
        loc="upper left", fontsize=8, frameon=False, title="register",
    )
    fig.suptitle(
        "HN-D vs HN-S kaomoji register  —  "
        "does vocabulary reflect the dominance split?",
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
    `probe_means`; extension probes use `extension_probe_scores_t0`.
    Drops rows missing any probe."""
    if not rows:
        return [], np.zeros((0, 0))

    sample = next((r for r in rows if "extension_probe_scores_t0" in r), None)
    if sample is None:
        return [], np.zeros((0, 0))
    ext_names = sorted(sample["extension_probe_scores_t0"].keys())
    names = CORE_PROBES + ext_names

    mat: list[list[float]] = []
    for r in rows:
        if "error" in r: continue
        try:
            row_vals = [r["probe_means"][p] for p in CORE_PROBES] + \
                       [r["extension_probe_scores_t0"][p] for p in ext_names]
        except KeyError:
            continue
        mat.append(row_vals)
    return names, np.asarray(mat, dtype=np.float32)


def fig_probe_correlations(by_model: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    im = None
    for ax, short in zip(axes, MODELS):
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
    for short in MODELS:
        path = MODEL_REGISTRY[short].emotional_data_path
        rows = _load(path)
        with_ext = sum(1 for r in rows if "extension_probe_scores_t0" in r)
        print(f"{short}: {len(rows)} rows ({with_ext} with extension scores)")
        if with_ext == 0:
            print(f"  WARN: no extension scores on {short}; run scripts/27 first")
        by_model[short] = rows

    out_dir = Path(__file__).resolve().parent.parent / "figures" / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_quadrant_means(by_model, out_dir / "fig_v3_extension_quadrant_means.png")
    fig_hn_dominance_split(by_model, out_dir / "fig_v3_extension_hn_dominance_split.png")
    fig_probe_correlations(by_model, out_dir / "fig_v3_extension_probe_correlations.png")

    # Drop the now-stale dominance scatter PNG if it lingers from a
    # pre-2026-05-01 run — its rule-3a-broken interpretation no longer
    # applies, and keeping a stale figure on disk invites mis-reading.
    stale = out_dir / "fig_v3_extension_dominance_scatter.png"
    if stale.exists():
        stale.unlink()
        print(f"  removed stale {stale}")


if __name__ == "__main__":
    main()
