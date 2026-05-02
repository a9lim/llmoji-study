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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


# Probes to highlight in the per-quadrant figure. Post-2026-05-03
# 3-probe migration, fearful.unflinching is a CORE probe (lives in
# probe_scores_t0[idx]); the other three live in extension_probe_*
# dicts as before. curious.disinterested no longer auto-bootstraps
# (PROBE_CATEGORIES dropped "register"/"epistemic") so it's out.
HIGHLIGHT_PROBES = [
    "powerful.powerless",      # PAD dominance — headline (extension)
    "fearful.unflinching",     # direct fear probe (now CORE post-migration)
    "disgusted.accepting",     # Plutchik disgust (extension)
    "surprised.unsurprised",   # Plutchik surprise (extension)
]

# Canonical 3-probe set (post-2026-05-03 migration). Mirrors
# `config.PROBES`. Drives `fig_v3_canonical_quadrant_means.png`,
# which is the per-quadrant analogue of the extension-probe figure
# but for the eager-scored canonical probes.
CANONICAL_PROBES = [
    "happy.sad",
    "angry.calm",
    "fearful.unflinching",
]


def _probe_value(r: dict, probe: str, *, agg: str = "t0") -> float | None:
    """Schema-spanning probe lookup. Post-2026-05-03 migration the
    canonical 3 probes (happy.sad / angry.calm / fearful.unflinching)
    live in the list-indexed `probe_scores_t0/_tlast` and the named
    `probe_means` dict. Extension probes (powerful.powerless,
    surprised.unsurprised, disgusted.accepting) live in the
    `extension_probe_scores_t0/_tlast` / `extension_probe_means`
    dicts written by scripts/local/27. Returns None on miss."""
    from llmoji_study.config import PROBES as CORE_PROBES_NEW
    if probe in CORE_PROBES_NEW:
        if agg in ("t0", "tlast"):
            seq = r.get(f"probe_scores_{agg}") or []
            idx = CORE_PROBES_NEW.index(probe)
            if len(seq) > idx:
                return float(seq[idx])
        elif agg == "mean":
            d = r.get("probe_means") or {}
            if probe in d:
                return float(d[probe])
    field = {"t0": "extension_probe_scores_t0",
             "tlast": "extension_probe_scores_tlast",
             "mean": "extension_probe_means"}.get(agg)
    if field:
        d = r.get(field) or {}
        if probe in d:
            return float(d[probe])
    return None

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


def _plot_quadrant_means(
    by_model: dict[str, list[dict]],
    probes: list[str],
    *,
    title: str,
    out: Path,
) -> None:
    """Per-probe per-quadrant mean probe score, **NB-subtracted** —
    each probe's mean over this experiment's NB rows is subtracted
    from every quadrant. Drops the saklas-bundled-neutrals baseline
    that's baked into raw probe scores (per `monitor.py` layer-mean
    centering) in favor of the project's own NB quadrant. NB bars
    are zero by construction; HP/LP/HN-D/HN-S/LN bars read as the
    affect lift over a domain-matched neutral observation."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    for ax, short in zip(axes, MODELS):
        rows = by_model[short]
        n_probes = len(probes)
        n_quads = len(QUADRANT_ORDER)
        bar_w = 0.16
        x = np.arange(n_probes)
        # Per-probe NB baseline (this experiment's NB rows, t0).
        nb_baseline: dict[str, float] = {}
        for probe in probes:
            nb_vals = []
            for r in rows:
                if "error" in r or _quad(r["prompt_id"]) != "NB":
                    continue
                v = _probe_value(r, probe, agg="t0")
                if v is not None:
                    nb_vals.append(v)
            nb_baseline[probe] = float(np.mean(nb_vals)) if nb_vals else 0.0
        for qi, q in enumerate(QUADRANT_ORDER):
            scores: list[float] = []
            for probe in probes:
                vals: list[float] = []
                for r in rows:
                    if "error" in r or _quad(r["prompt_id"]) != q:
                        continue
                    v = _probe_value(r, probe, agg="t0")
                    if v is not None:
                        vals.append(v)
                raw = float(np.mean(vals)) if vals else 0.0
                scores.append(raw - nb_baseline[probe])
            offset = (qi - (n_quads - 1) / 2) * bar_w
            ax.bar(
                x + offset, scores, width=bar_w,
                color=QUADRANT_COLORS[q], edgecolor="black", linewidth=0.4,
                label=q if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [p.split(".")[0] for p in probes],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_title(f"{short}", fontsize=12)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("mean probe score at h_first  (NB-subtracted)")
    axes[0].legend(
        title="quadrant", loc="upper left", fontsize=8, ncols=5,
        frameon=False,
    )
    fig.suptitle(
        f"{title}\n(NB-subtracted — each probe's mean over NB rows is set as zero)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_quadrant_means(by_model: dict[str, list[dict]], out: Path) -> None:
    _plot_quadrant_means(
        by_model, HIGHLIGHT_PROBES,
        title="v3 extension probes: per-quadrant mean at h_first",
        out=out,
    )


def fig_canonical_quadrant_means(by_model: dict[str, list[dict]], out: Path) -> None:
    _plot_quadrant_means(
        by_model, CANONICAL_PROBES,
        title="v3 canonical probes: per-quadrant mean at h_first",
        out=out,
    )


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


# Core probes are now read dynamically from config.PROBES so the
# correlation matrix tracks the actual eager-probe set rather than
# a hardcoded 5-probe v1/v2 list.


def _build_probe_matrix(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """Build (probe_names, n_rows × n_probes matrix). Pulls each
    probe via the schema-spanning ``_probe_value`` so the same code
    handles pre-2026-05-03 (5-core + 12-extension) and post-migration
    (3-core + 3-extension) data shapes."""
    from llmoji_study.config import PROBES as CORE_PROBES_NEW
    if not rows:
        return [], np.zeros((0, 0))

    # Discover extension probes present on any row.
    ext_keys: set[str] = set()
    for r in rows:
        d = r.get("extension_probe_scores_t0") or {}
        if isinstance(d, dict):
            ext_keys.update(d.keys())
    # Drop overlap with core (e.g. fearful.unflinching used to live in
    # extension on legacy data and now lives in core).
    ext_names = sorted(k for k in ext_keys if k not in set(CORE_PROBES_NEW))
    names = list(CORE_PROBES_NEW) + ext_names

    mat: list[list[float]] = []
    for r in rows:
        if "error" in r:
            continue
        row_vals: list[float] = []
        ok = True
        for p in names:
            v = _probe_value(r, p, agg="mean")
            if v is None:
                # mean-aggregate may be missing for legacy rows that
                # only have t0/tlast. Fall back to t0.
                v = _probe_value(r, p, agg="t0")
            if v is None:
                ok = False
                break
            row_vals.append(v)
        if ok:
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
        from llmoji_study.config import PROBES as _CORE_NEW
        sep = len(_CORE_NEW) - 0.5
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

    out_dir = Path(__file__).resolve().parent.parent.parent / "figures" / "local" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_quadrant_means(by_model, out_dir / "fig_v3_extension_quadrant_means.png")
    fig_canonical_quadrant_means(by_model, out_dir / "fig_v3_canonical_quadrant_means.png")
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
