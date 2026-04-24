# Cross-Pilot, Valence-Replication, and Prompt-Matrix Analyses — Plan

> **Implementation note:** This project has no test suite (CLAUDE.md:
> "Not a library. No public API, no pypi release, no tests."). The
> writing-plans skill's TDD structure is adapted accordingly — each task
> is: design → implement → run → eyeball output → commit. Verification
> is via `print()` dimensions/values in the driver scripts and visual
> inspection of the generated figures.

**Goal:** Three free analyses on existing JSONL data that answer
open questions left by v1/v2/v3 without running new generations.

**Scope:** No new model calls. Reads `data/pilot_raw.jsonl` (900 rows,
v1+v2, 6 arms) and `data/emotional_raw.jsonl` (640 rows, v3 unsteered,
4 Russell quadrants). Produces figures in `figures/` and summary TSVs
in `data/`.

**Architecture:** One new module (`cross_pilot_analysis.py`) for the
pooled analysis that needs both datasets. Two new functions appended
to the existing `emotional_analysis.py` for the v3-only analyses
(#3 and #5), matching the existing module's conventions. Three new
driver scripts numbered `10_`, `11_`, `12_` continuing the existing
series.

**Tech Stack:** pandas, numpy, scipy.cluster.hierarchy,
sklearn.metrics.pairwise, matplotlib — already in use project-wide.

---

## File Structure

- Create: `llmoji/cross_pilot_analysis.py` — loader that unions v1/v2
  and v3 JSONL into a single DataFrame with a `source` column;
  per-(kaomoji, source) mean aggregation; pooled cosine heatmap.
- Modify: `llmoji/emotional_analysis.py` — add
  `plot_probe_correlation_matrix()` and `plot_prompt_kaomoji_matrix()`.
- Create: `scripts/10_cross_pilot_clustering.py` — driver for #1.
- Create: `scripts/11_emotional_probe_correlations.py` — driver for #3.
- Create: `scripts/12_emotional_prompt_matrix.py` — driver for #5.
- Modify: `CLAUDE.md` — append "Cross-pilot analyses" section under
  Status once all three are complete (done separately, not in this
  plan's scope).

## Key Design Decisions

### #1 uses `probe_scores_t0` for both datasets

Both JSONLs store `probe_scores_t0`. Under `stateless=True` this is
the whole-generation aggregate in both v1/v2 and v3 (CLAUDE.md's
`stateless=True` gotcha). v3 additionally has `probe_scores_tlast`
but it's identical — pooling on `t0` is correct and avoids schema
branching.

### #1 groups by `(first_word, source)`, not just `first_word`

Motivation from our design conversation: if the same kaomoji appears
under v1 `steered_sad` and v3 `LN` (unsteered), do those two rows
land in the same probe-space cluster, or does steering's probe shift
dominate? Per-kaomoji-pooled-across-sources throws this information
away; per-(kaomoji, source) preserves it. Row labels in the heatmap
are `"(｡•́︿•̀｡) | steered_sad"` form so source is visible at a glance.

The `source` values are: `baseline`, `kaomoji_prompted` (v1/v2),
`steered_happy`, `steered_sad`, `steered_angry`, `steered_calm` (v1/v2),
and `v3_HP`, `v3_LP`, `v3_HN`, `v3_LN` (v3 unsteered). 10 values.

### #1 filters `min_count=3` per-(kaomoji, source) tuple

Matches the existing figures' threshold. Expected survivor count:
30–60 rows (readable heatmap).

### #3 runs correlations on v3 only

v3 is the unsteered naturalistic regime; v1/v2 steered arms shift
the probe distributions artificially and would confound the test.
Pearson + Spearman on (happy.sad, angry.calm) pairs, reported:
(a) all 640 rows, (b) per quadrant, (c) full 5×5 correlation matrix
as a heatmap. Pre-registered interpretation threshold from my earlier
framing: |ρ| > 0.7 means v2's valence-collapse claim replicates on
naturalistic data; |ρ| < 0.4 means it was a steering artifact.

### #5 builds a (prompt × kaomoji) count matrix

80 prompts × top-K kaomoji (K = 12 to keep the heatmap legible).
Rows ordered by quadrant with visible dividers. Cells are emission
counts out of 8 seeds. The question: within a quadrant (e.g. LN's
20 prompts), do prompt-level emission profiles differ, or do all LN
prompts pull the same kaomoji distribution? Matrix heterogeneity
within a quadrant block = yes.

---

## Task 1: Cross-pilot pooled clustering module

**Files:**
- Create: `llmoji/cross_pilot_analysis.py`

- [ ] **Step 1: Write `load_pooled_rows()`**

Loads both JSONLs, stamps each row with a `source` string, selects
common columns, concatenates. Schema-difference handling: `pilot_raw`
lacks `probe_scores_tlast`; we use `probe_scores_t0` only, so we drop
`probe_scores_tlast` from the v3 side.

```python
# llmoji/cross_pilot_analysis.py
"""Pooled analysis across v1/v2 (pilot_raw.jsonl) and v3
(emotional_raw.jsonl). Both datasets store `probe_scores_t0` as the
whole-generation aggregate (under stateless=True — see CLAUDE.md
gotcha), so pooling on that column is apples-to-apples.

Grouping: per-(first_word, source) tuples. 'source' distinguishes
v1/v2 condition arms from v3 quadrants, so we can read whether the
same kaomoji carries the same probe signature across steered and
naturalistic regimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PROBES

# v3 prompt_id prefixes → pooled source name.
_V3_QUADRANT_SOURCE = {"HP": "v3_HP", "LP": "v3_LP", "HN": "v3_HN", "LN": "v3_LN"}

# Reuse the kaomoji-start-char filter from emotional_analysis.
KAOMOJI_START_CHARS = set("([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ")


def load_pooled_rows(v1_v2_path: str, v3_path: str) -> pd.DataFrame:
    """Union v1/v2 and v3 JSONL with a 'source' column identifying arm
    (v1/v2) or quadrant (v3). Explodes probe_scores_t0 into per-probe
    columns. Drops rows whose first_word doesn't look like a kaomoji."""
    v12: pd.DataFrame = pd.read_json(v1_v2_path, lines=True)
    v12 = v12.assign(source=v12["condition"])

    v3: pd.DataFrame = pd.read_json(v3_path, lines=True)
    quad = v3["prompt_id"].str[:2].str.upper()
    v3 = v3.assign(source=quad.map(_V3_QUADRANT_SOURCE))
    # v3 has probe_scores_tlast; we don't use it for pooling.
    if "probe_scores_tlast" in v3.columns:
        v3 = v3.drop(columns=["probe_scores_tlast"])

    common = [
        "source", "prompt_id", "seed", "prompt_text",
        "text", "first_word", "kaomoji", "kaomoji_label",
        "probe_scores_t0",
    ]
    df = pd.concat([v12[common], v3[common]], ignore_index=True)

    # Explode probe_scores_t0 into per-probe columns.
    stacked = np.asarray(df["probe_scores_t0"].tolist(), dtype=float)
    for i, probe in enumerate(PROBES):
        df[f"t0_{probe}"] = stacked[:, i]
    df = df.drop(columns=["probe_scores_t0"])

    # Filter to kaomoji-bearing rows (same logic as emotional_analysis).
    df = df[df["first_word"].astype(str).str.len() > 0]
    df = df[df["first_word"].astype(str).str[0].isin(KAOMOJI_START_CHARS)]
    return df.reset_index(drop=True)


def grouped_kaomoji_source_means(
    df: pd.DataFrame, *, min_count: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Group by (first_word, source), require n >= min_count, return
    (mean-probe-vector DataFrame, count Series). Index is a MultiIndex
    of (first_word, source)."""
    cols = [f"t0_{p}" for p in PROBES]
    grouped = df.groupby(["first_word", "source"])[cols].mean()
    counts = df.groupby(["first_word", "source"]).size()
    keep = counts[counts >= min_count].index
    grouped = grouped.loc[grouped.index.isin(keep)]
    counts = counts.loc[grouped.index]
    return grouped, counts
```

- [ ] **Step 2: Write `plot_pooled_cosine_heatmap()`**

Adapts `emotional_analysis.plot_kaomoji_cosine_heatmap` to the
(kaomoji, source) index. Row labels encode both fields. Color coding
on the labels by source (one color per source), separate from the
taxonomy-pole colors used in the v3 figures.

```python
# source-color palette. Distinct hues per source; grouped visually by
# family (baselines neutral, happy-pole warm, sad-pole cool, angry
# reds, calm greens, v3 quadrants mid-saturation).
SOURCE_COLORS = {
    "baseline":         "#888888",
    "kaomoji_prompted": "#444444",
    "steered_happy":    "#e08a1f",
    "steered_sad":      "#1f5fa8",
    "steered_angry":    "#b93128",
    "steered_calm":     "#2f8860",
    "v3_HP":            "#e6b260",
    "v3_LP":            "#b28c3d",
    "v3_HN":            "#d06c5a",
    "v3_LN":            "#5f7ca8",
}


def plot_pooled_cosine_heatmap(
    df: pd.DataFrame, out_path: str, *, min_count: int = 3,
) -> None:
    """Per-(kaomoji, source) mean probe-vector cosine similarity, with
    hierarchical-clustering row order. Row tick labels are colored by
    source."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    from .emotional_analysis import _use_cjk_font

    _use_cjk_font()

    grouped, counts = grouped_kaomoji_source_means(df, min_count=min_count)
    if len(grouped) < 3:
        print(f"  [pooled heatmap] only {len(grouped)} (kaomoji, source) with n≥{min_count}; skipping")
        return

    M = grouped.to_numpy()
    sim = cosine_similarity(M)
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    ordered_sim = sim[np.ix_(order, order)]

    idx = grouped.index.to_list()
    ordered_idx = [idx[i] for i in order]
    ordered_counts = [int(counts.loc[k]) for k in ordered_idx]
    labels = [f"{km}  [{src}]  n={c}"
              for (km, src), c in zip(ordered_idx, ordered_counts)]
    row_colors = [SOURCE_COLORS.get(src, "#666") for km, src in ordered_idx]

    n = len(ordered_idx)
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * n + 5), max(9, 0.28 * n + 4)))
    im = ax.imshow(ordered_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{km}" for km, src in ordered_idx],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), row_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.set_title(
        f"Pooled per-(kaomoji, source) probe-vector cosine similarity\n"
        f"(n ≥ {min_count}; {n} rows; v1/v2 + v3)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cb.ax.tick_params(labelsize=8)

    legend_handles = [Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(1.15, 0.0), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Write `pooled_summary_table()`**

Per-(kaomoji, source) summary TSV. One row per surviving tuple:
first_word, source, n, mean probe vector, row taxonomy label.

```python
def pooled_summary_table(
    df: pd.DataFrame, *, min_count: int = 3,
) -> pd.DataFrame:
    """Per-(kaomoji, source) summary: n, taxonomy label, and the
    5-probe mean vector."""
    from .taxonomy import TAXONOMY
    cols = [f"t0_{p}" for p in PROBES]
    rows: list[dict[str, Any]] = []
    for (km, src), g in df.groupby(["first_word", "source"]):
        if len(g) < min_count:
            continue
        means = g[cols].mean().to_dict()
        rows.append({
            "first_word": km,
            "source": src,
            "n": int(len(g)),
            "taxonomy_label": int(TAXONOMY.get(str(km), 0)),
            **{p: float(means[f"t0_{p}"]) for p in PROBES},
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["first_word", "source"]).reset_index(drop=True)
    return out
```

- [ ] **Step 4: Write driver `scripts/10_cross_pilot_clustering.py`**

```python
"""Cross-pilot pooled clustering driver. Reads pilot_raw.jsonl (v1/v2)
and emotional_raw.jsonl (v3), pools per-(kaomoji, source) aggregate
probe vectors, writes figures/fig_pool_cosine.png and
data/pool_summary.tsv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    FIGURES_DIR,
    PILOT_RAW_PATH,
)
from llmoji.cross_pilot_analysis import (
    grouped_kaomoji_source_means,
    load_pooled_rows,
    plot_pooled_cosine_heatmap,
    pooled_summary_table,
)


def main() -> None:
    if not PILOT_RAW_PATH.exists():
        print(f"no data at {PILOT_RAW_PATH}; run scripts/01_pilot_run.py first")
        return
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_pooled_rows(str(PILOT_RAW_PATH), str(EMOTIONAL_DATA_PATH))
    print(f"pooled {len(df)} kaomoji-bearing rows across v1/v2 + v3")
    print("rows per source:")
    print(df["source"].value_counts().to_string())

    grouped, counts = grouped_kaomoji_source_means(df, min_count=3)
    print(f"\n{len(grouped)} (kaomoji, source) tuples survive n≥3 filter")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = FIGURES_DIR / "fig_pool_cosine.png"
    plot_pooled_cosine_heatmap(df, str(fig_path))
    print(f"\nwrote {fig_path}")

    summary = pooled_summary_table(df, min_count=3)
    summary_path = DATA_DIR / "pool_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"wrote {summary_path} ({len(summary)} rows)")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run and eyeball**

```bash
python scripts/10_cross_pilot_clustering.py
```

Expected output:
- `pooled ~1200 kaomoji-bearing rows across v1/v2 + v3` (900 total v1/v2
  minus baseline no-kaomoji rows, plus 486 = 145+114+159+68 v3 kaomoji
  rows per CLAUDE.md's v3 emission counts — roughly 1100-1300 survivors).
- `~30–60 (kaomoji, source) tuples survive n≥3 filter`.
- Figure shows clustered cosine heatmap. Look for:
  - Whether `v3_HN (╯°□°)` rows cluster near `steered_angry` rows.
  - Whether `v3_LN (｡•́︿•̀｡)` clusters near `steered_sad (｡•́︿•̀｡)`.
  - Whether any (kaomoji, source) pair splits across clusters (signals
    steering shifts the probe signature of the kaomoji).

- [ ] **Step 6: Commit**

```bash
git add llmoji/cross_pilot_analysis.py scripts/10_cross_pilot_clustering.py
git commit -m "analysis: cross-pilot pooled (kaomoji, source) clustering"
```

---

## Task 2: v3 probe-correlation (valence-collapse replication)

**Files:**
- Modify: `llmoji/emotional_analysis.py` — append
  `plot_probe_correlation_matrix()` + small helper.
- Create: `scripts/11_emotional_probe_correlations.py`

- [ ] **Step 1: Add `compute_probe_correlations()` to `emotional_analysis.py`**

Returns a dict of overall + per-quadrant (Pearson, Spearman) on all
5-choose-2 = 10 probe pairs, for eye-check before the heatmap.

```python
# append to llmoji/emotional_analysis.py

def compute_probe_correlations(
    df: pd.DataFrame, *, timestep: str = "t0",
) -> dict[str, Any]:
    """Full pairwise Pearson + Spearman correlation between probe
    scores at the given timestep. Returns overall + per-quadrant.
    Run on v3 unsteered data only — steered data would shift the
    probe distributions and confound the collapse reading.
    """
    from scipy.stats import pearsonr, spearmanr
    from .config import PROBES

    cols = _probe_cols(timestep)
    out: dict[str, Any] = {"probes": list(PROBES), "by_subset": {}}

    def pair_stats(sub: pd.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n < 3:
            return {"n": n, "pearson": None, "spearman": None}
        vals = sub[cols].to_numpy()
        p = np.full((len(PROBES), len(PROBES)), np.nan)
        s = np.full((len(PROBES), len(PROBES)), np.nan)
        for i in range(len(PROBES)):
            for j in range(len(PROBES)):
                if i == j:
                    p[i, j] = 1.0
                    s[i, j] = 1.0
                else:
                    p[i, j] = float(pearsonr(vals[:, i], vals[:, j])[0])
                    s[i, j] = float(spearmanr(vals[:, i], vals[:, j])[0])
        return {"n": int(n), "pearson": p.tolist(), "spearman": s.tolist()}

    out["by_subset"]["all"] = pair_stats(df)
    for q in ("HP", "LP", "HN", "LN"):
        out["by_subset"][q] = pair_stats(df[df["quadrant"] == q])
    return out
```

- [ ] **Step 2: Add `plot_probe_correlation_matrix()` to `emotional_analysis.py`**

```python
def plot_probe_correlation_matrix(
    df: pd.DataFrame, out_path: str, *,
    method: str = "pearson", timestep: str = "t0",
) -> None:
    """Plot a 5-panel figure: overall probe correlation matrix + one
    per Russell quadrant. method='pearson' or 'spearman'. Uses t0
    (== whole-generation aggregate under stateless=True) — this is
    the same column v2's valence-collapse claim was derived from."""
    import matplotlib.pyplot as plt
    from .config import PROBES

    _use_cjk_font()
    stats = compute_probe_correlations(df, timestep=timestep)
    panels = [("all", "all v3 rows"), ("HP", "HP"), ("LP", "LP"),
              ("HN", "HN"), ("LN", "LN")]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, (key, title) in zip(axes, panels):
        sub = stats["by_subset"][key]
        mat = sub.get(method)
        if mat is None:
            ax.text(0.5, 0.5, f"n={sub['n']}\n(too few)", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{title}  n={sub['n']}")
            continue
        arr = np.asarray(mat)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(PROBES)))
        ax.set_yticks(range(len(PROBES)))
        ax.set_xticklabels(PROBES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(PROBES, fontsize=7)
        for i in range(len(PROBES)):
            for j in range(len(PROBES)):
                ax.text(j, i, f"{arr[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(arr[i, j]) > 0.5 else "#333")
        ax.set_title(f"{title}  n={sub['n']}")
    fig.suptitle(f"v3 probe-probe {method} correlations "
                 f"(t0 = whole-generation aggregate under stateless)",
                 fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, label=f"{method} r")
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Write driver `scripts/11_emotional_probe_correlations.py`**

```python
"""v3 probe-correlation analysis. Replicates v2's valence-collapse
claim on naturalistic unsteered data: does happy.sad × angry.calm
correlate as strongly in v3 as v2 said it should?

Pre-registered reading:
  |ρ(happy.sad, angry.calm)| > 0.7 on all 640 rows → v2 replicates;
  |ρ| < 0.4 → v2's collapse was a steering artifact, naturalistic
             data has richer structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import DATA_DIR, EMOTIONAL_DATA_PATH, FIGURES_DIR, PROBES
from llmoji.emotional_analysis import (
    compute_probe_correlations,
    load_rows,
    plot_probe_correlation_matrix,
)


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} v3 rows")

    stats = compute_probe_correlations(df, timestep="t0")
    i_hs = PROBES.index("happy.sad")
    i_ac = PROBES.index("angry.calm")

    print("\nhappy.sad × angry.calm correlation (critical pair):")
    for key in ("all", "HP", "LP", "HN", "LN"):
        sub = stats["by_subset"][key]
        n = sub["n"]
        if sub["pearson"] is None:
            print(f"  {key}: n={n}  (too few)")
            continue
        r = sub["pearson"][i_hs][i_ac]
        rho = sub["spearman"][i_hs][i_ac]
        print(f"  {key}: n={n}  pearson r = {r:+.3f}  spearman ρ = {rho:+.3f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_p = FIGURES_DIR / "fig_v3_corr_pearson.png"
    fig_s = FIGURES_DIR / "fig_v3_corr_spearman.png"
    plot_probe_correlation_matrix(df, str(fig_p), method="pearson")
    print(f"\nwrote {fig_p}")
    plot_probe_correlation_matrix(df, str(fig_s), method="spearman")
    print(f"wrote {fig_s}")

    stats_path = DATA_DIR / "v3_probe_correlations.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run and eyeball**

```bash
python scripts/11_emotional_probe_correlations.py
```

Expected:
- Console prints Pearson r and Spearman ρ for happy.sad × angry.calm,
  overall and per quadrant.
- Two heatmaps (one Pearson, one Spearman), 5 panels each.
- Full stats JSON at `data/v3_probe_correlations.json`.
- Pre-registered verdict printed or eyeballed: `|ρ_all| > 0.7` PASS
  v2 replicates; `< 0.4` FAIL v2 was a steering artifact.

- [ ] **Step 5: Commit**

```bash
git add llmoji/emotional_analysis.py scripts/11_emotional_probe_correlations.py
git commit -m "analysis: v3 probe-correlation replication of v2 valence-collapse"
```

---

## Task 3: v3 prompt × kaomoji emission matrix

**Files:**
- Modify: `llmoji/emotional_analysis.py` — append
  `plot_prompt_kaomoji_matrix()` + `prompt_kaomoji_matrix()`.
- Create: `scripts/12_emotional_prompt_matrix.py`

- [ ] **Step 1: Add `prompt_kaomoji_matrix()` to `emotional_analysis.py`**

Builds the (prompt × top-K kaomoji) count matrix. Returns a DataFrame
with prompts in quadrant-sorted order.

```python
# append to llmoji/emotional_analysis.py

def prompt_kaomoji_matrix(
    df: pd.DataFrame, *, top_k: int = 12, min_prompt_emissions: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(80-prompt × top-K kaomoji) emission-count matrix. Rows are
    ordered by quadrant (HP, LP, HN, LN) then by prompt_id within
    quadrant. Returns (matrix, row_meta) where row_meta has prompt_id,
    quadrant, prompt_text, total_emissions."""
    sub = _kaomoji_rows(df)
    if len(sub) == 0:
        return pd.DataFrame(), pd.DataFrame()

    top = sub["first_word"].value_counts().head(top_k).index.tolist()
    prompts = df[["prompt_id", "quadrant", "prompt_text"]].drop_duplicates("prompt_id")
    q_order = {"HP": 0, "LP": 1, "HN": 2, "LN": 3}
    prompts = prompts.assign(_qord=prompts["quadrant"].map(q_order))
    prompts = prompts.sort_values(["_qord", "prompt_id"]).drop(columns=["_qord"])

    mat = pd.DataFrame(
        0, index=prompts["prompt_id"].tolist(), columns=top, dtype=int,
    )
    for pid, group in sub.groupby("prompt_id"):
        if pid not in mat.index:
            continue
        counts = group["first_word"].value_counts()
        for km in top:
            mat.at[pid, km] = int(counts.get(km, 0))

    prompts = prompts.assign(
        total_emissions=[int(mat.loc[pid].sum()) for pid in prompts["prompt_id"]]
    )
    if min_prompt_emissions > 0:
        keep = prompts[prompts["total_emissions"] >= min_prompt_emissions]["prompt_id"]
        mat = mat.loc[keep]
        prompts = prompts[prompts["prompt_id"].isin(keep)]
    return mat, prompts.reset_index(drop=True)


def plot_prompt_kaomoji_matrix(
    df: pd.DataFrame, out_path: str, *, top_k: int = 12,
) -> None:
    """Heatmap of prompt-level emission counts, rows grouped by
    quadrant with horizontal dividers between quadrants."""
    import matplotlib.pyplot as plt

    _use_cjk_font()

    mat, meta = prompt_kaomoji_matrix(df, top_k=top_k)
    if mat.empty:
        print("  [prompt matrix] no rows; skipping")
        return

    quad_colors = {"HP": "#e6b260", "LP": "#b28c3d",
                   "HN": "#d06c5a", "LN": "#5f7ca8"}
    row_colors = [quad_colors[q] for q in meta["quadrant"]]

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(mat.columns) + 4),
                                    max(8, 0.18 * len(mat) + 3)))
    im = ax.imshow(mat.to_numpy(), cmap="magma", aspect="auto",
                   vmin=0, vmax=8)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    y_labels = [f"{pid}  [{q}]  {txt[:40]}"
                for pid, q, txt in zip(meta["prompt_id"],
                                       meta["quadrant"],
                                       meta["prompt_text"])]
    ax.set_yticklabels(y_labels, fontsize=6)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    # Quadrant divider lines.
    prev_q = None
    for i, q in enumerate(meta["quadrant"]):
        if prev_q is not None and q != prev_q:
            ax.axhline(i - 0.5, color="#333", linewidth=1)
        prev_q = q

    for i in range(len(mat)):
        for j in range(len(mat.columns)):
            v = int(mat.iat[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v >= 4 else "#333", fontsize=6)

    ax.set_title(
        f"v3 prompt × kaomoji emission counts "
        f"(top-{top_k} kaomoji; rows by quadrant)"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label="count out of 8 seeds")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Write driver `scripts/12_emotional_prompt_matrix.py`**

```python
"""v3 prompt × kaomoji emission matrix. Row = prompt (80 of them,
grouped by Russell quadrant); column = top-K kaomoji; cell = emission
count out of 8 seeds. Surfaces within-quadrant variation that the
per-kaomoji summary averages over."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.config import DATA_DIR, EMOTIONAL_DATA_PATH, FIGURES_DIR
from llmoji.emotional_analysis import (
    load_rows,
    plot_prompt_kaomoji_matrix,
    prompt_kaomoji_matrix,
)


def main() -> None:
    if not EMOTIONAL_DATA_PATH.exists():
        print(f"no data at {EMOTIONAL_DATA_PATH}; run scripts/03_emotional_run.py first")
        return

    df = load_rows(str(EMOTIONAL_DATA_PATH))
    print(f"loaded {len(df)} v3 rows")

    mat, meta = prompt_kaomoji_matrix(df, top_k=12)
    print(f"built {len(mat)} prompts × {len(mat.columns)} kaomoji matrix")
    print("\nper-quadrant total emissions in top-12 kaomoji:")
    for q in ("HP", "LP", "HN", "LN"):
        q_meta = meta[meta["quadrant"] == q]
        q_mat = mat.loc[q_meta["prompt_id"]]
        print(f"  {q}: prompts={len(q_meta)}  sum={int(q_mat.to_numpy().sum())}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = FIGURES_DIR / "fig_v3_prompt_kaomoji.png"
    plot_prompt_kaomoji_matrix(df, str(fig_path), top_k=12)
    print(f"\nwrote {fig_path}")

    # TSV of the full matrix with prompt text for spreadsheet inspection.
    out = mat.copy()
    out.insert(0, "quadrant", meta.set_index("prompt_id").loc[out.index, "quadrant"])
    out.insert(1, "prompt_text", meta.set_index("prompt_id").loc[out.index, "prompt_text"])
    out.insert(2, "total_emissions",
               meta.set_index("prompt_id").loc[out.index, "total_emissions"])
    tsv_path = DATA_DIR / "v3_prompt_kaomoji_matrix.tsv"
    out.to_csv(tsv_path, sep="\t", index=True, index_label="prompt_id")
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run and eyeball**

```bash
python scripts/12_emotional_prompt_matrix.py
```

Expected:
- `loaded 640 v3 rows`.
- `built 80 prompts × 12 kaomoji matrix` (or close; may drop prompts
  that never emit a kaomoji — LP and HN have a lot of skips, see
  CLAUDE.md quadrant emission rates).
- Heatmap with four quadrant bands (20 prompts each). Within-quadrant
  heterogeneity is visible if cells vary within a band.
- Specifically eyeball: in the LN band, do the "dog died" and "year
  since dad passed" rows (both LN) use different kaomoji?

- [ ] **Step 4: Commit**

```bash
git add llmoji/emotional_analysis.py scripts/12_emotional_prompt_matrix.py
git commit -m "analysis: v3 prompt × kaomoji emission matrix"
```

---

## Self-Review

Spec coverage against my 3-analysis recommendation:
- #1 Cross-pilot pooled probe-space clustering → Task 1 ✓
- #3 v3 replication of v2 valence-collapse → Task 2 ✓
- #5 Prompt-level per-kaomoji matrix → Task 3 ✓

Placeholder scan: no TBDs, no "implement later", no stubs. All
code steps have complete code. Type names consistent across tasks
(`load_pooled_rows`, `grouped_kaomoji_source_means`,
`compute_probe_correlations`, `prompt_kaomoji_matrix` —
all defined where used, called with matching signatures).

Gotchas respected:
- `probe_scores_t0` treated as the whole-generation aggregate (CLAUDE.md
  `stateless=True` gotcha) — no claims framed as "token 0" anywhere.
- Matplotlib font fallback reuses `emotional_analysis._use_cjk_font`;
  no new chain introduced.
- Labels re-extracted in place by `04_emotional_analysis.py`
  before Task 3 runs (if the user hasn't run `04_` recently, they
  should — but Task 3 doesn't strictly require fresh labels; it
  reads `first_word` via the same kaomoji-start filter).

Execution handoff: inline by default — the user is running this
conversation, the scope is small, and they have the venv already.
