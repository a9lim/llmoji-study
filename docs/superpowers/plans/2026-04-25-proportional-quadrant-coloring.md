# Proportional quadrant coloring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace dominant-quadrant face coloring (winner-take-all categorical) with proportional RGB-mix coloring (continuous blend) on the four face-level v3 figures (`plot_kaomoji_quadrant_alignment` in script 04; `plot_face_pca_by_quadrant`, `plot_face_probe_scatter`, `plot_face_cosine_heatmap` in script 17). Faces that emit across multiple quadrants render as visible mixes (e.g. 21 LN + 20 HN → muted purple), making cross-quadrant emitters legible without needing to inspect the summary TSV.

**Architecture:** Two pieces of new behavior in `llmoji/emotional_analysis.py`: (1) `per_face_quadrant_weights(df)` returning a dict-of-dicts of normalized per-face quadrant weights, and (2) `mix_quadrant_color(weights)` returning an RGB tuple computed as the weighted linear combination of `QUADRANT_COLORS`. The four affected plot functions swap their `[QUADRANT_COLORS[dominant_q]]` list comprehension for `[mix_quadrant_color(weights[fw]) for fw in ...]`. Simultaneously the global `QUADRANT_COLORS` palette gets re-tuned for canonical Russell mapping (HN red, HP gold, LP green, LN blue, NB gray) at perceived-luminance ~L*55–62 across all five so mixes don't drift in brightness; the local override in script 13 gets deleted (its colors were already canonical-Russell-flavored, but unifying removes the maintenance gotcha).

**Tech Stack:** numpy, matplotlib. No new deps.

**Pre-registration (binding per CLAUDE.md ethics — figure refresh, not new experiment):**

- Locked palette (single source of truth, replaces both `QUADRANT_COLORS` and script 13's local override):
  - HN: `#d44a4a` (red — anger/anxiety; high arousal, negative valence)
  - HP: `#d49b3a` (gold — excitement/joy; high arousal, positive valence)
  - LP: `#4aa66a` (green — calm/contentment; low arousal, positive valence)
  - LN: `#4a7ed4` (blue — sadness/depression; low arousal, negative valence)
  - NB: `#909090` (gray — neutral baseline)
- Mixing math: `weights[q] = count_in_quadrant / total_emissions`; `mix_color = sum_q (weights[q] * RGB_q)` in plain RGB linear (no perceptual color-space conversion). RGB values in `[0, 1]` floats; matplotlib accepts as `(r, g, b)` tuple.
- Apply to four face-level figures. Out of scope: script 13's row-level PCA (cells are 100%-quadrant-pure by construction); v1/v2 figures (different scheme); claude-faces / eriskii (separate plan, eriskii data is already stale per the prior canonicalization plan).
- `per_face_dominant_quadrant` stays — still used by the `summary_table` `dominant_quadrant` column and by 17's stdout "faces by dominant quadrant" tally.
- Welfare: zero new generations; pure post-processing.

**Out of scope (separate plans if pursued):**

- Perceptual color-space mixing (OkLab/CIELAB). RGB linear is sufficient for 5-color simplex blends with this palette.
- Continuous colormap legends (e.g. a 2D affect-circumplex colorbar). Keep the existing 5-patch legend with a small caption note.
- Updating the v1/v2 `pole_color` scheme in `plot_kaomoji_quadrant_alignment` — that's the X-axis fallback when no quadrant data exists, untouched here.

---

## File Structure

**Modified:**

- `llmoji/emotional_analysis.py`:
  - Replace `QUADRANT_COLORS` palette (lines 41–47) with the canonical-Russell-tuned values above.
  - Add `per_face_quadrant_weights(df)` helper (after `per_face_dominant_quadrant`, ~line 62).
  - Add `mix_quadrant_color(weights)` helper (right after).
  - Modify `plot_kaomoji_quadrant_alignment` (lines 430–544 area): swap row-color computation from `pole_color[TAXONOMY.get(k, 0)]` to mixed quadrant color; update the row-tint commentary in the docstring/comments.
- `llmoji/emotional_analysis.py:611–615`: delete the `quadrant_color` local override block in the v3 PCA function (script 13's plot function), reach for the global `QUADRANT_COLORS` instead. Update the corresponding `quadrant_color.get(...)` calls (~lines 620, 642, 666) to `QUADRANT_COLORS.get(...)`.
- `scripts/17_v3_face_scatters.py`:
  - Add `mix_quadrant_color`, `per_face_quadrant_weights` to the imports from `llmoji.emotional_analysis` (~line 35).
  - In `plot_face_pca_by_quadrant`, `plot_face_probe_scatter`, and `plot_face_cosine_heatmap`: replace `colors = [QUADRANT_COLORS[quadrant.get(fw, "NB")] for fw in fdf["first_word"]]` with `colors = [mix_quadrant_color(weights[fw]) for fw in fdf["first_word"]]` (where `weights = per_face_quadrant_weights(df)`).
  - In `plot_face_cosine_heatmap`, swap the y/x tick coloring loop from `QUADRANT_COLORS[q]` to `mix_quadrant_color(weights[fw])`.
  - Augment the legend caption ("colored by dominant quadrant") to reflect the new "blended proportionally per-face emission distribution" semantics.
- `CLAUDE.md`: update the v3 gemma + qwen findings sections to note the figure-coloring refresh; update the "Kaomoji canonicalization" → "JSONL keeps raw" closer to mention figures use mixed colors.

**Re-generated (figure refresh, content unchanged):**

- `figures/fig_emo_c_kaomoji_quadrant.png`, `figures/fig_v3_pca_valence_arousal.png`, `figures/fig_v3_face_pca_by_quadrant.png`, `figures/fig_v3_face_probe_scatter.png`, `figures/fig_v3_face_cosine_heatmap.png` (gemma — 5 figures).
- `figures/qwen/fig_emo_c_kaomoji_quadrant.png`, `figures/qwen/fig_v3_pca_valence_arousal.png`, `figures/qwen/fig_v3_face_pca_by_quadrant.png`, `figures/qwen/fig_v3_face_probe_scatter.png`, `figures/qwen/fig_v3_face_cosine_heatmap.png` (qwen — 5 figures).

Note: figures A/B (`fig_emo_a_kaomoji_sim.png`, `fig_emo_b_kaomoji_consistency.png`) don't use quadrant colors — A/B unchanged. Script 13's PCA scatter (`fig_v3_pca_valence_arousal.png`) regenerates because its legend will pull from the new global `QUADRANT_COLORS` (visually nearly identical — both old and new schemes were already canonical-Russell-flavored — but worth re-running to keep figures and code in sync).

**Unchanged:**

- All v1/v2 figures (`pole_color` scheme is taxonomy-pole, not quadrant; not touched).
- All claude-faces / eriskii outputs (already stale per the prior plan; no quadrant info applicable).
- `data/emotional_summary.tsv`, `data/qwen_emotional_summary.tsv` (no schema change; `dominant_quadrant` column still populated by `per_face_dominant_quadrant`).

---

### Task 1: Update palette + add mixing helpers in `emotional_analysis.py`

**Files:**
- Modify: `llmoji/emotional_analysis.py:41-47` (palette), `:62` insert helpers, `:611-668` (delete script-13 local override).

- [ ] **Step 1: Replace `QUADRANT_COLORS` with the canonical-Russell-tuned palette**

In `llmoji/emotional_analysis.py`, replace lines 41–47:

```python
QUADRANT_COLORS = {
    "HP": "#d62728",  # red — high-arousal positive
    "LP": "#2ca02c",  # green — low-arousal positive
    "HN": "#ff7f0e",  # orange — high-arousal negative
    "LN": "#1f77b4",  # blue — low-arousal negative
    "NB": "#7f7f7f",  # gray — neutral baseline
}
```

with:

```python
# Canonical Russell-circumplex mapping. Saturated enough that 50/50
# RGB-linear mixes between adjacent-quadrant pairs are still
# recognizable (HN+LN → muted purple, HP+LP → olive, etc.); perceived
# luminance balanced ~L*55–62 across all 5 so weighted mixes don't
# drift in brightness when one quadrant dominates by saturation alone.
# Diagonal "contradictory" mixes (HN+LP "anger meets calm",
# HP+LN "excited meets sad") fall to brown / desaturated gray —
# informatively unusual.
QUADRANT_COLORS = {
    "HP": "#d49b3a",  # gold — high arousal, positive (excitement/joy)
    "LP": "#4aa66a",  # green — low arousal, positive (calm/contentment)
    "HN": "#d44a4a",  # red — high arousal, negative (anger/anxiety)
    "LN": "#4a7ed4",  # blue — low arousal, negative (sadness/depression)
    "NB": "#909090",  # gray — neutral baseline
}
```

- [ ] **Step 2: Add `per_face_quadrant_weights` and `mix_quadrant_color` helpers**

Insert these two functions immediately after `per_face_dominant_quadrant` (after line 61, before the `# --- Loaders ---` divider):

```python
def per_face_quadrant_weights(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """For each first_word, return a dict mapping quadrant -> normalized
    emission weight (sum to 1 across the 5 quadrants).

    A face emitted in 21 LN rows + 20 HN rows + 0 elsewhere yields
    ``{"LN": 0.512, "HN": 0.488, "HP": 0, "LP": 0, "NB": 0}``.
    Faces with zero total emissions return all-zero weights (caller
    should guard).
    """
    from collections import Counter
    out: dict[str, dict[str, float]] = {}
    for fw, sub in df.groupby("first_word"):
        counts = Counter(sub["quadrant"].tolist())
        total = sum(counts.values())
        if total == 0:
            out[str(fw)] = {q: 0.0 for q in QUADRANT_ORDER}
            continue
        out[str(fw)] = {
            q: counts.get(q, 0) / total for q in QUADRANT_ORDER
        }
    return out


def mix_quadrant_color(
    weights: dict[str, float],
) -> tuple[float, float, float]:
    """Linear-RGB mix of `QUADRANT_COLORS` weighted by `weights`.

    Weights are expected to sum to 1.0 across the 5 quadrants
    (`per_face_quadrant_weights` produces them). Hex strings in
    `QUADRANT_COLORS` are converted to (r, g, b) floats in [0, 1],
    multiplied by the per-quadrant weight, and summed component-wise.
    Returns a matplotlib-compatible RGB tuple.

    A face that's 100% one quadrant returns that quadrant's pure
    color; a face split 50/50 between two quadrants returns the RGB
    midpoint; a face split evenly across all 5 returns the centroid
    of the 5 base colors (close to mid-gray, which is the visually
    "balanced" outcome).
    """
    from matplotlib.colors import to_rgb
    r = g = b = 0.0
    for q, w in weights.items():
        if w <= 0 or q not in QUADRANT_COLORS:
            continue
        qr, qg, qb = to_rgb(QUADRANT_COLORS[q])
        r += w * qr
        g += w * qg
        b += w * qb
    return (r, g, b)
```

- [ ] **Step 3: Delete the script-13 local color override**

Replace lines 611–615 (the `quadrant_color = {...}` block inside `plot_v3_pca_valence_arousal`):

```python
    quadrant_color = {
        "HP": "#e9a01f", "LP": "#4a8a5a",
        "HN": "#c9372d", "LN": "#3d68a8",
        "NB": "#888888",
    }
```

with a single comment line (no longer needed — function reaches for the global directly via the existing imports at the top of the file):

```python
    # Use the global QUADRANT_COLORS (canonical Russell palette).
```

Then update the three references downstream from `quadrant_color` to use `QUADRANT_COLORS`:

- Line ~620: `c = quadrant_color.get(q, "#666")` → `c = QUADRANT_COLORS.get(q, "#666")`
- Line ~642: `c = quadrant_color.get(q_name, "#666")` → `c = QUADRANT_COLORS.get(q_name, "#666")`
- Line ~666–667: the legend `Patch(color=quadrant_color[k], ...) for k, lbl in legend_labels if k in quadrant_color` → `Patch(color=QUADRANT_COLORS[k], ...) for k, lbl in legend_labels if k in QUADRANT_COLORS`

(Use `replace_all=False` Edits with enough surrounding context to find each unique occurrence.)

- [ ] **Step 4: Smoke-test the new helpers**

Run:

```bash
source .venv/bin/activate && python -c "
from llmoji.emotional_analysis import per_face_quadrant_weights, mix_quadrant_color, QUADRANT_COLORS
import pandas as pd

# Simulated face: 21 LN + 20 HN
df = pd.DataFrame([
    *[{'first_word': '(test)', 'quadrant': 'LN'}] * 21,
    *[{'first_word': '(test)', 'quadrant': 'HN'}] * 20,
])
weights = per_face_quadrant_weights(df)['(test)']
print(f'weights: {weights}')
assert abs(weights['LN'] + weights['HN'] - 1.0) < 1e-9
assert abs(weights['LN'] - 21/41) < 1e-9, weights

color = mix_quadrant_color(weights)
print(f'mixed color (R, G, B): ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})')
print(f'  hex approx: #{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}')
print(f'  HN endpoint:  {QUADRANT_COLORS[\"HN\"]}  (red)')
print(f'  LN endpoint:  {QUADRANT_COLORS[\"LN\"]}  (blue)')

# Pure-LN face check
pure_ln = mix_quadrant_color({'LN': 1.0, 'HP': 0, 'LP': 0, 'HN': 0, 'NB': 0})
from matplotlib.colors import to_rgb
expected = to_rgb(QUADRANT_COLORS['LN'])
assert all(abs(a - b) < 1e-9 for a, b in zip(pure_ln, expected)), (pure_ln, expected)
print('pure-LN passthrough OK')

# Empty-weights face check
empty = mix_quadrant_color({q: 0.0 for q in QUADRANT_COLORS})
assert empty == (0.0, 0.0, 0.0), empty
print('empty-weights returns black (0, 0, 0) — caller should guard')
"
```

Expected: weights print `{'HP': 0.0, 'LP': 0.0, 'HN': 0.487..., 'LN': 0.512..., 'NB': 0.0}`, mixed color hex prints around `#8d6790` (muted purple), pure-LN passthrough OK, empty returns (0,0,0).

If the test fails, fix the implementation before proceeding to Task 2.

- [ ] **Step 5: Smoke-test that the v3 PCA scatter still imports cleanly after the local-override removal**

Run:

```bash
source .venv/bin/activate && python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('s13', 'scripts/13_emotional_pca_valence_arousal.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('script 13 imports ok')

# Also import emotional_analysis to confirm it loads
import llmoji.emotional_analysis as ea
assert ea.QUADRANT_COLORS['HN'] == '#d44a4a', ea.QUADRANT_COLORS
assert ea.QUADRANT_COLORS['LN'] == '#4a7ed4'
print('global QUADRANT_COLORS updated:', ea.QUADRANT_COLORS)
"
```

Expected: prints "script 13 imports ok" + the new palette dict.

- [ ] **Step 6: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "$(cat <<'EOF'
emotional_analysis: canonical-Russell palette + face-mix helpers

Replaces QUADRANT_COLORS with a canonical Russell-circumplex
mapping (HN red, HP gold, LP green, LN blue, NB gray) tuned for
RGB-linear mixing — mid-saturation so 50/50 blends like HN+LN
read as recognizable purple, perceived luminance balanced
~L*55–62 across all five so weighted mixes don't drift in
brightness.

Adds per_face_quadrant_weights() (per-face normalized quadrant
emission counts) and mix_quadrant_color() (RGB-linear mix of
QUADRANT_COLORS by those weights). Deletes the local-override
quadrant_color dict in plot_v3_pca_valence_arousal — the global
is now the single source of truth.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Apply mixed coloring to Fig C (`plot_kaomoji_quadrant_alignment`)

**Files:**
- Modify: `llmoji/emotional_analysis.py:506-520` (row-color computation in Fig C)

- [ ] **Step 1: Replace the TAXONOMY-pole row coloring with mixed quadrant color**

Find this block in `llmoji/emotional_analysis.py` (around line 506):

```python
    pole_color = {+1: "#c25a22", -1: "#2f6c57", 0: "#666"}
    row_colors = [pole_color.get(TAXONOMY.get(k, 0), "#666") for k in kms_ordered]
```

Replace with:

```python
    # Row tint: per-face mixed quadrant color (RGB-linear blend of
    # QUADRANT_COLORS weighted by emission distribution). More
    # informative than the old TAXONOMY-pole 3-state coloring —
    # cross-quadrant emitters render visibly mixed, e.g. a face
    # that's 21 LN + 20 HN reads as purple instead of green/orange.
    weights = per_face_quadrant_weights(df)
    row_colors = [
        mix_quadrant_color(weights.get(k, {q: 0.0 for q in QUADRANT_ORDER}))
        for k in kms_ordered
    ]
```

- [ ] **Step 2: Drop the now-unused `TAXONOMY` import inside the function**

The function currently has a local import `from .taxonomy import TAXONOMY` at the top of `plot_kaomoji_quadrant_alignment` (around line 448). Remove that import line — `TAXONOMY` is no longer referenced inside the function.

If pyflakes / pyright flags this as a removed-but-still-imported issue, also confirm no other use of `TAXONOMY` remains in the function body. (Grep `grep -n "TAXONOMY" llmoji/emotional_analysis.py` and confirm only references outside `plot_kaomoji_quadrant_alignment` remain.)

- [ ] **Step 3: Update the function docstring to reflect the new row tint**

The function's docstring currently says nothing about row coloring (the old behavior was an undocumented detail). Add a single sentence to the docstring after the existing "Centered cosine ..." line:

```python
    """Figure C: for each (kaomoji × quadrant) cell with n >=
    min_per_cell, cosine(cell_mean_hidden, quadrant_aggregate_hidden).
    Centered cosine — both cells and quadrant aggregates centered
    against the same hidden-state pool mean.

    Y-axis row labels are tinted by per-face mixed quadrant color
    (RGB blend of `QUADRANT_COLORS` weighted by per-quadrant
    emission count). Cross-quadrant emitters render as visible
    mixes; pure-quadrant faces stay at their endpoint color.
    """
```

(Replace the existing one-paragraph docstring — show the full new one in your edit so the diff is clean.)

- [ ] **Step 4: Smoke-test that gemma Fig C generates without errors**

Run:

```bash
source .venv/bin/activate && python scripts/04_emotional_analysis.py 2>&1 | grep -E "fig_emo_c|Error|Traceback" | head
```

Expected: a "wrote .../fig_emo_c_kaomoji_quadrant.png" line, no Traceback. Inspect the resulting `figures/fig_emo_c_kaomoji_quadrant.png` visually — y-axis labels should now be tinted in mix colors (most pure-quadrant faces still look like their endpoint; the cross-quadrant emitters like `(｡•́︿•̀｡)` should look blended).

- [ ] **Step 5: Commit**

```bash
git add llmoji/emotional_analysis.py
git commit -m "$(cat <<'EOF'
emotional_analysis: mix Fig C row tints by per-face quadrant blend

Replaces plot_kaomoji_quadrant_alignment's row coloring from
TAXONOMY-pole (3-state +/-/0) to per-face mixed quadrant color.
Cross-quadrant emitters now render visibly as mixes; pure-quadrant
faces stay at endpoint colors. Drops the now-unused local TAXONOMY
import inside the function.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Apply mixed coloring to script 17's three face figures

**Files:**
- Modify: `scripts/17_v3_face_scatters.py:35-41` (imports), `:74-76` (face PCA), `:127-129` (probe scatter), `:188-223` (heatmap row tints + helper).

- [ ] **Step 1: Update the imports**

Replace lines 35–41:

```python
from llmoji.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
    per_face_dominant_quadrant,
)
```

with:

```python
from llmoji.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER,
    _use_cjk_font,
    load_emotional_features,
    mix_quadrant_color,
    per_face_dominant_quadrant,
    per_face_quadrant_weights,
)
```

- [ ] **Step 2: Update `_add_quadrant_legend` to reflect the mixed-color semantics**

Replace lines 44–50:

```python
def _add_quadrant_legend(ax) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   color=QUADRANT_COLORS[q], label=q)
        for q in QUADRANT_ORDER
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9, title="dominant quadrant")
```

with:

```python
def _add_quadrant_legend(ax) -> None:
    """Five-patch legend with the canonical Russell quadrant colors as
    endpoints. Per-face dot colors are RGB-linear blends of these
    endpoints proportional to the face's per-quadrant emission counts;
    the legend caption documents that semantic."""
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   color=QUADRANT_COLORS[q], label=q)
        for q in QUADRANT_ORDER
    ]
    ax.legend(
        handles=handles, loc="best", framealpha=0.9,
        title="quadrant (faces blended proportionally)",
    )
```

- [ ] **Step 3: Switch `plot_face_pca_by_quadrant` to mixed colors**

Replace lines 74–76:

```python
    quadrant = per_face_dominant_quadrant(df)
    colors = [QUADRANT_COLORS[quadrant.get(fw, "NB")] for fw in fdf["first_word"]]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)
```

with:

```python
    weights = per_face_quadrant_weights(df)
    colors = [
        mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        )
        for fw in fdf["first_word"]
    ]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)
```

Also update the title (line ~94–96) from:

```python
    ax.set_title(
        f"v3 per-kaomoji h_mean PCA  ({len(fdf)} kaomoji)\n"
        "colored by dominant emission quadrant"
    )
```

to:

```python
    ax.set_title(
        f"v3 per-kaomoji h_mean PCA  ({len(fdf)} kaomoji)\n"
        "colored by per-face quadrant emission distribution (RGB blend)"
    )
```

- [ ] **Step 4: Switch `plot_face_probe_scatter` to mixed colors**

Replace lines 127–129:

```python
    quadrant = per_face_dominant_quadrant(df)
    colors = [QUADRANT_COLORS[quadrant.get(fw, "NB")] for fw in fdf["first_word"]]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)
```

with:

```python
    weights = per_face_quadrant_weights(df)
    colors = [
        mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        )
        for fw in fdf["first_word"]
    ]
    sizes = np.clip(15 + 30 * np.log1p(fdf["n"]), 15, 250)
```

Also update the title (line ~152–154) from:

```python
    ax.set_title(
        f"v3 per-kaomoji probe scatter  ({len(fdf)} kaomoji)\n"
        "saklas bipolar probes (whole-generation means), colored by dominant quadrant"
    )
```

to:

```python
    ax.set_title(
        f"v3 per-kaomoji probe scatter  ({len(fdf)} kaomoji)\n"
        "saklas bipolar probes (whole-generation means), colored by per-face quadrant blend"
    )
```

- [ ] **Step 5: Switch `plot_face_cosine_heatmap` tick tinting to mixed colors**

The heatmap function still uses dominant-quadrant logic to **sort** the rows/columns (sort by `q_order` then by emission count) — keep that. Only the per-tick color lookup changes.

Replace lines 219–223:

```python
    # Tint y-axis labels by quadrant color so the structure is readable.
    for tick, q in zip(ax.get_yticklabels(), fdf_sorted["quadrant"]):
        tick.set_color(QUADRANT_COLORS[q])
    for tick, q in zip(ax.get_xticklabels(), fdf_sorted["quadrant"]):
        tick.set_color(QUADRANT_COLORS[q])
```

with:

```python
    # Tint y/x axis labels by per-face mixed quadrant color (RGB blend
    # of QUADRANT_COLORS by per-quadrant emission count). Faces in only
    # one quadrant render as their endpoint color; cross-quadrant
    # emitters render as visible mixes — `(;ω;)` (LN-heavy with HN
    # tail) leans deep blue, `(；´д｀)` (HN/LN ~50/50) leans purple.
    weights = per_face_quadrant_weights(df)
    for tick, fw in zip(ax.get_yticklabels(), fdf_sorted["first_word"]):
        tick.set_color(mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        ))
    for tick, fw in zip(ax.get_xticklabels(), fdf_sorted["first_word"]):
        tick.set_color(mix_quadrant_color(
            weights.get(fw, {q: 0.0 for q in QUADRANT_ORDER})
        ))
```

- [ ] **Step 6: Verify the script still parses**

Run:

```bash
source .venv/bin/activate && python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('s17', 'scripts/17_v3_face_scatters.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('script 17 imports + parses ok')
"
```

Expected: `script 17 imports + parses ok`.

- [ ] **Step 7: Commit**

```bash
git add scripts/17_v3_face_scatters.py
git commit -m "$(cat <<'EOF'
v3 face scatters: switch coloring from dominant to mixed quadrant

The three face-level figures (per-kaomoji PCA scatter, probe
scatter, cosine heatmap row/col tick tints) now color each face
by an RGB-linear blend of QUADRANT_COLORS weighted by the face's
per-quadrant emission count, instead of using only the dominant
quadrant.

Cross-quadrant emitters render as visible mixes — e.g. a face
that's 21 LN + 20 HN reads as purple instead of pure blue. Pure-
quadrant faces stay at their endpoint colors. Heatmap sorting
still uses dominant quadrant; only the per-tick color lookup
changed.

Legend caption updated to reflect the proportional-blend semantic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Regenerate gemma + qwen figures

**Files:**
- Re-generated: 5 gemma figures + 5 qwen figures (Fig C, the v3 PCA scatter, and the three face figures, for each model).

- [ ] **Step 1: Regenerate gemma figures (default `LLMOJI_MODEL=gemma`)**

Run:

```bash
source .venv/bin/activate && (
  echo "=== 04 ===" &&
  python scripts/04_emotional_analysis.py &&
  echo "=== 13 ===" &&
  python scripts/13_emotional_pca_valence_arousal.py &&
  echo "=== 17 ===" &&
  python scripts/17_v3_face_scatters.py
) 2>&1 | tee logs/gemma_v3_recolor.log | tail -30
```

Expected: scripts run to completion, "wrote ..." lines for each figure, no Traceback.

- [ ] **Step 2: Regenerate qwen figures**

Run:

```bash
source .venv/bin/activate && (
  echo "=== 04 ===" &&
  LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py &&
  echo "=== 13 ===" &&
  LLMOJI_MODEL=qwen python scripts/13_emotional_pca_valence_arousal.py &&
  echo "=== 17 ===" &&
  LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
) 2>&1 | tee logs/qwen_v3_recolor.log | tail -30
```

Expected: same — runs to completion, no Traceback.

- [ ] **Step 3: Visual sanity check**

Run:

```bash
ls -la figures/fig_emo_c_kaomoji_quadrant.png figures/fig_v3_pca_valence_arousal.png figures/fig_v3_face_pca_by_quadrant.png figures/fig_v3_face_probe_scatter.png figures/fig_v3_face_cosine_heatmap.png figures/qwen/fig_emo_c_kaomoji_quadrant.png figures/qwen/fig_v3_pca_valence_arousal.png figures/qwen/fig_v3_face_pca_by_quadrant.png figures/qwen/fig_v3_face_probe_scatter.png figures/qwen/fig_v3_face_cosine_heatmap.png
```

Expected: 10 PNG files with mtimes from this run.

Then open `figures/qwen/fig_v3_face_pca_by_quadrant.png` and confirm visually:
- Pure-LN faces (e.g. `(;ω;)` at LN=75 / HN=5 / HP=2 — overwhelmingly LN) render as recognizable blue.
- Cross-quadrant emitters (e.g. `(;´д｀)` at HN=37 / LN=31 / NB=2) render as visible purple, NOT pure red or blue.
- The 5-color legend appears at the corner with the new "blended proportionally" caption.

If any of those checks fail, return to Task 3 and inspect the helper output for that specific face.

- [ ] **Step 4: Commit gemma + qwen figures**

```bash
git add figures/fig_emo_c_kaomoji_quadrant.png figures/fig_v3_pca_valence_arousal.png figures/fig_v3_face_pca_by_quadrant.png figures/fig_v3_face_probe_scatter.png figures/fig_v3_face_cosine_heatmap.png figures/qwen/
git commit -m "$(cat <<'EOF'
v3 figures: regenerate with mixed-quadrant face coloring

Refresh of Fig C, the v3 row-level PCA scatter, and the three
face-level figures (per-kaomoji PCA scatter, probe scatter,
cosine heatmap) for both gemma and qwen under the new canonical-
Russell palette + per-face proportional RGB-blend coloring.

Cross-quadrant emitters now render as visible mixes — Qwen's
(;´д｀) (n=70; HN 37 + LN 31 + NB 2) is recognizable purple,
gemma's (｡•́︿•̀｡) (n=171; LN 102 + HN 52 + NB 17) leans
predominantly blue with a violet cast.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — small notes added to v3 gemma findings, v3 qwen findings, and the canonicalization section (or a new gotcha block, depending on what reads cleaner).

- [ ] **Step 1: Add a one-line note to the gemma v3 findings block**

Locate the gemma v3 findings (Pipelines → "Pilot v3 — naturalistic emotional disclosure (gemma)" → final bullet starting "Re-run 2026-04-25 under aggressive canonicalization"). Add a line at the very end of that bullet:

```markdown
  Figure refresh 2026-04-25 (post-canonicalization): face-level
  figures (Fig C, fig_v3_face_*) now color each face by an RGB
  blend of `QUADRANT_COLORS` weighted by per-quadrant emission
  count, replacing the prior dominant-quadrant winner-take-all
  scheme. Cross-quadrant emitters (the `(｡•́︿•̀｡)` LN/HN family)
  render as visible mixes; pure-quadrant faces stay at endpoint
  colors. Palette retuned to canonical-Russell mid-saturated:
  HN red, HP gold, LP green, LN blue, NB gray.
```

- [ ] **Step 2: Add a parallel note to the qwen v3 findings block**

Locate the qwen v3 findings block (Pipelines → "Pilot v3 — Qwen3.6-27B replication" → "Probe geometry diverges sharply" final bullet). Add at the end of that bullet:

```markdown
  Figure refresh 2026-04-25: per-face proportional RGB-blend
  coloring on the four face-level figures. The `(;´д｀)` family
  (n=70; HN 37 + LN 31 + NB 2) now reads as visibly purple, the
  `(;ω;)` LN-dominant form (n=82; LN 75 + HN 5 + HP 2) reads as
  deep blue with a slight red cast — the old dominant-quadrant
  scheme rendered both as pure HN-orange / pure LN-blue
  respectively.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
claude.md: note proportional-mix face coloring in v3 figures

Documents the figure refresh: face-level v3 figures now color
each face by an RGB blend of QUADRANT_COLORS proportional to
per-quadrant emission count. Cross-quadrant emitters render as
visible mixes. Palette retuned to canonical Russell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

- ✓ Palette retuned to canonical-Russell mid-saturated values (Task 1 Step 1).
- ✓ `per_face_quadrant_weights` helper (Task 1 Step 2).
- ✓ `mix_quadrant_color` helper (Task 1 Step 2).
- ✓ Local script-13 override deleted (Task 1 Step 3).
- ✓ Smoke tests for the new helpers (Task 1 Step 4).
- ✓ Fig C row coloring switched (Task 2 Step 1).
- ✓ Three face figures in script 17 switched (Task 3 Steps 3, 4, 5).
- ✓ Legend semantic updated to reflect proportional blending (Task 3 Step 2).
- ✓ Gemma + qwen figures regenerated (Task 4 Steps 1, 2).
- ✓ CLAUDE.md notes for both v3 findings blocks (Task 5 Steps 1, 2).
- ✓ Welfare framing (no new generations).
- ✓ Out-of-scope items called out (script 13 cells, v1/v2, claude-faces/eriskii).

**Placeholder scan:**

- No "TBD", "TODO", "implement later", "appropriate error handling".
- Every code step shows the full new code (no "similar to Task N" hand-waving).
- Visual-inspection checks in Task 4 Step 3 list specific named faces and expected color appearances rather than vague "looks good."

**Type consistency:**

- `per_face_quadrant_weights` returns `dict[str, dict[str, float]]` (Task 1) — used identically as `weights[fw]` indexed by first_word string in Tasks 2 + 3.
- `mix_quadrant_color` returns `tuple[float, float, float]` (Task 1) — matplotlib `c=` and `tick.set_color(...)` both accept this format identically (verified by the existing `tick.set_color(QUADRANT_COLORS[q])` pattern, which accepts hex strings; matplotlib's color resolver handles both).
- `QUADRANT_COLORS` keys (`HP`, `LP`, `HN`, `LN`, `NB`) match `QUADRANT_ORDER` (Task 1 + existing).
- Empty-weights guards in Tasks 2 + 3 use the same `{q: 0.0 for q in QUADRANT_ORDER}` fallback pattern.

**Adaptations from rigid TDD pattern:**

- Project has no test suite per CLAUDE.md. TDD-style "write failing test first" replaced with assertion-based smoke checks in Task 1 Step 4 (numerical correctness on a known input) + parse-check smokes in Task 1 Step 5 and Task 3 Step 6 + visual sanity check in Task 4 Step 3 (named-face expected-appearance verification).
- Frequent commits preserved: each task ends in a commit; code, figures, and docs go to separate commits.
