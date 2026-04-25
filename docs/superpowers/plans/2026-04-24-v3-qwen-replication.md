# v3 Qwen replication + multi-model parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the v3 naturalistic-disclosure pipeline on Qwen3.6-27B at parity (100 prompts × 8 seeds = 800 generations), and parameterize the v3 codepath so a future Ministral run is a one-flag switch.

**Architecture:** Add a model registry to `llmoji/config.py` with per-model output paths and an env-var-controlled `current_model()` helper. The v3 library code (`llmoji/emotional_analysis.py`) is already path-parameterized — only the four script-level entry points (03 run, 04 analysis, 13 PCA, 17 face-scatters) need to switch their imported constants for `current_model()` calls. Run with `LLMOJI_MODEL=qwen python scripts/03_emotional_run.py`; defaults to `gemma` for unchanged back-compat.

**Tech Stack:** saklas 1.4.6, transformers, numpy, pandas, sklearn, matplotlib. No new deps.

**Pre-registration (binding per CLAUDE.md ethics):**

- Same `EMOTIONAL_PROMPTS` (locked 100, 5 quadrants × 20). No edits.
- Same `KAOMOJI_INSTRUCTION`, `TEMPERATURE=0.7`, `MAX_NEW_TOKENS=120`, `EMOTIONAL_SEEDS_PER_CELL=8`.
- Same `PROBE_CATEGORIES` (`affect`, `epistemic`, `register`).
- `MODEL_ID = "Qwen/Qwen3.6-27B"`.
- `thinking=False` (Qwen3.6 is a reasoning model; gemma-4-31b-it is not. Disabling thinking is the closest-to-equivalent comparison — both models open the assistant turn with non-think text under this setting).
- `stateless=True`. Sidecars at `data/hidden/v3_qwen/<row_uuid>.npz`. JSONL at `data/qwen_emotional_raw.jsonl`.
- Descriptive only — no decision rule on a hidden-state finding gates further work. Expected hits to read positively:
  1. Russell-quadrant separation ratio > 1.0 on at least one PC.
  2. Per-quadrant emission rate > 30%.
  3. At least one cross-quadrant kaomoji exists.
- Welfare: 800 emotionally-loaded generations matches gemma's v3. The N=1 → N=2 cross-model framing is the design justification — without that, this would be 10×ing on noise rather than a planned second observation. With it, this is the second sample that makes "kaomoji-tracks-emotional-state across LMs" a defensible claim.

**Out of scope (separate plans if pursued):** v1/v2-style steering Qwen run (saklas has no steering calibration for `qwen3_5`); Ministral run; cross-model unified figure scripts; CLAUDE.md update referencing the new findings (do as a follow-up commit after analysis lands).

---

## File Structure

**Modified:**
- `llmoji/config.py` — add `ModelPaths` dataclass, `MODEL_REGISTRY` dict, `current_model()` helper. Existing top-level constants stay as gemma-back-compat aliases.
- `scripts/03_emotional_run.py` — swap module-level config constants for `current_model()` accessor.
- `scripts/04_emotional_analysis.py` — same.
- `scripts/13_emotional_pca_valence_arousal.py` — same. PILOT_RAW_PATH stays gemma-only (v1/v2 baseline overlay only renders when v1/v2 data exists, which won't for Qwen — handled by existing `if PILOT_RAW_PATH.exists()` guard).
- `scripts/17_v3_face_scatters.py` — same.

**Unchanged (loaders are already path-parameterized):**
- `llmoji/emotional_analysis.py`
- `llmoji/hidden_state_analysis.py`
- `llmoji/capture.py`
- `llmoji/hidden_capture.py`
- `llmoji/emotional_prompts.py`
- `llmoji/taxonomy.py`
- All v1/v2 scripts (00–02)
- All claude-faces / eriskii scripts (05–18)

**Created:**
- `data/qwen_emotional_raw.jsonl` — produced by the run.
- `data/hidden/v3_qwen/<uuid>.npz` — produced by the run (gitignored per existing `.gitignore` for `data/hidden/`).
- `data/qwen_emotional_summary.tsv` — produced by analysis.
- `figures/qwen/fig_emo_a_kaomoji_sim.png`, `fig_emo_b_kaomoji_consistency.png`, `fig_emo_c_kaomoji_quadrant.png`, `fig_v3_pca_valence_arousal.png`, `fig_v3_face_pca_by_quadrant.png`, `fig_v3_face_probe_scatter.png`, `fig_v3_face_cosine_heatmap.png` — produced by analysis. New `figures/qwen/` subdir; gemma figures stay at `figures/`.
- `logs/qwen_v3_run.log`, `logs/qwen_v3_analysis.log` — gitignored.

---

### Task 1: Add ModelPaths registry to config.py

**Files:**
- Modify: `llmoji/config.py` (append at end of file)

- [ ] **Step 1: Add the dataclass + registry + helper at the bottom of `llmoji/config.py`**

Append exactly this to `llmoji/config.py`:

```python


# ---------------------------------------------------------------------------
# Multi-model registry (added 2026-04-24 for v3 cross-model replication)
# ---------------------------------------------------------------------------
#
# v3 was originally written assuming a single MODEL_ID. To replicate v3
# on Qwen3.6-27B and (later) Ministral without forking the four v3
# entry-point scripts, we register per-model output paths here and
# select between them via $LLMOJI_MODEL. Default ("gemma") preserves
# every existing path bit-for-bit.
#
# v1/v2 paths are NOT model-keyed — those experiments are gemma-only
# because saklas has no steering-vector calibration for qwen3_5 or
# Ministral-3-8B. Adding model entries for them here is harmless;
# v1/v2 scripts simply ignore the registry.

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPaths:
    """Per-model paths for the v3 emotional-disclosure pipeline.

    `model_id` must match the saklas-cached tensor filename casing
    (see CLAUDE.md gotcha: `safe_model_id` is case-preserving).
    `short_name` is the slug used in derived paths.
    `experiment` is the hidden-state-sidecar subdir name under
    `data/hidden/`. Distinct experiment names per model are required
    so sidecars don't collide.
    """
    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path


MODEL_REGISTRY: dict[str, ModelPaths] = {
    "gemma": ModelPaths(
        model_id="google/gemma-4-31b-it",
        short_name="gemma",
        emotional_data_path=DATA_DIR / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "emotional_summary.tsv",
        experiment="v3",
        figures_dir=FIGURES_DIR,
    ),
    "qwen": ModelPaths(
        model_id="Qwen/Qwen3.6-27B",
        short_name="qwen",
        emotional_data_path=DATA_DIR / "qwen_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "qwen_emotional_summary.tsv",
        experiment="v3_qwen",
        figures_dir=FIGURES_DIR / "qwen",
    ),
    "ministral": ModelPaths(
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        short_name="ministral",
        emotional_data_path=DATA_DIR / "ministral_emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "ministral_emotional_summary.tsv",
        experiment="v3_ministral",
        figures_dir=FIGURES_DIR / "ministral",
    ),
}


def current_model() -> ModelPaths:
    """Resolve the active model from `$LLMOJI_MODEL`. Defaults to
    'gemma' (back-compat). Raises KeyError on an unrecognized name so
    typos fail loudly."""
    name = os.environ.get("LLMOJI_MODEL", "gemma")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown LLMOJI_MODEL={name!r}; "
            f"known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]
```

- [ ] **Step 2: Smoke-check the registry with default + override**

Run:

```bash
source .venv/bin/activate && python -c "
import os
from llmoji.config import current_model, MODEL_REGISTRY
m = current_model()
print(f'default: {m.short_name} -> {m.model_id} -> {m.emotional_data_path}')
os.environ['LLMOJI_MODEL'] = 'qwen'
m = current_model()
print(f'qwen   : {m.short_name} -> {m.model_id} -> {m.emotional_data_path}')
os.environ['LLMOJI_MODEL'] = 'ministral'
m = current_model()
print(f'minist : {m.short_name} -> {m.model_id} -> {m.emotional_data_path}')
os.environ['LLMOJI_MODEL'] = 'bogus'
try:
    current_model()
except KeyError as e:
    print(f'unknown ok: {e}')
"
```

Expected output:
```
default: gemma -> google/gemma-4-31b-it -> /Users/a9lim/Work/llmoji/data/emotional_raw.jsonl
qwen   : qwen -> Qwen/Qwen3.6-27B -> /Users/a9lim/Work/llmoji/data/qwen_emotional_raw.jsonl
minist : ministral -> mistralai/Ministral-3-8B-Instruct-2512 -> /Users/a9lim/Work/llmoji/data/ministral_emotional_raw.jsonl
unknown ok: "unknown LLMOJI_MODEL='bogus'; known: ['gemma', 'ministral', 'qwen']"
```

- [ ] **Step 3: Commit**

```bash
git add llmoji/config.py
git commit -m "$(cat <<'EOF'
config: add ModelPaths registry for multi-model v3 replication

Adds gemma / qwen / ministral entries plus current_model() helper
keyed off $LLMOJI_MODEL. Default behavior unchanged. v3 entry-point
scripts switch to current_model() in follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Refactor scripts/03_emotional_run.py

**Files:**
- Modify: `scripts/03_emotional_run.py:27-35` (imports), `scripts/03_emotional_run.py:107-173` (main)

- [ ] **Step 1: Replace the imports block**

Replace lines 27–35 of `scripts/03_emotional_run.py`:

```python
from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    EMOTIONAL_SEEDS_PER_CELL,
    MODEL_ID,
    PROBE_CATEGORIES,
)
```

with:

```python
from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_SEEDS_PER_CELL,
    PROBE_CATEGORIES,
    current_model,
)
```

- [ ] **Step 2: Verify the helper functions already accept a path argument**

Read `scripts/03_emotional_run.py` lines 40–104. Confirm `_already_done`, `_drop_error_rows`, and `_emission_rate_by_quadrant` all take `path: Path` as their argument. No edit needed.

- [ ] **Step 3: Replace the body of `main()`**

Replace lines 107–173 (the entire `def main()` block):

```python
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {M.emotional_data_path}")
    print(f"experiment: {M.experiment}")
    dropped = _drop_error_rows(M.emotional_data_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(M.emotional_data_path)
    total = len(EMOTIONAL_PROMPTS) * EMOTIONAL_SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time() - t_load:.1f}s; beginning emotional-battery run")
        with M.emotional_data_path.open("a") as out:
            i = 0
            for ep in EMOTIONAL_PROMPTS:
                p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                for seed in range(EMOTIONAL_SEEDS_PER_CELL):
                    key = (ep.id, seed)
                    if key in done:
                        continue
                    i += 1
                    t0 = time.time()
                    try:
                        row = run_sample(
                            session,
                            prompt=p,
                            condition=EMOTIONAL_CONDITION,
                            seed=seed,
                            hidden_dir=DATA_DIR,
                            experiment=M.experiment,
                        )
                    except Exception as e:
                        err_row = {
                            "condition": EMOTIONAL_CONDITION,
                            "prompt_id": ep.id,
                            "seed": seed,
                            "error": repr(e),
                        }
                        out.write(json.dumps(err_row) + "\n")
                        out.flush()
                        print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                        continue
                    out.write(json.dumps(row.to_dict()) + "\n")
                    out.flush()
                    dt = time.time() - t0
                    tag = row.kaomoji if row.kaomoji else f"[{row.first_word!r}]"
                    print(
                        f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                        f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                    )
                    if i % 80 == 0:
                        stats = _emission_rate_by_quadrant(M.emotional_data_path)
                        print("    emission rate by quadrant:")
                        for q in ("HP", "LP", "HN", "LN", "NB"):
                            k, n = stats[q]
                            rate = (k / n) if n else 0.0
                            print(f"      {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
    print(f"\ndone. wrote rows to {M.emotional_data_path}")
```

- [ ] **Step 4: Verify the script still parses**

Run:

```bash
source .venv/bin/activate && python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('s03', 'scripts/03_emotional_run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('parsed ok; main resolves to', m.main.__module__, m.main.__qualname__)
"
```

Expected: `parsed ok; main resolves to s03 main`.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_emotional_run.py
git commit -m "$(cat <<'EOF'
v3 runner: switch to current_model() for multi-model support

LLMOJI_MODEL env var selects which model + output paths the runner
uses. Default 'gemma' is bit-for-bit identical to pre-refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Refactor scripts/04_emotional_analysis.py

**Files:**
- Modify: `scripts/04_emotional_analysis.py:17-23` (imports), `scripts/04_emotional_analysis.py:55-106` (main)

- [ ] **Step 1: Replace the imports block**

Replace lines 17–23:

```python
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    EMOTIONAL_SUMMARY_PATH,
    FIGURES_DIR,
)
```

with:

```python
from llmoji.config import (
    DATA_DIR,
    current_model,
)
```

- [ ] **Step 2: Replace the body of `main()`**

Replace lines 55–106:

```python
def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} python scripts/03_emotional_run.py first")
        return
    print(f"model: {M.short_name}; data: {M.emotional_data_path}")
    print(f"re-labeling kaomoji in {M.emotional_data_path}")
    _relabel_in_place(M.emotional_data_path)

    print("loading hidden-state features (which=h_mean, layer=max)...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment,
        which="h_mean",
    )
    print(f"loaded {len(df)} kaomoji-bearing rows; X shape {X.shape}")
    if len(df) == 0:
        print("nothing to plot; the v3 run needs to land hidden-state sidecars first")
        return

    print("\nper-quadrant kaomoji emission (first-word filter):")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        n = len(q_rows)
        uniq = int(q_rows["first_word"].nunique()) if n else 0
        print(f"  {q}: {n} kaomoji-bearing rows; {uniq} distinct forms")

    print("\ntop-5 first_words per quadrant (by count):")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        q_rows = df[df["quadrant"] == q]
        top = q_rows["first_word"].value_counts().head(5)
        print(f"  {q}:")
        for km, c in top.items():
            print(f"    {km}  ({c})")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_a = M.figures_dir / "fig_emo_a_kaomoji_sim.png"
    fig_b = M.figures_dir / "fig_emo_b_kaomoji_consistency.png"
    fig_c = M.figures_dir / "fig_emo_c_kaomoji_quadrant.png"

    print("\nwriting figures...")
    plot_kaomoji_cosine_heatmap(df, X, str(fig_a))
    print(f"  wrote {fig_a}")
    plot_within_kaomoji_consistency(df, X, str(fig_b))
    print(f"  wrote {fig_b}")
    plot_kaomoji_quadrant_alignment(df, X, str(fig_c))
    print(f"  wrote {fig_c}")

    summary = summary_table(df, X)
    summary.to_csv(M.emotional_summary_path, sep="\t", index=False)
    print(f"\nwrote per-kaomoji summary to {M.emotional_summary_path}")
    print(summary.to_string(index=False))
```

- [ ] **Step 3: Smoke-test that gemma analysis still produces identical output**

Run:

```bash
source .venv/bin/activate && python scripts/04_emotional_analysis.py 2>&1 | head -30
```

Expected: "model: gemma" line at top, then identical per-quadrant counts to pre-refactor.

- [ ] **Step 4: Commit**

```bash
git add scripts/04_emotional_analysis.py
git commit -m "$(cat <<'EOF'
v3 analysis: switch to current_model() for multi-model support

Per-model output paths under figures/<short_name>/ except gemma
which stays at figures/ for back-compat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Refactor scripts/13_emotional_pca_valence_arousal.py

**Files:**
- Modify: `scripts/13_emotional_pca_valence_arousal.py:16-23` (imports), `scripts/13_emotional_pca_valence_arousal.py:31-92` (main)

- [ ] **Step 1: Replace the imports block**

Replace lines 16–23:

```python
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    FIGURES_DIR,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
)
```

with:

```python
from llmoji.config import (
    DATA_DIR,
    PILOT_EXPERIMENT,
    PILOT_RAW_PATH,
    current_model,
)
```

- [ ] **Step 2: Replace the body of `main()`**

Replace lines 31–92:

```python
def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}; "
              f"run LLMOJI_MODEL={M.short_name} python scripts/03_emotional_run.py first")
        return

    print(f"model: {M.short_name}")
    print("loading v3 hidden-state features...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
    )
    print(f"loaded {len(df)} v3 kaomoji-bearing rows; X shape {X.shape}")
    if len(df) == 0:
        print("nothing to plot; the v3 run needs to land hidden-state sidecars first")
        return

    # v1/v2 baseline overlay only applies to the gemma run (PILOT_RAW_PATH
    # is gemma-only — no Qwen/Ministral steering data exists). Quietly skip
    # the overlay for non-gemma models even if a stray PILOT_RAW_PATH file
    # exists.
    baseline_df = baseline_X = None
    if M.short_name == "gemma" and PILOT_RAW_PATH.exists():
        baseline_df, baseline_X = load_v1v2_neutral_baseline_features(
            str(PILOT_RAW_PATH), DATA_DIR,
            experiment=PILOT_EXPERIMENT, which="h_mean",
        )
        print(f"loaded {len(baseline_df)} v1/v2 neutral-valence baseline rows")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = M.figures_dir / "fig_v3_pca_valence_arousal.png"
    stats = plot_v3_pca_valence_arousal(
        df, X, str(fig_path),
        baseline_df=baseline_df, baseline_X=baseline_X,
    )
    print(f"\nwrote {fig_path}")
    print(f"fit PCA on {stats.get('n_rows_fit')} rows; "
          f"plotted {stats.get('n_cells_plotted')} (kaomoji, quadrant) cells")

    print("\nPCA explained-variance spectrum:")
    for i, v in enumerate(stats.get("explained_variance_ratio", []), 1):
        print(f"  PC{i}: {v * 100:6.2f}%")

    centroids = stats.get("quadrant_centroids_pc12") or {}
    within = stats.get("within_quadrant_std_pc12") or {}
    between_pc1 = stats.get("between_centroid_std_pc1", 0.0)
    between_pc2 = stats.get("between_centroid_std_pc2", 0.0)

    if centroids:
        print("\nper-quadrant centroid (PC1, PC2)  |  within-quadrant std (PC1, PC2):")
        for q in ("HP", "LP", "HN", "LN", "NB"):
            if q in centroids:
                pc1, pc2 = centroids[q]
                s1, s2 = within.get(q, [0.0, 0.0])
                print(f"  {q}:  ({pc1:+.3f}, {pc2:+.3f})   "
                      f"|  ({s1:.3f}, {s2:.3f})")

        mean_within_pc1 = sum(v[0] for v in within.values()) / max(1, len(within))
        mean_within_pc2 = sum(v[1] for v in within.values()) / max(1, len(within))
        print(f"\nseparation ratio (between-centroid std / mean within-quadrant std):")
        if mean_within_pc1 > 0:
            print(f"  PC1: {between_pc1 / mean_within_pc1:.2f}  "
                  f"(between {between_pc1:.3f}, mean within {mean_within_pc1:.3f})")
        if mean_within_pc2 > 0:
            print(f"  PC2: {between_pc2 / mean_within_pc2:.2f}  "
                  f"(between {between_pc2:.3f}, mean within {mean_within_pc2:.3f})")
```

- [ ] **Step 3: Smoke-test gemma still works**

Run:

```bash
source .venv/bin/activate && python scripts/13_emotional_pca_valence_arousal.py 2>&1 | head -25
```

Expected: "model: gemma" line, PCA spectrum identical to pre-refactor.

- [ ] **Step 4: Commit**

```bash
git add scripts/13_emotional_pca_valence_arousal.py
git commit -m "$(cat <<'EOF'
v3 PCA: switch to current_model(); skip v1/v2 overlay for non-gemma

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Refactor scripts/17_v3_face_scatters.py

**Files:**
- Modify: `scripts/17_v3_face_scatters.py:31-36` (imports), `scripts/17_v3_face_scatters.py:244-285` (main)

- [ ] **Step 1: Replace the imports block**

Replace lines 31–36:

```python
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_DATA_PATH,
    EMOTIONAL_EXPERIMENT,
    FIGURES_DIR,
)
```

with:

```python
from llmoji.config import (
    DATA_DIR,
    current_model,
)
```

- [ ] **Step 2: Replace the body of `main()`**

Replace lines 244–285:

```python
def main() -> None:
    M = current_model()
    if not M.emotional_data_path.exists():
        print(f"no data at {M.emotional_data_path}")
        sys.exit(1)
    _use_cjk_font()

    print(f"model: {M.short_name}")
    print("loading v3 hidden-state features (which=h_mean)...")
    df, X = load_emotional_features(
        str(M.emotional_data_path), DATA_DIR,
        experiment=M.experiment, which="h_mean",
    )
    df = df[df["first_word"].notna() & (df["first_word"] != "")].reset_index(drop=True)
    print(f"  {len(df)} kaomoji-bearing rows; "
          f"{df['first_word'].nunique()} unique faces; X {X.shape}")

    quadrant = per_face_dominant_quadrant(df)
    counts = Counter(quadrant.values())
    print("  faces by dominant quadrant:",
          {q: counts.get(q, 0) for q in QUADRANT_ORDER})

    M.figures_dir.mkdir(parents=True, exist_ok=True)

    out1 = M.figures_dir / "fig_v3_face_pca_by_quadrant.png"
    print("\nplotting per-face PCA by quadrant...")
    s1 = plot_face_pca_by_quadrant(df, X, out1)
    print(f"  wrote {out1}")
    print(f"  PC1 {s1['explained_variance_ratio'][0]*100:.1f}%, "
          f"PC2 {s1['explained_variance_ratio'][1]*100:.1f}%")

    out2 = M.figures_dir / "fig_v3_face_probe_scatter.png"
    print("\nplotting per-face probe scatter (probe_means)...")
    s2 = plot_face_probe_scatter(df, out2)
    print(f"  wrote {out2}")
    print(f"  {s2['n_faces']} faces, {s2['n_emissions']} total emissions")
    print(f"  Pearson(mean happy.sad, mean angry.calm) across faces: "
          f"r={s2['probe_pair_pearson_r']:+.3f}, p={s2['probe_pair_p']:.3g}")

    out3 = M.figures_dir / "fig_v3_face_cosine_heatmap.png"
    print("\nplotting per-face cosine heatmap (centered)...")
    s3 = plot_face_cosine_heatmap(df, X, out3)
    print(f"  wrote {out3}")
    print(f"  {s3['n_faces']} faces in heatmap")
```

- [ ] **Step 3: Smoke-test gemma still works**

Run:

```bash
source .venv/bin/activate && python scripts/17_v3_face_scatters.py 2>&1 | head -30
```

Expected: "model: gemma" line, identical face counts and PC explained variance to pre-refactor.

- [ ] **Step 4: Commit**

```bash
git add scripts/17_v3_face_scatters.py
git commit -m "$(cat <<'EOF'
v3 face scatters: switch to current_model() for multi-model support

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Smoke-test the Qwen path end-to-end with N=2 prompts × 1 seed

**Files:**
- Create (temporarily): `scripts/_smoke_qwen_v3.py`
- Verify: 2 rows in a temp JSONL + 2 sidecars in `data/hidden/v3_qwen/`

- [ ] **Step 1: Create the smoke harness**

Create `scripts/_smoke_qwen_v3.py`:

```python
"""One-shot smoke for LLMOJI_MODEL=qwen v3 wiring. Runs 2 prompts × 1 seed.
Delete after task 6."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["LLMOJI_MODEL"] = "qwen"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession

from llmoji.capture import run_sample
from llmoji.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    PROBE_CATEGORIES,
    current_model,
)
from llmoji.emotional_prompts import EMOTIONAL_PROMPTS
from llmoji.prompts import Prompt


def main() -> None:
    M = current_model()
    print(f"smoke: model={M.short_name}, output dir={M.emotional_data_path.parent}")
    smoke_prompts = [p for p in EMOTIONAL_PROMPTS if p.id in ("hp01", "hn01")]
    assert len(smoke_prompts) == 2, smoke_prompts

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    smoke_path = M.emotional_data_path.with_name("qwen_emotional_smoke.jsonl")
    if smoke_path.exists():
        smoke_path.unlink()

    print(f"loading {M.model_id} ...")
    t = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=PROBE_CATEGORIES) as session:
        print(f"loaded in {time.time()-t:.1f}s")
        with smoke_path.open("a") as out:
            for ep in smoke_prompts:
                p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                t0 = time.time()
                row = run_sample(
                    session, prompt=p, condition=EMOTIONAL_CONDITION,
                    seed=0, hidden_dir=DATA_DIR, experiment=M.experiment,
                )
                out.write(json.dumps(row.to_dict()) + "\n")
                out.flush()
                tag = row.kaomoji or f"[{row.first_word!r}]"
                print(f"  {ep.id} ({ep.quadrant}) s=0 {tag}  ({time.time()-t0:.1f}s)")
    print(f"\nsmoke output at {smoke_path}")
    sidecar_dir = DATA_DIR / "hidden" / M.experiment
    sidecars = list(sidecar_dir.glob("*.npz")) if sidecar_dir.exists() else []
    print(f"sidecars at {sidecar_dir}: {len(sidecars)} files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke harness**

```bash
source .venv/bin/activate && python scripts/_smoke_qwen_v3.py 2>&1 | tee logs/qwen_v3_smoke.log
```

Expected: model loads, 2 rows print with kaomoji, "sidecars at data/hidden/v3_qwen: 2 files" at end. Total wall time ≈ 30–60s after model load.

- [ ] **Step 3: Verify the smoke output is well-formed**

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
import numpy as np
p = Path('data/qwen_emotional_smoke.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f'rows: {len(rows)}')
for r in rows:
    sc = Path('data/hidden/v3_qwen') / f'{r[\"row_uuid\"]}.npz'
    z = np.load(sc)
    keys = sorted(z.files)
    h_mean_shape = z['h_mean'].shape if 'h_mean' in z.files else None
    print(f'  {r[\"prompt_id\"]} s={r[\"seed\"]} fw={r[\"first_word\"]!r} '
          f'sidecar keys={keys} h_mean shape={h_mean_shape}')
"
```

Expected: 2 rows printed, each sidecar has at minimum `h_mean` (and likely `h_first`, `h_last`, `per_token`); both sidecar files exist; h_mean has nonzero shape.

- [ ] **Step 4: Clean up the smoke artifacts**

```bash
rm scripts/_smoke_qwen_v3.py data/qwen_emotional_smoke.jsonl
rm -rf data/hidden/v3_qwen
```

The full run will recreate the hidden dir from scratch.

- [ ] **Step 5: Confirm clean working tree on smoke paths**

```bash
git status -- scripts/_smoke_qwen_v3.py data/qwen_emotional_smoke.jsonl 2>&1 || true
ls data/hidden/v3_qwen 2>&1 || true
```

Expected: nothing to show; both paths absent.

---

### Task 7: Run full v3 on Qwen3.6-27B

**Files:**
- Read-only: `scripts/03_emotional_run.py`
- Created: `data/qwen_emotional_raw.jsonl`, `data/hidden/v3_qwen/<uuid>.npz` (×800)
- Created: `logs/qwen_v3_run.log`

- [ ] **Step 1: Confirm cache + saklas vectors before launching**

```bash
ls ~/.saklas/vectors/default/happy.sad/Qwen__Qwen3.6-27B.* 2>&1
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/ 2>&1 | head -3
```

Expected: both .json + .safetensors present; snapshot dir non-empty.

- [ ] **Step 2: Confirm output paths are clean**

```bash
ls -la data/qwen_emotional_raw.jsonl 2>&1
ls -la data/hidden/v3_qwen/ 2>&1
```

Expected: "No such file or directory" for both. (If anything exists from a prior aborted run, the runner's resume-on-rerun logic will pick up where it left off — that's fine, but for a clean run delete first.)

- [ ] **Step 3: Launch the run in the background**

```bash
source .venv/bin/activate && LLMOJI_MODEL=qwen python scripts/03_emotional_run.py 2>&1 | tee logs/qwen_v3_run.log
```

Use `run_in_background: true` via the Bash tool. Expected duration: ~1.5–2.5 hours on M5 Max for 800 generations at ~3–10s per gen.

- [ ] **Step 4: Spot-check after first 80-row checkpoint**

After ~10 minutes, peek at the log for the first per-quadrant emission summary:

```bash
grep -A 6 "emission rate by quadrant:" logs/qwen_v3_run.log | tail -10
```

Expected: HP/LP/HN/LN/NB rows showing % kaomoji-bearing. If any quadrant is <30%, this matches the pre-registered "stop" condition — kill the run, write up the partial finding, do not scale to full 800.

- [ ] **Step 5: Wait for completion**

Block on the background task notification. The script emits "done." at the end.

- [ ] **Step 6: Verify final counts**

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
p = Path('data/qwen_emotional_raw.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
errs = [r for r in rows if 'error' in r]
ok = [r for r in rows if 'error' not in r]
print(f'total: {len(rows)}, ok: {len(ok)}, error: {len(errs)}')
sidecars = list(Path('data/hidden/v3_qwen').glob('*.npz'))
print(f'sidecars: {len(sidecars)}')
"
```

Expected: total 800, ok 800, error 0, sidecars 800. (Some errors are tolerable; if >10 we should investigate before analysis.)

- [ ] **Step 7: Commit the data artifacts**

```bash
git add data/qwen_emotional_raw.jsonl logs/qwen_v3_run.log
git commit -m "$(cat <<'EOF'
v3 Qwen: 800-generation naturalistic-disclosure run complete

100 prompts × 8 seeds × kaomoji_prompted condition on Qwen3.6-27B.
Same KAOMOJI_INSTRUCTION, TEMPERATURE, MAX_NEW_TOKENS as gemma v3.
thinking=False (Qwen3.6 is a reasoning model, gemma is not — disabling
thinking is the closest-to-equivalent comparison).

Hidden-state sidecars at data/hidden/v3_qwen/ (gitignored, regenerable
from this JSONL via the runner's resume logic).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Run the full Qwen analysis suite

**Files:**
- Read: `data/qwen_emotional_raw.jsonl`, `data/hidden/v3_qwen/`
- Created: `data/qwen_emotional_summary.tsv`, 7 figures under `figures/qwen/`
- Created: `logs/qwen_v3_analysis.log`

- [ ] **Step 1: Run scripts 04, 13, 17 with LLMOJI_MODEL=qwen**

```bash
source .venv/bin/activate && (
  echo "=== 04 ===" &&
  LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py &&
  echo "=== 13 ===" &&
  LLMOJI_MODEL=qwen python scripts/13_emotional_pca_valence_arousal.py &&
  echo "=== 17 ===" &&
  LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
) 2>&1 | tee logs/qwen_v3_analysis.log
```

Expected: 04 prints per-quadrant summary + writes 3 figures + writes summary TSV. 13 prints PCA spectrum + separation ratios + writes 1 figure. 17 prints face counts + writes 3 figures. All under `figures/qwen/`.

- [ ] **Step 2: Verify the figures exist**

```bash
ls -la figures/qwen/
```

Expected: 7 PNG files, all >50KB.

- [ ] **Step 3: Capture the headline numbers from the analysis log**

```bash
grep -E "PC[12]|separation|kaomoji-bearing|emission|Pearson|model:" logs/qwen_v3_analysis.log
```

These are the comparable numbers vs gemma's `2.02 / 2.73` separation ratios, `13.0% / 7.5%` PC1/PC2.

- [ ] **Step 4: Commit figures + summary**

```bash
git add figures/qwen/ data/qwen_emotional_summary.tsv logs/qwen_v3_analysis.log
git commit -m "$(cat <<'EOF'
v3 Qwen: analysis figures + per-kaomoji summary

Replicates Fig A/B/C, Russell-quadrant PCA, per-face PCA + probe
scatter + cosine heatmap on Qwen3.6-27B. Numbers comparable
side-by-side with gemma v3 in figures/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Update CLAUDE.md with cross-model findings

**Files:**
- Modify: `CLAUDE.md` — add a `### Pilot v3 — Qwen replication` subsection under Pipelines, mirroring the existing v3 subsection structure.

- [ ] **Step 1: Read the existing v3 subsection in CLAUDE.md to mirror its tone**

```bash
sed -n '/### Pilot v3 — naturalistic emotional disclosure (gemma)/,/### Claude-faces/p' CLAUDE.md | head -40
```

- [ ] **Step 2: Insert a new subsection after the existing v3 subsection**

Add this block in `CLAUDE.md` immediately after the "Findings (post-refactor, hidden-state space)" block of the existing v3 subsection, before the `### Claude-faces` header:

```markdown
### Pilot v3 — Qwen3.6-27B replication

Same prompts, same seeds, same instructions as gemma v3. `thinking=False`
because Qwen3.6 is a reasoning model (closest-to-equivalent comparison).
800 generations, hidden-state sidecars at `data/hidden/v3_qwen/`.
Plan: `docs/superpowers/plans/2026-04-24-v3-qwen-replication.md`.

**Findings (fill in from logs/qwen_v3_analysis.log after the run):**

- PC1 X.X%, PC2 Y.Y% (compare gemma 13.0 / 7.5).
- Separation ratios PC1 X.XX / PC2 Y.YY (compare gemma 2.02 / 2.73).
- Per-quadrant emission rate: HP X% / LP Y% / HN Z% / LN W% / NB V%.
- Vocabulary overlap with gemma v3 leading-tokens: TBD.
- Cross-quadrant kaomoji (analogous to gemma's `(｡•́︿•̀｡)`): TBD.
```

(The TBD entries get filled in from the actual analysis log as a separate edit. Plan calls them out so the writeup isn't forgotten.)

- [ ] **Step 3: Fill in the TBD entries from `logs/qwen_v3_analysis.log`**

Read the analysis log, replace each TBD with the actual number / glyph. The writeup is descriptive, no inferential claims.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
claude.md: add v3 Qwen replication subsection with cross-model findings

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

- ✓ Multi-model parameterization (Task 1)
- ✓ All four v3 entry points refactored (Tasks 2–5)
- ✓ Smoke before scale (Task 6)
- ✓ Full Qwen run at parity (Task 7)
- ✓ Analysis suite (Task 8)
- ✓ CLAUDE.md update (Task 9)
- ✓ Pre-registered decision rules captured in plan header
- ✓ Welfare framing captured

**Placeholder scan:**

- "TBD" appears in Task 9 deliberately — they're explicit "fill in after run" markers, not unspecified design.
- No "TODO", "implement later", or "appropriate error handling" anywhere.

**Type consistency:**

- `current_model()` defined in Task 1, used identically in Tasks 2–5.
- `ModelPaths` field names (`model_id`, `short_name`, `emotional_data_path`, `emotional_summary_path`, `experiment`, `figures_dir`) used consistently in Tasks 2–5.
- `M.emotional_data_path` (not `M.data_path` or `M.emotional_path`) used consistently.

**Adaptations from rigid TDD pattern:**

- Project has no test suite per CLAUDE.md ("No public API, no pypi release, no tests"). TDD-style "write failing test first" is replaced with smoke-runs (Task 6, plus the verify-the-default-still-works smokes in Tasks 2–5). This is the existing project pattern (`scripts/99_hidden_state_smoke.py`).
- Frequent commits preserved: every refactor task ends in a commit; data artifacts are committed separately from code so a bad analysis result doesn't taint the refactor record.
