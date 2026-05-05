"""Harness-side per-provider per-project eriskii axis means/stds,
read from the user's own ~/.claude + ~/.codex journals.

Why this is a side-script and not part of the canonical eriskii
pipeline: as of the 2026-04-27 HF-corpus refactor, the cross-
contributor corpus pools per-machine before upload, so per-row
``project_slug`` doesn't exist in the public dataset and the per-
project section of ``scripts/64_eriskii_replication.py`` was
removed. The journals on this machine still carry ``cwd`` though,
and the ``llmoji`` package's source adapters preserve it as
``project_slug``. This script reads those locally, splits by
provider (claude_code vs codex), embeds the ``assistant_text`` of
each emission with MiniLM (response-based, matching the pre-
refactor approach), projects onto the 21 eriskii axes, and renders
one project × axis heatmap per provider.

Outputs (per provider, where ``<p>`` ∈ {claude, codex}):
  data/harness/<p>/per_project_axes.tsv
  figures/harness/<p>/per_project_axes_mean.png
  figures/harness/<p>/per_project_axes_std.png

Usage:
  python scripts/65_per_project_axes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji.sources.journal import iter_journal

from llmoji_study.claude_faces import EMBED_MODEL
from llmoji_study.config import DATA_DIR, ERISKII_AXES, FIGURES_DIR
from llmoji_study.eriskii import compute_axis_vectors, project_onto_axes
from llmoji_study.eriskii_anchors import AXIS_ANCHORS

MIN_PER_PROJECT = 10
MIN_TEXT_LEN = 20

# (journal path, source token, short provider name used for output dir)
JOURNALS = [
    (Path.home() / ".claude" / "kaomoji-journal.jsonl", "claude_code", "claude"),
    (Path.home() / ".codex" / "kaomoji-journal.jsonl", "codex", "codex"),
]


def _load_emissions(path: Path, source: str) -> pd.DataFrame:
    rows = []
    if not path.exists():
        print(f"  skip {path} (missing)")
        return pd.DataFrame(rows)
    for sr in iter_journal(path, source=source):
        txt = (sr.assistant_text or "").strip()
        if len(txt) < MIN_TEXT_LEN:
            continue
        rows.append({
            "project_slug": sr.project_slug,
            "kaomoji": sr.first_word or sr.kaomoji,
            "assistant_text": txt,
        })
    print(f"  {path.name}: {len(rows)} usable rows")
    return pd.DataFrame(rows)


def _heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="project_slug", columns="axis", values=value_col)
    pivot = pivot.reindex(columns=ERISKII_AXES)
    # Sort projects by total n (kept on df), tied-broken by name.
    order = (
        df.groupby("project_slug")["n"].first().sort_values(ascending=False).index
    )
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(pivot) + 1.6), dpi=140)
    if value_col == "mean":
        vmax = float(np.nanmax(np.abs(pivot.values)))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ERISKII_AXES)))
    ax.set_xticklabels(ERISKII_AXES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [f"{p}  (n={int(df[df.project_slug == p]['n'].iloc[0])})" for p in pivot.index],
        fontsize=9,
    )
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(Path.cwd())}")


def _project_for_provider(
    embedder, rows: pd.DataFrame, axis_vectors: dict[str, np.ndarray]
) -> pd.DataFrame:
    counts = rows.project_slug.value_counts()
    keep = counts[counts >= MIN_PER_PROJECT].index.tolist()
    rows = rows[rows.project_slug.isin(keep)].reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame()
    print(f"  {len(keep)} projects clear n≥{MIN_PER_PROJECT}; "
          f"{len(rows)} emissions retained")

    E = embedder.encode(
        rows.assistant_text.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    E = np.asarray(E)
    P = project_onto_axes(E, axis_vectors, ERISKII_AXES)
    proj = pd.DataFrame(P, columns=ERISKII_AXES)
    proj["project_slug"] = rows.project_slug.values

    out_rows = []
    for slug, sub in proj.groupby("project_slug"):
        n = len(sub)
        for axis in ERISKII_AXES:
            v = sub[axis].values
            out_rows.append({
                "project_slug": slug,
                "axis": axis,
                "n": n,
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=0)),
            })
    return pd.DataFrame(out_rows)


def main() -> None:
    print(f"loading {EMBED_MODEL}...")
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        device = "cpu"
    print(f"  device={device}")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    print("computing axis vectors...")
    axis_vectors = compute_axis_vectors(embedder, AXIS_ANCHORS)

    for path, source, provider in JOURNALS:
        print(f"\n=== provider: {provider} ===")
        rows = _load_emissions(path, source)
        if rows.empty:
            print(f"  no rows for {provider}; skip")
            continue
        out = _project_for_provider(embedder, rows, axis_vectors)
        if out.empty:
            print(f"  no projects clear n≥{MIN_PER_PROJECT} for {provider}; skip figure")
            continue

        out_tsv = DATA_DIR / "harness" / provider / "per_project_axes.tsv"
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_tsv, sep="\t", index=False)
        print(f"  wrote {out_tsv.relative_to(Path.cwd())}")

        prov_dir = FIGURES_DIR / "harness" / provider
        _heatmap(
            out, "mean",
            f"{provider}: per-project eriskii axis means "
            f"(n≥{MIN_PER_PROJECT}, response embedding)",
            prov_dir / "per_project_axes_mean.png",
        )
        _heatmap(
            out, "std",
            f"{provider}: per-project eriskii axis std "
            f"(n≥{MIN_PER_PROJECT}, response embedding)",
            prov_dir / "per_project_axes_std.png",
        )

        print(f"\n  top / bottom axis per project ({provider}, mean):")
        for slug, sub in out.groupby("project_slug"):
            sub = sub.sort_values("mean")
            top = sub.iloc[-1]
            bot = sub.iloc[0]
            n = int(sub["n"].iloc[0])
            print(f"    {slug:25s} n={n:4d}  "
                  f"top={top.axis:>14s} ({top['mean']:+.3f})  "
                  f"bot={bot.axis:>14s} ({bot['mean']:+.3f})")


if __name__ == "__main__":
    main()
