# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Re-extract ``first_word`` on the saved pilot rows under llmoji v2.

The v0 / v1 pilot (300 rows, ``data/claude_disclosure_pilot.jsonl``) was
analyzed under llmoji v1.3.0's extractor, which rejected wing-hand
``\\(^o^)/`` patterns as markdown-escape artifacts. Several framed-HP
"non-emission" rows were actually wing-hand kaomoji the v1 extractor
missed.

Under llmoji v2.0.0 (``KAOMOJI_START_CHARS`` adds ``\\``, ``⊂``, ``✧``;
``is_kaomoji_candidate`` allows backslash at position 0), those
patterns extract correctly. This script re-extracts ``first_word`` on
each row's saved ``response_text`` and rewrites the JSONL.

Original ``first_word`` values are preserved as ``first_word_v1`` for
audit. Idempotent: re-running on already-migrated rows is a no-op.

Also re-writes the per-(category × condition) summary TSV at
``data/claude_disclosure_pilot_summary.tsv``.

Usage:
  python scripts/harness/21_reextract_pilot_first_word.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji import __version__ as llmoji_version
from llmoji.taxonomy import canonicalize_kaomoji, extract

from llmoji_study.config import DATA_DIR


JSONL_PATH = DATA_DIR / "claude_disclosure_pilot.jsonl"
SUMMARY_PATH = DATA_DIR / "claude_disclosure_pilot_summary.tsv"


def main() -> None:
    if not JSONL_PATH.exists():
        raise SystemExit(f"missing {JSONL_PATH}")
    print(f"llmoji version: {llmoji_version}")

    rows: list[dict] = []
    with JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n_total = len(rows)
    n_changed = 0
    n_recovered = 0  # rows where v1 had '' but v2 has a kaomoji
    n_already_migrated = 0
    for r in rows:
        if "error" in r:
            continue
        text = r.get("response_text", "")
        old_fw = r.get("first_word", "")
        if "first_word_v1" in r:
            n_already_migrated += 1
            # Already migrated; re-extract anyway (idempotent under same
            # llmoji version) but compare against the v1 value, not the
            # current first_word which is already v2.
            v1_fw = r["first_word_v1"]
        else:
            v1_fw = old_fw
        new_fw = extract(text).first_word
        if new_fw != old_fw:
            n_changed += 1
            r["first_word_v1"] = v1_fw
            r["first_word"] = new_fw
            if v1_fw == "" and new_fw != "":
                n_recovered += 1
        elif "first_word_v1" not in r:
            # Same value but mark v1 lineage explicitly so future
            # re-runs know this row was checked under v2.
            r["first_word_v1"] = v1_fw

    print(f"rows: {n_total}")
    print(f"  already migrated: {n_already_migrated}")
    print(f"  changed first_word: {n_changed}")
    print(f"  recovered (v1='' → v2 kaomoji): {n_recovered}")

    JSONL_PATH.write_text("\n".join(
        json.dumps(r, ensure_ascii=False) for r in rows
    ) + "\n")
    print(f"rewrote {JSONL_PATH}")

    # Show the recovered rows for sanity.
    if n_recovered > 0:
        print(f"\nrecovered rows ({n_recovered}):")
        for r in rows:
            if r.get("first_word_v1", "") == "" and r.get("first_word", ""):
                print(f"  {r['quadrant']:<4} {r['condition']:<6} s={r['seed']:<2} "
                      f"v1='' → v2={r['first_word']!r}  "
                      f"canon={canonicalize_kaomoji(r['first_word'])!r}")

    # Re-summarize.
    from collections import Counter

    QUADS = ("HP", "LP", "NB")
    CONDS = ("direct", "framed")

    by_cat_cond: dict[tuple[str, str], Counter] = {}
    for r in rows:
        if "error" in r:
            continue
        cat = r["quadrant"]
        cond = r["condition"]
        fw = canonicalize_kaomoji(r.get("first_word") or "") or ""
        by_cat_cond.setdefault((cat, cond), Counter())[fw] += 1

    import math

    def _jsd(a: Counter, b: Counter) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        n_a, n_b = sum(a.values()), sum(b.values())
        if n_a == 0 or n_b == 0:
            return 0.0
        p = {k: a.get(k, 0) / n_a for k in keys}
        q = {k: b.get(k, 0) / n_b for k in keys}
        m = {k: 0.5 * (p[k] + q[k]) for k in keys}

        def kl(x, y):
            return sum(x[k] * math.log2(x[k] / y[k])
                       for k in x if x[k] > 0 and y[k] > 0)

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    out_lines: list[str] = []
    out_lines.append("\t".join([
        "category", "condition", "n", "n_unique_faces", "non_emission_rate",
        "modal_face", "modal_count", "modal_share",
        "jsd_vs_other_cond_in_bits",
        "modal_agrees_with_other_cond",
    ]))
    print(f"\nper (category × condition) summary (post-v2 re-extraction):")
    print(f"  {'cat':<4} {'cond':<6} {'n':>3} {'unique':>6} {'non-emit':>9} "
          f"{'modal':>15} {'count':>5} {'share':>6} {'JSD':>6} {'modal-agree':>11}")
    for cat in QUADS:
        for cond in CONDS:
            counts = by_cat_cond.get((cat, cond), Counter())
            n = sum(counts.values())
            n_emit = sum(c for f, c in counts.items() if f)
            non_emit = (n - n_emit) / n if n > 0 else 0.0
            unique = sum(1 for _, c in counts.items() if c > 0)
            modal_face, modal_count = (counts.most_common(1)[0]
                                       if counts else ("", 0))
            modal_share = (modal_count / n) if n > 0 else 0.0
            other = "framed" if cond == "direct" else "direct"
            other_counts = by_cat_cond.get((cat, other), Counter())
            jsd = _jsd(counts, other_counts)
            other_modal = (other_counts.most_common(1)[0][0]
                           if other_counts else "")
            modal_agree = (modal_face == other_modal) if (modal_face and other_modal) else None
            out_lines.append("\t".join([
                cat, cond, str(n), str(unique), f"{non_emit:.3f}",
                modal_face, str(modal_count), f"{modal_share:.3f}",
                f"{jsd:.4f}",
                "" if modal_agree is None else ("True" if modal_agree else "False"),
            ]))
            print(f"  {cat:<4} {cond:<6} {n:>3} {unique:>6} {non_emit:>9.3f} "
                  f"{modal_face!r:>15} {modal_count:>5} {modal_share:>6.3f} "
                  f"{jsd:>6.4f} {str(modal_agree):>11}")
    SUMMARY_PATH.write_text("\n".join(out_lines) + "\n")
    print(f"\nrewrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
