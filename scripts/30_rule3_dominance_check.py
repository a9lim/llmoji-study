"""Rule 3 (revised) verdict — HN-D vs HN-S dominance check.

Implements the rule-3-redesign decision rule from
``docs/2026-05-01-rule3-redesign.md``. Reads each model's v3 JSONL,
applies the registry HN split (HN→HN-D / HN-S, drops untagged HN
rows), and tests:

  Rule 3b — fear test:
      mean(fearful.unflinching | HN-S) − mean(... | HN-D) > 0
  on at least 2 of 3 aggregates (t0, tlast, mean) per model, with
  bootstrap 95% CI excluding zero on the same.

  Rule 3a — dominance test (DROPPED 2026-05-01, reported for record):
      mean(powerful.powerless | HN-D) − mean(... | HN-S) > 0

Composite verdict:
  - all 3 models PASS rule 3b → cross-model dominance representation
    confirmed via the fear axis
  - 2 / 3 PASS → directional confirmation but model-specific
    weakness; report which model and where
  - 1 / 3 or fewer → null finding; either tagging didn't separate
    the registers cleanly or PAD dominance isn't linearly readable
    in these models

Works on whatever data is currently on disk — partial-supp results
are interpretable. Per-model HN-D / HN-S row counts are printed so
the verdict is read in context.

Outputs:
  stdout  ASCII verdict block per model + composite
  tsv     data/rule3_dominance_check.tsv  (one row per model × aggregate × probe)
  md      figures/local/cross_model/rule3_dominance_check.md
          (markdown summary for inclusion in writeups)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji_study.config import MODEL_REGISTRY
from llmoji_study.emotional_analysis import _hn_split_map

MODELS = ("gemma", "qwen", "ministral")
AGGREGATES = ("t0", "tlast", "mean")
FIELD_BY_AGG = {
    "t0":    "extension_probe_scores_t0",
    "tlast": "extension_probe_scores_tlast",
    "mean":  "extension_probe_means",
}
PROBES = {
    "rule3b": "fearful.unflinching",   # active gating probe
    "rule3a": "powerful.powerless",    # dropped — reported for record
}
N_BOOT = 2000
RNG_SEED = 0


# -------------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------------


def _load_hn_rows(model: str) -> tuple[list[dict], list[dict]]:
    """Return (HN-D rows, HN-S rows) for a model. Drops untagged-HN
    (registry pad_dominance == 0) and any rows missing extension scores
    on the active probes."""
    M = MODEL_REGISTRY[model]
    if not M.emotional_data_path.exists():
        return [], []
    rows = [json.loads(l) for l in open(M.emotional_data_path) if l.strip()]
    hn_split = _hn_split_map()
    d_rows: list[dict] = []
    s_rows: list[dict] = []
    for r in rows:
        if "error" in r:
            continue
        if r.get("prompt_id", "")[:2].upper() != "HN":
            continue
        tag = hn_split.get(r["prompt_id"])
        if tag is None:
            continue  # untagged borderline; drop from rule 3
        # Need at least the gating probe on at least one aggregate.
        has_any = any(
            PROBES["rule3b"] in (r.get(field) or {})
            for field in FIELD_BY_AGG.values()
        )
        if not has_any:
            continue
        if tag == "HN-D":
            d_rows.append(r)
        else:
            s_rows.append(r)
    return d_rows, s_rows


def _values(rows: list[dict], probe: str, agg: str) -> np.ndarray:
    """Pull the per-row scalar for (probe, aggregate) from a row list,
    skipping rows missing that field."""
    field = FIELD_BY_AGG[agg]
    out: list[float] = []
    for r in rows:
        v = (r.get(field) or {}).get(probe)
        if v is None:
            continue
        out.append(float(v))
    return np.asarray(out, dtype=np.float64)


# -------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------


def _bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean(a) − mean(b) with 95% CI. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
    return float(a.mean() - b.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples. Pooled SD."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n_a, n_b = len(a), len(b)
    s2_pooled = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    if s2_pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(s2_pooled))


# -------------------------------------------------------------------------
# Per-model evaluation
# -------------------------------------------------------------------------


def evaluate_model(model: str) -> pd.DataFrame:
    """One row per (rule, aggregate) for one model. Columns include
    point estimate, CI bounds, n_d / n_s, Cohen's d, directional flag,
    CI-excludes-zero flag."""
    d_rows, s_rows = _load_hn_rows(model)
    out_rows: list[dict] = []
    for rule_key, probe in PROBES.items():
        for agg in AGGREGATES:
            d_vals = _values(d_rows, probe, agg)
            s_vals = _values(s_rows, probe, agg)
            if len(d_vals) < 5 or len(s_vals) < 5:
                continue
            # Direction: rule3b expects HN-S > HN-D; rule3a expected HN-D > HN-S.
            if rule_key == "rule3b":
                point, lo, hi = _bootstrap_diff_ci(s_vals, d_vals)
                d = _cohens_d(s_vals, d_vals)
                expected_sign = "+ (S > D)"
            else:
                point, lo, hi = _bootstrap_diff_ci(d_vals, s_vals)
                d = _cohens_d(d_vals, s_vals)
                expected_sign = "+ (D > S)"
            directional = point > 0
            ci_excl_zero = (lo > 0) or (hi < 0)
            out_rows.append({
                "model": model,
                "rule": rule_key,
                "probe": probe,
                "aggregate": agg,
                "n_d": len(d_vals),
                "n_s": len(s_vals),
                "expected": expected_sign,
                "diff": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "cohens_d": d,
                "directional": directional,
                "ci_excludes_zero": ci_excl_zero,
            })
    return pd.DataFrame(out_rows)


def per_model_verdict(df: pd.DataFrame, rule: str = "rule3b") -> str:
    """PASS / mid / fail for a single model on a single rule, given the
    per-aggregate rows in df. Threshold (revised 2026-05-01): direction
    correct AND CI excludes zero on at least 2 of 3 aggregates."""
    sub = df[df["rule"] == rule]
    if len(sub) == 0:
        return "n/a"
    aggregates_pass = int((sub["directional"] & sub["ci_excludes_zero"]).sum())
    aggregates_directional = int(sub["directional"].sum())
    if aggregates_pass >= 2:
        return "PASS"
    if aggregates_directional >= 2:
        return "mid (directional but CI ambiguous)"
    if aggregates_directional == 0:
        return "fail (wrong direction)"
    return "fail (mixed)"


# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------


def _format_block(df: pd.DataFrame, model: str) -> str:
    sub = df[df["model"] == model]
    if len(sub) == 0:
        return f"=== {model} ===\n  no data\n"
    lines = [f"=== {model} ==="]
    lines.append(f"  HN-D / HN-S row counts: "
                 f"D={int(sub['n_d'].iloc[0])}, S={int(sub['n_s'].iloc[0])}")
    for rule_key in ("rule3b", "rule3a"):
        rs = sub[sub["rule"] == rule_key]
        if len(rs) == 0:
            continue
        probe = rs["probe"].iloc[0]
        rule_tag = ("(active gating)" if rule_key == "rule3b"
                    else "(DROPPED — record only)")
        lines.append(f"\n  {rule_key}: {probe}  {rule_tag}")
        lines.append(f"    {'agg':6s} {'diff':>10s}  {'95% CI':>22s}  {'cohen_d':>8s}  flags")
        for _, r in rs.iterrows():
            flags = []
            if r["directional"]: flags.append("dir✓")
            else:                flags.append("dir✗")
            if r["ci_excludes_zero"]: flags.append("CI-excl-0")
            else:                     flags.append("CI∋0")
            lines.append(
                f"    {r['aggregate']:6s} {r['diff']:+10.4f}  "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                f"{r['cohens_d']:+8.3f}  {' '.join(flags)}"
            )
        verdict = per_model_verdict(sub, rule_key)
        lines.append(f"    verdict: {verdict}")
    return "\n".join(lines) + "\n"


def _format_composite(df: pd.DataFrame) -> str:
    verdicts_3b = {m: per_model_verdict(df[df.model == m], "rule3b") for m in MODELS}
    n_pass = sum(1 for v in verdicts_3b.values() if v == "PASS")
    n_mid = sum(1 for v in verdicts_3b.values() if v.startswith("mid"))
    n_fail = sum(1 for v in verdicts_3b.values() if v.startswith("fail"))
    n_na = sum(1 for v in verdicts_3b.values() if v == "n/a")

    if n_pass == 3:
        outcome = "RULE 3b CONFIRMED — directional + CI on all 3 models"
    elif n_pass + n_mid == 3 and n_pass >= 2:
        outcome = (f"RULE 3b PARTIAL — {n_pass} clean PASS, {n_mid} directional "
                   "but CI ambiguous; investigate the weak case")
    elif n_pass >= 1:
        outcome = (f"RULE 3b WEAK — {n_pass} PASS, {n_mid} mid, {n_fail} fail; "
                   "redesign tagging or accept reduced-confidence rule")
    else:
        outcome = "RULE 3b NULL — no model cleanly passes; redesign or accept null"
    if n_na:
        outcome += f"  ({n_na} model(s) without data)"

    lines = [f"=== composite ===", ""]
    for m in MODELS:
        lines.append(f"  {m:10s}  rule3b: {verdicts_3b[m]}")
    lines.append("")
    lines.append(f"  outcome: {outcome}")
    return "\n".join(lines) + "\n"


def _format_markdown(df: pd.DataFrame) -> str:
    """Markdown summary for inclusion in writeups (findings.md etc.)."""
    lines = [
        "# Rule 3 dominance check",
        "",
        ("Auto-generated by `scripts/30_rule3_dominance_check.py`. Reads "
         "current v3 JSONLs, applies registry HN split, computes per-model "
         "verdicts on the revised rule 3b. See "
         "`docs/2026-05-01-rule3-redesign.md` for the design."),
        "",
    ]
    for m in MODELS:
        sub = df[df["model"] == m]
        if len(sub) == 0:
            lines += [f"## {m}", "", "no data on disk", ""]
            continue
        lines += [
            f"## {m}",
            "",
            f"HN-D / HN-S row counts: D={int(sub['n_d'].iloc[0])}, "
            f"S={int(sub['n_s'].iloc[0])}",
            "",
        ]
        for rule_key in ("rule3b", "rule3a"):
            rs = sub[sub["rule"] == rule_key]
            if len(rs) == 0:
                continue
            probe = rs["probe"].iloc[0]
            tag = ("(active gating)" if rule_key == "rule3b"
                   else "(DROPPED — record only)")
            lines += [f"### {rule_key} — `{probe}` {tag}", ""]
            lines.append(
                "| aggregate | diff | 95% CI | Cohen's d | directional | CI excludes 0 |"
            )
            lines.append("| --- | ---: | --- | ---: | :---: | :---: |")
            for _, r in rs.iterrows():
                lines.append(
                    f"| {r['aggregate']} | {r['diff']:+.4f} | "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                    f"{r['cohens_d']:+.3f} | "
                    f"{'✓' if r['directional'] else '✗'} | "
                    f"{'✓' if r['ci_excludes_zero'] else '✗'} |"
                )
            lines.append(f"\n**verdict:** {per_model_verdict(sub, rule_key)}\n")
    lines += ["## composite", ""]
    verdicts_3b = {m: per_model_verdict(df[df.model == m], "rule3b") for m in MODELS}
    for m in MODELS:
        lines.append(f"- **{m}:** {verdicts_3b[m]}")
    n_pass = sum(1 for v in verdicts_3b.values() if v == "PASS")
    if n_pass == 3:
        outcome = ("**RULE 3b CONFIRMED** — directional + CI excludes zero on all "
                   "three models. PAD dominance representation reads cleanly via "
                   "`fearful.unflinching` against the registry HN-D / HN-S split.")
    elif n_pass >= 2:
        outcome = (f"**RULE 3b PARTIAL** — {n_pass} of 3 models cleanly pass; "
                   "investigate the weaker model.")
    elif n_pass >= 1:
        outcome = "**RULE 3b WEAK** — only one model cleanly passes."
    else:
        outcome = "**RULE 3b NULL** — no clean PASS."
    lines += ["", outcome, ""]
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    print("Rule 3 (revised) — HN-D vs HN-S dominance check\n")
    frames = []
    for m in MODELS:
        frames.append(evaluate_model(m))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df) == 0:
        print("no data on disk for any model; nothing to evaluate")
        return

    for m in MODELS:
        print(_format_block(df, m))
    print(_format_composite(df))

    tsv_path = repo / "data" / "rule3_dominance_check.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")

    md_path = repo / "figures" / "local" / "cross_model" / "rule3_dominance_check.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_format_markdown(df))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
