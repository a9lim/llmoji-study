"""Re-score existing v3 sidecars with the extension probes.

No new generations. Loads the v3 JSONL (per `$LLMOJI_MODEL`), opens
each row's hidden-state sidecar, and scores every probe that's
loaded in the saklas session against saved h_first / h_last / h_mean
snapshots via saklas's own `score_single_token` path. Writes
new dict-keyed fields:

  extension_probe_scores_t0      {probe_name: score} from h_first
  extension_probe_scores_tlast   {probe_name: score} from h_last
  extension_probe_means          {probe_name: score} from h_mean

Each dict EXCLUDES the five core PROBES (which are already on the
row); it INCLUDES every additional probe the session bootstrapped.
On a clean install that's just the three packs registered by
`probe_extensions.py` (powerful.powerless / surprised.unsurprised /
disgusted.accepting). On a9's machine (and anyone with a working
saklas repo), the user-side ~/.saklas/vectors/default/ also has
`fearful.unflinching`, `curious.disinterested`, and
`individualist.collectivist` from a previously-installed saklas
version that shipped them — those have been auto-bootstrapping with
every recent run because they're tagged `affect`. The rescore picks
them up automatically; the column set is therefore machine-
dependent, but each row records exactly which probes contributed,
so downstream analysis can subset.

`fearful.unflinching` in particular is the cleanest direct test of
the v3 follow-on anger/fear question — it's a fear probe extracted
on the same model under the same protocol as happy.sad, and the
v3 sidecars already contain enough information to score it.

The existing `probe_scores_t0` / `probe_scores_tlast` lists and
`probe_means` dict are unchanged — the extension is strictly
additive so existing analysis scripts keep working.

Output: rewrites the JSONL in place, atomically (via tempfile +
rename). Skips rows that already have all three extension fields
populated UNLESS `--force` is passed; with `--force`, scores every
row (useful when new probes get added since the last rescore).

Prereqs:
  1. `python scripts/26_register_extension_probes.py` — done once,
     materializes our three packs and extracts the contrastive
     vectors.
  2. v3 sidecars exist under data/hidden/<experiment>/<uuid>.npz.

Caveat: saklas's per-token aggregation under `stateless=True`
collapses `per_generation` to a single value (see CLAUDE.md gotcha),
so the original capture wrote real per-token scores via
`session.last_per_token_scores`. The post-hoc rescore here doesn't
have access to per-token hidden states beyond the three aggregates
— `extension_probe_scores_t0` is therefore from the h_first
snapshot (state at first generated token), not strictly the same as
the original `probe_scores_t0` which came from per-token scoring at
t=0. In practice these match closely for the t=0 case (h_first IS
the t=0 hidden state); `tlast` similarly matches because h_last is
the final-token hidden state. `means` is a fresh path (scored from
h_mean, not aggregated from per-token scores) — flag when
interpreting results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from saklas import SaklasSession

from llmoji_study.config import DATA_DIR, PROBE_CATEGORIES, PROBES, current_model
from llmoji_study.hidden_state_analysis import recompute_probe_scores
from llmoji_study.hidden_state_io import hidden_state_path, load_hidden_states
from llmoji_study.probe_extensions import (
    extension_probe_names,
    register_extension_probes,
)


CORE_PROBES = set(PROBES)
T0_FIELD = "extension_probe_scores_t0"
TLAST_FIELD = "extension_probe_scores_tlast"
MEAN_FIELD = "extension_probe_means"


def _resolve_extension_names(session: Any) -> list[str]:
    """Probes the session has loaded that aren't part of the core
    PROBES set. Order: explicit registered extensions first, then any
    other auto-discovered probes alphabetically."""
    loaded: set[str] = set()
    monitor = getattr(session, "_monitor", None)
    if monitor is not None:
        # `probe_names` is a property -> list[str]; `profiles` is dict.
        names = getattr(monitor, "probe_names", None)
        if isinstance(names, list):
            loaded = set(names)
        else:
            profiles = getattr(monitor, "profiles", None)
            if isinstance(profiles, dict):
                loaded = set(profiles.keys())

    if not loaded:
        # Fallback: trust the explicit registration list. This loses
        # auto-discovered probes but at least keeps the explicit set.
        loaded = set(extension_probe_names()) | CORE_PROBES

    explicit = [p for p in extension_probe_names() if p in loaded]
    discovered = sorted(loaded - CORE_PROBES - set(explicit))
    return explicit + discovered


def _row_already_done(row: dict[str, Any], names: list[str]) -> bool:
    for field in (T0_FIELD, TLAST_FIELD, MEAN_FIELD):
        existing = row.get(field) or {}
        if not all(name in existing for name in names):
            return False
    return True


def _score_one(
    session: Any, sidecar: Path, ext_names: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Score one sidecar's three snapshots and return three name->score
    dicts filtered to extension probes only."""
    capture = load_hidden_states(sidecar, full_trace=False)

    def _ext_only(scores: dict[str, float]) -> dict[str, float]:
        return {name: float(scores[name]) for name in ext_names if name in scores}

    t0_scores = recompute_probe_scores(capture, session, which="h_first")
    tlast_scores = recompute_probe_scores(capture, session, which="h_last")
    mean_scores = recompute_probe_scores(capture, session, which="h_mean")
    return (
        _ext_only(t0_scores),
        _ext_only(tlast_scores),
        _ext_only(mean_scores),
    )


def _parse_args(argv: list[str]) -> tuple[bool, Path | None, str | None]:
    """Parse `--force`, `--jsonl <path>`, `--experiment <name>` from argv.
    Returns (force, override_jsonl, override_experiment).
    Used by introspection-pilot data + any other off-main-run JSONL.
    """
    force = "--force" in argv
    jsonl_override: Path | None = None
    exp_override: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--jsonl" and i + 1 < len(argv):
            jsonl_override = Path(argv[i + 1])
            i += 2
            continue
        if a == "--experiment" and i + 1 < len(argv):
            exp_override = argv[i + 1]
            i += 2
            continue
        i += 1
    return force, jsonl_override, exp_override


def main() -> None:
    force, jsonl_override, exp_override = _parse_args(sys.argv[1:])

    M = current_model()
    jsonl = jsonl_override if jsonl_override is not None else M.emotional_data_path
    experiment = exp_override if exp_override is not None else M.experiment
    if not jsonl.exists():
        print(f"ERR: {jsonl} does not exist; nothing to rescore")
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"model: {M.short_name}")
    print(f"jsonl: {jsonl}")
    print(f"total rows: {len(rows)}")

    # Make sure our packs are registered (idempotent) before saklas
    # bootstraps. If 26 hasn't been run yet this becomes the first-
    # time extraction path — same effect, slightly slower first call.
    register_extension_probes()

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=PROBE_CATEGORIES,
    ) as session:
        print(f"  loaded in {time.time() - t_load:.1f}s")

        ext_names = _resolve_extension_names(session)
        registered = set(extension_probe_names())
        explicit = [n for n in ext_names if n in registered]
        autodiscovered = [n for n in ext_names if n not in registered]
        print(f"  extension probes (explicit, this repo): {explicit}")
        if autodiscovered:
            print(f"  extension probes (auto-discovered from ~/.saklas): "
                  f"{autodiscovered}")
        print(f"  total extension probes: {len(ext_names)}")
        if not ext_names:
            print("ERR: no extension probes resolvable from the session.")
            sys.exit(1)

        needs_rescore = [
            r for r in rows
            if "error" not in r
            and r.get("row_uuid")
            and (force or not _row_already_done(r, ext_names))
        ]
        print(f"  rows needing rescore: {len(needs_rescore)}")
        if not needs_rescore:
            print("  nothing to do.")
            return

        print("  rescoring sidecars ...")
        # Index original rows by row_uuid for cheap merge.
        by_uuid = {r["row_uuid"]: r for r in rows if r.get("row_uuid")}
        n_done = 0
        n_missing = 0
        n_err = 0
        t_start = time.time()
        for r in needs_rescore:
            uuid = r["row_uuid"]
            sidecar = hidden_state_path(DATA_DIR, experiment, uuid)
            if not sidecar.exists():
                n_missing += 1
                continue
            try:
                t0_scores, tlast_scores, mean_scores = _score_one(
                    session, sidecar, ext_names,
                )
            except Exception as e:
                n_err += 1
                print(f"  ERR uuid={uuid[:8]}: {e!r}")
                continue
            target = by_uuid[uuid]
            target.setdefault(T0_FIELD, {}).update(t0_scores)
            target.setdefault(TLAST_FIELD, {}).update(tlast_scores)
            target.setdefault(MEAN_FIELD, {}).update(mean_scores)
            n_done += 1
            if n_done % 50 == 0:
                rate = n_done / (time.time() - t_start)
                print(f"    [{n_done}/{len(needs_rescore)}]  {rate:.1f} rows/s")

    print(f"\nrescored {n_done}; missing sidecars: {n_missing}; errors: {n_err}")

    # Atomic rewrite via tempfile + rename.
    tmp = jsonl.with_suffix(jsonl.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(jsonl)
    print(f"wrote {jsonl}")


if __name__ == "__main__":
    main()
