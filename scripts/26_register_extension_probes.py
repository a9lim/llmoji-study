"""Register the v3 extension probes into saklas's default namespace and
extract per-model contrastive vectors.

Run once per model after pulling the repo. Idempotent — repeated runs
short-circuit on the per-pack hash check (statements unchanged → no
work, no re-extract). Respects ``$LLMOJI_MODEL`` so the contrastive
extraction lands at the right per-model cache path.

The extension is three concept packs (`powerful.powerless`,
`surprised.unsurprised`, `disgusted.accepting`) — see
`llmoji_study/probe_extensions.py` for the rationale and the on-disk
sources at `llmoji_study/probe_packs/<name>/`. All three are tagged
`affect`, so the existing `PROBE_CATEGORIES` setting picks them up
automatically once they're on disk under
``~/.saklas/vectors/default/``.

What this script does NOT do: re-score existing v3 sidecars with the
new probes. That's `scripts/27_v3_extension_probe_rescore.py`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession

from llmoji_study.config import PROBE_CATEGORIES, current_model
from llmoji_study.probe_extensions import (
    EXTENSION_PROBES,
    extension_probe_names,
    register_extension_probes,
)


def main() -> None:
    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")

    print("\nstep 1: copying probe packs into ~/.saklas/vectors/default/")
    changes = register_extension_probes()
    for name, changed in changes.items():
        status = "registered/updated" if changed else "already up-to-date"
        print(f"  {name:30s} {status}")

    if not any(changes.values()):
        print("  (all packs were already in place; nothing to materialize)")

    print("\nstep 2: bootstrapping contrastive vectors via saklas")
    print(f"  loading {M.model_id} ...")
    t0 = time.time()
    # Loading the session triggers `bootstrap_probes` for every
    # concept under each PROBE_CATEGORIES tag. Newly-registered packs
    # have no per-model `*.safetensors`, so saklas falls into the
    # extract path — short forward passes over each pack's
    # statements.json (~80 examples, ~5-10s on M5 Max). Existing core
    # probes are already cached and skip extraction.
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=PROBE_CATEGORIES,
    ) as session:
        print(f"  session ready in {time.time() - t0:.1f}s")
        # Quick verification: every extension probe should appear in
        # the session's probe set. `probe_names` is a property on
        # `_monitor` (returns list[str]); `profiles` is the parallel
        # dict.
        loaded: set[str] = set()
        monitor = getattr(session, "_monitor", None)
        if monitor is not None:
            names = getattr(monitor, "probe_names", None)
            if isinstance(names, list):
                loaded = set(names)
            else:
                profiles = getattr(monitor, "profiles", None)
                if isinstance(profiles, dict):
                    loaded = set(profiles.keys())
        missing = [p for p in extension_probe_names() if p not in loaded]
        if missing:
            print(f"  WARNING: extension probes not loaded into session: {missing}")
            print(f"  loaded probes: {sorted(loaded)}")
            sys.exit(1)
        print(f"  verified: all {len(EXTENSION_PROBES)} extension probes "
              f"present in session.")

    print("\ndone. extension probes are extracted and cached. next step:")
    print("  python scripts/27_v3_extension_probe_rescore.py")


if __name__ == "__main__":
    main()
