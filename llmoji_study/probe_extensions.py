"""Project-side probe-pack extensions to saklas's bundled `default/` namespace.

Saklas ships 21 concept packs out of the box (see
`saklas/data/vectors/`). For the v3 follow-on work on anger/fear
disambiguation we added three more — `powerful.powerless` (PAD's
dominance axis, framed as felt agency / coping potential rather than
saklas's existing register-flavored `authoritative.submissive`),
`surprised.unsurprised` (Plutchik's surprise axis, missing from the
bundled set), and `disgusted.accepting` (Plutchik's disgust axis,
also missing). All three are tagged `affect` so they auto-pick-up
via the existing `PROBE_CATEGORIES` setting.

The packs live in this repo at `llmoji_study/probe_packs/<name>/`
(statements + scenarios committed; pack.json synthesized at register
time so on-disk hashes always match). The registration helper copies
them into `~/.saklas/vectors/default/` — the same location
`materialize_bundled()` uses for the shipped packs — without
overwriting any user-side state.

Idempotent: re-running is a no-op once the target packs are present
and their `files` hashes still match what's in this repo. If the
source statements change, re-running will detect the mismatch and
re-copy.

Usage:
    from llmoji_study.probe_extensions import register_extension_probes
    register_extension_probes()                     # before from_pretrained
    SaklasSession.from_pretrained(..., probes=PROBE_CATEGORIES)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from saklas.io.packs import (
    PackMetadata,
    hash_file,
    synthesize_pack_metadata,
)
from saklas.io.paths import vectors_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PACKS_DIR = REPO_ROOT / "llmoji_study" / "probe_packs"

# Per-pack metadata that is NOT in the on-disk source files. Hashes
# are computed at register time from the actual statements/scenarios
# files, not stored here.
EXTENSION_PROBES: dict[str, dict] = {
    "powerful.powerless": {
        "description": (
            "Bipolar axis: powerful (+) vs powerless (-). PAD-style dominance / "
            "felt agency / coping potential. Distinct from saklas's bundled "
            "authoritative.submissive, which captures commanding-vs-deferential "
            "register; this pack targets the felt internal-state dimension that "
            "PAD adds beyond Russell's V-A circumplex."
        ),
        "tags": ["affect"],
        "version": "1.0.0",
        "license": "CC-BY-SA-4.0",
        "recommended_alpha": 0.5,
    },
    "surprised.unsurprised": {
        "description": (
            "Bipolar axis: surprised (+) vs unsurprised (-). Plutchik's surprise / "
            "anticipation axis, framed as 'I did not see this coming' vs 'this is "
            "exactly what I expected'. Targets the appraisal-theoretic novelty "
            "dimension underrepresented by V-A circumplex models."
        ),
        "tags": ["affect"],
        "version": "1.0.0",
        "license": "CC-BY-SA-4.0",
        "recommended_alpha": 0.5,
    },
    "disgusted.accepting": {
        "description": (
            "Bipolar axis: disgusted (+) vs accepting (-). Plutchik's disgust axis. "
            "Covers bodily/sensory disgust, moral disgust, and contamination "
            "framings; the negative pole is composure / non-aversion rather than "
            "the antithetical 'delighted' to keep the contrast clean of valence "
            "load already covered by happy.sad."
        ),
        "tags": ["affect"],
        "version": "1.0.0",
        "license": "CC-BY-SA-4.0",
        "recommended_alpha": 0.5,
    },
}


def _source_dir(probe_name: str) -> Path:
    src = PROBE_PACKS_DIR / probe_name
    if not src.is_dir():
        raise FileNotFoundError(
            f"missing source pack at {src}; expected statements.json + scenarios.json"
        )
    if not (src / "statements.json").is_file():
        raise FileNotFoundError(f"missing {src / 'statements.json'}")
    return src


def _target_dir(probe_name: str) -> Path:
    return vectors_dir() / "default" / probe_name


def _needs_register(probe_name: str) -> bool:
    """True if the pack is missing or its source files have changed
    since the last register. Compared by sha256, not mtime — survives
    `cp -p` / git checkout / fresh clone without spurious re-extracts.
    """
    src = _source_dir(probe_name)
    tgt = _target_dir(probe_name)
    pack_json = tgt / "pack.json"
    if not pack_json.is_file():
        return True
    try:
        with open(pack_json) as f:
            recorded = json.load(f).get("files", {})
    except (OSError, json.JSONDecodeError):
        return True
    for fname in ("statements.json", "scenarios.json"):
        if not (src / fname).is_file():
            continue
        recorded_hash = recorded.get(fname)
        if recorded_hash != hash_file(src / fname):
            return True
        if not (tgt / fname).is_file() or hash_file(tgt / fname) != recorded_hash:
            return True
    return False


def _register_one(probe_name: str, *, force: bool = False) -> bool:
    """Materialize a single pack into ~/.saklas/vectors/default/<name>/.

    Returns True if any work was done, False if already up-to-date.
    Caller should NOT delete the target's per-model `*.safetensors` /
    `*.json` sidecars on a refresh — saklas detects the statements
    hash mismatch and re-extracts on its own.
    """
    if not force and not _needs_register(probe_name):
        return False
    src = _source_dir(probe_name)
    tgt = _target_dir(probe_name)
    tgt.mkdir(parents=True, exist_ok=True)

    # Copy statements + scenarios. Don't touch existing tensor sidecars
    # in the target — they're per-model extraction outputs, expensive
    # to recompute, and saklas's bootstrap will mark them stale via
    # statements_sha256 if the new statements differ.
    for fname in ("statements.json", "scenarios.json"):
        s = src / fname
        if not s.is_file():
            continue
        shutil.copy2(s, tgt / fname)

    # Synthesize pack.json with hashes drawn from what's now in tgt.
    meta_extra = EXTENSION_PROBES[probe_name]
    meta = synthesize_pack_metadata(
        name=probe_name,
        source="llmoji-study-extension",
        pack_dir=tgt,
        description=meta_extra["description"],
        tags=tuple(meta_extra["tags"]),
        version=meta_extra["version"],
        license=meta_extra["license"],
        recommended_alpha=meta_extra["recommended_alpha"],
    )
    PackMetadata(
        name=meta.name,
        description=meta.description,
        long_description=meta.long_description,
        version=meta.version,
        license=meta.license,
        tags=meta.tags,
        recommended_alpha=meta.recommended_alpha,
        source=meta.source,
        files=meta.files,
        format_version=meta.format_version,
    ).write(tgt)
    return True


def register_extension_probes(*, force: bool = False) -> dict[str, bool]:
    """Register every extension pack defined in EXTENSION_PROBES.

    Returns ``{probe_name: changed}``; ``changed=False`` means the
    target was already up-to-date. Cheap to call on every saklas
    session start; returns instantly when no packs need refreshing.
    """
    return {
        name: _register_one(name, force=force)
        for name in EXTENSION_PROBES
    }


def extension_probe_names() -> list[str]:
    """Names of every extension probe (in registration order)."""
    return list(EXTENSION_PROBES)
