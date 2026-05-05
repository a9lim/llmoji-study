"""Face_likelihood layout discovery — shared by scripts 51/52/53/54.

Post-2026-05-05 layout:
  data/local/<model>/face_likelihood[_<variant>]_summary.tsv
  data/local/<model>/face_likelihood[_<variant>].parquet
  data/harness/face_likelihood_<encoder>[_pilot]_summary.tsv

Encoder name in the returned dicts:
  - canonical local file → ``<model>``
  - local sub-config variant (e.g. rinna_jp_3_6b's _jp/_jpfull/_jpfull30) →
    ``<model>_<variant>``
  - local pilot-only file → ``<model>`` (resolved against the full file
    when both exist; ``prefer_full`` flag controls the tiebreak)
  - harness file → ``<encoder>`` (e.g. ``haiku``, ``opus``)

The variant token ``pilot`` is reserved as the pilot-flag — never an encoder
identity. All other variants survive into the encoder name.
"""
from __future__ import annotations

import re
from pathlib import Path

from llmoji_study.config import DATA_DIR


_LOCAL_SUMMARY_RE = re.compile(
    r"^face_likelihood(?:_(?P<variant>.+?))?_summary\.tsv$"
)
_LOCAL_PARQUET_RE = re.compile(
    r"^face_likelihood(?:_(?P<variant>.+?))?\.parquet$"
)
_HARNESS_SUMMARY_RE = re.compile(
    r"^face_likelihood_(?P<enc>.+?)(?P<pilot>_pilot)?_summary\.tsv$"
)
_HARNESS_PARQUET_RE = re.compile(
    r"^face_likelihood_(?P<enc>.+?)(?P<pilot>_pilot)?\.parquet$"
)

# Excluded encoder-name prefixes — historical aliases that no longer
# correspond to a real encoder run.
_EXCLUDE_PREFIXES = ("vote_", "gemma_vs_qwen", "gemma-")


def _classify_local(path: Path, regex: re.Pattern) -> tuple[str, bool] | None:
    """Map a local file path → (encoder_name, is_pilot) or None to skip."""
    m = regex.match(path.name)
    if not m:
        return None
    model = path.parent.name
    variant = m.group("variant") or ""
    is_pilot = (variant == "pilot")
    if is_pilot or not variant:
        encoder = model
    else:
        encoder = f"{model}_{variant}"
    if encoder.startswith(_EXCLUDE_PREFIXES):
        return None
    return encoder, is_pilot


def _classify_harness(path: Path, regex: re.Pattern) -> tuple[str, bool] | None:
    m = regex.match(path.name)
    if not m:
        return None
    encoder = m.group("enc")
    is_pilot = bool(m.group("pilot"))
    if encoder.startswith(_EXCLUDE_PREFIXES):
        return None
    return encoder, is_pilot


def _resolve(found: dict[str, dict[bool, str]], prefer_full: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    order = [False, True] if prefer_full else [True, False]
    for name, by_pilot in found.items():
        for is_pilot in order:
            if is_pilot in by_pilot:
                out[name] = by_pilot[is_pilot]
                break
    return out


def discover_summaries(prefer_full: bool) -> dict[str, str]:
    """{encoder_name: path/to/face_likelihood*_summary.tsv} for every encoder."""
    found: dict[str, dict[bool, str]] = {}
    for p in sorted((DATA_DIR / "local").glob("*/face_likelihood*_summary.tsv")):
        cls = _classify_local(p, _LOCAL_SUMMARY_RE)
        if cls is None:
            continue
        encoder, is_pilot = cls
        found.setdefault(encoder, {})[is_pilot] = str(p)
    for p in sorted((DATA_DIR / "harness").glob("face_likelihood_*_summary.tsv")):
        cls = _classify_harness(p, _HARNESS_SUMMARY_RE)
        if cls is None:
            continue
        encoder, is_pilot = cls
        found.setdefault(encoder, {})[is_pilot] = str(p)
    return _resolve(found, prefer_full)


def discover_parquets(prefer_full: bool) -> dict[str, str]:
    """{encoder_name: path/to/face_likelihood*.parquet}.

    Harness encoders (haiku, opus) only have summary TSVs — they
    won't appear here. Local encoders all produce parquets.
    """
    found: dict[str, dict[bool, str]] = {}
    for p in sorted((DATA_DIR / "local").glob("*/face_likelihood*.parquet")):
        cls = _classify_local(p, _LOCAL_PARQUET_RE)
        if cls is None:
            continue
        encoder, is_pilot = cls
        found.setdefault(encoder, {})[is_pilot] = str(p)
    for p in sorted((DATA_DIR / "harness").glob("face_likelihood_*.parquet")):
        cls = _classify_harness(p, _HARNESS_PARQUET_RE)
        if cls is None:
            continue
        encoder, is_pilot = cls
        found.setdefault(encoder, {})[is_pilot] = str(p)
    return _resolve(found, prefer_full)


def summary_path_for(encoder: str, prefer_full: bool = True) -> Path | None:
    """Resolve a single encoder name → its summary TSV path, or None.

    Used by script 56 (ensemble predict) which iterates over a fixed
    list of model names rather than auto-discovering. Honors the same
    prefer_full tiebreak as ``discover_summaries``.
    """
    discovered = discover_summaries(prefer_full)
    p = discovered.get(encoder)
    return Path(p) if p else None
