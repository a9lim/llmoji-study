"""Prompt dataclass shared across pilot scripts.

The v1/v2 PROMPTS list (30 valence-tagged prompts) was removed
2026-05-04 along with the gated v1/v2 pilot scripts (01, 02). Only the
dataclass remains — used by 03 / 32 / 43 / 50 for typing and
construction of v3-style prompts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    valence: int
    text: str
