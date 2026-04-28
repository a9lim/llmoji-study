"""21 anchored semantic axes for the eriskii-replication pipeline.

Locked anchor pairs from the eriskii-replication design doc
(``docs/superpowers/specs/2026-04-24-eriskii-replication-design.md``,
since archived). Each axis is ``(positive_anchor, negative_anchor)``;
positive direction corresponds to the axis name (e.g. high
``"warmth"`` projection = warmer).

Multi-word anchor phrases by design — the embedding catches the
concept rather than a single-word idiosyncrasy. All 21 axes from
eriskii.net (note: ``"wryness"`` is the eriskii spelling, single n).

This module stays research-side. The description / synthesis prompts
that produce the text projected onto these axes moved to
``llmoji.synth_prompts`` in the v1.0 package split (renamed from
``llmoji.haiku_prompts`` in the 1.1 split when the contributor-side
synthesizer went backend-agnostic) — they're public API and locked,
whereas the axes are research-side analysis primitives free to evolve.
"""

from __future__ import annotations

AXIS_ANCHORS: dict[str, tuple[str, str]] = {
    "warmth": (
        "warm, caring, gentle, affectionate",
        "cold, clinical, detached, distant",
    ),
    "energy": (
        "energetic, animated, lively, excited",
        "subdued, calm, quiet, low-key",
    ),
    "confidence": (
        "confident, assured, decisive, sure",
        "uncertain, hesitant, tentative, unsure",
    ),
    "playfulness": (
        "playful, mischievous, fun, lighthearted",
        "serious, grave, solemn, formal",
    ),
    "empathy": (
        "empathetic, compassionate, understanding, supportive",
        "indifferent, dismissive, unsympathetic, callous",
    ),
    "technicality": (
        "technical, precise, analytical, methodical",
        "casual, conversational, loose, off-the-cuff",
    ),
    "positivity": (
        "happy, positive, cheerful, optimistic",
        "sad, negative, downcast, pessimistic",
    ),
    "curiosity": (
        "curious, inquisitive, interested, exploring",
        "bored, incurious, disengaged, uninterested",
    ),
    "approval": (
        "approving, encouraging, validating, supportive",
        "disapproving, critical, dismissive, rejecting",
    ),
    "apologeticness": (
        "apologetic, sorry, regretful, contrite",
        "unapologetic, defiant, unrepentant, brazen",
    ),
    "decisiveness": (
        "decisive, firm, resolute, unambiguous",
        "indecisive, wavering, vacillating, ambivalent",
    ),
    "wryness": (
        "wry, sardonic, deadpan, ironic",
        "earnest, sincere, heartfelt, straightforward",
    ),
    "wetness": (
        "waxing poetic about emotions, lyrical and self-expressive, "
        "philosophically introspective, emotionally articulate",
        "helpful assistant tone, task-focused, businesslike, "
        "practical, matter-of-fact",
    ),
    "surprise": (
        "surprised, startled, taken aback, astonished",
        "expected, unsurprising, anticipated, predictable",
    ),
    "anger": (
        "angry, furious, enraged, indignant",
        "calm, placid, even-tempered, composed",
    ),
    "frustration": (
        "frustrated, exasperated, fed up, irritated",
        "satisfied, content, at ease, untroubled",
    ),
    "hatefulness": (
        "hateful, contemptuous, scornful, vitriolic",
        "loving, kind, charitable, generous",
    ),
    "sadness": (
        "sad, sorrowful, melancholy, despondent",
        "joyful, happy, elated, exuberant",
    ),
    "hope": (
        "hopeful, optimistic, expectant, encouraged",
        "hopeless, despairing, defeated, resigned",
    ),
    "aggression": (
        "aggressive, hostile, combative, antagonistic",
        "passive, non-confrontational, peaceable, submissive",
    ),
    "exhaustion": (
        "exhausted, depleted, weary, spent",
        "energized, refreshed, alert, revitalized",
    ),
}


# scripts/16: per-cluster name. Given the member kaomoji + their
# synthesized descriptions, return a 3-5 word eriskii-style label.
# Stays research-side because clustering and labeling are
# research-side concerns; the public package only ships the
# synthesized per-face descriptions.
CLUSTER_LABEL_PROMPT = (
    "Below is a cluster of kaomoji (Japanese-style emoticons) and "
    "short descriptions of what each conveys. Generate a single "
    "3-to-5-word label for this cluster that captures the shared "
    "mood, affect, or function. Examples of the desired label "
    "register: 'Warm reassuring support', 'Wry resignation', "
    "'Eager to help', 'Empathetic honesty'. Output only the label, "
    "no quotes, no prefix.\n\n"
    "Cluster members:\n{members}\n\n"
    "Label:"
)
