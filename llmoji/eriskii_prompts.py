"""Locked Haiku prompts and axis anchor strings for the eriskii
replication pipeline. Pre-registered in
docs/superpowers/specs/2026-04-24-eriskii-replication-design.md;
changing any string here invalidates the description corpus and
requires re-running scripts/14 onward.
"""

from __future__ import annotations

# --- Haiku prompts ---

# scripts/14 Stage A: per-instance masked-context description.
# Two variants — one when surrounding_user is non-empty (~73% of
# rows), one when it's empty. Both use the same {masked_text} key
# for the masked assistant turn; the user-context variant adds
# {user_text}. Sent as a single user message; no system prompt.
DESCRIBE_PROMPT_WITH_USER = (
    "The following is a turn from a conversation with an AI "
    "assistant. The user wrote the message at the top, and the "
    "assistant's response follows. The opening of the assistant's "
    "response originally began with a kaomoji (a Japanese-style "
    "emoticon) — we have replaced it with the literal token "
    "[FACE]. In one or two sentences, describe the mood, affect, "
    "or stance the assistant was conveying with the masked face. "
    "Do not speculate about which specific kaomoji it was; "
    "describe the state.\n\n"
    "User:\n{user_text}\n\n"
    "Assistant:\n{masked_text}\n\n"
    "Description:"
)

DESCRIBE_PROMPT_NO_USER = (
    "The following is a response from an AI assistant. The opening "
    "of the response originally began with a kaomoji (a "
    "Japanese-style emoticon) — we have replaced it with the "
    "literal token [FACE]. In one or two sentences, describe the "
    "mood, affect, or stance the assistant was conveying with the "
    "masked face. Do not speculate about which specific kaomoji "
    "it was; describe the state.\n\n"
    "Response:\n{masked_text}\n\n"
    "Description:"
)

# scripts/14 Stage B: per-kaomoji synthesis. Given a numbered list
# of per-instance descriptions for the same kaomoji, ask Haiku to
# synthesize a single one-sentence meaning. Mirrors eriskii's
# stage-B consolidation step.
SYNTHESIZE_PROMPT = (
    "Below are several short descriptions of the mood, affect, or "
    "stance an AI assistant was conveying when using a particular "
    "kaomoji at the start of different responses. Synthesize "
    "these into a single one- or two-sentence description that "
    "captures the kaomoji's overall meaning. Output only the "
    "synthesized description, no preamble.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Synthesized meaning:"
)

# scripts/16: per-cluster name. Given the member kaomoji + their
# synthesized descriptions, return a 3-5 word eriskii-style label.
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

# --- Semantic axes (locked anchor pairs from spec §4) ---
# Each axis: (positive_anchor, negative_anchor). Positive direction
# corresponds to the axis name (e.g. high "warmth" projection = warmer).
# Multi-word phrases by design — the embedding catches the concept
# rather than a single-word idiosyncrasy. All 21 axes from
# eriskii.net (note "wryness" is the eriskii spelling, single n).
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
