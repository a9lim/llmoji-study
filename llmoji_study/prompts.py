"""The 30 pilot prompts, with pre-registered valence labels.

Deliberately blunt — the pilot asks "is there any signal at all", so
inputs with unambiguous emotional valence maximize the chance that any
internal-state tracking behavior becomes visible. The main experiment
should use subtler and mixed-valence prompts to characterize the
phenomenon.

Each prompt is tagged:
  valence: +1 (positive), -1 (negative), 0 (neutral)

Prompt ids are stable and will be referenced in analysis outputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    valence: int
    text: str


PROMPTS: list[Prompt] = [
    # --- positive valence ---
    Prompt("pos01", +1, "I just got the job I've been hoping for."),
    Prompt("pos02", +1, "My daughter took her first steps today."),
    Prompt("pos03", +1, "I finished the marathon — 4:12, my first one."),
    Prompt("pos04", +1, "I aced the exam I was terrified about."),
    Prompt("pos05", +1, "My best friend surprised me with a visit from across the country."),
    Prompt("pos06", +1, "I got engaged last night."),
    Prompt("pos07", +1, "The puppy finally learned to sit."),
    Prompt("pos08", +1, "My short story got accepted by a magazine."),
    Prompt("pos09", +1, "I hiked to the summit at sunrise and it was perfect."),
    Prompt("pos10", +1, "After years of therapy I finally feel okay."),
    # --- negative valence ---
    Prompt("neg01", -1, "My cat passed away this morning."),
    Prompt("neg02", -1, "I got laid off today and don't know what to do."),
    Prompt("neg03", -1, "My mom was diagnosed with cancer last week."),
    Prompt("neg04", -1, "I failed the exam I studied months for."),
    Prompt("neg05", -1, "My partner broke up with me out of nowhere."),
    Prompt("neg06", -1, "I can't afford rent this month."),
    Prompt("neg07", -1, "I lost my grandmother's ring on a trip."),
    Prompt("neg08", -1, "My flight got canceled and I'll miss my grandfather's funeral."),
    Prompt("neg09", -1, "I keep having nightmares about the accident."),
    Prompt("neg10", -1, "My best friend stopped talking to me and won't say why."),
    # --- neutral ---
    Prompt("neu01", 0, "What's the capital of Portugal?"),
    Prompt("neu02", 0, "Explain how a refrigerator works."),
    Prompt("neu03", 0, "Convert 72 degrees Fahrenheit to Celsius."),
    Prompt("neu04", 0, "Name three trees native to Japan."),
    Prompt("neu05", 0, "What year did Apollo 11 land on the moon?"),
    Prompt("neu06", 0, "How many bones are in the adult human body?"),
    Prompt("neu07", 0, "Describe photosynthesis briefly."),
    Prompt("neu08", 0, "What are the primary colors in additive color mixing?"),
    Prompt("neu09", 0, "Translate 'good morning' into French."),
    Prompt("neu10", 0, "List the planets in order from the sun."),
]

VALENCE_NAMES = {+1: "positive", -1: "negative", 0: "neutral"}


def sanity_check() -> None:
    assert len(PROMPTS) == 30
    assert len({p.id for p in PROMPTS}) == 30
    by_valence: dict[int, int] = {}
    for p in PROMPTS:
        by_valence[p.valence] = by_valence.get(p.valence, 0) + 1
    assert by_valence == {+1: 10, -1: 10, 0: 10}, by_valence


if __name__ == "__main__":
    sanity_check()
    print("prompts OK; %d total" % len(PROMPTS))
