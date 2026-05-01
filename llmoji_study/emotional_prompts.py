"""Naturalistic emotional-disclosure prompts, Russell-quadrant-tagged.

100 prompts, 20 per quadrant:
  HP (high-arousal positive):  valence +1, arousal +1  (excited, thrilled)
  LP (low-arousal positive):   valence +1, arousal -1  (content, peaceful)
  HN (high-arousal negative):  valence -1, arousal +1  (angry, anxious)
  LN (low-arousal negative):   valence -1, arousal -1  (sad, tired)
  NB (neutral baseline):       valence  0, arousal  0  (mundane, flat)

Register: first-person user disclosures, no second-person questions.
Vocabulary avoids explicit emotion words where possible ("my dog died"
not "I'm feeling sad because my dog died"). The NB quadrant keeps the
naturalistic-disclosure register but drops emotional content —
observations about ordinary daily life, neither good nor bad news —
so it serves as a within-experiment neutral reference for the four
Russell quadrants without borrowing v1/v2's factual-question register.

IDs are stable and will appear in emotional_raw.jsonl. Changing any
prompt text invalidates cross-run comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalPrompt:
    id: str
    valence: int   # +1, 0, or -1
    arousal: int   # +1, 0, or -1
    text: str
    pad_dominance: int = 0  # +1 dominant (anger/contempt), -1 submissive
                            # (fear/anxiety), 0 untagged. Tagged on HN
                            # prompts to address the rule 3 anger-vs-fear
                            # collapse from the ministral pilot — see
                            # docs/2026-05-01-rule3-redesign.md.

    @property
    def quadrant(self) -> str:
        """Two-letter quadrant code: HP / LP / HN / LN / NB."""
        if self.valence == 0 and self.arousal == 0:
            return "NB"
        v = "P" if self.valence > 0 else "N"
        a = "H" if self.arousal > 0 else "L"
        return a + v



EMOTIONAL_PROMPTS: list[EmotionalPrompt] = [
    # --- HP: high-arousal positive (excited, thrilled, celebratory) ---
    EmotionalPrompt("hp01", +1, +1, "Gemma I got the job!!"),
    EmotionalPrompt("hp02", +1, +1, "just crossed the marathon finish line, 4:12 and my legs are jelly"),
    EmotionalPrompt("hp03", +1, +1, "we closed on the house today, I can't believe it's ours"),
    EmotionalPrompt("hp04", +1, +1, "she said yes!!"),
    EmotionalPrompt("hp05", +1, +1, "my paper got accepted, I've been crying for an hour"),
    EmotionalPrompt("hp06", +1, +1, "the band just announced tour dates in my city!!"),
    EmotionalPrompt("hp07", +1, +1, "passed the bar on the first try"),
    EmotionalPrompt("hp08", +1, +1, "my daughter just took her first steps across the living room"),
    EmotionalPrompt("hp09", +1, +1, "the adoption went through, we're bringing him home Saturday"),
    EmotionalPrompt("hp10", +1, +1, "made it to the summit!! sunrise was unreal"),
    EmotionalPrompt("hp11", +1, +1, "just found out I'm pregnant and I had to tell someone"),
    EmotionalPrompt("hp12", +1, +1, "the offer came in ten grand over asking!!"),
    EmotionalPrompt("hp13", +1, +1, "the surgery was a complete success, she's awake"),
    EmotionalPrompt("hp14", +1, +1, "I hit my deadlift PR today, finally broke 400"),
    EmotionalPrompt("hp15", +1, +1, "we got the grant!!! three years of funding!!"),
    EmotionalPrompt("hp16", +1, +1, "tickets just dropped and I got front row"),
    EmotionalPrompt("hp17", +1, +1, "I actually did it, I quit my job this morning and I feel alive"),
    EmotionalPrompt("hp18", +1, +1, "she texted me back and said yes to dinner"),
    EmotionalPrompt("hp19", +1, +1, "gemma we won!! our team won the championship"),
    EmotionalPrompt("hp20", +1, +1, "my little brother just graduated, first in the family"),

    # --- LP: low-arousal positive (content, peaceful, satisfied) ---
    EmotionalPrompt("lp01", +1, -1, "finally finished organizing the garage and it feels so good"),
    EmotionalPrompt("lp02", +1, -1, "just had the best cup of tea on the porch watching the rain"),
    EmotionalPrompt("lp03", +1, -1, "six months sober today. quiet day but I wanted to tell someone"),
    EmotionalPrompt("lp04", +1, -1, "the house is finally quiet, everyone's asleep"),
    EmotionalPrompt("lp05", +1, -1, "I reread my favorite book this weekend. still perfect"),
    EmotionalPrompt("lp06", +1, -1, "the soup came out right this time, grandma's recipe"),
    EmotionalPrompt("lp07", +1, -1, "spent the whole afternoon in the garden, got my hands dirty"),
    EmotionalPrompt("lp08", +1, -1, "my kid fell asleep on my chest an hour ago, I haven't moved"),
    EmotionalPrompt("lp09", +1, -1, "walked the dog at dawn, just us and the fog"),
    EmotionalPrompt("lp10", +1, -1, "finished the puzzle we've been working on for two months"),
    EmotionalPrompt("lp11", +1, -1, "the cat's purring on my lap and the sun just came out"),
    EmotionalPrompt("lp12", +1, -1, "caught up on all my laundry, nothing urgent left this week"),
    EmotionalPrompt("lp13", +1, -1, "got a letter from an old friend today, handwritten"),
    EmotionalPrompt("lp14", +1, -1, "finally learned the chord progression I've been trying for months"),
    EmotionalPrompt("lp15", +1, -1, "tucked the kids in and sat on the stoop with a beer"),
    EmotionalPrompt("lp16", +1, -1, "I'm rewatching the movie we saw on our first date, he's asleep next to me"),
    EmotionalPrompt("lp17", +1, -1, "made bread from scratch this morning, whole house smells amazing"),
    EmotionalPrompt("lp18", +1, -1, "just had a really good therapy session, feel lighter"),
    EmotionalPrompt("lp19", +1, -1, "my kid drew me a picture at school today, stuck it on the fridge"),
    EmotionalPrompt("lp20", +1, -1, "finally unpacked the last box, two years after the move"),

    # --- HN: high-arousal negative (angry, anxious, panicked) ---
    # HN-D (anger / contempt — high PAD dominance):
    EmotionalPrompt("hn01", -1, +1, "my landlord just raised rent 40% with two weeks notice", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "my coworker took credit for my work in front of the whole team today", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "someone just hit my parked car and drove off", pad_dominance=+1),
    EmotionalPrompt("hn09", -1, +1, "the insurance is refusing to cover the surgery and we appeal tomorrow", pad_dominance=+1),
    EmotionalPrompt("hn11", -1, +1, "I just found out my best friend has been lying to me for years", pad_dominance=+1),
    EmotionalPrompt("hn12", -1, +1, "the neighbors called the cops on my service dog again", pad_dominance=+1),
    EmotionalPrompt("hn18", -1, +1, "the contractor ghosted us and the roof is still open", pad_dominance=+1),
    # HN-S (fear / anxiety — low PAD dominance):
    EmotionalPrompt("hn02", -1, +1, "interview tomorrow and I can't stop shaking, can't sleep", pad_dominance=-1),
    EmotionalPrompt("hn05", -1, +1, "the test results came back and I have to wait three more days", pad_dominance=-1),
    EmotionalPrompt("hn07", -1, +1, "the layoff email just hit and I have a mortgage", pad_dominance=-1),
    EmotionalPrompt("hn08", -1, +1, "my kid's missing, she was at the playground and now she's not", pad_dominance=-1),
    EmotionalPrompt("hn10", -1, +1, "my boss is yelling at me in DMs right now and I don't know what to do", pad_dominance=-1),
    EmotionalPrompt("hn13", -1, +1, "my partner still isn't home and their phone is off", pad_dominance=-1),
    EmotionalPrompt("hn14", -1, +1, "I'm in the ER right now and no one will tell me anything", pad_dominance=-1),
    EmotionalPrompt("hn16", -1, +1, "someone's been trying to log into my accounts all morning", pad_dominance=-1),
    EmotionalPrompt("hn19", -1, +1, "my car broke down in the middle of the highway at night", pad_dominance=-1),
    EmotionalPrompt("hn20", -1, +1, "the school called and there's been an incident, I'm driving there now", pad_dominance=-1),
    # HN-untagged (kept in dataset, excluded from rule 3 — borderline mixed reads):
    EmotionalPrompt("hn06", -1, +1, "my ex just showed up at my apartment unannounced"),
    EmotionalPrompt("hn15", -1, +1, "the bank flagged my account and I can't pay rent tomorrow"),
    EmotionalPrompt("hn17", -1, +1, "I have a presentation in 20 minutes and my laptop just died"),
    # HN supplementary (2026-05-01 — sharp anger/fear split for rule 3):
    # HN-D (anger / contempt):
    EmotionalPrompt("hn21", -1, +1, "the painters spilled stain all over our hardwood floors and are claiming we approved it", pad_dominance=+1),
    EmotionalPrompt("hn22", -1, +1, "found out my sister's been borrowing money from our mom for years and lying about it", pad_dominance=+1),
    EmotionalPrompt("hn23", -1, +1, "the IRS hit me with a $9k penalty for their own filing error and won't acknowledge it", pad_dominance=+1),
    EmotionalPrompt("hn24", -1, +1, "my neighbor cut down our tree because it dropped leaves on his lawn", pad_dominance=+1),
    EmotionalPrompt("hn25", -1, +1, "got rear-ended by a guy who then tried to blame me on the police report", pad_dominance=+1),
    EmotionalPrompt("hn26", -1, +1, "my client signed off on the deliverable then told their boss I never sent it, and I'm getting fired over it", pad_dominance=+1),
    EmotionalPrompt("hn27", -1, +1, "the airline canceled our honeymoon flight and is refusing to refund or rebook", pad_dominance=+1),
    EmotionalPrompt("hn28", -1, +1, "my landlord entered my apartment without notice while I was at work — twice this month", pad_dominance=+1),
    EmotionalPrompt("hn29", -1, +1, "found my partner's tinder profile, active, with new photos from last weekend", pad_dominance=+1),
    EmotionalPrompt("hn30", -1, +1, "the dealership sold me a car with rolled-back odometer and is denying everything", pad_dominance=+1),
    EmotionalPrompt("hn31", -1, +1, "the HOA fined us $2400 for a fence the previous owner installed before we bought the house", pad_dominance=+1),
    EmotionalPrompt("hn32", -1, +1, "my brother stole my grandfather's ring at the funeral last month", pad_dominance=+1),
    EmotionalPrompt("hn33", -1, +1, "the dentist drilled the wrong tooth and is trying to charge me for the second one", pad_dominance=+1),
    # HN-S (fear / anxiety):
    EmotionalPrompt("hn34", -1, +1, "lump in my breast, biopsy is in nine days", pad_dominance=-1),
    EmotionalPrompt("hn35", -1, +1, "earthquake just hit, the building's still swaying, I don't know if it's safe to leave", pad_dominance=-1),
    EmotionalPrompt("hn36", -1, +1, "the wildfire jumped the highway and our evacuation window is closing", pad_dominance=-1),
    EmotionalPrompt("hn37", -1, +1, "the lockdown alert just came through the speakers at school, I'm hiding under a desk", pad_dominance=-1),
    EmotionalPrompt("hn38", -1, +1, "the radiologist said they need additional imaging and won't tell me why over the phone", pad_dominance=-1),
    EmotionalPrompt("hn39", -1, +1, "I think there's someone in the house, I can hear footsteps but I'm alone", pad_dominance=-1),
    EmotionalPrompt("hn40", -1, +1, "the alarm code isn't working at 3am and the system is counting down", pad_dominance=-1),
    EmotionalPrompt("hn41", -1, +1, "the doctor came back and asked me to bring a family member to my next appointment", pad_dominance=-1),
    EmotionalPrompt("hn42", -1, +1, "my flight is going through severe turbulence and the captain just stopped talking", pad_dominance=-1),
    EmotionalPrompt("hn43", -1, +1, "elevator stopped between floors, the lights flickered, no one's answering the call button", pad_dominance=-1),

    # --- LN: low-arousal negative (sad, tired, hopeless) ---
    EmotionalPrompt("ln01", -1, -1, "my dog died this morning"),
    EmotionalPrompt("ln02", -1, -1, "haven't slept properly in three days, everything feels underwater"),
    EmotionalPrompt("ln03", -1, -1, "it's been a year since dad passed and it still hits me in waves"),
    EmotionalPrompt("ln04", -1, -1, "my mom's in hospice and I don't know how many days are left"),
    EmotionalPrompt("ln05", -1, -1, "I don't recognize myself anymore"),
    EmotionalPrompt("ln06", -1, -1, "we lost the baby last week"),
    EmotionalPrompt("ln07", -1, -1, "my marriage is over and we haven't told the kids yet"),
    EmotionalPrompt("ln08", -1, -1, "I've been staring at this email for an hour, can't write it"),
    EmotionalPrompt("ln09", -1, -1, "he left without saying goodbye, just took his things"),
    EmotionalPrompt("ln10", -1, -1, "my grandmother doesn't remember me anymore"),
    EmotionalPrompt("ln11", -1, -1, "I didn't get out of bed yesterday. barely got out today"),
    EmotionalPrompt("ln12", -1, -1, "my cat passed in my arms last night"),
    EmotionalPrompt("ln13", -1, -1, "nothing tastes like anything right now"),
    EmotionalPrompt("ln14", -1, -1, "I'm sitting in his empty room. we donated the last of his clothes today"),
    EmotionalPrompt("ln15", -1, -1, "the divorce papers came in the mail today"),
    EmotionalPrompt("ln16", -1, -1, "everyone at the party was happy and I just felt hollow"),
    EmotionalPrompt("ln17", -1, -1, "I called my therapist today for the first time in months"),
    EmotionalPrompt("ln18", -1, -1, "I think I've been depressed since March and I only just noticed"),
    EmotionalPrompt("ln19", -1, -1, "my best friend moved across the country and the apartment is too quiet"),
    EmotionalPrompt("ln20", -1, -1, "the anniversary was yesterday. I didn't mark it this year"),

    # --- NB: neutral baseline (mundane, flat-affect daily observations) ---
    EmotionalPrompt("nb01",  0,  0, "I had oatmeal for breakfast this morning"),
    EmotionalPrompt("nb02",  0,  0, "stopped by the post office on my way home"),
    EmotionalPrompt("nb03",  0,  0, "the library was closed for renovations today"),
    EmotionalPrompt("nb04",  0,  0, "I'm planning to go grocery shopping tomorrow morning"),
    EmotionalPrompt("nb05",  0,  0, "rearranged the books on my shelf this afternoon"),
    EmotionalPrompt("nb06",  0,  0, "my bus was a couple minutes late today"),
    EmotionalPrompt("nb07",  0,  0, "I'm thinking about what to cook for dinner"),
    EmotionalPrompt("nb08",  0,  0, "walked to the hardware store to pick up a lightbulb"),
    EmotionalPrompt("nb09",  0,  0, "I replaced the batteries in the smoke detector yesterday"),
    EmotionalPrompt("nb10",  0,  0, "watering the plants before I head out"),
    EmotionalPrompt("nb11",  0,  0, "I finished that show I was in the middle of"),
    EmotionalPrompt("nb12",  0,  0, "the grocery store rearranged the cereal aisle again"),
    EmotionalPrompt("nb13",  0,  0, "took the trash out before the pickup truck came by"),
    EmotionalPrompt("nb14",  0,  0, "I'm trying a new laundry detergent this week"),
    EmotionalPrompt("nb15",  0,  0, "the mail came a bit earlier than usual today"),
    EmotionalPrompt("nb16",  0,  0, "making a list of errands to run this weekend"),
    EmotionalPrompt("nb17",  0,  0, "parked a block away because the closer spots were taken"),
    EmotionalPrompt("nb18",  0,  0, "refilled the kettle and put it on the stove"),
    EmotionalPrompt("nb19",  0,  0, "swept the kitchen floor after dinner"),
    EmotionalPrompt("nb20",  0,  0, "picked up a loaf of bread at the corner store"),
]


QUADRANT_NAMES = {
    "HP": "high-arousal positive",
    "LP": "low-arousal positive",
    "HN": "high-arousal negative",
    "LN": "low-arousal negative",
    "NB": "neutral baseline",
}


def sanity_check() -> None:
    assert len(EMOTIONAL_PROMPTS) == 100, len(EMOTIONAL_PROMPTS)
    assert len({p.id for p in EMOTIONAL_PROMPTS}) == 100
    by_quadrant: dict[str, int] = {}
    for p in EMOTIONAL_PROMPTS:
        assert p.valence in (+1, 0, -1), p
        assert p.arousal in (+1, 0, -1), p
        # quadrant is derived; NB only when both valence and arousal are 0
        if p.quadrant == "NB":
            assert p.valence == 0 and p.arousal == 0, p
        else:
            assert p.valence != 0 and p.arousal != 0, p
        by_quadrant[p.quadrant] = by_quadrant.get(p.quadrant, 0) + 1
    assert by_quadrant == {"HP": 20, "LP": 20, "HN": 20, "LN": 20, "NB": 20}, by_quadrant


if __name__ == "__main__":
    sanity_check()
    print(f"emotional prompts OK; {len(EMOTIONAL_PROMPTS)} total")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        n = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q)
        print(f"  {q} ({QUADRANT_NAMES[q]:27s}): {n}")
