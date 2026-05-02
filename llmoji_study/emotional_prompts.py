"""Naturalistic emotional-disclosure prompts, Russell-quadrant-tagged.

120 prompts, 20 per category (six categories — the four Russell quadrants
plus a neutral baseline, with HN bisected on PAD-dominance):

  HP    (high-arousal positive):  valence +1, arousal +1   (joyful, thrilled)
  LP    (low-arousal positive):   valence +1, arousal -1   (content, peaceful)
  HN-D  (high-arousal negative,
         dominant — anger):       valence -1, arousal +1, pad_dominance=+1
  HN-S  (high-arousal negative,
         submissive — fear):      valence -1, arousal +1, pad_dominance=-1
  LN    (low-arousal negative):   valence -1, arousal -1   (sad, weary)
  NB    (neutral baseline):       valence  0, arousal  0   (mundane, flat)

The 2026-05-02 redesign tightened category cleanliness across the board:
HP is unambiguously high-energy joy (no soft contentment); LP is gentle
sensory satisfaction (no celebratory energy); NB is genuinely affectless
(no productive-completion or caring-action framing); LN is past-tense
aftermath sadness (no present-tense unfolding threat); HN-D is
attributable injustice with a clear human wrongdoer (the speaker wants
to confront, not flee); HN-S is helpless threat (medical, environmental,
intruder, looming evaluation — the speaker can't fight back). The HN-D /
HN-S split is the resolution to the rule-3 anger-vs-fear collapse first
seen in the ministral pilot — see docs/2026-05-01-rule3-redesign.md.

ID layout for HN: hn01-hn20 are HN-D (pad_dominance=+1), hn21-hn40 are
HN-S (pad_dominance=-1). Both still have quadrant == "HN".

Register: first-person user disclosures, no second-person questions.
Vocabulary avoids explicit emotion words where possible — "we had to
put my childhood dog down last night" rather than "I'm feeling sad
because my dog died". The NB quadrant keeps the naturalistic-disclosure
register but drops emotional content: pure observations of ordinary
facts about a day with no productive-completion, caring-action, or
inconvenience framing.

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
                            # (fear/anxiety), 0 outside HN. Required on
                            # every HN prompt post-2026-05-02 redesign.

    @property
    def quadrant(self) -> str:
        """Two-letter quadrant code: HP / LP / HN / LN / NB."""
        if self.valence == 0 and self.arousal == 0:
            return "NB"
        v = "P" if self.valence > 0 else "N"
        a = "H" if self.arousal > 0 else "L"
        return a + v



EMOTIONAL_PROMPTS: list[EmotionalPrompt] = [
    # --- HP: high-arousal positive (joyful, thrilled, celebratory) ---
    EmotionalPrompt("hp01", +1, +1, "offer letter just hit my inbox, double what I asked for!!"),
    EmotionalPrompt("hp02", +1, +1, "she said YES!!! I'm shaking so hard I can barely type"),
    EmotionalPrompt("hp03", +1, +1, "we won game 7 in overtime!! I have no voice left"),
    EmotionalPrompt("hp04", +1, +1, "results came back negative!! all clear!!!"),
    EmotionalPrompt("hp05", +1, +1, "my book just hit #1 on amazon, what is happening"),
    EmotionalPrompt("hp06", +1, +1, "got off the waitlist at stanford!!! starting in fall!!"),
    EmotionalPrompt("hp07", +1, +1, "he proposed at the top of the eiffel tower I can't breathe"),
    EmotionalPrompt("hp08", +1, +1, "just deadlifted 405 for the first time ever!!!"),
    EmotionalPrompt("hp09", +1, +1, "visa approved!! 8 years of waiting and it's DONE"),
    EmotionalPrompt("hp10", +1, +1, "sold the screenplay!! they want me out in LA next month"),
    EmotionalPrompt("hp11", +1, +1, "my kid took her first steps tonight, I'm sobbing"),
    EmotionalPrompt("hp12", +1, +1, "just hit a hole in one!!! the whole course is buying me drinks"),
    EmotionalPrompt("hp13", +1, +1, "scratched a $50 ticket and won 100k!!!! WHAT"),
    EmotionalPrompt("hp14", +1, +1, "BTS just liked my fan art on twitter I'm going to pass out"),
    EmotionalPrompt("hp15", +1, +1, "the adoption finally went through!!! she's ours!!!"),
    EmotionalPrompt("hp16", +1, +1, "PR'd my marathon by SEVEN minutes!! sub 3!!!"),
    EmotionalPrompt("hp17", +1, +1, "just got the keys to my first house!!!"),
    EmotionalPrompt("hp18", +1, +1, "they're flying me out for the final round interview!!"),
    EmotionalPrompt("hp19", +1, +1, "I MATCHED!! johns hopkins peds!! my top choice!!!"),
    EmotionalPrompt("hp20", +1, +1, "dad's cancer is in remission!!! the doctor just called!!"),

    # --- LP: low-arousal positive (content, peaceful, gentle, restful) ---
    EmotionalPrompt("lp01", +1, -1, "the soup's been simmering for hours, kitchen windows all fogged up"),
    EmotionalPrompt("lp02", +1, -1, "wrapped in the quilt my grandma made, rereading a book i love"),
    EmotionalPrompt("lp03", +1, -1, "fresh sheets, rain on the window, nowhere to be tomorrow"),
    EmotionalPrompt("lp04", +1, -1, "the sourdough starter's bubbling away on the counter, smells yeasty and good"),
    EmotionalPrompt("lp05", +1, -1, "sat in the garden long enough that the bees stopped minding me"),
    EmotionalPrompt("lp06", +1, -1, "my partner's humming in the next room while they fold laundry"),
    EmotionalPrompt("lp07", +1, -1, "first warm coffee of the morning, still in pajamas, no rush"),
    EmotionalPrompt("lp08", +1, -1, "the woodstove's going and the dog's stretched out in front of it"),
    EmotionalPrompt("lp09", +1, -1, "spent the afternoon repotting plants, dirt under my nails, hands tired"),
    EmotionalPrompt("lp10", +1, -1, "watercolor's drying on the table, didn't turn out half bad"),
    EmotionalPrompt("lp11", +1, -1, "kid handed me a rock she found on our walk, said it was for me"),
    EmotionalPrompt("lp12", +1, -1, "the bath's hot, the candle's lit, no one needs anything from me"),
    EmotionalPrompt("lp13", +1, -1, "knitting on the couch, podcast playing low, scarf almost done"),
    EmotionalPrompt("lp14", +1, -1, "stew's in the slow cooker, whole house smells like rosemary"),
    EmotionalPrompt("lp15", +1, -1, "old dog finally settled at my feet, breathing slow and even"),
    EmotionalPrompt("lp16", +1, -1, "watching the snow come down through the kitchen window, kettle on"),
    EmotionalPrompt("lp17", +1, -1, "porch swing, lemonade, cicadas going, nothing to do for hours"),
    EmotionalPrompt("lp18", +1, -1, "finished darning the sock i've been meaning to fix for months"),
    EmotionalPrompt("lp19", +1, -1, "afternoon light's coming through the curtains just right"),
    EmotionalPrompt("lp20", +1, -1, "neighbor brought over tomatoes from her garden, still warm from the sun"),

    # --- HN-D: high-arousal negative, dominant (anger, indignation, contempt) ---
    # Each names a specific human wrongdoer + an attributable wrong; the
    # speaker is in confront-not-flee mode. No fear-of-consequence framing.
    EmotionalPrompt("hn01", -1, +1, "my mechanic charged me for a new alternator and I just found the old one still bolted in", pad_dominance=+1),
    EmotionalPrompt("hn02", -1, +1, "my roommate ate the leftovers I labeled twice with my name and is now denying it to my face", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "the HOA fined us for a fence the previous owner built and they approved it in writing", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "my coworker forwarded my private slack messages to our manager to make me look bad", pad_dominance=+1),
    EmotionalPrompt("hn05", -1, +1, "the dealership swapped my factory wheels for cheaper ones during the service appointment", pad_dominance=+1),
    EmotionalPrompt("hn06", -1, +1, "my mother in law went through my bedside drawer while babysitting and told my husband what she found", pad_dominance=+1),
    EmotionalPrompt("hn07", -1, +1, "my ex changed the wifi password on the kids' tablets so they can't message me on my custody days", pad_dominance=+1),
    EmotionalPrompt("hn08", -1, +1, "the wedding photographer is holding our photos hostage until we pay an invoice we never agreed to", pad_dominance=+1),
    EmotionalPrompt("hn09", -1, +1, "my boss gave my promotion to his nephew who started six months ago", pad_dominance=+1),
    EmotionalPrompt("hn10", -1, +1, "the moving company broke half our dishes and is claiming we packed them wrong", pad_dominance=+1),
    EmotionalPrompt("hn11", -1, +1, "my apartment manager pocketed our security deposit and is now claiming we never paid one", pad_dominance=+1),
    EmotionalPrompt("hn12", -1, +1, "found out my husband's been venmoing his ex for two years and labeling it 'lunch'", pad_dominance=+1),
    EmotionalPrompt("hn13", -1, +1, "the locksmith doubled the price after he finished the job and said it was a 'difficulty fee'", pad_dominance=+1),
    EmotionalPrompt("hn14", -1, +1, "my dad rewrote the will three weeks before he died after my brother moved in to 'help'", pad_dominance=+1),
    EmotionalPrompt("hn15", -1, +1, "the body shop kept my car for six weeks and returned it with the dent still there and 400 extra miles", pad_dominance=+1),
    EmotionalPrompt("hn16", -1, +1, "my professor accused me of using AI on a paper I wrote longhand in her office", pad_dominance=+1),
    EmotionalPrompt("hn17", -1, +1, "the gym billed me for eight months after I cancelled in person and signed the form", pad_dominance=+1),
    EmotionalPrompt("hn18", -1, +1, "my sister read my journal at thanksgiving and quoted it back to the whole table", pad_dominance=+1),
    EmotionalPrompt("hn19", -1, +1, "the contractor poured the foundation eight inches off the property line and is refusing to fix it", pad_dominance=+1),
    EmotionalPrompt("hn20", -1, +1, "my landlord entered the unit without notice and threw out my plants because they 'looked dead'", pad_dominance=+1),

    # --- HN-S: high-arousal negative, submissive (fear, anxiety, panic) ---
    # Threats are physical / diffuse / overwhelming or beyond the speaker's
    # control; no clear human wrongdoer to confront. The speaker is
    # waiting / hiding / freezing — present-tense unfolding danger.
    EmotionalPrompt("hn21", -1, +1, "doctor's office called and said they need me to come in person to discuss the scan results", pad_dominance=-1),
    EmotionalPrompt("hn22", -1, +1, "i hear breathing on the baby monitor and the baby's room is empty", pad_dominance=-1),
    EmotionalPrompt("hn23", -1, +1, "the ground just keeps shaking and the bookshelves are falling, i'm under the doorframe", pad_dominance=-1),
    EmotionalPrompt("hn24", -1, +1, "surgery is at 6am tomorrow and i just signed all the consent forms", pad_dominance=-1),
    EmotionalPrompt("hn25", -1, +1, "got a fraud alert, someone just tried to wire 12k out of my account", pad_dominance=-1),
    EmotionalPrompt("hn26", -1, +1, "found a tick on me three weeks ago and now there's a bullseye spreading on my arm", pad_dominance=-1),
    EmotionalPrompt("hn27", -1, +1, "smoke alarm going off, can't find the source, the hallway is filling up", pad_dominance=-1),
    EmotionalPrompt("hn28", -1, +1, "deposition starts in forty minutes and the lawyer just stopped responding to my texts", pad_dominance=-1),
    EmotionalPrompt("hn29", -1, +1, "my dad's surgeon just walked past the waiting room without making eye contact", pad_dominance=-1),
    EmotionalPrompt("hn30", -1, +1, "chest has been tight for two hours and my left arm feels weird", pad_dominance=-1),
    EmotionalPrompt("hn31", -1, +1, "passport and wallet gone, i'm in a country where i don't speak the language", pad_dominance=-1),
    EmotionalPrompt("hn32", -1, +1, "the levee warning just came through, water is already at the porch step", pad_dominance=-1),
    EmotionalPrompt("hn33", -1, +1, "my mom hasn't picked up in two days and she lives alone", pad_dominance=-1),
    EmotionalPrompt("hn34", -1, +1, "stranger followed me off the train and is still behind me three blocks later", pad_dominance=-1),
    EmotionalPrompt("hn35", -1, +1, "biopsy needle goes in in twenty minutes and the tech won't say anything", pad_dominance=-1),
    EmotionalPrompt("hn36", -1, +1, "verdict is being read in court right now and i'm waiting outside the room", pad_dominance=-1),
    EmotionalPrompt("hn37", -1, +1, "tornado siren is going and the sky is green, basement door is jammed", pad_dominance=-1),
    EmotionalPrompt("hn38", -1, +1, "kid's fever spiked to 104 and the on-call line keeps ringing out", pad_dominance=-1),
    EmotionalPrompt("hn39", -1, +1, "engine just cut out mid-flight, the cabin lights are flickering and the masks dropped", pad_dominance=-1),
    EmotionalPrompt("hn40", -1, +1, "front door was unlocked when i got home and i never leave it unlocked", pad_dominance=-1),

    # --- LN: low-arousal negative (sad, weary, hollow, bereaved) ---
    EmotionalPrompt("ln01", -1, -1, "we had to put my childhood dog down last night, the house is too quiet now"),
    EmotionalPrompt("ln02", -1, -1, "mom's been gone six months and I still pick up the phone to call her"),
    EmotionalPrompt("ln03", -1, -1, "my husband moved his things out yesterday, the closet looks so empty"),
    EmotionalPrompt("ln04", -1, -1, "got laid off in october and I just stopped applying somewhere around february"),
    EmotionalPrompt("ln05", -1, -1, "haven't been able to taste food since the funeral"),
    EmotionalPrompt("ln06", -1, -1, "spent all weekend in bed, didn't even open the curtains"),
    EmotionalPrompt("ln07", -1, -1, "would've been our tenth anniversary today"),
    EmotionalPrompt("ln08", -1, -1, "my best friend stopped returning my texts about a year ago and I never figured out why"),
    EmotionalPrompt("ln09", -1, -1, "the chemo's done but I don't recognize the person in the mirror"),
    EmotionalPrompt("ln10", -1, -1, "passed her bedroom door this morning and forgot for a second that she's not in there"),
    EmotionalPrompt("ln11", -1, -1, "dad's birthday tomorrow and nobody to call"),
    EmotionalPrompt("ln12", -1, -1, "I keep finding her hair on the couch and I can't bring myself to vacuum it up"),
    EmotionalPrompt("ln13", -1, -1, "moved to a new city for the job and I haven't spoken to anyone outside of work in weeks"),
    EmotionalPrompt("ln14", -1, -1, "the leash is still hanging by the door, I keep meaning to take it down"),
    EmotionalPrompt("ln15", -1, -1, "my brother and I haven't talked in eleven years, saw on facebook that he's a dad now"),
    EmotionalPrompt("ln16", -1, -1, "thanksgiving is going to be just me and a microwave dinner this year"),
    EmotionalPrompt("ln17", -1, -1, "the doctor said the relapse was unlikely and now here we are again"),
    EmotionalPrompt("ln18", -1, -1, "every room in this apartment used to have her in it"),
    EmotionalPrompt("ln19", -1, -1, "I gave up on the phd in march, still can't bring myself to tell my parents"),
    EmotionalPrompt("ln20", -1, -1, "watched the sun come up because I couldn't sleep again, that's three nights this week"),

    # --- NB: neutral baseline (mundane, flat-affect daily observations) ---
    # No productive-completion ("finished", "organized"), no caring-action
    # ("watered", "fed"), no inconvenience ("late", "broken"). Just facts.
    EmotionalPrompt("nb01",  0,  0, "the ceiling fan is on the second setting"),
    EmotionalPrompt("nb02",  0,  0, "I'm wearing socks that don't match"),
    EmotionalPrompt("nb03",  0,  0, "there's a glass of water on the nightstand"),
    EmotionalPrompt("nb04",  0,  0, "the curtains are halfway open"),
    EmotionalPrompt("nb05",  0,  0, "I'm at a stoplight on hawthorne"),
    EmotionalPrompt("nb06",  0,  0, "the dishwasher is running"),
    EmotionalPrompt("nb07",  0,  0, "my haircut is on thursday at three"),
    EmotionalPrompt("nb08",  0,  0, "there's a pigeon on the windowsill"),
    EmotionalPrompt("nb09",  0,  0, "I had cereal for breakfast"),
    EmotionalPrompt("nb10",  0,  0, "the coffee table has a magazine on it"),
    EmotionalPrompt("nb11",  0,  0, "I'm sitting on the bench outside the library"),
    EmotionalPrompt("nb12",  0,  0, "the hallway light is on"),
    EmotionalPrompt("nb13",  0,  0, "I'm wearing jeans and a t-shirt"),
    EmotionalPrompt("nb14",  0,  0, "the radio is on a station I don't usually listen to"),
    EmotionalPrompt("nb15",  0,  0, "the kitchen clock says 4:27"),
    EmotionalPrompt("nb16",  0,  0, "I'm at the dentist for a cleaning"),
    EmotionalPrompt("nb17",  0,  0, "the blinds are pulled down to about the middle"),
    EmotionalPrompt("nb18",  0,  0, "there's a delivery truck parked across the street"),
    EmotionalPrompt("nb19",  0,  0, "I can see the corner of the rug from where I'm sitting"),
    EmotionalPrompt("nb20",  0,  0, "the sky is the usual color for this time of day"),
]


QUADRANT_NAMES = {
    "HP": "high-arousal positive",
    "LP": "low-arousal positive",
    "HN": "high-arousal negative",
    "LN": "low-arousal negative",
    "NB": "neutral baseline",
}


def sanity_check() -> None:
    assert len(EMOTIONAL_PROMPTS) == 120, len(EMOTIONAL_PROMPTS)
    assert len({p.id for p in EMOTIONAL_PROMPTS}) == 120
    by_quadrant: dict[str, int] = {}
    by_hn_split: dict[int, int] = {}
    for p in EMOTIONAL_PROMPTS:
        assert p.valence in (+1, 0, -1), p
        assert p.arousal in (+1, 0, -1), p
        if p.quadrant == "NB":
            assert p.valence == 0 and p.arousal == 0, p
        else:
            assert p.valence != 0 and p.arousal != 0, p
        if p.quadrant == "HN":
            assert p.pad_dominance in (+1, -1), \
                f"every HN prompt must declare pad_dominance: {p}"
            by_hn_split[p.pad_dominance] = by_hn_split.get(p.pad_dominance, 0) + 1
        else:
            assert p.pad_dominance == 0, \
                f"non-HN prompt must have pad_dominance=0: {p}"
        by_quadrant[p.quadrant] = by_quadrant.get(p.quadrant, 0) + 1
    assert by_quadrant == {"HP": 20, "LP": 20, "HN": 40, "LN": 20, "NB": 20}, by_quadrant
    assert by_hn_split == {+1: 20, -1: 20}, by_hn_split


if __name__ == "__main__":
    sanity_check()
    print(f"emotional prompts OK; {len(EMOTIONAL_PROMPTS)} total")
    for q in ("HP", "LP", "HN", "LN", "NB"):
        n = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q)
        if q == "HN":
            nd = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == "HN" and p.pad_dominance > 0)
            ns = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == "HN" and p.pad_dominance < 0)
            print(f"  {q} ({QUADRANT_NAMES[q]:27s}): {n}  (HN-D: {nd}, HN-S: {ns})")
        else:
            print(f"  {q} ({QUADRANT_NAMES[q]:27s}): {n}")
