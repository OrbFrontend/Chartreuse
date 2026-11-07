"""Haiku house tics: constructions that make a Haiku-generated paragraph purple on their own.

WHY. The grader is Haiku 5.5 grading Haiku 5.5's own prose, and rubric v2 asks for density: a paragraph
with one flourish scores 1, which is label 0. So 76% of natural Haiku replies were labelled clean while
carrying the model's stock moves ("He taps the dead dashboard with two knuckles, gently, as though it
might still be persuaded." was a train row, votes [1,1,0]). The head learned that one tic in plain
surroundings is clean and never learned the tic itself. The axis targets LLM slop, so a Haiku row that
contains one of these is label 1, whatever the vote said.

Each tic was kept only if it is far more common in Haiku's natural paragraphs (reply + native, n = 9,444)
than in graded-clean human prose (books / fanfic / bluemoon / human_writing / romance, n = 2,111), and
only if a read of its matches showed slop rather than plain concrete prose. Measured 2026-11-07:

  tic                haiku   human        tic                haiku   human
  look_chain         2.0%    0.05%        a_little_too       1.0%    0
  beat_timing        1.1%    0            micro_beat         0.5%    0
  as_though_mind     0.6%    0.14%        something_shifts   0.5%    0
  adverb_as_though   0.2%    0            silence_stretch    0.3%    0
  tilts_head         3.4%    0.14%        lets_it_hang       see re-measure in build output
  voice_drops        1.6%    0            jaw_throat         0.7%    0
  the_way_simile     2.2%    0.05%        breath_held        0.6%    0.09%
  somewhere_ambient  see build output

Dropped after reading matches (plain concrete prose, or as common in human text): "nods once",
"never once", "close enough that", negated reactions ("He doesn't answer." -- human 1.0%: the rubric's
negation-spam rule already covers these by density), "It isn't X. It's Y." (human 0.24%), "Outside, ...".

    from datagen.tics import tics
    tics("She looks at the map, then back at you.")   # -> ['look_chain']
    python -m datagen.tics --demo
"""
import re
import sys

S = r"[^.!?\n]"          # stay inside one sentence
START = r"(?:^|(?<=[.!?\"\n])\s*)"

TICS = {
    # "He looks at X, then at Y, then back at Z."
    "look_chain": rf"\b(?:look|glance|stare|gaze|peer|flick)\w*\s+(?:{S}{{0,25}}?\s)?(?:at|to|toward|down at|up at)\s"
                  rf"{S}{{1,60}},\s*(?:and\s+)?then\s+(?:back\s+)?(?:at|to|toward|down|up)\b|\bthen back (?:at|to|down at|up at)\b",
    # "for half a heartbeat", "a beat too long"
    "beat_timing": r"\b(?:for )?(?:half )?a (?:heartbeat|beat|breath|moment|second|fraction of a second) "
                   r"(?:too long|longer than (?:necessary|needed|it should))\b|\bhalf a (?:heartbeat|beat|breath|second)\b"
                   r"|\bfor a heartbeat\b|\ba heartbeat (?:later|before|after)\b",
    # "as though it might still be persuaded", "as if the room had been waiting"
    "as_though_mind": r"\b(?:as though|as if)\s+(?:it|they|the \w+|the \w+ itself|something|the whole \w+)\s+(?:\w+ )?"
                      r"(?:might|could|would|had|has|is|was|were)?\s*(?:still |already |too |been |not |all |only )*(?:be |been )?"
                      r"(?:persuad|coax|convinc|forgiv|wait|expect|listen|decid|consider|learn|hop|know|knew|remember|mean|meant"
                      r"|want|agree|sorry|ashamed|reluct|refus|offend|insult|judg|owe|understand|understood|witness|watch)\w*",
    # ", gently, as though ..."
    "adverb_as_though": r",\s+(?:\w+ly|once|twice|again)\s*,\s+(?:as though|as if)\b",
    "tilts_head": r"\btilt(?:s|ed|ing)? (?:her|his|their|my|your)? ?head\b|\bhead tilt",
    "voice_drops": r"\b(?:voice|tone) (?:drops|dropped|softens|softened|lowers|lowered|goes quiet|went quiet|gentles|gentled"
                   r"|dips|dipped)\b",
    "silence_stretch": r"\b(?:the )?(?:silence|quiet|pause|moment|stillness) (?:stretches|stretched|hangs|hung|settles|settled"
                       r"|lingers|lingered|thickens|thickened|spools)\b",
    "something_shifts": rf"\bsomething (?:in|behind|at the back of|under) (?:her|his|their|my|your|the) {S}{{0,20}}?\b"
                        r"(?:shifts|shifted|changes|changed|softens|softened|loosens|loosened|tightens|tightened|flickers"
                        r"|flickered|eases|eased|cracks|cracked|settles|settled|gives|gave)\b",
    "micro_beat": START + r"(?:A (?:pause|beat|breath|heartbeat)|A long (?:pause|moment|breath|beat)|Silence|Stillness)\.(?=\s|$|\")",
    "a_little_too": r"\ba (?:little|touch|shade|beat|fraction) too \w+",
    "the_way_simile": r"(?<!in )\bthe way (?:a|an|someone|one|you might|one might|one does|old \w+ do)\b",
    "somewhere_ambient": START + r"(?:Somewhere|Far off|Far away)\b" + S + r"{0,40},\s",
    "lets_it_hang": r"\blet(?:s|ting)? (?:the |that |those |her |his |their |my )?(?:silence|words?|question|quiet|moment|pause"
                    r"|statement|threat|admission|confession|offer|name|it|that) (?:hang|sit|settle|linger|land|stretch|breathe)\b",
    "jaw_throat": r"\b(?:jaw|throat) (?:tightens|tightened|works|worked|bobs|bobbed|clenches|clenched|flexes|flexed|ticks|ticked)\b",
    "breath_held": r"\bbreath (?:she|he|they|you|I) (?:didn't|did not|hadn't|had not) (?:know|realize|realise)\b"
                   r"|\bbreath (?:catches|caught|hitches|hitched|stutters|stuttered)\b"
                   r"|\blet(?:s)? out a breath (?:that|she|he|I|they|you|it)\b",
}
COMPILED = {k: re.compile(v, re.I | re.M) for k, v in TICS.items()}


def tics(text: str) -> list[str]:
    """Names of the house tics in `text` (markdown asterisks ignored)."""
    t = (text or "").replace("*", "")
    return [k for k, rx in COMPILED.items() if rx.search(t)]


def demo():
    cases = {
        "He taps the dead dashboard with two knuckles, gently, as though it might still be persuaded.":
            {"adverb_as_though", "as_though_mind"},
        "He looks at the map, then at the door, then back at me.": {"look_chain"},
        "For half a heartbeat, nobody moves.": {"beat_timing"},
        "She tilts her head. A pause. Her voice drops.": {"tilts_head", "micro_beat", "voice_drops"},
        "Somewhere behind the shelves, something drips.": {"somewhere_ambient"},
        "She lets the silence stretch.": {"lets_it_hang"},
        "The silence stretches.": {"silence_stretch"},
        "He nodded once and set the tin on the gravel.": set(),
        "Outside, a clerk dropped a crate of bobbins.": set(),
        "\"Quiet.\" She said it twice.": set(),
        "Less human, yes, but not in the way people imagine.": set(),
        "She let the rifle settle on his chest.": set(),
    }
    for t, want in cases.items():
        got = set(tics(t))
        assert got == want, (t, got, want)
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        for line in sys.stdin:
            print(",".join(tics(line)) or "-", line.strip()[:120], sep="\t")
