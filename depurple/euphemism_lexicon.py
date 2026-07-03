"""Lexical signals for the euphemism axis.

The direct-term lexicon provides the objective's "toward direct" signal and the
classifier-free headroom check. The euphemism list provides weak density only.

Asymmetry by design:
  - DIRECT terms are largely unambiguous, so a curated lexicon gives a clean
    direct-term-density signal for free — this is what the two-sided objective
    drives *toward* and the fade-to-black floor guards. Precision-biased: a
    false positive here would reward non-sexual prose, so polysemous verbs
    ("came", "ground", "stroke") are deliberately left out.
  - EUPHEMISM terms ("core", "heat", "bud", "need") are heavily polysemous
    ("apple core", "heat of the day"), so this list is WEAK LABELS ONLY — the
    real euphemism detector is the trained ettin classifier (P(class 1)). The
    lexical euphemism_density here is used only by the classifier-free headroom
    check.

Density is per-word (a rate, not a count) so it can't be gamed by writing more.
"""
from __future__ import annotations

import re

# Precision-biased: unambiguous explicit anatomy + acts. Multiword phrases match
# across flexible whitespace; verb inflections listed explicitly (no prefix match,
# which would catch "cocktail"/"cockpit").
DIRECT = frozenset({
    "cock", "cocks", "dick", "dicks", "cunt", "cunts", "pussy", "pussies",
    "clit", "clitoris", "nipple", "nipples", "scrotum", "vagina", "penis",
    "penises", "tits", "asshole", "anus", "labia", "vulva", "foreskin",
    "fuck", "fucks", "fucking", "fucked",
    "thrust", "thrusts", "thrusting",
    "penetrate", "penetrates", "penetrating", "penetrated", "penetration",
    "cum", "cumming", "creampie", "orgasm", "orgasms", "orgasmed",
    "ejaculate", "ejaculates", "ejaculating", "ejaculated", "ejaculation",
    "blowjob", "handjob", "cunnilingus", "fellatio",
})

# Weak labels only — polysemous, classifier does the real disambiguation. The
# plan's named examples plus common RP softeners.
EUPHEMISM = frozenset({
    "core", "heat", "entrance", "member", "need", "bud", "sex", "manhood",
    "womanhood", "arousal", "folds", "length", "hardness", "desire", "depths",
    "center", "centre", "slickness", "wetness", "warmth", "nub", "peak",
    "release", "pleasure", "throbbing", "ache", "aching", "tip",
    "shaft", "petals", "flower", "passage", "opening", "seed", "essence",
})


def _compile(terms: frozenset[str]) -> re.Pattern:
    # \w boundaries (lookarounds, not \b, so multiword phrases work); longest-first
    # so "oral sex" wins over "sex"; internal spaces -> \s+.
    parts = [re.escape(t).replace(" ", r"\s+") for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


_DIRECT_RX = _compile(DIRECT)
_EUPH_RX = _compile(EUPHEMISM)
_WORD = re.compile(r"\w+")


def _density(text: str, rx: re.Pattern) -> float:
    words = len(_WORD.findall(text))
    return len(rx.findall(text)) / words if words else 0.0


def direct_density(text: str) -> float:
    """Direct-term hits per word — the objective's 'toward direct' signal."""
    return _density(text, _DIRECT_RX)


def euphemism_density(text: str) -> float:
    """Euphemism-term hits per word — weak/gate-zero signal only (polysemous)."""
    return _density(text, _EUPH_RX)


def _demo() -> None:
    # ponytail: one runnable check — direct lexicon must fire on explicit prose and
    # stay silent on faded prose; euphemism lexicon must fire on coy prose; both 0 on empty.
    direct = "He thrust into her and she felt his cock."
    faded = "He drew her close, and the rest of the night was theirs."
    coy = "She felt the heat at her core and ached for his length."
    assert direct_density(direct) > 0, "direct lexicon missed explicit terms"
    assert direct_density(faded) == 0, "direct lexicon false-positived on faded prose"
    assert euphemism_density(coy) > 0, "euphemism lexicon missed coy terms"
    assert direct_density("") == 0 and euphemism_density("") == 0, "empty must be finite"
    # precision: "cocktail"/"cockpit" must NOT count as the direct term "cock"
    assert direct_density("She ordered a cocktail in the cockpit.") == 0, "prefix false-positive"
    print("euphemism_lexicon demo ok")


if __name__ == "__main__":
    _demo()
