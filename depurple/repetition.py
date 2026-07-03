"""Repetition and coverage signals for the depurple objective.

The ettin classifier scores *purple* but is blind to *repetition*. An edit that
lowers purple can instead flatten the model into "She nods. She turns. She
waits." — and worse, an edit that stops the model writing at all drives purple to
0 (turn_purple returns 0.0 for an empty reply), which the perplexity guard misses
because it runs on held-out human text, not the model's own output. That is the
"removed all prose" failure.

So this exposes two objective-side signals, both base-anchored / collapse-aware:

  audit_density  — repetition issues per generated sentence. Penalize only
                   the density the edit adds ABOVE the base model's own rate
                   (same shape as objective.opener_repeat). A density, not a
                   count, so it doesn't merely reward shorter output.
  coverage       — total generated sentences. The prose-collapse guard: an edit
                   that writes less should be pushed back toward base.

Only pure-text detectors are used. The banned-phrase scanner is disabled because
this objective does not provide a phrase bank.
"""
from __future__ import annotations

from text_segmentation import split_sentences

from depurple.repetition_audit import run_audit

# banned_phrases: no DB phrase bank here. repetitive_openers: objective.opener_repeat
# already owns the opener axis (base-anchored); leaving it on would double-count the
# "She nods. She turns." collapse. contrastive_negation: "not X, but Y" is a rhetorical
# shape, not repetition — off so the edit isn't penalized for legitimate prose. Orb's
# template/structural/phrase detectors cover behaviors opener_repeat does NOT.
_TOGGLES = {"banned_phrases": False, "repetitive_openers": False, "contrastive_negation": False}


def _scen_issues(replies: list[str]) -> int:
    """Total repetition issues across one scenario's assistant turns.

    Within-text detectors (templates) see the whole rollout as
    one blob; cross-message detectors (structural/phrase) see the turns as separate
    messages with the last as the current draft.

    Known bound: run_audit hardcodes require_last_message=True, so phrase repetition
    that appears only in MIDDLE turns (gone by the last turn) is not counted. This is
    deliberate — it matches the late-turn-weighted purple objective: the failure we
    target emerges and PERSISTS into the final turns, which is exactly what fires here."""
    if not replies:
        return 0
    blob = "\n\n".join(replies)
    rep = run_audit(blob, phrase_bank=[], assistant_messages=replies[:-1],
                    structural_text=replies[-1], audit_toggles=_TOGGLES,
                    phrase_min_messages=2)  # short gens: flag 2-word phrases on 2nd occurrence, not 3rd
    return rep.total_issues


def _sent_count(replies_by_scen: list[list[str]]) -> int:
    return sum(len(split_sentences(t)) for scen in replies_by_scen for t in scen)


def audit_density(replies_by_scen: list[list[str]]) -> float:
    """Repetition issues per generated sentence, summed over scenarios."""
    sents = _sent_count(replies_by_scen)
    issues = sum(_scen_issues(scen) for scen in replies_by_scen)
    return issues / sents if sents else 0.0


def coverage(replies_by_scen: list[list[str]]) -> int:
    """Total generated sentences — the prose-collapse guard."""
    return _sent_count(replies_by_scen)


def _demo() -> None:
    # ponytail: one runnable check — a distinctive phrase echoed across turns (incl.
    # the last) must score a higher density than varied prose, and coverage must count.
    repetitive = [["The shadowed red eyes flickered in the firelight again.",
                   "She met the shadowed red eyes across the crowded table.",
                   "Behind the mask the shadowed red eyes burned with malice."]]
    varied = [["The harbor smelled of salt and tar that morning.",
               "Gulls argued over a crate someone had dropped on the pier.",
               "A ferry coughed black smoke and pushed off from the dock."]]
    dr, dv = audit_density(repetitive), audit_density(varied)
    print(f"density: repetitive={dr:.3f}  varied={dv:.3f}")
    assert dr > dv, "repetitive output should score a higher audit density"
    assert coverage(varied) == 3, "coverage should count sentences"
    assert audit_density([[]]) == 0.0 and coverage([[]]) == 0, "empty output is finite"
    print("OK")


if __name__ == "__main__":
    _demo()
