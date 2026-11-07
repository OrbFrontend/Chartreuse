"""Filler augmentation for the paragraph axis -> interim/aug_items.jsonl (graded by grade_paragraphs --set aug).

WHY. The v2-rubric head (2026-11-07) has not learned filler in otherwise eventful prose. Insert two
filler beats into 150 clean test replies and the v2 grader calls 89% of them purple, but the head flags
5%. Its filler positives are almost all the 394 one-template `camouflaged` paragraphs. So this module
takes clean natural TRAIN paragraphs and makes two copies of each, both in the base's group:
  filler    k real Haiku filler / negation sentences inserted at sentence breaks
  concrete  the same k slots filled with real Haiku concrete action sentences instead
The concrete arm is the control. Without it the head could learn "an inserted sentence that does not
fit = purple". With it, the inserted sentences differ only in whether they carry an event. k is drawn
from 1-3: under rubric v2 one beat is a 1, so label 0, and two or more is a 2, so the head sees the
density line from both sides. Neither arm is labelled by construction: grade_paragraphs --set aug grades
both blind under rubric.txt, and build_dataset adds them to TRAIN only by majority vote, ties dropped.

THE INVENTORY. Every sentence is a real sentence from a Haiku train-split paragraph. Val and test are
never mined, so no group is shared with calibration or test.
  filler    the tier-1 families of OrbRewriter's lib/filler: breath, shift, tag, comment
  negation  nonaction (`She doesn't look away.`, `Nobody moves.`), negated inventories (`No door.`),
            negative definitions (`It isn't fear.`)
  concrete  5-12 words with a pronoun subject and an event verb, from v2-unanimous clean paragraphs; no
            filler or negation, no figurative `like` / `as`, no atmosphere words
A sentence that would not travel never enters it: a name, a quote, a digit or a possessive
(modern_slop_sets.injectable), or a qualifier (`Not yet.`). Beats are at most 9 words, with no comma or
simile, so they teach filler rather than ornament. A beat must fit the paragraph's tense (a tenseless beat
always fits) and name only pronouns its narration already has.

ROUNDS. Each round is drawn, graded and stored on its own (aug_items[_rN].jsonl), so a new round never
invalidates an earlier round's grades.
  1  PROBE_HOLDOUT strings kept out, uniform draw, 12 uses each
  2  formulaic beats in (`A pause.` is 60 of the ~870 train-split beats), drawn by train-split frequency,
     40 uses each; PROBE_FAMILIES kept out instead
The holdouts served a 2026-11-07 transfer probe (results in data/purple_para/README.md); they stay
because they are how the graded items were drawn. build_dataset trains on every graded row of every round.
Round 2 prefers base paragraphs round 1 did not use, but those are mostly the ones without slots or a
fitting beat, so most bases carry one pair per round.

    DEPURPLE_AXIS=purple_para python -m datagen.augment_filler --round 2 --stats   # inventory + samples
    DEPURPLE_AXIS=purple_para python -m datagen.augment_filler --round 1          # -> interim/aug_items.jsonl
    DEPURPLE_AXIS=purple_para python -m datagen.augment_filler --round 2          # -> interim/aug_items_r2.jsonl
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --set aug --round 1   # and --round 2
    DEPURPLE_AXIS=purple_para python -m datagen.augment_filler --demo
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter

from core.paths import INTERIM, UNIT
from text_segmentation import find_quote_spans, split_sentences

HAIKU = INTERIM / "haiku_paragraphs.jsonl"
HUMAN_GRADED = INTERIM / "human_graded.jsonl"
GRADES = INTERIM / "haiku_grades.jsonl"
OUT = INTERIM / "aug_items.jsonl"     # round 1; round N writes aug_items_rN.jsonl
NATURAL = ("reply", "native", "real")
N_BASES = 1500
K_WEIGHTS = {1: 0.25, 2: 0.45, 3: 0.30}
NEG_SHARE = 0.5                  # share of filler-arm beats drawn from the negation inventory
MAX_USES = 12
MAX_FILLER_WORDS = 9         # beats stay short and plain: an ornate one would teach ornament, which the head has
MIN_SENTS = 4
SEED = 0
# PROBE_HOLDOUT / PROBE_FAMILIES: what rounds 1 / 2 were drawn without (see ROUNDS in the docstring). Fixed:
# changing either would no longer reproduce the graded items.
PROBE_FAMILIES = ("shift", "negative_definition")
ROUNDS = {1: {"seed": 0, "holdout_strings": True, "holdout_families": (), "weighted": False, "max_uses": 12},
          2: {"seed": 1, "holdout_strings": False, "holdout_families": PROBE_FAMILIES, "weighted": True, "max_uses": 40}}


def items_path(rnd: int):
    return OUT if rnd == 1 else INTERIM / f"aug_items_r{rnd}.jsonl"
PROBE_HOLDOUT = frozenset(s.lower().rstrip(".") for s in [
    "A pause.", "A beat.", "Silence.", "Just that.", "Nothing more.", "Not yet.", "No answer.", "Not a word.",
    "Nothing else.", "No sound.", "A knock at the door.", "Two minutes.", "Footsteps on the stairs.",
    "Rain on the glass.", "A phone buzzes.", "Three missed calls.", "The kettle clicks off.", "A car horn outside.",
    "Ten past six.", "Boots in the hall.", "No one moves.", "No one answers.", "It isn't anger.", "It isn't fear.",
    "No light.", "No warning.", "Nothing changes.", "The door stays closed.", "Nobody comes.", "No one says anything."])

# ---- families: lib/filler (OrbRewriter, 2026-11-07) tier 1 + nonaction, read on norm() text ----
MAX_BEAT_WORDS = 14
_GENERIC = r"(?:moment|part|thing|point|difference|truth|kind of \w+|sort of \w+)"
_GADJ = r"(?:small|only|worst|best|hardest|strangest|strange|quiet|real|whole|last|first|important)"
_SUBJ = r"(?:she|he|they|i|you|we|it)"
_NEG = r"(?:does not|doesn't|did not|didn't|do not|don't|will not|won't|can't|cannot|could not|couldn't)"
_NONACT_V = (r"(?:look away|look up|look back|move|speak|answer|reply|respond|pull away|step back|step away|"
             r"come closer|blink|flinch|turn(?: away)?|look at (?:him|her|you|them|me)|let go|stop|smile|laugh|"
             r"reach for it|hurry|soften|say anything|say a word|break (?:the )?(?:silence|eye contact|gaze))")
_FRAME = r"(?:for a (?:long )?(?:moment|while|breath|beat|heartbeat|second)s?,? )"
FAMILIES = {
    "breath": [r"^(?:a|one|another) (?:breath|pause|beat|heartbeat|moment|silence)(?: of silence| passes| hangs?| held)?$",
               r"^(?:you|i|she|he|they|we) (?:breathe|breathes|exhale|exhales|inhale|inhales)(?: (?:in|out|slowly|once))?$",
               r"^(?:the )?silence (?:stretches|holds|lingers|settles|hangs|thickens|deepens|carries|between|after|"
               r"that follows|returns|is)\b"],
    "shift": [r"^something (?:in|about|inside|behind|beneath|under) [^,.]{1,40} (?:shifts|shifted|changes|changed|gives|"
              r"gave|softens|softened|breaks|cracks|loosens|settles|tightens|stirs|eases|has (?:changed|shifted))\b"],
    "tag": [r"^(?:just that|nothing more|nothing else|then another(?: \w+)?|only that|and then nothing)$"],
    "comment": [rf"^(?:and |but )?(?:that's|that is|this is|it's|it is|that was|this was|it was) (?:the|a) (?:{_GADJ} )?"
                rf"{_GENERIC}(?: (?:that|which|you|i|she|he|they|we) [a-z' ]{{1,25}})?$",
                r"^(?:and |but )?(?:that's|that is|it's|it is|this is) enough$",
                r"^that(?:'s| is) all(?: (?:she|he|they|i|you|we) (?:says|said))?$"],
    # negation: what is NOT -- rubric v2's negation_spam
    "nonaction": [rf"^{_SUBJ} {_NEG} (?:\w+ )?{_NONACT_V}\b[^,;]{{0,20}}$",
                  r"^neither of (?:us|them|you) (?:speaks?|spoke|moves?|moved|looks? away|says? anything|breaks? (?:it|the silence))\b",
                  rf"^{_FRAME}?(?:nothing|no one|nobody) (?:moves|moved|speaks|spoke|answers|answered|says anything|"
                  rf"said anything|happens|happened|comes|came)(?: (?:again|at all|for a (?:long )?(?:moment|while)))?$",
                  rf"^{_FRAME}(?:she|he|they|i|you|we|neither) (?:does(?: not|n't)|did(?: not|n't)|do(?: not|n't)|says? nothing|said nothing)\b"],
    "negated_inventory": [r"^no (?:[a-z]+ ){0,2}[a-z]+$"],
    "negative_definition": [r"^(?:and |but )?(?:it|this|that)(?:'s| is| was)(?: not|n't) (?:\w+ ){0,3}\w+$",
                            r"^(?:it|this|that) (?:isn't|wasn't) (?:\w+ ){0,3}\w+$"],
}
FILLER_FAMILIES = ("breath", "shift", "tag", "comment")
NEGATION_FAMILIES = ("nonaction", "negated_inventory", "negative_definition")
QUALIFIERS = frozenset({"almost", "not yet", "not quite", "not really", "not anymore", "no more", "still nothing"})
_COMPILED = {k: [re.compile(p) for p in v] for k, v in FAMILIES.items()}

# ---- concrete: a pronoun subject and a finite verb, nothing atmospheric ----
_ATMOS = re.compile(r"\b(?:like|as if|as though|silence|quiet|breath|moment|something|nothing|air|weight|heavy|light|"
                    r"shadow|feel|feels|felt|seem|seems|seemed|almost|somehow|still)\b")
_NEGWORD = re.compile(r"\b(?:not|no|never|nobody|nothing|neither|nor)\b|n't\b")
_PRON = re.compile(r"\b(i|me|my|you|your|we|us|our|she|her|he|him|his|they|them|their)\b")
_PAST_IRR = frozenset("""was were had did said went came saw took made felt knew thought got gave found told left kept held
    stood sat ran began brought heard meant turned caught threw drew fell grew wore broke chose spoke woke rose drove ate
    drank swam sang lay led lost paid sent spent built bought fought taught sought slept swept wept dealt felt hung shook
    struck stuck swung dug spun slid hid bit lit met fed fled bent lent knelt""".split())
_PRES_AUX = frozenset("is are am has have does do".split())
# verbs that make a sentence a thought, a want or a meta move rather than an event (`I hate how small it sounds.`)
_NONEVENT_V = frozenset("""know knew think thought want wanted need needed hate hated love loved like liked wonder wondered
    realize realized hope hoped remember remembered can could will would should must might may have has had ask asks
    asked am is are was were feel feels felt seem seems seemed guess mean means meant notice notices noticed try tries
    tried decide decides decided let lets wish wishes wished""".split())
_NARR_START = re.compile(r"^[*_]*(?:she|he|they|i|you|we|her|his|their|my|your)\s+[a-z']+", re.I)
_SV = re.compile(r"\b(i|you|he|she|it|we|they)\s+(?:\w+ly\s+)?([a-z']+)")
_AUX = re.compile(r"\b(is|are|am|has|was|were|had)\b")


def norm(s: str) -> str:
    return re.sub(r"[*_\"“”]", "", s.lower().replace("’", "'")).strip().rstrip(".!?…").strip()


def family(sentence: str) -> str | None:
    n = norm(sentence)
    if not n or len(n.split()) > MAX_BEAT_WORDS or n in QUALIFIERS:
        return None
    for k, pats in _COMPILED.items():
        if any(p.search(n) for p in pats):
            return k
    return None


def unquoted(text: str) -> str:
    """The narration: quoted speech blanked."""
    out, last = [], 0
    for a, b in find_quote_spans(text):
        out.append(text[last:a])
        last = b
    return "".join(out) + text[last:]


def tense(text: str) -> str | None:
    """past / present from pronoun-subject verbs and be/have auxiliaries in the narration, None when it has
    neither (`A breath.`, `No door.`). A heuristic: the grader is the label, a rare mismatch only adds noise
    to both arms alike."""
    t = unquoted(text).lower().replace("’", "'")
    p = q = 0
    for subj, v in _SV.findall(t):
        if v in _PAST_IRR or v.endswith("ed"):
            p += 1
        elif v in _PRES_AUX or v.endswith("'s") or v.endswith("'re") or v.endswith("'m") or \
                (subj in ("he", "she", "it") and v.endswith("s") and not v.endswith("ss")):
            q += 1
    for a in _AUX.findall(t):
        p, q = (p + 1, q) if a in ("was", "were", "had") else (p, q + 1)
    return None if p == q == 0 else "past" if p > q else "present"


def pronouns(text: str) -> set[str]:
    return set(_PRON.findall(unquoted(text).lower()))


def travels(sentence: str) -> bool:
    """Nothing from its own scene: no name (a capital past the first word, `I` aside), digit, quote, possessive,
    markup or brackets."""
    w = sentence.split()
    return (not re.search(r"[0-9\"“”<>\[\]()#:;]|'s\b|’s\b", sentence)
            and not any(x[:1].isupper() and x.strip(".,!?") not in ("I", "I'm", "I'd", "I've", "I'll") for x in w[1:]))


def clean_sentence(s: str) -> str:
    return s.strip().strip("*_").strip()


def beat(sentence: str) -> bool:
    n = norm(sentence)
    return len(n.split()) <= MAX_FILLER_WORDS and "," not in n and not re.search(r"\b(?:like|as)\b", n)


def concrete(sentence: str) -> bool:
    n = norm(sentence)
    w = n.split()
    return (5 <= len(w) <= 12 and w[1] not in _NONEVENT_V and sentence.rstrip("*_ ").endswith(".") and family(sentence) is None
            and not _ATMOS.search(n) and not _NEGWORD.search(n) and n.split()[0] in
            ("i", "you", "he", "she", "they", "we") and tense(sentence) is not None and "," not in n)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def key(text: str) -> str:
    return " ".join(text.lower().split())


def v2_votes() -> dict[str, list[int]]:
    from datagen.grade_paragraphs import RUBRIC, rubric_tag
    tag = rubric_tag(RUBRIC.read_text(encoding="utf-8"))
    return {key(g["text"]): g["audit_scores"] for g in read_jsonl(GRADES) if g.get("rubric") == tag}


def mine(rows: list[dict], votes: dict, holdout_strings: bool = True,
         holdout_families: tuple = ()) -> tuple[dict[str, list[str]], Counter]:
    """Distinct injectable sentences per inventory, from train-split Haiku rows only, and each one's count."""
    inv, freq = {"filler": {}, "negation": {}, "concrete": {}}, Counter()
    for r in rows:
        if r.get("split") != "train":
            continue
        a = votes.get(key(r["text"]), [])
        clean_par = r["variant"] in NATURAL + ("terse",) and len(a) >= 2 and all(s < 2 for s in a)
        for s in split_sentences(unquoted(r["text"])):
            c = clean_sentence(s)
            if not c or not travels(c) or (holdout_strings and norm(c) in PROBE_HOLDOUT) or not c[0].isupper():
                continue
            f = family(c)
            if f and (not beat(c) or f in holdout_families):
                continue
            if f in FILLER_FAMILIES + NEGATION_FAMILIES or (clean_par and concrete(c)):
                freq[inv["filler" if f in FILLER_FAMILIES else "negation" if f else "concrete"].setdefault(norm(c), c)] += 1
            if f in FILLER_FAMILIES:
                inv["filler"].setdefault(norm(c), c)
            elif f in NEGATION_FAMILIES:
                inv["negation"].setdefault(norm(c), c)
            elif clean_par and concrete(c):
                inv["concrete"].setdefault(norm(c), c)
    return {k: sorted(v.values()) for k, v in inv.items()}, freq


def slots(text: str) -> list[int]:
    """Insertion offsets: the end of every sentence but the last, never inside quoted speech, never inside an
    action marker left open."""
    quotes, out, pos = find_quote_spans(text), [], 0
    sents = [s.strip() for s in split_sentences(text) if s.strip()]
    for s in sents[:-1]:
        a = text.find(s, pos)
        if a < 0:
            continue
        b = pos = a + len(s)
        while b < len(text) and text[b] in "*_\"”":
            b += 1
        if 0 < b < len(text.rstrip()) and not any(qa < b < qb for qa, qb in quotes) and text[:b].count("*") % 2 == 0:
            out.append(b)
    return out


def insert(text: str, at: list[int], beats: list[str]) -> str:
    for b, s in sorted(zip(at, beats), reverse=True):
        s2 = f"*{s}*" if text[b - 1:b] == "*" else s       # the paragraph's action-marker convention
        text = text[:b] + " " + s2 + text[b:]
    return text


def fits(sentence: str, tz: str | None, pron: set[str]) -> bool:
    t = tense(sentence)
    return t in (None, tz) and pronouns(sentence) <= pron


def make(bases: list[dict], inv: dict, rng: random.Random, max_uses: int = MAX_USES,
         n: int | None = None, freq: Counter | None = None) -> tuple[list[dict], Counter]:
    """Two items (filler, concrete) per base, until `n` bases are written or the pool runs out. With `freq`,
    beats are drawn by their corpus frequency instead of uniformly (concrete sentences stay uniform)."""
    used, out, st = Counter(), [], Counter()
    ks, kw = list(K_WEIGHTS), list(K_WEIGHTS.values())
    for r in bases:
        if n is not None and len(out) >= 2 * n:
            break
        text = r["text"]
        sl = slots(text)
        k = rng.choices(ks, kw)[0]
        if len(sl) < k:
            st["too few slots"] += 1
            continue
        tz, pron = tense(text), pronouns(text)
        fit = {n: [c for c in inv[n] if used[c] < max_uses and fits(c, tz, pron) and norm(c) not in key(text)]
               for n in inv}
        beats = []
        for _ in range(k):
            pool = fit["negation"] if fit["negation"] and rng.random() < NEG_SHARE else fit["filler"]
            pool = [c for c in pool if c not in beats]
            if not pool:
                break
            beats.append(rng.choices(pool, [freq[c] for c in pool])[0] if freq else rng.choice(pool))
        conc = rng.sample(fit["concrete"], k) if len(fit["concrete"]) >= k else []
        if len(beats) < k or not conc:
            st["no fitting inventory"] += 1
            continue
        at = sorted(rng.sample(sl, k))
        used.update(beats + conc)
        for arm, ins in (("filler", beats), ("concrete", conc)):
            t = insert(text, at, ins)
            out.append({"id": "a" + hashlib.md5(key(t).encode()).hexdigest()[:12], "text": t, "arm": arm, "k": k,
                        "inserted": ins, "families": [family(x) for x in ins] if arm == "filler" else [],
                        "group_id": r["group_id"], "base_id": base_id(text), "base_variant": r["variant"],
                        "is_real": r.get("is_real", False)})
        st[f"written k={k}"] += 1
    return out, st


def base_id(text: str) -> str:
    return "b" + hashlib.md5(key(text).encode()).hexdigest()[:12]


def narrative(text: str) -> bool:
    """Fiction or RP, not a how-to: at least two sentences open on a pronoun subject, no markdown structure."""
    if re.search(r"^\s*(?:[-*] |\d+\. |#)|\*\*|```", text, re.M):
        return False
    return sum(bool(_NARR_START.match(s.strip())) for s in split_sentences(text)) >= 2


def bases(rows: list[dict], votes: dict, human: list[dict] = ()) -> list[dict]:
    """Clean narrative train paragraphs, at least MIN_SENTS sentences, no filler/negation beat already:
    Haiku natural rows that are v2 unanimous 0, and graded-clean human rows outside build_dataset's held-out
    doc groups (bluemoon RP is the register the head serves)."""
    from datagen.build_dataset import HELDOUT_HUMAN_FRAC, bucket
    from datagen.grade_paragraphs import RUBRIC, rubric_tag
    tag = rubric_tag(RUBRIC.read_text(encoding="utf-8"))
    hum = [{**r, "split": "train", "variant": r["variant"], "_a": r["audit_scores"]} for r in human
           if r.get("rubric") == tag and r["label"] == 0 and not bucket(r["group_id"], HELDOUT_HUMAN_FRAC, "heldout")]
    out = []
    for r in list(rows) + hum:
        a = r.get("_a") or votes.get(key(r["text"]), [])
        if (r.get("split") == "train" and (r["variant"] in NATURAL or r["variant"].startswith("human:"))
                and len(a) >= 2 and all(s < 2 for s in a)
                and len(split_sentences(r["text"])) >= MIN_SENTS and narrative(r["text"])
                and not any(family(clean_sentence(s)) for s in split_sentences(unquoted(r["text"])))):
            out.append({k: v for k, v in r.items() if k != "_a"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="print the inventory and samples, write nothing")
    ap.add_argument("--n", type=int, default=N_BASES, help="base paragraphs (2 items each)")
    ap.add_argument("--round", type=int, default=max(ROUNDS), choices=sorted(ROUNDS),
                    help="1: probe strings held out, uniform; 2: formulaic beats in, frequency-weighted")
    a = ap.parse_args()
    if UNIT != "paragraph":
        raise SystemExit("FATAL: augment_filler is for a paragraph axis (DEPURPLE_AXIS=purple_para)")
    rows, votes = read_jsonl(HAIKU), v2_votes()
    if not rows or not votes:
        raise SystemExit("FATAL: run datagen.import_paragraphs and grade_paragraphs --set haiku first")
    cfg = ROUNDS[a.round]
    inv, freq = mine(rows, votes, cfg["holdout_strings"], cfg["holdout_families"])
    pool = bases(rows, votes, read_jsonl(HUMAN_GRADED))
    print(f"inventory: " + ", ".join(f"{k} {len(v)}" for k, v in inv.items()) + f"; eligible bases {len(pool)}")
    rng = random.Random(cfg["seed"])
    if a.stats:
        for k, v in inv.items():
            print(f"\n{k}: " + " | ".join(rng.sample(v, min(25, len(v)))))
        fams = Counter(family(c) for c in inv["filler"] + inv["negation"])
        print("\nfamilies:", dict(fams))
    rng.shuffle(pool)
    # later rounds take the base paragraphs no earlier round used first (stable partition of the shuffled pool)
    seen = {it["base_id"] for r in range(1, a.round) for it in read_jsonl(items_path(r))}
    pool = [r for r in pool if base_id(r["text"]) not in seen] + [r for r in pool if base_id(r["text"]) in seen]
    items, st = make(pool, inv, rng, max_uses=cfg["max_uses"], n=a.n, freq=freq if cfg["weighted"] else None)
    for it in items:
        it["round"] = a.round
    print(f"items {len(items)}  {dict(sorted(st.items()))}  bases by variant "
          f"{dict(Counter(it['base_variant'] for it in items[::2]).most_common())}")
    used = Counter(c for it in items for c in it["inserted"])
    print(f"distinct inserted: filler-arm {len({c for it in items if it['arm'] == 'filler' for c in it['inserted']})}, "
          f"concrete-arm {len({c for it in items if it['arm'] == 'concrete' for c in it['inserted']})}; "
          f"most used {used.most_common(5)}")
    if a.stats:
        for it in items[:6]:
            print(f"\n[{it['arm']} k={it['k']}] {it['text'][:500]}")
        return
    out = items_path(a.round)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    print(f"-> {out}")


def demo():
    for s, f in [("A breath.", "breath"), ("Nothing more.", "tag"), ("She doesn't look away.", "nonaction"),
                 ("Nobody moves.", "nonaction"), ("No door.", "negated_inventory"), ("It isn't fear.", "negative_definition"),
                 ("And that's the worst part.", "comment"), ("Something in her softens.", "shift")]:
        assert family(s) == f, (s, family(s))
    for s in ["Not yet.", "She sets the cup down and walks to the door.", "He did not answer the letter for three years, "
              "and by then the house had been sold."]:
        assert family(s) is None, s
    assert tense("She sets the cup down.") == "present" and tense("She set the cup down. He was gone.") == "past"
    assert tense("A breath.") is None and tense('He waits. "I was there," she says.') == "present"
    assert concrete("She sets the cup on the counter.") and not concrete("She feels the weight of it.")
    assert not concrete("I hate how small it sounds.") and not concrete("You ask what it is.")
    assert narrative(base_t := "She opens the door. He looks up. They talk.") and not narrative("- step one. She goes. He goes.")
    assert not concrete("She doesn't move.") and not travels("Then Mira sets it down.") and travels("I set it down.")
    t = 'He opens the door. "Come in. Sit." She sits. *He pours tea.* They drink.'
    sl = slots(t)
    assert all(not any(a < b < c for a, c in find_quote_spans(t)) for b in sl) and t.index("They") - 1 in sl
    got = insert(t, [sl[0], sl[-1]], ["A breath.", "Just that."])
    assert got.startswith("He opens the door. A breath. ") and got.endswith("*He pours tea.* *Just that.* They drink.")
    assert fits("She doesn't look away.", "present", {"she"}) and not fits("He doesn't look away.", "present", {"she"})
    assert not fits("She pulled it down.", "present", {"she"})
    base = {"text": "She opens the door. He looks up from the desk. She drops the file on it. He reads the first page.",
            "group_id": "g1", "variant": "reply"}
    inv = {"filler": ["A breath.", "Just this."], "negation": ["She doesn't look away.", "No door."],
           "concrete": ["She pulls out a chair.", "He sets his pen down.", "She takes off her coat."]}
    items, st = make([base] * 4, inv, random.Random(1))
    assert items and all(it["group_id"] == "g1" for it in items) and {it["arm"] for it in items} == {"filler", "concrete"}
    for f, c in zip(items[::2], items[1::2]):
        assert f["k"] == c["k"] and len(f["inserted"]) == f["k"] and all(x in f["text"] for x in f["inserted"])
        assert all(x in c["text"] for x in c["inserted"]) and not c["families"]
    assert all(norm(x) not in PROBE_HOLDOUT for x in sum(inv.values(), []))
    # frequency-weighted draw: the common beat dominates, the cap still binds
    w, _ = make([base] * 300, {**inv, "negation": []}, random.Random(2), max_uses=30,
                freq=Counter({"A breath.": 50, "Just this.": 1}))
    got = Counter(x for it in w if it["arm"] == "filler" for x in it["inserted"])
    assert 25 <= got["A breath."] <= 30 and got["Just this."] < got["A breath."], got   # the cap binds
    # mine(): holdout switches
    rows_ = [{"text": "She sets the cup on the counter. A pause. Something in the room shifts. It wasn't relief. "
                      "He looks up from the desk.", "split": "train", "variant": "reply"}]
    inv1, f1 = mine(rows_, {}, True, ())
    inv2, f2 = mine(rows_, {}, False, PROBE_FAMILIES)
    assert "A pause." not in inv1["filler"] and "Something in the room shifts." in inv1["filler"]
    assert "A pause." in inv2["filler"] and "Something in the room shifts." not in inv2["filler"]
    assert "It wasn't relief." in inv1["negation"] and "It wasn't relief." not in inv2["negation"] and f2["A pause."] == 1
    print("demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
