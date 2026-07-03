"""Generate matched synthetic pairs for the active style axis.

DEPURPLE_AXIS picks the style axis; the rewrite direction is NOT a flag — it's implied by
which seed dir a sentence comes from, and ONE run does both:

    purple:  raw/good/*.txt -> worsen  (plain seed 0 -> purple twin 1, intensity-banded)
             raw/bad/*.txt  -> improve (purple seed 1 -> honest twin 0)
    euphemism: raw/good/*.txt -> worsen  (direct seed 0 -> rename-only euphemized twin 1)
             raw/bad/*.txt  -> improve (coy seed 1 -> rename-only direct twin 0),
             kept if the rename signature holds — the real seed IS one pole, either direction.

The label IS the cell, not a post-hoc score. Content is held constant across a pair so the
model learns *style*, not topic. Purple positives are intensity-banded mild->overwrought so
they aren't all cartoonish. Both axes trust the real seed as one pole and generate the other.

Each axis is one entry in REGISTRY (its prompts + a build hook, and an optional prepare hook
for prompt-fill state like the intensity band). Dispatch is data-driven — SPEC = REGISTRY[AXIS] —
so adding an axis is one Axis(...) block here plus seeds under data/<axis>/raw/ and a scorer;
no `if AXIS == ...` to hunt down. Every axis emits the SAME unified, pole-ordered pair schema:

    data/<axis>/interim/<axis>_pairs.jsonl
    {"neg":<label-0 pole>, "pos":<label-1 pole>, "group_id", "source", "real":"neg"|"pos", ...}

`real` names which pole came from the real seed (the other is generated); axis-specific extras
(e.g. purple's "intensity") ride along as extra keys. build_dataset/direction.py read neg/pos
generically, so a new axis needs no reader changes.

LLM via OpenAI-compatible endpoint. Config from env:
    OPENAI_API_KEY   (optional — local endpoints ignore it; defaults to sk-noauth)
    OPENAI_BASE_URL  (optional — set for OpenRouter / vLLM / local; unset = api.openai.com)
    GEN_MODEL        (optional — omit for single-model local servers that ignore the
                      "model" field; if the endpoint actually needs one, it'll reject the
                      request and that error surfaces per-seed below)

    python -m datagen.generate_synthetic --dry-run   # print the prompt(s) for the first seed, no call
    python -m datagen.generate_synthetic --limit 12  # small pilot, eyeball the output
    python -m datagen.generate_synthetic             # full run (resumes; skips pairs already done)

Resumable: a pair's group_id is a stable hash of the seed, and seeds already present in the output
are skipped — re-runs never duplicate or re-pay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import syntok.segmenter as segmenter

from core.paths import SEED_DIR, WORSE_DIR, AXIS, pairs_path

# Unified pair file, one name convention for every axis (data/<axis>/interim/<axis>_pairs.jsonl).
OUT = Path(pairs_path(AXIS))

# Intensity bands and their share of positives. Weighted toward mild/moderate on purpose: the gate's
# job is catching the *subtle* floweriness LLM output actually produces, not just cartoon excess.
INTENSITY = {
    "mild": (35, "a SUBTLE touch of purple — keep the sentence almost entirely intact and at roughly "
                 "the SAME LENGTH; add ONE unearned ornament (a decorative modifier, a faint note of "
                 "sentiment, a small atmospheric flourish) while keeping the ORIGINAL vocabulary — do "
                 "NOT just swap a word for a fancier synonym; rare words alone are not purple, excess "
                 "ornament is. The floweriness should be faint enough a reader might barely notice it"),
    "moderate": (40, "clearly overwritten — stack a few modifiers past the point of information, add "
                     "one strained metaphor or a note of sentimentality the moment doesn't warrant"),
    "overwrought": (25, "maximally purple — pile on ornate/archaic diction, mixed or collapsing "
                        "metaphor, melodrama, and empty atmospheric padding; thoroughly overwrought"),
}

SYSTEM = (
    "You rewrite sentences into PURPLE PROSE for a training dataset. Purple = ornament in excess of "
    "what the thought requires: the prose calls attention to its own decoration instead of its "
    "meaning. Techniques: modifier stacking, ornate/archaic/inflated diction, strained or mixed "
    "metaphor, sentimentality/melodrama, cliche-as-decoration, and empty atmospheric padding "
    "('the air grew thick with tension'). You add excess on purpose — that is the goal here."
)

USER_TMPL = (
    "Rewrite the sentence below as PURPLE PROSE at this intensity: {desc}.\n\n"
    "Rules:\n"
    "- Keep the SAME core meaning and concrete content (same people/things/events). Decorate it; "
    "don't change what happens.\n"
    "- Output EXACTLY ONE sentence. No line breaks.\n"
    "- Output ONLY the rewritten sentence — no preamble, no quotes, no explanation.\n\n"
    "Sentence: {orig}"
)

# --- "improve" / make-it-better prompt (purple axis, raw/bad seeds) ------------------
# worsen seeds the BETTER pole (plain) and writes WORSE (purple). improve goes the other way:
# seed a real purple sentence you hate (raw/bad, label 1) and write its honest twin (better,
# label 0). Same unified {neg:0, pos:1} pair schema -> build_dataset needs no change.
REVERSE_SYSTEM = (
    "You rewrite bad, overwritten PURPLE PROSE into BETTER, honest prose for a training "
    "dataset. Purple = ornament in excess of what the thought requires. Strip the excess: "
    "cut stacked modifiers, ornate/archaic diction, strained metaphor, sentimentality, and "
    "atmospheric padding. Replace tired figurative cliches ('barely a whisper', 'the air "
    "grew thick with tension', etc.) with the plain, literal thing they gesture at. Keep the same "
    "concrete content and events — say it cutely and honestly, not blandly."
)
REVERSE_USER = (
    "Rewrite the sentence below as BETTER, honest prose: remove the ornament that exceeds "
    "the thought and replace any tired cliche with the literal thing it means. Keep the "
    "SAME core meaning and concrete content (same people/things/events).\n"
    "- Output EXACTLY ONE sentence. No line breaks.\n"
    "- May add fluff/character to it if needed.\n"
    "- If the sentence describes only voice or eyes, rewrite then add a following analogy after a comma.\n"
    "- If the sentence is a contrastive negation (Not X; but Y), just write direct intent (Y).\n"
    "- Output ONLY the rewritten sentence — no preamble, no quotes, no explanation.\n\n"
    "Sentence: {orig}"
)

# --- de-euphemism mode (DEPURPLE_AXIS=euphemism) ---------------------------------------
# Directional like purple: the real direct seed IS the label-0 pole (trust the original),
# one worsen call renames it into the euphemized label-1 twin, kept if the signature holds.
EUPH_COY_SYSTEM = (
    "You rewrite explicit adult sentences into COY, EUPHEMISTIC wording for a writing-style "
    "dataset — the evasive register romance writers use. RENAME ONLY: replace each direct "
    "anatomical or sex-act word with a euphemism (heat, core, entrance, member, need, seed, "
    "length, folds, wetness, skin, anatomy, length, rosebud, center, petals, opening, etc.). Keep the sentence's structure, length, and exactly what happens "
    "identical. Never fade to black, summarize, or omit the act."
)
EUPH_COY_USER = (
    "Rewrite the sentence below in COY, EUPHEMISTIC wording. Replace every direct anatomical "
    "or sex-act word with a softer euphemism, but keep the SAME structure, and events.\n"
    "- Do NOT fade to black, summarize, or skip the act.\n"
    "- Replace creative adjectives with with boring ones.\n"
    "- Output EXACTLY ONE sentence. No line breaks.\n"
    "- Output ONLY the sentence — no preamble, no quotes, no explanation.\n\n"
    "Sentence: {orig}"
)

# improve pass (de-euphemism): mirror of the coy prompt. The real COY seed IS the label-1
# pole; one call renames it back into DIRECT, explicit wording (the label-0 twin).
EUPH_DIRECT_SYSTEM = (
    "You rewrite COY, EUPHEMISTIC adult sentences into GROSS, EXPLICIT wording for a "
    "writing-style dataset — the plain register, not the evasive one. RENAME ONLY: replace each "
    "euphemism (heat, core, entrance, member, need, seed, length, folds, petals, opening, sex, "
    "etc.) with the filthy and direct anatomical or sex-act word it stands in for. Keep the sentence's "
    "structure, length, and exactly what happens identical. Never fade to black or summarize."
)
EUPH_DIRECT_USER = (
    "Rewrite the sentence below in DIRECT, EXPLICIT wording. Replace every euphemism with the "
    "plain anatomical or sex-act word it stands in for, but keep the SAME structure and events.\n"
    "- Do NOT fade to black, summarize, or skip the act.\n"
    "- Output EXACTLY ONE sentence. No line breaks.\n"
    "- Output ONLY the sentence — no preamble, no quotes, no explanation.\n\n"
    "Example: A bald, prepubescent gash—the jailbait's sterile cleft seizes up in an un-fucked, un-raped spasm.\n\n"
    "Sentence: {orig}"
)

# Omission/summary markers: a euphemized rewrite that fades to black or labels the whole act
# ("made love") instead of renaming its parts is rejected — we want renames, not summaries.
_FADE = re.compile(
    r"fade to black|rest of the night|the morning after|next morning|made love|became one"
    r"|spent themselves|left (?:to|nothing to) the imagination|everything else|what followed",
    re.IGNORECASE)

# --- per-axis prompts -------------------------------------------------------------------
# Each axis maps a direction -> (system, user_template, temperature). "worsen" raises the
# style axis (better->worse, label 1 / pos); "improve" lowers it (worse->better, label 0 /
# neg). Direction is NOT a flag — it's implied by the seed dir (raw/good->worsen,
# raw/bad->improve) and one run does both. user_template takes {orig}; purple/worsen also
# takes {desc} (the intensity band, filled by the axis's prepare hook). An axis is worsen-only
# exactly when it omits its "improve" entry.
PURPLE_PROMPTS = {
    "worsen":  (SYSTEM, USER_TMPL, 0.95),
    "improve": (REVERSE_SYSTEM, REVERSE_USER, 0.95),
}
EUPH_PROMPTS = {
    "worsen":  (EUPH_COY_SYSTEM, EUPH_COY_USER, 0.9),
    "improve": (EUPH_DIRECT_SYSTEM, EUPH_DIRECT_USER, 0.9),
}


@dataclass(frozen=True)
class Axis:
    """One style axis's synthetic-generation config. `prompts` drives the LLM call; `build`
    turns a validated rewrite into the unified pair record (and applies any axis-specific accept
    guard, returning None to skip); `prepare` fills the {desc} prompt slot (default: none).
    Adding an axis = one Axis(...) entry in REGISTRY below — no `if AXIS == ...` anywhere."""
    prompts: dict[str, tuple[str, str, float]]
    build: Callable[[str, str, str, str], dict | None]      # (source, seed, direction, generated)
    prepare: Callable[[str, str], str] = lambda seed, direction: ""


def improves(ax: "Axis") -> bool:
    """Does this axis run an improve pass? Derived from its prompts (the single source for
    directions), so no separate list drifts out of sync: an axis is worsen-only exactly when it
    has no "improve" entry. Adding that entry + real seeds in raw/bad + an improve-side accept
    guard makes it symmetric — that's all symmetry needs, no fork to hunt."""
    return "improve" in ax.prompts


def stable_gid(orig: str) -> str:
    return "pair_" + hashlib.sha1(orig.encode("utf-8")).hexdigest()[:12]


def pick_intensity(orig: str) -> str:
    """Deterministic per-seed so resumed runs assign the same band; weighted by the shares above."""
    bucket = int(hashlib.sha1(b"i" + orig.encode()).hexdigest(), 16) % 100
    cum = 0
    for name, (share, _) in INTENSITY.items():
        cum += share
        if bucket < cum:
            return name
    return "moderate"


def load_seeds(seed_dir=SEED_DIR) -> list[tuple[str, str]]:
    """(source, sentence) from every *.txt in the seed dir, deduped, order stable."""
    seen, out = set(), []
    for f in sorted(seed_dir.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append((f.stem, s))
    return out


def n_sentences(text: str) -> int:
    return sum(1 for para in segmenter.process(text) for _ in para)


# Real LLM floweriness inflates a sentence; it doesn't triple it. Reject rewrites longer than this
# multiple of the original — those are the cartoonish over-generations, not the subtle drift we target.
MAX_LEN_RATIO = 2.5


def clean_rewrite(raw: str, orig: str) -> str | None:
    """Validate the model output is a usable single-sentence purple twin, or None to skip."""
    s = (raw or "").strip().strip('"').strip("'").strip()
    if "\n" in s or len(s) < 12:
        return None
    if s[-1] not in ".?!":
        return None
    if s.lower() == orig.lower():  # model refused / echoed the input
        return None
    if len(s) > MAX_LEN_RATIO * len(orig):  # over-inflated; not the subtle case we're after
        return None
    if n_sentences(s) != 1:        # must be exactly one sentence
        return None
    return s


def messages_for(ax: Axis, direction, orig, desc=""):
    """The axis's `direction` prompt, formatted into chat messages. desc fills the intensity
    slot for purple/worsen; prompts without a {desc} placeholder ignore it."""
    system, user, _temp = ax.prompts[direction]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(orig=orig, desc=desc)}]


def rewrite(client, model, ax: Axis, direction, seed, desc=""):
    """One prompt -> one validated single-sentence rewrite of seed (or None to skip)."""
    temp = ax.prompts[direction][2]
    return clean_rewrite(chat(client, model, messages_for(ax, direction, seed, desc),
                              temperature=temp), seed)


def rename_ok(seed: str, generated: str) -> bool:
    """Symmetric rename sanity check for the euphemism axis, either direction: trust the real
    seed as one pole and judge only the generated twin. No lexicon — gemma 4 follows the rename
    instruction reliably, and a lexicon can't keep up with its vocabulary (it falsely rejects
    good renames). Only the cheap, unambiguous failures are caught: an unchanged echo, a big
    length change (summary/expansion), or a fade-to-black / omission marker in the OUTPUT.
    Direction-agnostic: worsen passes (direct seed, euph out), improve passes (coy seed, direct out)."""
    if generated.strip().lower() == seed.strip().lower():
        return False
    if not 0.6 <= len(generated) / max(len(seed), 1) <= 1.6:
        return False
    return _FADE.search(generated) is None


# Sidecar log of *attempted* seeds (kept OR validation-rejected), one group_id per line. Without
# it, resume only knows the kept pairs, so every previously-rejected seed gets re-attempted and
# re-paid every run. Delete this file to force a full retry of the stochastic rejects.
ATTEMPTED = OUT.with_suffix(".attempted")


def already_done() -> set[str]:
    done = set()
    if OUT.exists():
        done |= {json.loads(l)["group_id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    if ATTEMPTED.exists():
        done |= {l.strip() for l in ATTEMPTED.read_text(encoding="utf-8").splitlines() if l.strip()}
    return done


def make_client():
    from openai import OpenAI
    # ponytail: local endpoint ignores the key, but the OpenAI client requires a non-empty string
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-noauth"), base_url=os.environ.get("OPENAI_BASE_URL"))


# Body to disable reasoning. Default suits local gemma (chat_template_kwargs); deepseek wants
# {"thinking": {"type": "disabled"}}. Override per-endpoint via THINKING_OFF_BODY (JSON).
# ponytail: env knob over endpoint-sniffing — one value, default keeps the committed gemma path.
_THINK_OFF = json.loads(os.environ.get(
    "THINKING_OFF_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}'))


def chat(client, model, messages, temperature=0.95, max_tokens=300) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        # Reasoning models otherwise burn the whole budget thinking and return empty content.
        extra_body=_THINK_OFF,
    )
    return resp.choices[0].message.content


# --- per-axis prepare/build hooks (referenced by REGISTRY below) ------------------------
def _purple_prepare(seed: str, direction: str) -> str:
    """Fill the {desc} slot with the seed's intensity band on worsen; improve carries no band."""
    return INTENSITY[pick_intensity(seed)][1] if direction == "worsen" else ""


def _purple_build(source, seed, direction, generated) -> dict:
    """Purple-style: the real seed IS one labeled pole; the LLM wrote the opposite pole.
    worsen: plain seed(neg) -> purple(pos), intensity-banded. improve: purple seed(pos) ->
    honest(neg) (shorter, so the over-inflation guard never trips; the echo guard in
    clean_rewrite catches a non-improvement). Pole-ordered by direction; `real` marks the seed
    side, `intensity` rides along as an axis extra."""
    if direction == "worsen":
        neg, pos, real, intensity = seed, generated, "neg", pick_intensity(seed)
    else:
        neg, pos, real, intensity = generated, seed, "pos", "better"
    return {"neg": neg, "pos": pos, "group_id": stable_gid(seed),
            "source": source, "real": real, "intensity": intensity}


def _euph_build(source, seed, direction, generated) -> dict | None:
    """Directional both ways: the real seed IS one labeled pole (trust it), the LLM renamed it
    into the other-register twin — kept only if the rename signature holds (else None). worsen:
    real direct seed(neg) -> euphemized(pos). improve: real coy seed(pos) -> direct(neg). `real`
    marks the seed side so build_dataset tracks is_real on the right pole."""
    if not rename_ok(seed, generated):
        return None
    if direction == "worsen":
        neg, pos, real = seed, generated, "neg"
    else:
        neg, pos, real = generated, seed, "pos"
    return {"neg": neg, "pos": pos, "group_id": stable_gid(seed),
            "source": source, "real": real}


# --- the axis registry: adding an axis is one entry here --------------------------------
REGISTRY = {
    "purple":    Axis(PURPLE_PROMPTS, _purple_build, _purple_prepare),
    "euphemism": Axis(EUPH_PROMPTS,   _euph_build),
}
try:
    SPEC = REGISTRY[AXIS]                    # the active axis's generation config
except KeyError:
    raise SystemExit(f"FATAL: no synthetic-generation config for DEPURPLE_AXIS={AXIS!r} "
                     f"(known axes: {', '.join(REGISTRY)})")


def generate_one(client, model, source, seed, direction):
    """One seed -> one unified pair record (or None to skip), via the active axis's hooks."""
    desc = SPEC.prepare(seed, direction)
    out = rewrite(client, model, SPEC, direction, seed, desc)
    return SPEC.build(source, seed, direction, out) if out is not None else None


def seeds_to_do(worsen_only) -> list[tuple[str, str, str]]:
    """(source, seed, direction) for every seed. Direction is implied by the dir: directional
    axes worsen raw/good AND improve raw/bad in the same run; worsen-only axes only worsen
    raw/good (the real seed is the label-0 pole)."""
    if worsen_only:
        return [(src, s, "worsen") for src, s in load_seeds(SEED_DIR)]
    return ([(src, s, "worsen") for src, s in load_seeds(SEED_DIR)]
            + [(src, s, "improve") for src, s in load_seeds(WORSE_DIR)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap number of new pairs (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API calls")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt(s) for the first seed, no call")
    args = ap.parse_args()

    worsen_only = not improves(SPEC)
    seeds = seeds_to_do(worsen_only)
    if not seeds:
        where = SEED_DIR if worsen_only else f"{SEED_DIR}/ or {WORSE_DIR}/"
        sys.exit(f"no seeds in {where} — run sample_seeds.py or drop *.txt there first")

    if args.dry_run:
        src, seed, direction = seeds[0]
        print(f"# {AXIS} ({direction}): {len(seeds)} seeds. First ({src}):\n")
        desc = SPEC.prepare(seed, direction)
        print(f"=== {AXIS}/{direction} ===")
        for m in messages_for(SPEC, direction, seed, desc):
            print(f"[{m['role']}]\n{m['content']}\n")
        return

    done = already_done()
    todo = [(s, o, d) for s, o, d in seeds if stable_gid(o) not in done]
    # Deterministic shuffle so a --limit pilot samples across sources/directions, not just the first file.
    random.Random(0).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(seeds)} seeds, {len(done)} already done, generating {len(todo)} ...", flush=True)
    if not todo:
        return

    client = make_client()
    model = os.environ.get("GEN_MODEL")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with OUT.open("a", encoding="utf-8") as fh, ATTEMPTED.open("a", encoding="utf-8") as af, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate_one, client, model, s, o, d): o for s, o, d in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                # Infra failure (endpoint down / rate limit), not a rewrite outcome — don't log it
                # as attempted, or a transient blip would permanently skip the seed.
                skipped += 1
                print(f"  ! error: {e}", flush=True)
                continue
            # Log every completed attempt (kept or validation-rejected) so resume never re-pays it.
            af.write(stable_gid(futs[fut]) + "\n")
            af.flush()
            if rec is None:
                skipped += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1
            if kept % 25 == 0:
                print(f"  {kept} kept ...", flush=True)
    print(f"done: {kept} pairs written, {skipped} skipped -> {OUT}")


def demo():
    """Self-check rewrite validation and intensity selection without network calls."""
    o = "A plain sentence of light."  # realistic-length orig so the ratio check doesn't trip
    assert clean_rewrite('"A gilded and gleaming sentence of purest light."', o) \
        == "A gilded and gleaming sentence of purest light."
    assert clean_rewrite("Two sentences. Here is another.", o) is None  # not single
    assert clean_rewrite("no terminal mark", o) is None
    assert clean_rewrite("Same.", "same.") is None                        # echoed input
    assert clean_rewrite("A" * 50 + " gilded sentence here.", "short orig.") is None  # over-inflated
    # reverse direction: a shorter honest rewrite of a purple seed passes (the over-inflation
    # guard never trips when shorter; the echo guard still catches a non-improvement).
    worse = "Her voice was barely a whisper against the dying, sorrowful light of a fading dusk."
    assert clean_rewrite("She spoke quietly in the dusk.", worse) == "She spoke quietly in the dusk."
    assert clean_rewrite(worse, worse) is None                            # echo / no improvement
    assert stable_gid("hello") == stable_gid("hello")                     # stable
    assert pick_intensity("hello") == pick_intensity("hello") in INTENSITY  # deterministic + valid
    # weights roughly honored across many seeds
    from collections import Counter
    c = Counter(pick_intensity(str(i)) for i in range(3000))
    assert c["moderate"] > c["overwrought"]

    # euphemism accept filter (symmetric): trust the seed, judge the generated twin.
    # worsen — seed is direct, a clean euphemized rename passes; fade-to-black / echo fail.
    direct = "He thrust his cock into her as she gasped."
    coy = "He pressed his length into her heat as she gasped."
    assert rename_ok(direct, coy)
    assert not rename_ok(direct, "He slid into her heat as the rest of the night blurred.")
    assert not rename_ok(direct, direct)
    # improve (de-euph) — seed is coy, a clean direct rename passes; the FADE guard reads the OUTPUT,
    # so a coy seed that itself mentions "the morning after" is fine as long as the output is direct.
    assert rename_ok(coy, direct)
    assert not rename_ok("They made love as the morning after crept in.", "He fucked her; the morning after crept in.")

    # build hooks emit the unified pole-ordered schema; `real` marks which pole was the seed.
    pw = _purple_build("src", "Plain seed.", "worsen", "Purple twin of the seed.")
    assert (pw["neg"], pw["pos"], pw["real"]) == ("Plain seed.", "Purple twin of the seed.", "neg")
    assert pw["intensity"] in INTENSITY
    pi = _purple_build("src", "Purple seed.", "improve", "Honest twin.")
    assert (pi["neg"], pi["pos"], pi["real"], pi["intensity"]) == ("Honest twin.", "Purple seed.", "pos", "better")
    # euphemism build runs the rename guard then pole-orders by direction (None when it fails).
    ew = _euph_build("src", direct, "worsen", coy)
    assert ew and (ew["neg"], ew["pos"], ew["real"]) == (direct, coy, "neg")
    assert _euph_build("src", direct, "worsen", "He slid into her heat as the rest of the night blurred.") is None
    ei = _euph_build("src", coy, "improve", direct)
    assert ei and (ei["neg"], ei["pos"], ei["real"]) == (direct, coy, "pos")

    # registry wiring: every axis/direction prompt formats with no leftover placeholder; the
    # intensity desc reaches purple/worsen (the only prompt with a {desc} slot) and nowhere else.
    for ax in REGISTRY.values():
        for dir_ in ax.prompts:
            filled = messages_for(ax, dir_, "SEEDTEXT.", desc="DESCTEXT")[1]["content"]
            assert "SEEDTEXT." in filled and "{orig}" not in filled and "{desc}" not in filled
    purple = REGISTRY["purple"]
    assert "DESCTEXT" in messages_for(purple, "worsen", "x.", desc="DESCTEXT")[1]["content"]
    assert "DESCTEXT" not in messages_for(purple, "improve", "x.", desc="DESCTEXT")[1]["content"]
    # direction availability derives from an axis's prompts, not a hardcoded list. Both axes are
    # symmetric; a worsen-only axis would simply omit its "improve" entry.
    assert improves(REGISTRY["purple"]) and improves(REGISTRY["euphemism"])
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
