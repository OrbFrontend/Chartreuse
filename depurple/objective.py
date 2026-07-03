"""Multi-turn objective: roll out scripted roleplay conversations against a
generation model, score every assistant turn for purple-ness with the existing
ettin classifier, and report the per-turn curve + a late-turn-weighted scalar.

Single-turn creative prompts give no signal (the classifier learned 'human' from
the same WritingPrompts corpus the LLMs trained on). Purple emerges over TURNS in
roleplay, so we measure it there. Each candidate edit regenerates the whole
conversation (assistant turns feed on their own prior output).

    python depurple/objective.py     # baseline on Qwen2.5-3B-Instruct: does purple rise with depth?
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

from depurple._model import MODEL as GEN_MODEL
from depurple._model import CLASSIFIER, AXIS, AXES, classifier_dir
from depurple.euphemism_lexicon import direct_density, euphemism_density
from text_segmentation import split_narration_sentences, split_sentences

_AX = "" if AXIS == "purple" else f"-{AXIS}"
SCEN = Path(f"depurple/eval_scenarios{_AX}.jsonl")   # axis-routed; purple keeps the legacy name
CLS_DIR = CLASSIFIER                         # per-axis scorer (purple -> models/ettin400m-purple)
PLAIN = Path("human_writing.txt")
MAX_NEW = 350

# Two-sided euphemism objective (plan step 5): obj = P(euphemism) - BETA*direct_density.
# Minimizing P(euphemism) alone rewards fade-to-black; the direct-density term drives
# TOWARD direct language. BETA scales the per-word direct rate (~0.0-0.05) up to the
# P(euph) range (~0-1); tunable knob — sweep if direct stays pinned low.
BETA = 5.0
INTIMATE_FRAC = 0.5     # default intimate boundary: score the back half of each scenario


# Inline scorer (mirrors app.score) so we don't pull the FastAPI serving stack into
# the training venv. P(class 1) per sentence (purple, or euphemism on that axis), fp32
# to match the calibrated numbers. Lazy-loaded and keyed by classifier dir so a JOINT run
# can hold one scorer per axis at once. Gate-zero (step 0) runs on lexicons alone, BEFORE
# the euphemism classifier exists, so importing this module must not require the classifier.
_scorers: dict[str, tuple] = {}


def _load_scorer(cls_dir: str = CLS_DIR):
    if cls_dir not in _scorers:
        tok = AutoTokenizer.from_pretrained(cls_dir)
        # bf16 here, not the calibrated fp32: a JOINT run holds one scorer PER axis resident
        # at once, and two fp32 ettin400m (~1.6GB each) starve gemma's generate activations on
        # a 24GB card. This scorer is the search/eyeball signal, not the deployed 0.66 gate
        # (serve.py/calibrate.py own that), so bf16 softmax drift (~1e-3) doesn't move it.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        m = AutoModelForSequenceClassification.from_pretrained(cls_dir, dtype=dtype).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()
        _scorers[cls_dir] = (tok, m)
    return _scorers[cls_dir]


@torch.inference_mode()
def score(sentences: list[str], batch: int = 64, cls_dir: str = CLS_DIR) -> list[float]:
    ctok, cmodel = _load_scorer(cls_dir)
    probs: list[float] = []
    for i in range(0, len(sentences), batch):
        enc = ctok(sentences[i:i + batch], return_tensors="pt", padding=True,
                   truncation=True, max_length=128).to(cmodel.device)
        probs.extend(cmodel(**enc).logits.softmax(-1)[:, 1].tolist())
    return probs


def _scen_path(axis: str) -> Path:
    return Path(f"depurple/eval_scenarios{'' if axis == 'purple' else '-' + axis}.jsonl")


def load_scenarios(kind: str | None = None, axis: str | None = None) -> list[dict]:
    """All scenarios for an axis (default: the primary axis), or only those whose id prefix
    matches `kind` ('bait' = purple-elicitation, 'collapse' = repetition/overstaying probe).
    A joint run passes axis= to read each axis's own eval_scenarios file."""
    path = SCEN if axis is None else _scen_path(axis)
    scens = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [s for s in scens if kind is None or s["id"].startswith(kind + "_")]


class RolloutAborted(Exception):
    """Mid-rollout abort raised when partial replies already fail structural guards.

    Carries the reject `value` so the objective can return through Optuna's normal
    completed-trial path. rollout_many does not catch this; it propagates through
    evaluate or rollout_all to the objective."""
    def __init__(self, value: float):
        super().__init__(f"rollout aborted -> reject {value:+.3f}")
        self.value = value


# Scenarios generated together in one left-padded generate() call per turn. Generation
# (not classifier scoring) is the rollout bottleneck and runs batch-1 otherwise; bump this
# when VRAM is spare, lower it on OOM. 16 covers the full bait set (14) in one shot.
ROLLOUT_BATCH = int(os.environ.get("ROLLOUT_BATCH", "16"))


@torch.inference_mode()
def rollout_many(model, tok, scenarios: list[dict], max_new: int = MAX_NEW,
                 batch: int = ROLLOUT_BATCH,
                 on_batch: Callable[[list[list[str]]], None] | None = None) -> list[list[str]]:
    """Greedy multi-turn rollouts for many scenarios at once — the speedup when VRAM is
    spare. Up to `batch` scenarios share one left-padded generate() per turn; shorter
    scenarios drop out as the conversation deepens. batch=1 is byte-identical to the old
    per-scenario rollout (no padding), and EVERY caller (optimize baseline + trials, eyeball,
    state_probe) goes through this one path, so greedy float-noise from padding — if any —
    hits base and edit alike and the base-anchored comparisons stay consistent.

    `on_batch`, if given, is called after each batch finishes ALL its turns, with the
    replies-so-far (only the completed scenarios are filled; the rest are still []). It may
    raise RolloutAborted to stop the rollout mid-way (mid-rollout brain-damage early-out).
    Brain damage emerges at DEPTH, so checking per-batch (full depth) not per-turn."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    replies_by_scen: list[list[str]] = [[] for _ in scenarios]
    for lo in range(0, len(scenarios), batch):
        group = list(range(lo, min(lo + batch, len(scenarios))))
        msgs = {i: [{"role": "system", "content": scenarios[i]["system"]}] for i in group}
        for t in range(max(len(scenarios[i]["user_turns"]) for i in group)):
            active = [i for i in group if t < len(scenarios[i]["user_turns"])]
            for i in active:
                msgs[i].append({"role": "user", "content": scenarios[i]["user_turns"][t]})
            # return_dict=False -> plain list[int]; transformers>=5 defaults this to True (BatchEncoding)
            enc = [tok.apply_chat_template(msgs[i], add_generation_prompt=True, return_dict=False) for i in active]
            mlen = max(len(e) for e in enc)
            ids = torch.full((len(enc), mlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(enc), mlen), dtype=torch.long)
            for k, e in enumerate(enc):                      # left-pad: generated tokens all start at col mlen
                ids[k, mlen - len(e):] = torch.tensor(e)
                attn[k, mlen - len(e):] = 1
            out = model.generate(input_ids=ids.to(model.device), attention_mask=attn.to(model.device),
                                 max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
            for k, i in enumerate(active):
                reply = tok.decode(out[k, mlen:], skip_special_tokens=True).strip()
                msgs[i].append({"role": "assistant", "content": reply})
                replies_by_scen[i].append(reply)
        if on_batch is not None:          # after this batch's full depth -> may raise RolloutAborted
            on_batch(replies_by_scen)
    return replies_by_scen


@torch.inference_mode()
def rollout(model, tok, scen: dict, max_new: int = MAX_NEW) -> list[str]:
    """Greedy multi-turn rollout for one scenario. Returns the assistant reply per user turn.
    Thin batch-1 wrapper over rollout_many (no padding -> identical to the old single path)."""
    return rollout_many(model, tok, [scen], max_new, batch=1)[0]


def turn_purple(reply: str, cls_dir: str = CLS_DIR) -> float:
    """Mean P(purple) over the reply's sentences."""
    sents = split_sentences(reply)
    return sum(score(sents, cls_dir=cls_dir)) / len(sents) if sents else 0.0


def evaluate(model, tok, scenarios: list[dict], cls_dir: str = CLS_DIR,
             on_batch: Callable[[list[list[str]]], None] | None = None
             ) -> tuple[list[float], list[list[str]]]:
    """Per-turn mean P(purple), averaged across scenarios -> the degradation curve.
    Also returns the assistant replies grouped by scenario (the rollout already
    generated them) so the caller can score opener collapse and cross-turn
    repetition without regenerating. Flatten with [r for scen in replies for r in scen].
    `on_batch` is forwarded to rollout_many (mid-rollout early-out)."""
    if not scenarios:
        return [], []
    n = max(len(s["user_turns"]) for s in scenarios)
    sums, cnts = [0.0] * n, [0] * n
    replies_by_scen = rollout_many(model, tok, scenarios, on_batch=on_batch)
    for replies in replies_by_scen:
        for t, reply in enumerate(replies):
            sums[t] += turn_purple(reply, cls_dir=cls_dir)
            cnts[t] += 1
    curve = [sums[t] / cnts[t] if cnts[t] else 0.0 for t in range(n)]
    return curve, replies_by_scen


def opener_repeat(replies: list[str]) -> float:
    """Mean share of sentences per reply that reuse an opening word already seen in
    that reply. 0 = every sentence opens differently; ~1 = one opener throughout.
    Catches 'She nods. She turns. She doesn't.' structural collapse — not hedge-slop
    (that varies its openers, so it reads ~base here)."""
    rates = []
    for r in replies:
        firsts = [s.split()[0].lower() for s in split_sentences(r) if s.split()]
        if len(firsts) < 2:
            continue
        repeats = sum(n - 1 for n in Counter(firsts).values())   # every dup beyond first
        rates.append(repeats / (len(firsts) - 1))                # 0..1
    return sum(rates) / len(rates) if rates else 0.0


_STRUCT_RE = re.compile(r"^\s*(\*\*|#{1,3}\s|\d+\.\s|[-*•]\s)")


def struct_frac(replies: list[str]) -> float:
    """Fraction of non-empty lines that open like markdown structure (bold header, # header,
    numbered/bulleted list). Narrative rollouts should sit near zero; high values indicate
    report-mode collapse that can look good to the style classifier. Base-anchored in
    optimize like the other guards."""
    lines = [ln for r in replies for ln in r.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(bool(_STRUCT_RE.match(ln)) for ln in lines) / len(lines)


_WORD_RE = re.compile(r"[a-z']+")
# function words whose natural repetition isn't degeneration (excluded from lexical-fixation count)
_STOPWORDS = frozenset(
    "a an the and or but if then of to in into on at by for with from up out down off over "
    "is are was were be been being am do does did has have had will would can could my your his "
    "her its their our i you he she it we they me him them this that these those there here as "
    "so not no yes like just".split())


def _seq_rep(toks: list[str], n: int) -> float:
    """Fraction of n-grams that are repeats (1 - distinct/total). 0 = all unique."""
    if len(toks) <= n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def intra_repeat(replies: list[str]) -> float:
    """Within-reply degeneration density (mean over replies). Per reply, the worse of:
      - repeated 2/3/4-gram fraction — the 'My ass is X. My ass is Y.' stem-loop;
      - share of content words held by the single most-repeated one — lexical fixation
        ('ass' 9x in one turn).
    0 = varied prose; high = the edit loops/fixates inside ONE turn. opener_repeat sees only the
    first word of each sentence and the cross-message audit only sees repetition across turns;
    this catches the intra-reply word/phrase spam both miss.
    Base-anchored in optimize so only the repetition the EDIT adds is punished."""
    rates = []
    for r in replies:
        toks = _WORD_RE.findall(r.lower())
        if len(toks) < 4:
            continue
        seq = max(_seq_rep(toks, n) for n in (2, 3, 4))
        content = [t for t in toks if t not in _STOPWORDS]
        fix = max(Counter(content).values()) / len(content) if content else 0.0
        rates.append(max(seq, fix))
    return sum(rates) / len(rates) if rates else 0.0


def late_weighted(curve: list[float]) -> float:
    """Scalar objective: mean weighted by turn index (later turns count more)."""
    w = [i + 1 for i in range(len(curve))]
    return sum(c * wi for c, wi in zip(curve, w)) / sum(w)


# --- euphemism axis: score the intimate turns, two-sided (plan step 5) ---
# Euphemism isn't monotone-escalating like purple, so we don't late-weight; we score
# the turns where the scene is intimate (a scenario may declare intimate_from, else the
# back half) and drive P(euphemism) down AND direct-term density up.

def rollout_all(model, tok, scenarios: list[dict],
                on_batch: Callable[[list[list[str]]], None] | None = None) -> list[list[str]]:
    """Replies per scenario without classifier scoring.

    Used by classifier-free checks and euphemism scoring paths. `on_batch` is
    forwarded to rollout_many for mid-rollout early-out."""
    return rollout_many(model, tok, scenarios, on_batch=on_batch)


def intimate_replies(scenarios: list[dict], replies_by_scen: list[list[str]]) -> list[str]:
    """Flattened replies on the intimate turns only (where euphemism appears)."""
    out: list[str] = []
    for s, replies in zip(scenarios, replies_by_scen):
        start = s.get("intimate_from", int(len(replies) * INTIMATE_FRAC))
        out.extend(replies[start:])
    return out


def euph_scores(replies: list[str], cls_dir: str = CLS_DIR) -> tuple[float, float]:
    """(mean P(euphemism), direct-term density) over the NARRATION of intimate replies —
    spoken dialogue is stripped (split_narration_sentences). Scoring dialogue made the
    optimizer ablate softening/politeness out of speech, so characters turned blunt and
    mean; the euphemism axis only cares how the ACT is narrated, so we score narration
    alone. Fade-to-black OR all-dialogue (no narration) -> (0.0, 0.0): P(euph) bottoms out,
    but direct density is also 0, so the optimizer's direct-density floor (optimize.py)
    catches the cheat. coverage() in optimize.py still sees full replies, so the collapse
    guard is unaffected."""
    sents = [x for r in replies for x in split_narration_sentences(r)]
    if not sents:
        return 0.0, 0.0
    euph = sum(score(sents, cls_dir=cls_dir)) / len(sents)
    direct = direct_density(" ".join(sents))
    return euph, direct


def two_sided(euph: float, direct: float) -> float:
    """Scalar objective for the euphemism axis (minimized)."""
    return euph - BETA * direct


@torch.inference_mode()
def _ppl_on_ids(model, ids, ctx: int, bos, n_chunks: int | None = None) -> float:
    """Mean-chunk perplexity of a token id sequence under `model`. n_chunks caps how many
    ctx-sized chunks to score (None = all). Each chunk is its own context (gemma needs BOS)."""
    limit = len(ids) if n_chunks is None else min(n_chunks * ctx, len(ids))
    losses = []
    for i in range(0, limit, ctx):
        chunk = ids[i:i + ctx].to(model.device)
        if len(chunk) < 2:
            break
        if bos is not None:
            chunk = torch.cat([chunk.new_tensor([bos]), chunk])
        out = model(chunk[None], labels=chunk[None])   # HF shifts labels internally
        losses.append(out.loss.item())
    return float(torch.tensor(losses).mean().exp()) if losses else float("inf")


@torch.inference_mode()
def perplexity(model, tok, n_chunks: int = 20, ctx: int = 512) -> float:
    """Perplexity on held-out plain good prose — the capability guard (can the edit still
    READ good prose). Blind to the edit's OWN output coherence; see gen_perplexity."""
    text = PLAIN.read_text(errors="ignore")[:200_000]
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return _ppl_on_ids(model, ids, ctx, tok.bos_token_id, n_chunks)


@torch.inference_mode()
def gen_perplexity(model, tok, replies: list[str], ctx: int = 512) -> float:
    """Perplexity of the model's OWN generated text under the current weights. Run with the
    BASE weights resident (restore the snapshot first) and it becomes the output-coherence
    guard perplexity() can't be: a brain-damaged edit reads human prose fine, yet its own
    rollouts are improbable under the sane base model. Scores 'ass ass ass' and
    broken/looping/non-sequitur text high; coherent plain OR purple prose low. inf on
    (near-)empty output — caller ratio-caps so Optuna never sees inf (coverage owns collapse)."""
    text = "\n\n".join(r for r in replies if r.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) < 2:
        return float("inf")
    return _ppl_on_ids(model, ids, ctx, tok.bos_token_id)


# Below this many sentences per intimate turn the act isn't really narrated -> fade-to-black,
# the null where orthogonalization is the wrong tool (plan step 0 / fallback). Tunable knob.
COVERAGE_FLOOR = 1.5


def gate_zero(model, tok, axis: str | None = None) -> None:
    """Check that the unedited model has a euphemism gap to close.

    On intimate turns, lexically require euphemistic density above direct density
    and enough narration to rule out fade-to-black. The check is classifier-free.
    axis= reads that axis's scenarios for joint runs."""
    scen = load_scenarios(axis=axis)
    if not scen:
        raise SystemExit(f"no scenarios for axis {axis or AXIS} — author intimate-escalation RP first (plan step 5)")
    intimate = intimate_replies(scen, rollout_all(model, tok, scen))
    text = " ".join(intimate)
    ed, dd = euphemism_density(text), direct_density(text)
    n_sents = sum(len(split_sentences(r)) for r in intimate)
    cov = n_sents / len(intimate) if intimate else 0.0
    print(f"intimate turns={len(intimate)}  sentences={n_sents}  sents/turn={cov:.2f}")
    print(f"euphemism density={ed:.4f}   direct density={dd:.4f}")
    assert cov >= COVERAGE_FLOOR, ("fade-to-black: intimate turns barely narrate — "
                                   "orthogonalization is the wrong tool, see plan fallback")
    assert ed > dd, "intimate prose already direct (or non-euphemistic) — nothing to remove"
    print("KILL-TEST 0 PASSED: euphemistic, narrated, headroom toward direct")


def _kill_test(model, tok, axis: str) -> None:
    """Run one axis's baseline signal check on the unedited model.

    Euphemism checks that there is a gap to close. Purple checks that P(purple)
    rises over turns. Reads this axis's classifier and scenarios so joint runs
    check every axis, not just the primary one."""
    if axis == "euphemism":
        gate_zero(model, tok, axis=axis)
        return
    cls = classifier_dir(axis)
    scen = load_scenarios("bait", axis=axis)   # purple rise is a bait-only property; collapse probes don't escalate
    curve, _ = evaluate(model, tok, scen, cls_dir=cls)
    print(f"per-turn mean P({axis}):")
    for t, c in enumerate(curve):
        print(f"  turn {t+1:2d}: {c:.3f}  {'#' * round(c * 50)}")
    print(f"late-weighted scalar: {late_weighted(curve):.3f}")
    print(f"baseline perplexity (plain prose): {perplexity(model, tok):.2f}")
    early = sum(curve[:2]) / 2
    late = sum(curve[-2:]) / 2
    print(f"early(1-2)={early:.3f}  late(-2)={late:.3f}  rise={late - early:+.3f}")
    assert late > early, f"{axis} does NOT rise with depth — multi-turn eval has no signal"
    print(f"KILL-TEST 2 PASSED: {axis} emerges over turns")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    for axis in AXES:                          # joint run: each axis runs its own baseline kill-test
        if len(AXES) > 1:
            print(f"=== baseline kill-test: {axis} ===")
        _kill_test(model, tok, axis)


def demo() -> None:
    """Self-check intimate-turn selection and two-sided scoring without model or network calls."""
    scen = [{"user_turns": [0, 1, 2, 3]}, {"user_turns": [0, 1], "intimate_from": 1}]
    rbs = [["a", "b", "c", "d"], ["e", "f"]]
    assert intimate_replies(scen, rbs) == ["c", "d", "f"]  # back-half default + explicit boundary
    assert two_sided(0.5, 0.0) == 0.5                       # no direct content: pure P(euph)
    assert two_sided(0.5, 0.1) < two_sided(0.5, 0.0)       # direct density lowers (improves) the objective
    spam = "My ass is exposed. My ass is big. My ass is cold. My ass, my ass, my ass."  # stem-loop + fixation
    varied = "The harbor smelled of salt. Gulls argued over a crate. A ferry pushed off the dock."
    assert intra_repeat([spam]) > intra_repeat([varied]), "intra-reply spam must score higher than varied prose"
    assert intra_repeat([varied]) < 0.2 and intra_repeat([""]) == 0.0, "varied prose / empty must stay low"
    mixed = 'She traced his jaw. "You disgust me," she spat. The fire guttered low.'
    narr = " ".join(split_narration_sentences(mixed))  # euph_scores scores this, not the spoken line
    assert "disgust" not in narr and "traced his jaw" in narr, "dialogue must be stripped from euph scoring"
    print("objective demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
