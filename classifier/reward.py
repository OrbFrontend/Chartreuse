"""Paragraph-head P(purple) for a reward function (OrbRewriter GRPO): one score per text, hard to game.

WHY A SEPARATE SCORER. A sentence head is fed sentences at 128 tokens; OrbRewriter's grpo.Purple does
exactly that. A paragraph head (purple_para) fed that way never sees what it was trained on: filler and
negation spam only show across a paragraph. Pointing PURPLE_MODEL at it is wrong, so this module scores
the way the head was trained and closes the holes a policy would find (measured 2026-11-07 on the
test split, 3-seed ensemble):

  - paragraph splitting: a purple paragraph rewritten one sentence per paragraph drops 0.74 -> 0.35 if
    paragraphs are scored separately and averaged. Scoring the whole text as one unit closes that
    (0.76) but puts multi-paragraph text out of distribution: the head trained on single paragraphs,
    and 3 unrelated clean paragraphs joined score 0.33 against 0.21 alone. So paragraphs are merged
    with their neighbours (joined by a newline) until each chunk has MIN_WORDS words, the chunks are
    scored, and the word-weighted mean is returned. At 60 words a split purple paragraph scores 0.73
    (0.66 at 40) and 3 joined clean ones 0.28. Test paragraphs have a median of 60 (clean) to 78
    (purple) words, so a real paragraph is usually its own chunk.
  - truncation: the head sees max_length tokens (chartreuse.json, 512). A slop tail past 600 clean tokens
    scored 0.29-0.50 against 0.61-0.69 inside the window. A chunk longer than that is scored in
    sentence-aligned windows, weighted by tokens, so no part of it goes unread. Test paragraphs peak at
    381 tokens, so windows are rare.
  - seeds: production is ONE model (models/ettin400m-purple_para, seed 42; since 2026-11-07 trained with
    human-only negatives, so it scores LLM provenance as well as slop -- intended). Seed variance is real (three seeds of one
    recipe flagged 46%, 72% and 55% of a filler probe), so a list of dirs is still accepted and averaged,
    but nothing trains more than one seed.
Not closed: comma / adverb stripping moves P by only 0.02 / 0.005, so nothing is needed there. Longer
PURPLE paragraphs score higher (r = +0.34), and clean ones do not (r = -0.05), so shortening a purple
paragraph lowers P; that is mostly the edit we want, but a length term should still guard deletion.

    from classifier.reward import ParagraphPurple
    pp = ParagraphPurple("models/ettin400m-purple_para")
    pp.score(["She waits. A pause. Nothing more.", ...])   # -> [P(purple), ...]

    python -m classifier.reward --demo
    python -m classifier.reward MODEL_DIR[+MODEL_DIR...] "text" ...
"""
import re
import sys

import torch

from core import model_meta

BLANK = re.compile(r"\n\s*\n")
MIN_WORDS = 60


def chunks(text: str, min_words: int = MIN_WORDS) -> list[str]:
    """Paragraphs, each merged with the next (by a newline) until it has `min_words` words; a short tail
    joins the chunk before it. Splitting a paragraph into one-line paragraphs therefore changes nothing."""
    out = []
    for p in (x.strip() for x in BLANK.split((text or "").strip())):
        if not p:
            continue
        if out and len(out[-1].split()) < min_words:
            out[-1] += "\n" + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < min_words:
        last = out.pop()
        out[-1] += "\n" + last
    return out or [""]


def windows(ids: list[int], sent_ends: list[int], size: int) -> list[tuple[int, int]]:
    """Spans of at most `size` token positions over `ids`, cut at sentence ends where possible."""
    out, a = [], 0
    while a < len(ids):
        b = min(a + size, len(ids))
        if b < len(ids):
            cut = [e for e in sent_ends if a < e <= b]
            b = cut[-1] if cut and cut[-1] - a >= size // 2 else b
        out.append((a, b))
        a = b
    return out


class ParagraphPurple:
    def __init__(self, model_dirs, device: str | None = None, dtype=torch.bfloat16, batch: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dirs = [model_dirs] if isinstance(model_dirs, str) else list(model_dirs)
        metas = [model_meta.read(d) for d in dirs]
        if any(m["unit"] != "paragraph" for m in metas):
            raise ValueError(f"not a paragraph head: {[d for d, m in zip(dirs, metas) if m['unit'] != 'paragraph']} "
                             "(chartreuse.json unit) -- a sentence head is scored per sentence, not here")
        self.max_length = min(m["max_length"] for m in metas)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(dirs[0])
        self.models = [AutoModelForSequenceClassification.from_pretrained(d, dtype=dtype).to(self.device).eval()
                       for d in dirs]
        self.batch = batch

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        """(piece, weight in words) for every chunk; a chunk over max_length is cut into sentence-aligned
        windows sharing its weight by tokens."""
        out = []
        for c in chunks(text):
            wins = self._windows(c)
            total = sum(n for _, n in wins)
            out += [(p, max(1, len(c.split())) * n / total) for p, n in wins]
        return out

    def _windows(self, t: str) -> list[tuple[str, int]]:
        enc = self.tok(t, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        room = self.max_length - self.tok.num_special_tokens_to_add()
        if len(ids) <= room:
            return [(t, max(1, len(ids)))]
        ends = [i + 1 for i, (_, e) in enumerate(offs) if e and t[e - 1:e] in ".!?…\n"]
        return [(t[offs[a][0]:offs[b - 1][1]], b - a) for a, b in windows(ids, ends, room)]

    @torch.no_grad()
    def _probs(self, pieces: list[str]) -> list[float]:
        out = []
        for i in range(0, len(pieces), self.batch):
            e = self.tok(pieces[i:i + self.batch], return_tensors="pt", padding=True, truncation=True,
                         max_length=self.max_length).to(self.device)
            p = torch.stack([torch.softmax(m(**e).logits.float(), -1)[:, 1] for m in self.models]).mean(0)
            out += p.tolist()
        return out

    def score(self, texts: list[str]) -> list[float]:
        """P(purple) per text: seed-mean over word-weighted chunks (see MIN_WORDS), windowed past max_length."""
        per = [self._pieces(t) for t in texts]
        flat = self._probs([p for v in per for p, _ in v])
        out, k = [], 0
        for v in per:
            w = [n for _, n in v]
            out.append(sum(p * n for p, n in zip(flat[k:k + len(v)], w)) / sum(w))
            k += len(v)
        return out


def demo():
    long = " ".join(["word"] * 70)
    assert chunks("A.\n\nB.\n \nC.\nD.") == ["A.\nB.\nC.\nD."]                  # short paragraphs merge
    assert chunks(f"{long}\n\n{long}") == [long, long]                               # full ones stay apart
    assert chunks(f"{long}\n\nShort tail.") == [f"{long}\nShort tail."]             # a short tail joins back
    assert chunks("") == [""] and chunks("\n\n") == [""]
    assert windows(list(range(10)), [4, 8], 5) == [(0, 4), (4, 8), (8, 10)]
    assert windows(list(range(10)), [1], 5) == [(0, 5), (5, 10)]          # no usable sentence end: hard cut
    assert windows(list(range(3)), [], 5) == [(0, 3)]
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        pp = ParagraphPurple(sys.argv[1].split("+"))
        for t, s in zip(sys.argv[2:], pp.score(sys.argv[2:])):
            print(f"{s:.3f}  {t[:100]}")
