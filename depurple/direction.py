"""Extract a per-layer style direction from matched pairs.

Direction = mean(residual_pos) - mean(residual_neg), per layer, mean-pooled over
content tokens and averaged across matched pairs. The pair sides share content,
so topic variance mostly cancels and the remaining direction follows the active
style axis.

Self-check: on held-out pairs, the positive pole must project higher onto the
direction than the negative pole. If it does not, this axis is not usefully
linear in the sampled residual stream.

    python -m depurple.direction            # extract + self-check, saves the direction artifact
    python -m depurple.direction --diagnose # length/register confound report, no edit

DEPURPLE_PROJECT controls optional projection before saving the active direction.
The raw direction and projection axis are saved alongside it so diagnostics remain
auditable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.paths import pair_paths
from depurple._model import MODEL, DIRECTIONS, AXIS, PROJECT  # run via `python -m depurple.direction`

OUT = Path(DIRECTIONS)
MAX_TOK = 256
BATCH = 16
TEST_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _rows() -> list[dict]:
    """Union of every pair corpus for the axis (generated + curated); see core.pair_paths."""
    return [json.loads(l) for p in pair_paths(AXIS)
            for l in Path(p).read_text().splitlines() if l.strip()]


def load_pairs() -> list[tuple[str, str]]:
    """(low_side, high_side) per row — the high side (pos) is the one we ablate. Unified
    pole-ordered schema across every axis (neg = low pole, pos = high pole): purple neg/pos =
    (plain, purple); euphemism neg/pos = (real direct seed, gemma euphemized). Axis-agnostic —
    no per-axis key branch."""
    rows = _rows()
    return [(r["neg"], r["pos"]) for r in rows if r.get("neg") and r.get("pos")]


@torch.inference_mode()
def pooled(model, tok, texts: list[str]) -> torch.Tensor:
    """Mean-pooled residual stream over content tokens. -> [n_texts, n_layers+1, d]."""
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOK).to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple[L+1] of [B,T,d]
        h = torch.stack(hs, dim=1).float()                          # [B, L+1, T, d]
        m = enc["attention_mask"][:, None, :, None].float()         # [B, 1, T, 1]
        out.append((h * m).sum(2) / m.sum(2).clamp(min=1))          # [B, L+1, d]
    return torch.cat(out).cpu()


def _load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return tok, model


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _proj_axis(plain: torch.Tensor, purple: torch.Tensor, wc: torch.Tensor) -> torch.Tensor | None:
    """The per-layer axis [L+1, d] that PROJECT strips out of r, or None for baseline.
    plain    (PROJECT=plain):       the plain-mean location.
    length   (PROJECT=length):      cov(word_count, activation) over neg∪pos = the direction
             activations move along as sentences get longer (the confound the diagnostic found)."""
    if PROJECT == "plain":
        return _unit(plain.mean(0))
    if PROJECT == "length":
        acts = torch.cat([plain, purple], 0)                        # [2N, L+1, d]
        wcc = (wc - wc.mean())[:, None, None]                       # centered word counts
        return _unit((wcc * (acts - acts.mean(0))).mean(0))         # [L+1, d]
    return None


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc.norm() * yc.norm()).clamp(min=1e-8)
    return float((xc * yc).sum() / denom)


def diagnose() -> None:
    """Report whether the axis direction is contaminated by length or register. For
    r = μ_purple - μ_plain, report per layer: cos(r, μ̂_plain) (how much r is the plain-mean
    direction would remove), Pearson(word count, projection onto r) pooled over neg∪pos
    (is r a length detector?), and ‖r_proj‖/‖r‖ (how much of r survives that projection). Gates
    live in docs/ablation_refinements_plan.md; euphemism is the clean reference."""
    tok, model = _load()
    pairs = load_pairs()
    plain = pooled(model, tok, [p[0] for p in pairs])              # [N, L+1, d] ALL pairs
    purple = pooled(model, tok, [p[1] for p in pairs])
    r = purple.mean(0) - plain.mean(0)                             # [L+1, d]
    r_unit = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mu_hat = plain.mean(0)
    mu_hat = mu_hat / mu_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos = (r_unit * mu_hat).sum(-1)                                # [L+1]  gate #1
    r_proj = r - (r * mu_hat).sum(-1, keepdim=True) * mu_hat
    survive = r_proj.norm(dim=-1) / r.norm(dim=-1).clamp(min=1e-8)  # [L+1] gate #3

    wc = torch.tensor([len(p[0].split()) for p in pairs] +        # [2N] neg then pos
                      [len(p[1].split()) for p in pairs]).float()
    proj_all = (torch.cat([plain, purple], 0) * r_unit).sum(-1)    # [2N, L+1] projection onto r
    corr = torch.tensor([_pearson(wc, proj_all[:, L]) for L in range(proj_all.shape[1])])  # gate #2

    n = len(cos)
    lo, hi = int(0.15 * n), int(0.85 * n)                          # mid-stack window (where style lives)
    print(f"\n=== Stage 0 diagnostic: axis={AXIS}  {len(pairs)} pairs  layers 0..{n-1} ===")
    print("  layer   cos(r,mu_plain)   corr(words,proj)   ||r_proj||/||r||")
    for L in range(n):
        mark = " *" if lo <= L <= hi else "  "
        print(f"  {L:3d}{mark}     {cos[L]:+.3f}            {corr[L]:+.3f}             {survive[L]:.3f}")
    mid = slice(lo, hi + 1)
    print(f"\n  mid-stack ({lo}..{hi}): max|cos|={cos[mid].abs().max():.3f}  "
          f"max|corr|={corr[mid].abs().max():.3f}  min survive={survive[mid].min():.3f}")
    print(f"  GATE #1 cos<0.2 => #1 ~no-op:        {'YES (skip #1)' if cos[mid].abs().max() < 0.2 else 'no'}")
    print(f"  GATE #2 corr>0.4 => fix data first:  {'YES (Stage 4)' if corr[mid].abs().max() > 0.4 else 'no'}")
    print(f"  GATE #3 survive<0.5 => under-ablate:  {'YES (re-tune kernel)' if survive[mid].min() < 0.5 else 'no'}")


def main() -> None:
    tok, model = _load()

    pairs = load_pairs()
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_test = int(len(pairs) * TEST_FRAC)
    test_idx, train_idx = set(perm[:n_test]), perm[n_test:]
    print(f"{len(pairs)} pairs -> {len(train_idx)} train / {n_test} test")

    plain = pooled(model, tok, [pairs[i][0] for i in train_idx])    # [N, L+1, d]
    purple = pooled(model, tok, [pairs[i][1] for i in train_idx])
    raw_r = (purple.mean(0) - plain.mean(0))                        # [L+1, d]  raw class-mean gap
    wc = torch.tensor([len(pairs[i][0].split()) for i in train_idx] +
                      [len(pairs[i][1].split()) for i in train_idx]).float()  # neg then pos
    proj_axis = _proj_axis(plain, purple, wc)                       # #1 axis to strip, or None
    r = raw_r
    if proj_axis is not None:                                       # #1: remove the projected axis from r
        r = raw_r - (raw_r * proj_axis).sum(-1, keepdim=True) * proj_axis
        print(f"PROJECT={PROJECT}: r orthogonalized against per-layer {PROJECT} axis "
              f"(||r_proj||/||r||={(r.norm(dim=-1)/raw_r.norm(dim=-1).clamp(min=1e-8)).mean():.3f} mean)")
    direction = _unit(r)                                            # unit-normalize (raw or projected)

    # Self-check on held-out pairs: purple twin must project higher than plain twin.
    tp = pooled(model, tok, [pairs[i][0] for i in sorted(test_idx)])
    tq = pooled(model, tok, [pairs[i][1] for i in sorted(test_idx)])
    proj_plain = (tp * direction).sum(-1)                           # [N, L+1]
    proj_purple = (tq * direction).sum(-1)
    sep = (proj_purple - proj_plain).mean(0)                        # [L+1]
    acc = (proj_purple > proj_plain).float().mean(0)               # [L+1]
    best = int(acc.argmax())
    print(f"best layer {best}: acc={acc[best]:.3f} sep={sep[best]:.3f}")
    for L in range(0, len(acc), 4):
        print(f"  layer {L:2d}: acc={acc[L]:.3f} sep={sep[L]:+.3f}")

    OUT.parent.mkdir(exist_ok=True)
    # Keep raw r + the stripped axis regardless of PROJECT so the choice is auditable/reversible.
    torch.save({"direction": direction, "raw_r": raw_r, "proj_axis": proj_axis,
                "project": PROJECT, "acc": acc, "sep": sep,
                "best_layer": best, "model": MODEL, "axis": AXIS}, OUT)
    print(f"saved {OUT}  (project={PROJECT!r})")
    assert acc[best] > 0.8, (f"{AXIS} not linearly separable after PROJECT={PROJECT!r} "
                             f"(best acc {acc[best]:.3f}) -> this projection destroys the axis; abort")
    print(f"KILL-TEST PASSED: {AXIS} (project={PROJECT!r}) is a linear direction")


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        diagnose()
    else:
        main()
