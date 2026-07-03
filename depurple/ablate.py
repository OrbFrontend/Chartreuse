"""Directional ablation for residual-writing weight matrices.

Orthogonalizes attention o_proj, MLP down_proj, and related write paths against
one or more style directions with a tunable per-layer weight kernel. The edit is
baked into weights, so it adds no inference latency.

W' = W - w * d (d^T W)   removes the (scaled) component along unit vector d from
the matrix output. w in [0,1] is the per-layer dial: w=1 fully projects the
direction out, w<1 turns it down.

snapshot()/restore() let an Optuna loop reuse one loaded model across trials
(ablation is destructive and not idempotent, so restore before re-applying).

Target = Gemma 4 family. Its text decoder is
model.model.language_model.layers (the vision/audio towers also have o_proj/down_proj — we
must NOT touch those). We ablate every residual-*writing* matrix: attn o_proj, mlp
down_proj, the PLE per_layer_projection (d_ple -> d_model, present only on PLE builds like
e4b), and — on MoE builds — the fused expert down_proj [n_experts, d_model, d_ff] (an MoE
layer runs the dense mlp AND the experts and sums both into the residual, so both are write
paths; the router only routes and is left alone). All sit behind a post-norm before the
residual add, so the orthogonalization is approximate the same way for each.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class AblationSpec(NamedTuple):
    """One style axis to remove: its per-layer directions [L+1, d], the per-layer
    kernel weight (build_kernel output), and an optional fixed dir_layer (None =>
    use the per-layer direction directions[i+1] at layer i). A joint edit passes a
    list of these to apply_ablation. mlp_scale (<=1) further dials down the kernel
    weight on the MLP write path only. norm_preserve rescales each matrix column
    back to the norm it would have without this spec's removal while preserving
    orthogonality to the removed direction. The setting is per spec, so a joint
    edit can preserve one axis and not another."""
    directions: torch.Tensor
    weights: list[float]
    dir_layer: int | None = None
    mlp_scale: float = 1.0
    norm_preserve: bool = False


def decoder_layers(model):
    """The text decoder layer list. Multimodal gemma-4 (e4b) nests the text path
    under language_model; a text-only build exposes layers directly. Either way we
    only touch the text decoder, never the vision/audio towers."""
    inner = model.model
    inner = getattr(inner, "language_model", inner)
    return inner.layers


def _weight(m):
    """The residual-writing weight tensor: an nn.Linear exposes .weight [out, in]; the MoE
    experts module has no .weight — its write matrix is the fused .down_proj Parameter
    [n_experts, out, in]."""
    return m.weight if hasattr(m, "weight") else m.down_proj


def _targets(model):
    """(layer_index, kind, module) for every residual-writing matrix we ablate. kind in
    {'attn','mlp','ple'} so apply_ablation can scale the MLP (down_proj) write-path apart."""
    for i, layer in enumerate(decoder_layers(model)):
        yield i, "attn", layer.self_attn.o_proj
        yield i, "mlp", layer.mlp.down_proj
        if hasattr(layer, "experts"):                # MoE: experts.down_proj is fused [E, d_model, d_ff]
            yield i, "mlp", layer.experts
        if hasattr(layer, "per_layer_projection"):   # PLE write-path (e4b); see module docstring
            yield i, "ple", layer.per_layer_projection


def snapshot(model) -> list[torch.Tensor]:
    # CPU-resident: the snapshot is only read back by restore() (copy_ handles the
    # CPU->GPU transfer), so keeping the ~18GiB clone off the card buys that headroom
    # for KV cache / activations. Impact is near-zero (~2s after each trial).
    return [_weight(m).detach().to("cpu", copy=True) for _, _, m in _targets(model)]


def restore(model, snap: list[torch.Tensor]) -> None:
    for (_, _, m), w in zip(_targets(model), snap):
        _weight(m).data.copy_(w)


def build_kernel(n: int, max_weight: float, max_pos: float,
                 min_weight: float, min_dist: float) -> list[float]:
    """Triangular four-parameter weight bump over layer depth."""
    center = max_pos * (n - 1)
    half = max(min_dist * (n - 1), 1e-6)
    out = []
    for i in range(n):
        f = max(0.0, 1.0 - abs(i - center) / half)
        out.append(min_weight + (max_weight - min_weight) * f)
    return out


def apply_ablation(model, directions, weights: list[float] | None = None,
                   dir_layer: int | None = None, mlp_scale: float = 1.0,
                   norm_preserve: bool = False) -> None:
    """Orthogonalize every residual-writing matrix (o_proj, down_proj, PLE) against one
    OR MORE style directions. Two call forms:

        apply_ablation(model, directions[L+1, d], weights, dir_layer)   # one axis
        apply_ablation(model, [AblationSpec(...), AblationSpec(...)])   # several axes, jointly

    The result is order-independent. norm_preserve is per spec (the kwarg forces
    it on for all specs): each preserving axis has the write magnitude its removal
    cost restored, while a non-preserving axis keeps its shrinkage. The restore
    target is the column norm after only the non-preserving removals; column
    scaling keeps every removed d exactly zero, so the removal stays exact."""
    specs = directions if isinstance(directions, (list, tuple)) else [
        AblationSpec(directions, weights, dir_layer, mlp_scale, norm_preserve)]
    eff_np = [s.norm_preserve or norm_preserve for s in specs]   # per-spec; kwarg forces all on
    any_np = any(eff_np)
    for i, kind, m in _targets(model):
        W = _weight(m).data                     # [out=d_model, in]  (fused experts: [E, out, in])
        cdim = W.ndim - 2                        # column axis = OUTPUT dim: 0 (2-D) or 1 (fused 3-D)
        delta = None                            # sum over ALL specs
        delta_keep = None                       # sum over NON-preserving specs (their shrinkage is kept)
        for si, spec in enumerate(specs):
            w = spec.weights[i]
            if kind == "mlp":
                w *= spec.mlp_scale             # MLP write-path is more damaging -> ablate it gentler
            if w <= 0:
                continue
            d = spec.directions[spec.dir_layer if spec.dir_layer is not None else i + 1]
            d = (d / d.norm().clamp(min=1e-8)).to(W.dtype).to(W.device)
            if W.ndim == 3:                      # fused MoE experts: same rank-1 removal, per expert
                proj = torch.einsum("o,eoi->ei", d, W)   # [E, in], each from the ORIGINAL W[e]
                term = w * torch.einsum("o,ei->eoi", d, proj)
            else:
                proj = d @ W                     # [in], read from the ORIGINAL W
                term = w * torch.outer(d, proj)
            delta = term if delta is None else delta.add_(term)
            if any_np and not eff_np[si]:        # non-preserving axis: keep its magnitude loss
                delta_keep = term.clone() if delta_keep is None else delta_keep.add_(term)
        if delta is None:
            continue
        ref_norm = None
        if any_np:                               # #2 target: column norm after ONLY non-preserving removals
            W_ref = W if delta_keep is None else W - delta_keep
            ref_norm = W_ref.norm(dim=cdim, keepdim=True)
        W.sub_(delta)                            # one subtraction => no stacking across axes
        if ref_norm is not None:                 # restore the magnitude the preserving axes' removal cost
            W.mul_(ref_norm / W.norm(dim=cdim, keepdim=True).clamp(min=1e-8))


def _fake_moe(E=3, d=8, ff=16, n=2):
    """A tiny fake gemma-4 decoder carrying a Gemma4-style fused experts.down_proj [E, d, ff]
    Parameter, so the 3-D write-path can be exercised CPU-only."""
    import torch.nn as nn
    layers = nn.ModuleList()
    for _ in range(n):
        layer = nn.Module()
        layer.self_attn = nn.Module(); layer.self_attn.o_proj = nn.Linear(d, d, bias=False)
        layer.mlp = nn.Module(); layer.mlp.down_proj = nn.Linear(ff, d, bias=False)
        layer.experts = nn.Module(); layer.experts.down_proj = nn.Parameter(torch.randn(E, d, ff))
        layers.append(layer)
    model = nn.Module(); model.model = nn.Module()
    model.model.language_model = nn.Module(); model.model.language_model.layers = layers
    return model, layers


def _check_moe() -> None:
    """No-network math check for the fused-expert (26B-A4B) write-path: build a tiny fake
    decoder, run the REAL apply_ablation over it, and assert every expert's residual
    projection is removed — with AND without norm-preservation (#2)."""
    n, d = 2, 8
    directions = torch.randn(n + 1, d)

    model, layers = _fake_moe(d=d, n=n)
    apply_ablation(model, directions, weights=[1.0] * n)
    for i, layer in enumerate(layers):
        dv = directions[i + 1] / directions[i + 1].norm()
        resid = torch.einsum("o,eoi->ei", dv, layer.experts.down_proj.data).norm().item()
        assert resid < 1e-4, f"fused expert down_proj not ablated (layer {i}): {resid}"

    # #2: column-renorm must keep the removal EXACT (cdim=1 for the 3-D fused path) AND restore
    # every per-expert column norm to its pre-edit value.
    model2, layers2 = _fake_moe(d=d, n=n)
    pre = [layer.experts.down_proj.data.norm(dim=1, keepdim=True).clone() for layer in layers2]
    apply_ablation(model2, directions, weights=[1.0] * n, norm_preserve=True)
    for i, layer in enumerate(layers2):
        dv = directions[i + 1] / directions[i + 1].norm()
        W = layer.experts.down_proj.data
        resid = torch.einsum("o,eoi->ei", dv, W).norm().item()
        assert resid < 1e-4, f"norm-preserve broke fused ablation (layer {i}): {resid}"
        assert torch.allclose(W.norm(dim=1, keepdim=True), pre[i], atol=1e-5), \
            f"norm-preserve did not restore fused column norms (layer {i})"
    print("ablate.py MoE fused-expert self-check OK (+ norm-preserve)")


def _check_mixed_np() -> None:
    """Per-axis norm-preserve in a JOINT edit -- the thing a single global flag can't express:
    spec A (norm_preserve=False) KEEPS its column-shrinkage, spec B (norm_preserve=True) has its
    magnitude RESTORED. The mixed edit must equal the (unchanged) full non-stacking removal of BOTH,
    column-rescaled to the A-ONLY norms -- so B's removal is magnitude-compensated, A's is not, and
    the removal span is untouched. (Overlapping dirs aren't each individually zeroed under
    non-stacking -- that's demo() part 2 -- so we check the rescale, not per-direction residuals.)"""
    n, d = 2, 8
    dA, dB = torch.randn(n + 1, d), torch.randn(n + 1, d)
    m, layers = _fake_moe(d=d, n=n)
    snap = snapshot(m)
    apply_ablation(m, [AblationSpec(dA, [1.0] * n, None)])           # A only -> reference norms
    ref = [l.self_attn.o_proj.weight.data.norm(dim=0, keepdim=True).clone() for l in layers]
    restore(m, snap)
    apply_ablation(m, [AblationSpec(dA, [1.0] * n, None),            # full non-stacking removal of BOTH
                       AblationSpec(dB, [1.0] * n, None)])
    full = [l.self_attn.o_proj.weight.data.clone() for l in layers]
    restore(m, snap)
    apply_ablation(m, [AblationSpec(dA, [1.0] * n, None, norm_preserve=False),   # A off, B on
                       AblationSpec(dB, [1.0] * n, None, norm_preserve=True)])
    for i, l in enumerate(layers):
        W = l.self_attn.o_proj.weight.data
        assert torch.allclose(W.norm(dim=0, keepdim=True), ref[i], atol=1e-5), \
            f"mixed norm-preserve: column norms != A-only reference (layer {i})"
        expect = full[i] * (ref[i] / full[i].norm(dim=0, keepdim=True).clamp(min=1e-8))
        assert torch.allclose(W, expect, atol=1e-5), \
            f"mixed norm-preserve: not the non-stacking removal rescaled to ref (layer {i})"
    restore(m, snap)
    print("ablate.py mixed per-axis norm-preserve self-check OK")


def demo() -> None:
    """Math self-check, in two parts:

    (1) A single full ablation (w=1) zeros that direction's projection out of o_proj AND the
        PLE write-path — the basic orthogonalization is correct.
    (2) A JOINT edit of two OVERLAPPING directions equals the closed-form single subtraction
        `W - Σ_a d_a (d_aᵀ W)`, every projection read from the SAME original W, and genuinely
        DIFFERS from chaining (edit, then re-edit the already-edited matrix). That difference
        is the actual proof this is a non-stacking simultaneous removal. Overlap is essential:
        for orthogonal directions stacking and non-stacking coincide, so an orthogonal pair
        could not tell the two apart."""
    _check_moe()         # CPU-only, no model load: the 3-D fused-expert branch, before the heavy part
    _check_mixed_np()    # CPU-only: per-axis (mixed) norm-preserve in a joint edit
    from transformers import AutoModelForCausalLM

    from depurple._model import MODEL, load_directions
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    directions = load_directions()
    n = len(decoder_layers(model))
    L = n // 2

    # (1) single-axis full ablation zeros the projection in o_proj and the PLE write-path.
    snap = snapshot(model)
    apply_ablation(model, directions, weights=[1.0] * n, dir_layer=None)
    m = decoder_layers(model)[L].self_attn.o_proj
    d = (directions[L + 1] / directions[L + 1].norm()).to(m.weight.dtype).to(m.weight.device)
    resid = (d @ m.weight.data).norm().item()
    layer = decoder_layers(model)[L]
    ple_resid = None
    if hasattr(layer, "per_layer_projection"):       # the newly-added target must also be ablated
        ple_resid = (d @ layer.per_layer_projection.weight.data).norm().item()
        print(f"per_layer_projection post-ablation (should be ~0): {ple_resid:.4f}")
    restore(model, snap)
    restored = (d @ decoder_layers(model)[L].self_attn.o_proj.weight.data).norm().item()
    print(f"single-axis o_proj projection: post={resid:.4f}  restored={restored:.4f}")
    assert resid < restored * 0.05, "single-axis ablation did not remove the direction"
    if ple_resid is not None:
        assert ple_resid < restored * 0.05, "per_layer_projection not ablated"

    # (1b) mlp_scale dials the MLP write-path apart: scale=0 leaves down_proj untouched while
    # o_proj is still fully ablated (model is pristine here from the restore above).
    base_mlp = (d @ decoder_layers(model)[L].mlp.down_proj.weight.data).norm().item()
    apply_ablation(model, directions, weights=[1.0] * n, dir_layer=None, mlp_scale=0.0)
    mlp_resid = (d @ decoder_layers(model)[L].mlp.down_proj.weight.data).norm().item()
    op_resid = (d @ decoder_layers(model)[L].self_attn.o_proj.weight.data).norm().item()
    restore(model, snap)
    print(f"mlp_scale=0: down_proj proj kept={mlp_resid:.4f} (base {base_mlp:.4f})  o_proj removed={op_resid:.4f}")
    assert op_resid < restored * 0.05, "attn not ablated under mlp_scale=0"
    assert mlp_resid > base_mlp * 0.95, "down_proj should be untouched at mlp_scale=0"

    # (1c) norm-preserve (#2): column-renorm keeps the removal EXACT (proj ~0) AND restores each
    # o_proj column norm (2-D path, cdim=0). This is the assertion the plan's row-renorm couldn't meet.
    pre_col = decoder_layers(model)[L].self_attn.o_proj.weight.data.norm(dim=0, keepdim=True).clone()
    apply_ablation(model, directions, weights=[1.0] * n, dir_layer=None, norm_preserve=True)
    Wnp = decoder_layers(model)[L].self_attn.o_proj.weight.data
    np_resid = (d @ Wnp).norm().item()
    col_ok = torch.allclose(Wnp.norm(dim=0, keepdim=True), pre_col, rtol=0.03, atol=1e-3)
    restore(model, snap)
    print(f"norm-preserve: o_proj proj={np_resid:.4f} (should be ~0)  col-norms restored={col_ok}")
    assert np_resid < restored * 0.05, "norm-preserve broke the ablation (2-D column-renorm should keep proj~0)"
    assert col_ok, "norm-preserve did not restore o_proj column norms"

    # (2) second axis that OVERLAPS the real one per layer (cos = 0.6), so non-stacking and
    # chaining MUST diverge — an orthogonal axis would leave them identical.
    g = torch.Generator().manual_seed(0)
    dir2 = torch.randn(directions.shape, generator=g)
    for k in range(directions.shape[0]):
        a = directions[k].float() / directions[k].float().norm().clamp(min=1e-8)
        perp = dir2[k].float() - (dir2[k].float() @ a) * a
        perp = perp / perp.norm().clamp(min=1e-8)
        dir2[k] = 0.6 * a + 0.8 * perp           # unit; cos(dir1, dir2) = 0.6

    W0 = decoder_layers(model)[L].self_attn.o_proj.weight.detach().clone()
    d1 = (directions[L + 1] / directions[L + 1].norm()).to(W0.dtype).to(W0.device)
    d2 = (dir2[L + 1] / dir2[L + 1].norm()).to(W0.dtype).to(W0.device)
    p1 = d1 @ W0
    ns = W0 - (torch.outer(d1, p1) + torch.outer(d2, d2 @ W0))   # both proj from ORIGINAL W0
    ch = W0 - torch.outer(d1, p1)
    ch = ch - torch.outer(d2, d2 @ ch)                          # chaining: re-read the edited W

    # ponytail: reuse the part-1 snapshot — the model is already pristine (restored above),
    # so a 2nd snapshot just OOMs a 96GB card with another ~18GB copy of the residual matrices.
    apply_ablation(model, [AblationSpec(directions, [1.0] * n, None),
                           AblationSpec(dir2, [1.0] * n, None)])
    Wj = decoder_layers(model)[L].self_attn.o_proj.weight.data
    edit = (W0 - ns).norm().item()                  # scale = magnitude of the removal
    err_ns = (Wj - ns).norm().item() / edit          # joint == single-subtraction form (~0)
    gap_chain = (ns - ch).norm().item() / edit       # ... and is NOT the chained result (>0)
    restore(model, snap)
    print(f"joint vs single-subtraction: rel err={err_ns:.4f} (~0)   "
          f"joint vs chaining: rel gap={gap_chain:.4f} (>0 => non-stacking)")
    assert err_ns < 0.05, "joint edit is not the single-subtraction (non-stacking) form"
    assert gap_chain > 0.05, "overlap too small to distinguish non-stacking from chaining"
    print("ablate.py self-check OK")


if __name__ == "__main__":
    demo()
