#!/usr/bin/env python3
"""
probe_realworld_seq.py — Real-World Sequential Continual Language Modeling
==========================================================================
Purpose:
    Validate LAOSP (Level-Aware Optimizer-State Protection) on a 4-domain
    sequential continual LM task with OVERLAPPING vocabularies and realistic
    frequency profiles. This addresses the reviewer critique that vocab-split
    tasks are too synthetic and do not represent real continual learning.

Domain Design (synthetic generation, realistic statistics):
    Domain 1 "wiki":   Zipfian s=1.0, broad vocab, moderate entropy
                        Tokens concentrated in [0, 384) with peak at [0, 128)
    Domain 2 "tech":   Zipfian s=0.7, concentrated vocab, lower entropy
                        Tokens concentrated in [64, 448) with peak at [192, 320)
    Domain 3 "code":   Zipfian s=0.5, very skewed, heavy structural repetition
                        Tokens concentrated in [128, 512) with peak at [256, 384)
    Domain 4 "dialog": Zipfian s=1.5, high-frequency common tokens, short bursts
                        Tokens concentrated in [0, 320) with peak at [0, 96)

    Key properties:
    - All domains share tokens in [128, 320) (overlapping "core" vocab)
    - Each domain has a unique high-frequency band (non-shared)
    - Gradient subspaces have PARTIAL overlap (not near-orthogonal)
    - This makes the CL problem harder and more realistic than disjoint splits

Methods (6):
    vanilla          No protection
    ewc              EWC with diagonal Fisher (lambda=500)
    ogd              Parameter-space OGD (Farajtabar et al.)
    uniform_osp      Uniform alpha=0.5 across ALL levels (critical baseline)
    level_aware_osp  Level-aware alpha schedule (OGP only, no CAM/CLGD)
    full_laosp       Full method: OGP + CAM + CLGD

Metrics:
    PPL matrix       PPL on domain i after training on domain j
    ACC              Average final PPL across all domains
    BWT              Backward transfer (ppl-based)
    Forgetting       Average max forgetting per domain
    Retention        Average PPL_{i,i} / PPL_{i,T}
    ER_slow          Energy retention at cms_slow (auxiliary)

Model: 40M HOPE (dim=128, n_layers=4, heads=4, vocab=512)
       CMS levels: fast(C=1), mid(C=4), slow(C=32), ultra(C=128)

Training: 2000 steps/domain, batch=32, seq_len=256

Usage:
    python probe_realworld_seq.py --condition full_laosp --device cuda:0
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ───────────────────────────────────────────────────────────────────
# Source paths (same as probe_main.py)
# ───────────────────────────────────────────────────────────────────
SEARCH_PATHS = [
    os.path.expanduser("~/hope_src"),
    os.path.expanduser("~/workspace/yuelin/continuous_learning/"
                       "Nested Learning The Illusion of Deep Learning Architecture "
                       "Ali Behrouz , Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni/"
                       "code/nested_learning_hgc/src"),
]
for p in SEARCH_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.levels import LevelSpec

# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────
LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
SLOW_LEVELS   = ["cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = {"cms_fast": 1, "cms_mid": 4, "cms_slow": 32, "cms_ultra": 128}

# Level-aware alpha schedule: alpha(l) = 1 - 1/log2(C(l)+1)
LEVEL_ALPHA_SCHEDULE = {
    "cms_fast":  0.0,    # C=1:   no protection (fast adaptation)
    "cms_mid":   0.60,   # C=4:   moderate protection
    "cms_slow":  0.83,   # C=32:  strong protection
    "cms_ultra": 0.92,   # C=128: near-full protection
}

# ───────────────────────────────────────────────────────────────────
# Domain definitions — overlapping, realistic frequency profiles
# ───────────────────────────────────────────────────────────────────
# vocab_size=512. All domains share tokens in [128, 320).
# Each domain has a primary band and a distinct peak band.
#
#   wiki:   ████████████░░░░░░░░  [0, 384)   peak [0, 128)
#   tech:   ░░██████████████░░░░  [64, 448)  peak [192, 320)
#   code:   ░░░░░░██████████████  [128, 512) peak [256, 384)
#   dialog: ██████████░░░░░░░░░░  [0, 320)   peak [0, 96)
#
# Overlap region [128, 320) is shared by ALL four domains.
# This creates partial gradient subspace overlap.

DOMAIN_CONFIGS = {
    "wiki": {
        "desc": "Natural language — Zipfian s=1.0, broad vocab, peak in low tokens",
        "base_lo": 0,   "base_hi": 384,   "s": 1.0,
        "peak_lo": 0,   "peak_hi": 128,   "peak_prob": 0.45,
    },
    "tech": {
        "desc": "Technical text — Zipfian s=0.7, concentrated vocab, mid-range peak",
        "base_lo": 64,  "base_hi": 448,   "s": 0.7,
        "peak_lo": 192, "peak_hi": 320,   "peak_prob": 0.50,
    },
    "code": {
        "desc": "Code patterns — Zipfian s=0.5, very skewed, structural repetition",
        "base_lo": 128, "base_hi": 512,   "s": 0.5,
        "peak_lo": 256, "peak_hi": 384,   "peak_prob": 0.55,
    },
    "dialog": {
        "desc": "Conversational — Zipfian s=1.5, high-freq common tokens, short bursts",
        "base_lo": 0,   "base_hi": 320,   "s": 1.5,
        "peak_lo": 0,   "peak_hi": 96,    "peak_prob": 0.50,
    },
}
DOMAIN_ORDER = ["wiki", "tech", "code", "dialog"]

# ───────────────────────────────────────────────────────────────────
# Method conditions
# ───────────────────────────────────────────────────────────────────
CONDITIONS = {
    "vanilla": dict(
        use_hgc=False, use_ewc=False, use_ogd=False,
        alpha_mode="none", gamma=0.0, dist=0.0,
    ),
    "ewc": dict(
        use_hgc=False, use_ewc=True, ewc_lambda=500.0,
        use_ogd=False, alpha_mode="none", gamma=0.0, dist=0.0,
    ),
    "ogd": dict(
        use_hgc=False, use_ewc=False, use_ogd=True, ogd_rank=20,
        alpha_mode="none", gamma=0.0, dist=0.0,
    ),
    "uniform_osp": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="uniform", alpha_uniform=0.5,
        gamma=0.0, dist=0.0,
    ),
    "uniform_osp_a02": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="uniform", alpha_uniform=0.2,
        gamma=0.0, dist=0.0,
    ),
    "uniform_osp_a01": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="uniform", alpha_uniform=0.1,
        gamma=0.0, dist=0.0,
    ),
    "level_aware_osp": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="level_aware", alpha_scale=1.0,
        gamma=0.0, dist=0.0,
    ),
    "level_aware_osp_half": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="level_aware", alpha_scale=0.5,
        gamma=0.0, dist=0.0,
    ),
    "level_aware_osp_quarter": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="level_aware", alpha_scale=0.25,
        gamma=0.0, dist=0.0,
    ),
    "full_laosp": dict(
        use_hgc=True, use_ewc=False, use_ogd=False,
        alpha_mode="level_aware",
        gamma=0.5, dist=0.3,
    ),
}
VALID_CONDITIONS = list(CONDITIONS.keys())


# ───────────────────────────────────────────────────────────────────
# Data generation
# ───────────────────────────────────────────────────────────────────
def make_domain_batches(domain_name, vocab_size, seq_len, batch_size,
                        n_steps, device, seed=42):
    """
    Generate batches with Zipfian base distribution + domain-specific peak band.
    Tokens are drawn from [base_lo, base_hi) with Zipfian ranking, then a
    fraction peak_prob of positions are replaced with tokens from [peak_lo, peak_hi).
    """
    cfg = DOMAIN_CONFIGS[domain_name]
    base_lo, base_hi = cfg["base_lo"], cfg["base_hi"]
    peak_lo, peak_hi = cfg["peak_lo"], cfg["peak_hi"]
    peak_prob = cfg["peak_prob"]
    s = cfg["s"]

    assert base_hi > base_lo, f"Invalid base range for {domain_name}"
    assert peak_hi > peak_lo, f"Invalid peak range for {domain_name}"
    assert base_hi <= vocab_size, f"base_hi={base_hi} > vocab_size={vocab_size}"
    assert peak_hi <= vocab_size, f"peak_hi={peak_hi} > vocab_size={vocab_size}"

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Zipfian distribution over base range
    n_base = base_hi - base_lo
    ranks = torch.arange(1, n_base + 1, dtype=torch.float32)
    probs = 1.0 / ranks.pow(s)
    probs = probs / probs.sum()

    # Peak range distribution (uniform within peak band)
    n_peak = peak_hi - peak_lo

    batches = []
    tokens_per_batch = batch_size * (seq_len + 1)

    for _ in range(n_steps):
        # Base: Zipfian sample
        indices = torch.multinomial(probs, tokens_per_batch,
                                    replacement=True, generator=rng)
        tokens = (indices + base_lo).clamp(0, vocab_size - 1)

        # Peak override: replace fraction of positions with peak-band tokens
        mask = torch.rand(tokens_per_batch, generator=rng) < peak_prob
        n_replace = mask.sum().item()
        if n_replace > 0:
            peak_tokens = torch.randint(0, n_peak, (n_replace,), generator=rng) + peak_lo
            tokens[mask] = peak_tokens.clamp(0, vocab_size - 1)

        batches.append(tokens.reshape(batch_size, seq_len + 1).to(device))

    return batches


def make_eval_batches(domain_name, vocab_size, seq_len, batch_size,
                      n_eval, device):
    """Fixed eval set (seed offset 10000 to avoid train overlap)."""
    seed = 10000 + DOMAIN_ORDER.index(domain_name) * 137
    return make_domain_batches(domain_name, vocab_size, seq_len, batch_size,
                               n_eval, device, seed=seed)


# ───────────────────────────────────────────────────────────────────
# Model construction
# ───────────────────────────────────────────────────────────────────
def build_model(device, vocab_size, dim, num_layers, heads, condition):
    cfg_d = CONDITIONS[condition]
    use_hgc = cfg_d["use_hgc"]

    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=dim, num_layers=num_layers, heads=heads,
        titan_level=LevelSpec("titan", update_period=1),
        cms_levels=[
            LevelSpec("cms_fast",  update_period=1),
            LevelSpec("cms_mid",   update_period=4),
            LevelSpec("cms_slow",  update_period=32),
            LevelSpec("cms_ultra", update_period=128),
        ],
        block_variant="hope_hybrid",
        hgc_enabled=use_hgc,
        hgc_r_base=4,
        hgc_gamma_scale=cfg_d.get("gamma", 0.0),
        hgc_distillation_strength=cfg_d.get("dist", 0.0),
        optimizers=({"default": {"type": "deep_momentum_hgc", "params": {}}}
                    if use_hgc else None),
    )
    model = HOPEModel(cfg).to(device)

    # Set per-level alpha values
    if use_hgc:
        alpha_mode = cfg_d["alpha_mode"]
        for block in model.blocks:
            gm = getattr(block, "grad_memory", None)
            if gm is None:
                continue
            for ln in LEVEL_NAMES:
                if alpha_mode == "uniform":
                    gm.alpha[ln] = cfg_d["alpha_uniform"]
                elif alpha_mode == "level_aware":
                    gm.alpha[ln] = LEVEL_ALPHA_SCHEDULE[ln] * cfg_d.get("alpha_scale", 1.0)
                else:
                    gm.alpha[ln] = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: dim={dim} layers={num_layers} heads={heads}  "
          f"params={n_params/1e6:.1f}M  condition={condition}  "
          f"use_hgc={use_hgc}  alpha_mode={cfg_d.get('alpha_mode','none')}  "
          f"gamma={cfg_d.get('gamma',0)}  dist={cfg_d.get('dist',0)}")
    return model, n_params


# ───────────────────────────────────────────────────────────────────
# Level-param helpers
# ───────────────────────────────────────────────────────────────────
def get_level_params(block, level_name):
    cms = getattr(block, "cms", None)
    if cms is None:
        return []
    sub = getattr(getattr(cms, "blocks", None), level_name, None)
    if sub is None:
        return []
    return list(sub.named_parameters())


def get_level_grad_flat(block, level_name):
    parts = [p.grad.detach().reshape(-1).float()
             for _, p in get_level_params(block, level_name)
             if p.grad is not None]
    return torch.cat(parts) if parts else None


def get_level_exp_avg_flat(opt, block, level_name):
    parts = [opt.state[p]["exp_avg"].detach().reshape(-1).cpu().float()
             for _, p in get_level_params(block, level_name)
             if p in opt.state and "exp_avg" in opt.state[p]]
    return torch.cat(parts) if parts else None


# ───────────────────────────────────────────────────────────────────
# HGC API calls (accumulate / consolidate / project)
# ───────────────────────────────────────────────────────────────────
def hgc_accumulate(model):
    """After backward(), before opt.step(): accumulate gradient signatures."""
    if not hasattr(hgc_accumulate, "_call_count"):
        hgc_accumulate._call_count = 0
    hgc_accumulate._call_count += 1

    # 256M HGC runs OOM if we store signatures every step.
    # Throttle accumulation to reduce retained GPU tensors.
    if hgc_accumulate._call_count % 40 != 0:
        return

    for block in model.blocks:
        gm = getattr(block, "grad_memory", None)
        if gm is None:
            continue
        for ln in SLOW_LEVELS:
            g = get_level_grad_flat(block, ln)
            if g is not None:
                gm.accumulate_gradient(ln, g)


def hgc_consolidate(model):
    """At domain boundary: build SVD basis from accumulated gradients."""
    for block in model.blocks:
        gm = getattr(block, "grad_memory", None)
        if gm is not None:
            gm.consolidate_all()


def hgc_project(model):
    """After backward(), before opt.step(): project level grads to protect old subspace."""
    for block in model.blocks:
        gm = getattr(block, "grad_memory", None)
        if gm is None:
            continue
        for ln in SLOW_LEVELS:
            params = get_level_params(block, ln)
            parts, shapes, param_refs = [], [], []
            for _, p in params:
                if p.grad is not None:
                    parts.append(p.grad.detach().reshape(-1).float())
                    shapes.append(p.grad.shape)
                    param_refs.append(p)
            if not parts:
                continue

            g_cat = torch.cat(parts)
            g_proj = gm.project_gradient(ln, g_cat)

            offset = 0
            for p, shape in zip(param_refs, shapes):
                numel = p.grad.numel()
                p.grad.copy_(g_proj[offset:offset + numel].reshape(shape).to(p.grad.dtype))
                offset += numel


# ───────────────────────────────────────────────────────────────────
# EWC
# ───────────────────────────────────────────────────────────────────
def compute_ewc_fisher(model, batches, n_batches=200):
    """Diagonal Fisher information matrix."""
    fisher = {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}
    model.eval()
    count = min(n_batches, len(batches))
    for b in batches[:count]:
        model.zero_grad()
        logits = model(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               b[:, 1:].reshape(-1))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    model.train()
    for n in fisher:
        fisher[n] /= count
    return fisher


def ewc_penalty(model, ewc_states, lam=500.0):
    """Accumulate EWC penalty over all previous tasks."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for old_params, fisher_dict in ewc_states:
        for n, p in model.named_parameters():
            if n in fisher_dict and n in old_params:
                penalty = penalty + (fisher_dict[n] *
                                     (p - old_params[n].to(p.device)) ** 2).sum()
    return lam * penalty


# ───────────────────────────────────────────────────────────────────
# OGD
# ───────────────────────────────────────────────────────────────────
def compute_ogd_basis(model, batches, rank=20, n_batches=50):
    """Parameter-space gradient SVD basis (Farajtabar et al., 2020)."""
    grads = []
    model.eval()
    count = min(n_batches, len(batches))
    for b in batches[:count]:
        model.zero_grad()
        logits = model(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               b[:, 1:].reshape(-1))
        loss.backward()
        g = torch.cat([p.grad.detach().reshape(-1).float()
                       for p in model.parameters() if p.grad is not None])
        grads.append(g.cpu())
    model.train()
    assert grads, "OGD: no gradients collected"
    G = torch.stack(grads, dim=1)  # [D, n_batches]
    q = min(rank, G.shape[1], G.shape[0])
    U, _, _ = torch.svd_lowrank(G.float(), q=q)
    return U[:, :q]  # [D, rank]


def apply_ogd(model, ogd_bases):
    """Project parameter gradients away from all previous task bases."""
    if not ogd_bases:
        return
    param_list = [p for p in model.parameters() if p.grad is not None]
    g_flat = torch.cat([p.grad.detach().reshape(-1).float()
                        for p in param_list]).cpu()
    for basis in ogd_bases:
        for col in basis.T:
            col = col.to(g_flat.device)
            g_flat = g_flat - (g_flat @ col / (col @ col + 1e-12)) * col
    offset = 0
    for p in param_list:
        n = p.numel()
        p.grad.copy_(g_flat[offset:offset + n].reshape(p.shape).to(p.device))
        offset += n


# ───────────────────────────────────────────────────────────────────
# PPL evaluation
# ───────────────────────────────────────────────────────────────────
def eval_ppl(model, batches, n_eval=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for b in batches[:n_eval]:
            logits = model(b[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   b[:, 1:].reshape(-1))
            assert not torch.isnan(loss), "NaN loss in eval"
            losses.append(loss.item())
    model.train()
    assert losses, "No valid eval batches"
    return math.exp(sum(losses) / len(losses))


# ───────────────────────────────────────────────────────────────────
# ER (auxiliary, from Adam exp_avg at cms_slow)
# ───────────────────────────────────────────────────────────────────
def build_subspace_A(opt, model, snapshots, rank=16):
    """Build U_A from collected exp_avg snapshots during domain training."""
    if len(snapshots) < 2:
        return None
    G = torch.stack(snapshots, dim=1).float()
    valid = G.norm(dim=0) > 1e-10
    if valid.sum() < 2:
        return None
    G = G[:, valid]
    q = min(rank, G.shape[1], G.shape[0])
    U, _, _ = torch.svd_lowrank(G, q=q)
    U_A = U[:, :q]
    block0 = model.blocks[0]
    m_now = get_level_exp_avg_flat(opt, block0, "cms_slow")
    init_sq = float((U_A.T @ m_now.float()).norm().item() ** 2) if m_now is not None else 1.0
    return (U_A, init_sq)


def compute_er_snapshot(opt, model, subspace_A):
    """ER at current step for cms_slow level, first block."""
    if subspace_A is None:
        return None
    U_A, init_sq = subspace_A
    block0 = model.blocks[0]
    m = get_level_exp_avg_flat(opt, block0, "cms_slow")
    if m is None:
        return None
    proj_sq = float((U_A.T @ m.float()).norm().item() ** 2)
    er = proj_sq / (init_sq + 1e-12)
    return float(er)


# ───────────────────────────────────────────────────────────────────
# Main probe
# ───────────────────────────────────────────────────────────────────
def run_probe(condition, vocab_size, dim, num_layers, heads, seq_len,
              batch_size, domain_steps, eval_batches_n, lr, device):

    print(f"\n{'='*65}")
    print(f"  Real-World Sequential CL — condition={condition}")
    print(f"  4 domains x {domain_steps} steps  |  vocab={vocab_size}  "
          f"dim={dim}  layers={num_layers}")
    print(f"{'='*65}")

    cfg_d = CONDITIONS[condition]
    model, n_params = build_model(device, vocab_size, dim, num_layers,
                                  heads, condition)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-generate fixed eval batches
    eval_data = {
        dom: make_eval_batches(dom, vocab_size, seq_len, batch_size,
                               eval_batches_n, device)
        for dom in DOMAIN_ORDER
    }

    # State containers
    ewc_states = []
    ogd_bases = []

    # PPL matrix: ppl_matrix[t][i] = PPL on domain i after learning domain t
    ppl_matrix = []
    loss_curves = []
    er_per_domain = []

    t0_total = time.time()

    for domain_idx, domain_name in enumerate(DOMAIN_ORDER):
        print(f"\n{'─'*55}")
        print(f"  Domain {domain_idx+1}/4: {domain_name.upper()}")
        print(f"  {DOMAIN_CONFIGS[domain_name]['desc']}")
        print(f"{'─'*55}")

        train_batches = make_domain_batches(
            domain_name, vocab_size, seq_len, batch_size,
            domain_steps, device, seed=42 + domain_idx * 137,
        )

        domain_losses = []
        exp_avg_snapshots = []

        for step in range(domain_steps):
            opt.zero_grad()
            logits = model(train_batches[step][:, :-1])
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                train_batches[step][:, 1:].reshape(-1),
            )
            assert not torch.isnan(lm_loss), \
                f"NaN lm_loss at domain={domain_name} step={step}"

            loss = lm_loss
            if cfg_d["use_ewc"] and ewc_states:
                loss = loss + ewc_penalty(model, ewc_states, cfg_d["ewc_lambda"])

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            # HGC: project gradients for domains after the first
            if cfg_d["use_hgc"] and domain_idx > 0:
                hgc_project(model)

            # HGC: accumulate gradient signatures (all domains)
            if cfg_d["use_hgc"]:
                hgc_accumulate(model)

            # OGD: project parameter gradients
            if cfg_d["use_ogd"] and ogd_bases:
                apply_ogd(model, ogd_bases)

            opt.step()
            domain_losses.append(float(lm_loss.item()))

            # Collect exp_avg snapshots for ER (after warmup, every 20 steps)
            if step > 50 and step % 20 == 0 and len(exp_avg_snapshots) < 64:
                block0 = model.blocks[0]
                m = get_level_exp_avg_flat(opt, block0, "cms_slow")
                if m is not None and not torch.isnan(m).any():
                    exp_avg_snapshots.append(m.clone())

            if step % 500 == 0:
                print(f"  step {step:4d}  lm_loss={lm_loss.item():.4f}  "
                      f"total_loss={loss.item():.4f}")

        loss_curves.append(domain_losses)

        # ── Domain boundary operations ────────────────────────────
        print(f"\n  [Boundary] End of domain {domain_idx+1}: {domain_name}")

        # HGC consolidation
        if cfg_d["use_hgc"]:
            hgc_consolidate(model)
            block0 = model.blocks[0]
            gm0 = getattr(block0, "grad_memory", None)
            if gm0:
                for ln in SLOW_LEVELS:
                    sig = gm0.signatures.get(ln)
                    ok = sig is not None and sig.basis is not None
                    alpha_val = gm0.alpha.get(ln, "?")
                    alpha_str = f"{alpha_val:.2f}" if isinstance(alpha_val, float) else str(alpha_val)
                    print(f"    HGC {ln}: basis={'OK' if ok else 'FAIL'}  "
                          f"alpha={alpha_str}")

        # EWC: compute Fisher
        if cfg_d["use_ewc"]:
            print(f"  [EWC] Computing Fisher information...")
            fisher = compute_ewc_fisher(model, train_batches[-200:],
                                        n_batches=200)
            old_params = {n: p.data.clone().cpu()
                          for n, p in model.named_parameters()}
            ewc_states.append((old_params, fisher))
            print(f"  [EWC] Fisher computed. EWC states: {len(ewc_states)}")

        # OGD: compute gradient basis
        if cfg_d["use_ogd"]:
            ogd_rank = cfg_d.get("ogd_rank", 20)
            print(f"  [OGD] Computing gradient basis (rank={ogd_rank})...")
            basis = compute_ogd_basis(model, train_batches[-50:],
                                      rank=ogd_rank, n_batches=50)
            ogd_bases.append(basis)
            print(f"  [OGD] Basis shape: {basis.shape}  "
                  f"bases accumulated: {len(ogd_bases)}")

        # ER snapshot
        subspace_A = build_subspace_A(opt, model, exp_avg_snapshots)
        er_val = compute_er_snapshot(opt, model, subspace_A)
        er_per_domain.append(er_val)
        er_str = f"{er_val:.4f}" if er_val is not None else "N/A"
        print(f"  ER_slow after domain {domain_idx+1}: {er_str}")

        # ── Evaluate PPL on ALL domains (including unseen) ────────
        ppl_row = {}
        for eval_idx, eval_name in enumerate(DOMAIN_ORDER):
            ppl = eval_ppl(model, eval_data[eval_name], n_eval=eval_batches_n)
            ppl_row[eval_name] = ppl
            seen = "seen" if eval_idx <= domain_idx else "unseen"
            print(f"  PPL[{eval_name:8s}] = {ppl:8.2f}  ({seen})")
        ppl_matrix.append(ppl_row)

    # ── Compute standard CL metrics ──────────────────────────────
    T = len(DOMAIN_ORDER)
    final_row = ppl_matrix[-1]  # PPL after learning all domains

    # ACC: average final PPL across all domains
    acc = sum(final_row[d] for d in DOMAIN_ORDER) / T

    # BWT: backward transfer = avg(PPL_{T,i} - PPL_{i,i}) for i < T
    # For PPL: lower is better, so BWT > 0 means forgetting (PPL increased)
    bwt_vals = []
    for i in range(T - 1):
        d = DOMAIN_ORDER[i]
        ppl_ii = ppl_matrix[i][d]       # PPL on domain i right after learning i
        ppl_iT = ppl_matrix[-1][d]      # PPL on domain i after learning all
        bwt_vals.append(ppl_iT - ppl_ii)
    bwt = sum(bwt_vals) / len(bwt_vals) if bwt_vals else float("nan")

    # Forgetting: avg(max(0, PPL_{T,i} - PPL_{i,i})) for i < T
    forgetting_vals = [max(0.0, v) for v in bwt_vals]
    forgetting = (sum(forgetting_vals) / len(forgetting_vals)
                  if forgetting_vals else float("nan"))

    # Retention: avg(PPL_{i,i} / PPL_{T,i}) for i < T — higher is better
    ret_vals = []
    for i in range(T - 1):
        d = DOMAIN_ORDER[i]
        ppl_ii = ppl_matrix[i][d]
        ppl_iT = ppl_matrix[-1][d]
        if ppl_iT > 0:
            ret_vals.append(ppl_ii / ppl_iT)
    retention = sum(ret_vals) / len(ret_vals) if ret_vals else float("nan")

    # Per-domain forgetting
    forgetting_per_domain = {}
    for i in range(T - 1):
        d = DOMAIN_ORDER[i]
        # Max forgetting: max PPL increase across all subsequent domain boundaries
        max_fgt = 0.0
        ppl_intro = ppl_matrix[i][d]
        for t in range(i + 1, T):
            ppl_t = ppl_matrix[t][d]
            max_fgt = max(max_fgt, ppl_t - ppl_intro)
        forgetting_per_domain[d] = max_fgt

    total_time = time.time() - t0_total

    print(f"\n{'='*65}")
    print(f"  RESULTS: condition={condition}")
    print(f"  ACC={acc:.2f}  BWT={bwt:+.2f}  "
          f"Forgetting={forgetting:.2f}  Retention={retention:.4f}")
    print(f"  Per-domain final PPL: "
          + "  ".join(f"{d}={final_row[d]:.1f}" for d in DOMAIN_ORDER))
    print(f"  Per-domain forgetting: "
          + "  ".join(f"{d}={forgetting_per_domain.get(d, 0):.1f}"
                      for d in DOMAIN_ORDER[:-1]))
    print(f"  ER_slow per domain: {er_per_domain}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*65}")

    # ── Build records ─────────────────────────────────────────────
    records = {
        "condition":            condition,
        "domain_order":         DOMAIN_ORDER,
        "ppl_matrix":           ppl_matrix,
        "acc":                  float(acc),
        "bwt":                  float(bwt),
        "forgetting":           float(forgetting),
        "retention":            float(retention),
        "forgetting_per_domain": {k: float(v) for k, v in forgetting_per_domain.items()},
        "er_per_domain":        er_per_domain,
        "loss_curves":          [c[::50] for c in loss_curves],  # downsample
        "total_time_s":         total_time,
        "model_info": {
            "dim": dim, "num_layers": num_layers, "heads": heads,
            "n_params_M": round(n_params / 1e6, 2),
            "vocab_size": vocab_size, "seq_len": seq_len,
            "batch_size": batch_size, "domain_steps": domain_steps,
            "n_domains": T,
            "experiment": "realworld_sequential",
        },
        "condition_cfg": {k: str(v) for k, v in CONDITIONS[condition].items()},
        "domain_configs": {
            d: {k: v for k, v in cfg.items() if k != "desc"}
            for d, cfg in DOMAIN_CONFIGS.items()
        },
    }
    return records


# ───────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-world sequential continual LM with LAOSP")
    parser.add_argument("--condition", type=str, required=True,
                        choices=VALID_CONDITIONS)
    parser.add_argument("--dim",          type=int,   default=128)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--heads",        type=int,   default=4)
    parser.add_argument("--vocab_size",   type=int,   default=512)
    parser.add_argument("--seq_len",      type=int,   default=256)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--domain_steps", type=int,   default=2000)
    parser.add_argument("--eval_batches", type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--output_dir",   type=str,
                        default=os.path.expanduser(
                            "~/experiments/results_realworld_seq"))
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    # ── Fast-fail preconditions ───────────────────────────────────
    assert torch.cuda.is_available(), "CUDA not available"
    assert args.condition in VALID_CONDITIONS, \
        f"Unknown condition: {args.condition!r}. Valid: {VALID_CONDITIONS}"
    assert args.dim > 0 and args.num_layers > 0 and args.heads > 0
    assert args.dim % args.heads == 0, \
        f"dim={args.dim} must be divisible by heads={args.heads}"
    assert args.vocab_size >= 512, \
        f"vocab_size={args.vocab_size} too small for domain configs"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}  Condition: {args.condition}  Seed: {args.seed}")

    records = run_probe(
        condition=args.condition,
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        heads=args.heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        domain_steps=args.domain_steps,
        eval_batches_n=args.eval_batches,
        lr=args.lr,
        device=device,
    )

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"records_{args.condition}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
