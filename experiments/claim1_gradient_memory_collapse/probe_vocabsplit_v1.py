"""
probe_vocabsplit_v1.py
======================
Claim 1 probe using **vocabulary-split** tasks (cross-distribution):
  Task A = tokens in [0, vocab_size//2)
  Task B = tokens in [vocab_size//2, vocab_size)

This creates genuinely orthogonal tasks, revealing true gradient memory collapse.
Intended to produce the main figure for Claim 1 at 256M scale.

Usage:
  python probe_vocabsplit_v1.py --dim 512 --num_layers 12 --heads 8 \
      --task_a_steps 2000 --task_b_steps 500 \
      --output_dir ~/experiments/results_vocabsplit_256M --device cuda:0
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
SEARCH_PATHS = [
    os.path.expanduser("~/hope_src"),
    os.path.expanduser("~/workspace/yuelin/continuous_learning/Nested Learning The Illusion of Deep Learning Architecture Ali Behrouz , Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni/code/nested_learning_hgc/src"),
]
for p in SEARCH_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.levels import LevelSpec
from nested_learning.optim.deep import DeepMomentum

LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = [1, 4, 32, 128]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_hope(device, dim, num_layers, heads, vocab_size, seq_len):
    cfg = ModelConfig(
        vocab_size   = vocab_size,
        d_model      = dim,
        num_layers   = num_layers,
        num_heads    = heads,
        d_ff         = dim * 4,
        max_seq_len  = seq_len,
        block_variant= "hope_hybrid",
        cms_levels   = [
            LevelSpec("cms_fast",  update_period=1),
            LevelSpec("cms_mid",   update_period=4),
            LevelSpec("cms_slow",  update_period=32),
            LevelSpec("cms_ultra", update_period=128),
        ],
        deep_momentum_variant = "nl_l2_precond",
        deep_momentum_beta    = 0.9,
    )
    model = HOPEModel(cfg).to(device)
    return model


# ── Data: vocabulary-split synthetic sequences ────────────────────────────────

def make_vocab_split_batches(vocab_size, split, seq_len, batch_size, n_steps, device):
    """Generate random token sequences from a vocab half-split."""
    lo = 0 if split == "A" else vocab_size // 2
    hi = vocab_size // 2 if split == "A" else vocab_size
    batches = []
    total_needed = n_steps * batch_size * (seq_len + 1)
    tokens = torch.randint(lo, hi, (total_needed,), device=device)
    idx = 0
    for _ in range(n_steps):
        chunk = tokens[idx: idx + batch_size * (seq_len + 1)].view(batch_size, seq_len + 1)
        batches.append(chunk)
        idx += batch_size * (seq_len + 1)
    return batches


# ── Train / Eval ──────────────────────────────────────────────────────────────

def compute_teach_signal(model, logits, tokens):
    B, T, V = logits.shape
    targets = tokens[:, 1: T + 1]
    if targets.shape[1] < T:
        pad = torch.zeros(B, T - targets.shape[1], dtype=targets.dtype, device=targets.device)
        targets = torch.cat([targets, pad], dim=1)
    active = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
    active[:, :T - 1] = True
    active_f = active.float().unsqueeze(-1)
    residual = logits.detach().clone()
    residual *= active_f
    safe_t = torch.where(active, targets, torch.zeros_like(targets))
    residual.scatter_add_(-1, safe_t.unsqueeze(-1), -active_f)
    residual /= active_f.sum().clamp(min=1.0)
    hw = model.lm_head.weight.detach().to(residual.dtype)
    return residual @ hw


def train_step(model, tokens, outer_optimizer):
    model.train()
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    outer_optimizer.zero_grad()
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    teach = compute_teach_signal(model, logits.detach(), tokens)
    with torch.no_grad():
        model(inp, teach_signal=teach)
    outer_optimizer.step()
    return loss.item()


# ── Momentum / parameter probing ──────────────────────────────────────────────

def get_level_momentum_flat(model, level_name):
    parts = []
    for block in model.blocks:
        lm = getattr(block, "level_manager", None)
        if lm is None:
            continue
        opt = lm.optimizers.get(level_name)
        if opt is None or not isinstance(opt, DeepMomentum):
            continue
        for st in opt.state.values():
            if st.grad_avg is not None:
                parts.append(st.grad_avg.detach().float().reshape(-1))
    return torch.cat(parts) if parts else None


def get_level_params_flat(model, level_name):
    parts = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is None or level_name not in cms.blocks:
            continue
        for p in cms.blocks[level_name].parameters():
            parts.append(p.detach().float().reshape(-1))
    return torch.cat(parts) if parts else None


def compute_svd_basis_cpu(snapshots, rank):
    """Stack snapshots and SVD on CPU to avoid OOM."""
    G = torch.stack(snapshots, dim=1).float()
    G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    col_norms = G.norm(dim=0)
    G = G[:, col_norms > 1e-12]
    if G.shape[1] < 2:
        return None, None
    q = min(rank, G.shape[0], G.shape[1])
    try:
        U, S, _ = torch.linalg.svd(G, full_matrices=False)
    except Exception:
        Q, R = torch.linalg.qr(G)
        U, S, _ = torch.linalg.svd(R, full_matrices=False)
        U = Q @ U
    return U[:, :q].cpu(), S[:q].cpu()


def proj_norm_ratio(U, v):
    vf = v.float()
    nv = vf.norm()
    if nv < 1e-12:
        return 0.0
    return float((U.float().T @ vf).norm().item() / nv.item())


def param_drift(now, ref):
    ref_n = ref.float().norm()
    if ref_n < 1e-12:
        return 0.0
    return float((now.float() - ref.float()).norm().item() / ref_n.item())


# ── Main probe ────────────────────────────────────────────────────────────────

def run_probe(dim, num_layers, heads, vocab_size, seq_len, batch_size,
              task_a_steps, task_b_steps, lr_outer, grad_rank, eval_every,
              device):

    print(f"\n{'='*60}")
    print(f"  Claim 1 Probe — Vocabulary Split (cross-distribution)")
    print(f"  dim={dim}  layers={num_layers}  heads={heads}")
    print(f"  vocab={vocab_size}  task_a={task_a_steps}  task_b={task_b_steps}")
    print(f"  device={device}")
    print(f"{'='*60}")

    # Adaptive buffer: keep RAM bounded (each snapshot ~param_count * 4 bytes)
    MAX_GRAD_BUF = 32 if dim >= 512 else 64

    print("\n=== Building HOPEModel ===")
    model = build_hope(device, dim, num_layers, heads, vocab_size, seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M")
    vram_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    print(f"  VRAM after build: {vram_mb:.0f} MB")

    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)

    # ── Task A batches ────────────────────────────────────────────────────────
    print("\n=== Generating Task A batches (vocab lower half) ===")
    batches_A = make_vocab_split_batches(vocab_size, "A", seq_len, batch_size, task_a_steps, device)

    # ── Train Task A ──────────────────────────────────────────────────────────
    print("\n=== Phase 1: Train on Task A ===")
    grad_buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}

    loss = 0.0
    for step in range(task_a_steps):
        loss = train_step(model, batches_A[step], outer_optimizer)
        for ln in LEVEL_NAMES:
            if len(grad_buffers[ln]) < MAX_GRAD_BUF:
                m = get_level_momentum_flat(model, ln)
                if m is not None and not torch.isnan(m).any():
                    grad_buffers[ln].append(m.clone().cpu())
        if step % eval_every == 0:
            vram_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
            print(f"  A step {step:5d}  loss={loss:.4f}  VRAM={vram_mb:.0f}MB")

    print(f"  Task A final loss={loss:.4f}")

    # ── Extract gradient signatures U_A ──────────────────────────────────────
    print("\n=== Extracting gradient signatures U_A per level ===")
    subspaces_A: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        buf = grad_buffers[ln]
        if len(buf) < 2:
            print(f"  {ln}: SKIP (buf={len(buf)})")
            continue
        eff_rank = max(1, min(grad_rank, len(buf) - 1))
        U, S = compute_svd_basis_cpu(buf, eff_rank)
        if U is None:
            print(f"  {ln}: SKIP (SVD failed)")
            continue
        subspaces_A[ln] = (U, S)
        print(f"  {ln} (C={C}): U={U.shape}, top-SV={S[0].item():.4f}, buf={len(buf)}")

    # Snapshot params at end of Task A
    theta_A: Dict[str, torch.Tensor] = {}
    for ln in LEVEL_NAMES:
        pf = get_level_params_flat(model, ln)
        if pf is not None:
            theta_A[ln] = pf.cpu()

    # Eval Task A loss as forgetting baseline
    with torch.no_grad():
        sample = batches_A[0]
        inp, tgt = sample[:, :-1], sample[:, 1:]
        logits = model(inp)
        loss_A_baseline = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
        ).item()
    print(f"\n  Task A baseline loss (for forgetting): {loss_A_baseline:.4f}")

    # Check A/B subspace overlap
    print("\n=== Checking Task A/B subspace overlap (quick 50-step probe) ===")
    _state_bk = {k: v.clone() for k, v in model.state_dict().items()}
    batches_B_probe = make_vocab_split_batches(vocab_size, "B", seq_len, batch_size, 50, device)
    b_buf: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}
    _outer_tmp = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)
    for vstep in range(50):
        train_step(model, batches_B_probe[vstep], _outer_tmp)
        for ln in LEVEL_NAMES:
            m = get_level_momentum_flat(model, ln)
            if m is not None:
                b_buf[ln].append(m.clone().cpu())
    model.load_state_dict(_state_bk)
    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)

    overlap_AB: Dict[str, float] = {}
    for ln in LEVEL_NAMES:
        if ln not in subspaces_A or len(b_buf[ln]) < 2:
            continue
        U_A = subspaces_A[ln][0]
        G_B = torch.stack(b_buf[ln], dim=1).float()
        G_B = torch.nan_to_num(G_B, nan=0.0)
        try:
            U_B, _, _ = torch.linalg.svd(G_B, full_matrices=False)
        except Exception:
            continue
        r = min(U_A.shape[1], U_B.shape[1])
        ov = float((U_A[:, :r].T @ U_B[:, :r]).norm() / math.sqrt(r))
        overlap_AB[ln] = ov
        print(f"  {ln}: A-B overlap={ov:.4f}")

    # Initial projections
    init_proj: Dict[str, float] = {}
    for ln in LEVEL_NAMES:
        if ln not in subspaces_A:
            init_proj[ln] = float("nan")
            continue
        U_A = subspaces_A[ln][0]
        m = get_level_momentum_flat(model, ln)
        init_proj[ln] = proj_norm_ratio(U_A, m.cpu()) if m is not None else float("nan")
    print(f"\n  Init projections: { {k: f'{v:.3f}' for k,v in init_proj.items()} }")

    # ── Task B: measure collapse ──────────────────────────────────────────────
    print("\n=== Phase 2: Train on Task B — measuring collapse ===")
    batches_B = make_vocab_split_batches(vocab_size, "B", seq_len, batch_size, task_b_steps, device)

    beta = 0.9
    records: Dict = {
        "step": [], "loss_B": [],
        "loss_A_forgetting": [],    # periodically re-evaluate loss on Task A batches
        "loss_A_step": [],
        "overlap_AB": overlap_AB,
        "model_info": {
            "dim": dim, "num_layers": num_layers, "heads": heads,
            "vocab_size": vocab_size, "seq_len": seq_len, "batch_size": batch_size,
            "n_params_M": round(n_params / 1e6, 2),
            "task_type": "vocab_split",
            "optimizer": "DeepMomentum(nl_l2_precond)",
            "outer_optimizer": "AdamW",
            "block_variant": "hope_hybrid",
        },
    }
    for ln in LEVEL_NAMES:
        records[f"mom_proj_{ln}"] = []
        records[f"mom_proj_norm_{ln}"] = []
        records[f"param_drift_{ln}"] = []
        records[f"theory_decay_{ln}"] = []

    for step in range(task_b_steps):
        loss_b = train_step(model, batches_B[step], outer_optimizer)
        records["step"].append(step)
        records["loss_B"].append(loss_b)

        # Periodic Task A forgetting measurement
        if step % eval_every == 0:
            with torch.no_grad():
                sample = batches_A[step % len(batches_A)]
                inp, tgt = sample[:, :-1], sample[:, 1:]
                logits = model(inp)
                loss_A_now = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
                ).item()
            records["loss_A_forgetting"].append(loss_A_now)
            records["loss_A_step"].append(step)

        for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
            if ln not in subspaces_A:
                for k in ["mom_proj", "mom_proj_norm", "param_drift", "theory_decay"]:
                    records[f"{k}_{ln}"].append(None)
                continue
            U_A = subspaces_A[ln][0]
            m = get_level_momentum_flat(model, ln)
            p = proj_norm_ratio(U_A, m.cpu()) if m is not None else float("nan")
            ip = init_proj.get(ln, float("nan"))
            p_n = p / ip if (not math.isnan(ip) and ip > 1e-8) else float("nan")

            theta_now = get_level_params_flat(model, ln)
            drift = param_drift(theta_now.cpu(), theta_A[ln]) \
                    if (theta_now is not None and ln in theta_A) else float("nan")
            k_eff = step // C + 1
            theory = beta ** k_eff

            def _s(v): return None if math.isnan(v) else v
            records[f"mom_proj_{ln}"].append(_s(p))
            records[f"mom_proj_norm_{ln}"].append(_s(p_n))
            records[f"param_drift_{ln}"].append(_s(drift))
            records[f"theory_decay_{ln}"].append(theory)

        if step % 50 == 0:
            print(f"  B step {step:5d}  loss_B={loss_b:.4f}  loss_A={records['loss_A_forgetting'][-1]:.4f}" if records["loss_A_forgetting"] else
                  f"  B step {step:5d}  loss_B={loss_b:.4f}", end="")
            for ln in LEVEL_NAMES:
                v = records[f"mom_proj_norm_{ln}"][-1]
                if v is not None:
                    print(f"  {ln[4:]}={v:.3f}", end="")
            print()

    print("\n=== Probe complete ===")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",          type=int,   default=512)
    parser.add_argument("--num_layers",   type=int,   default=12)
    parser.add_argument("--heads",        type=int,   default=8)
    parser.add_argument("--vocab_size",   type=int,   default=512)
    parser.add_argument("--seq_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--task_a_steps", type=int,   default=2000)
    parser.add_argument("--task_b_steps", type=int,   default=500)
    parser.add_argument("--lr_outer",     type=float, default=2e-4)
    parser.add_argument("--grad_rank",    type=int,   default=16)
    parser.add_argument("--eval_every",   type=int,   default=100)
    parser.add_argument("--output_dir",   default="./results_vocabsplit")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = run_probe(
        dim          = args.dim,
        num_layers   = args.num_layers,
        heads        = args.heads,
        vocab_size   = args.vocab_size,
        seq_len      = args.seq_len,
        batch_size   = args.batch_size,
        task_a_steps = args.task_a_steps,
        task_b_steps = args.task_b_steps,
        lr_outer     = args.lr_outer,
        grad_rank    = args.grad_rank,
        eval_every   = args.eval_every,
        device       = device,
    )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.json"
    with open(rec_path, "w") as f:
        json.dump(records, f, default=lambda x: None if x is None else float(x))
    print(f"Saved records -> {rec_path}")
    print("=== Done. ===")


if __name__ == "__main__":
    main()
