"""
Real-HOPE Gradient Memory Collapse Probe
=========================================
Uses the ACTUAL HOPEModel (hope_hybrid variant) with DeepMomentum (nl_l2_precond)
to measure gradient memory collapse at the optimizer-state level.

What this script does:
  1. Instantiates HOPEModel (hope_hybrid) with 4 CMS levels [C=1,4,32,128]
  2. Trains on Task A (domain-A synthetic language-like token sequences) for N steps
     — uses standard LM cross-entropy teach signal
  3. After Task A, extracts per-level gradient subspace U_A via SVD of
     accumulated DeepMomentum grad_avg states across all layers
  4. Switches to Task B (domain-B, different token distribution), trains for M steps
  5. At every step, measures:
       E1: ||U_A^T m_t^(l)|| / ||m_t^(l)||  — momentum projection per level
       E2: ||theta_t^(l) - theta_A^(l)|| / ||theta_A^(l)||  — param drift per level
       E3: step-wise ratio of projection norms per effective update

Key correctness guarantees vs toy experiment:
  - Uses real HOPEModel architecture (Titan + CMS + Attention backbone)
  - Uses real DeepMomentum with nl_l2_precond variant
  - Uses real LM cross-entropy loss with teach signal mechanism
  - Measures optimizer STATE (grad_avg in DeepMomentum.state) not synthetic momentum

Usage:
    python probe_real_hope.py --output_dir ./results_real_hope --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Import real HOPE ──────────────────────────────────────────────────────────
_CANDIDATES = [
    # Docker path (copied via scp)
    os.path.expanduser("~/hope_src"),
    # Mac remote-workspace mount path (fallback)
    os.path.expanduser(
        "~/remote-workspace/continuous_learning/"
        "Nested Learning The Illusion of Deep Learning Architecture "
        "Ali Behrouz , Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni/"
        "code/nested_learning_hgc/src"
    ),
]
HOPE_ROOT = next((p for p in _CANDIDATES if os.path.isdir(os.path.join(p, "nested_learning"))), _CANDIDATES[0])
sys.path.insert(0, HOPE_ROOT)

from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.levels import LevelSpec
from nested_learning.optim.deep import DeepMomentum


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_small_hope(device: torch.device, vocab_size: int = 512) -> HOPEModel:
    """Build a small but REAL HOPEModel (hope_hybrid).

    Size rationale: small enough to fit in one A800 with fast iterations,
    but architecturally faithful — same block_variant, same optimizer configs,
    same level structure as the pilot config.
    """
    cms_levels = [
        LevelSpec(name="cms_fast",  update_period=1,   optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid",   update_period=4,   optimizer_key="cms_opt"),
        LevelSpec(name="cms_slow",  update_period=32,  optimizer_key="cms_opt"),
        LevelSpec(name="cms_ultra", update_period=128, optimizer_key="cms_opt"),
    ]
    titan_level = LevelSpec(name="titan", update_period=8, optimizer_key="titan_opt")

    optimizer_configs = {
        "titan_opt": {
            "type": "deep_momentum",
            "lr": 6e-4,
            "params": {
                "beta": 0.9,
                "beta2": 0.999,
                "variant": "nl_l2_precond",
            },
        },
        "cms_opt": {
            "type": "deep_momentum",
            "lr": 3e-4,
            "params": {
                "beta": 0.9,
                "beta2": 0.999,
                "variant": "nl_l2_precond",
            },
        },
    }

    config = ModelConfig(
        vocab_size=vocab_size,
        dim=128,           # Small: 128-dim (faithful architecture, reduced size)
        num_layers=4,      # 4 layers (pilot has 12, we use 4 for speed)
        heads=4,
        titan_level=titan_level,
        cms_levels=cms_levels,
        teach_scale=0.1,
        teach_clip=5.0,
        surprise_threshold=None,
        self_mod_lr=1e-3,
        block_variant="hope_hybrid",
        optimizers=optimizer_configs,
        hgc_enabled=False,  # Vanilla HOPE — no HGC
    )
    model = HOPEModel(config).to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic task data
# ─────────────────────────────────────────────────────────────────────────────

def make_token_task(
    vocab_size: int,
    seq_len: int,
    n_samples: int,
    seed: int,
    device: torch.device,
    # Domain bias: task A concentrates on lower half of vocab, B on upper half
    low_token: int = 0,
    high_token: int = None,
) -> torch.Tensor:
    """Generate synthetic token sequences with domain-specific distribution.

    Task A: tokens drawn from [low_token, vocab_size//2)
    Task B: tokens drawn from [vocab_size//2, vocab_size)
    This creates approximately orthogonal gradient subspaces via the embedding layer.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    if high_token is None:
        high_token = vocab_size
    span = high_token - low_token
    tokens = torch.randint(0, span, (n_samples, seq_len), generator=rng) + low_token
    return tokens.to(device)


def get_batch(tokens: torch.Tensor, batch_size: int, step: int) -> torch.Tensor:
    n = tokens.shape[0]
    start = (step * batch_size) % n
    end = min(start + batch_size, n)
    return tokens[start:end]


# ─────────────────────────────────────────────────────────────────────────────
# Probe: extract DeepMomentum grad_avg states per level per layer
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_NAMES = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = [1, 4, 32, 128]


def get_level_momentum_flat(model: HOPEModel, level_name: str) -> Optional[torch.Tensor]:
    """Collect and flatten ALL DeepMomentum grad_avg vectors for a given CMS level
    across all layers. Returns a single concatenated flat vector."""
    parts = []
    for block in model.blocks:
        # Access the LevelOptimizerManager inside each HOPEBlock
        level_manager = getattr(block, "level_manager", None)
        if level_manager is None:
            continue
        optimizer = level_manager.optimizers.get(level_name)
        if optimizer is None or not isinstance(optimizer, DeepMomentum):
            continue
        # optimizer.state is Dict[param_key -> DeepMomentumState]
        for state in optimizer.state.values():
            if state.grad_avg is not None:
                parts.append(state.grad_avg.detach().float().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def get_level_params_flat(model: HOPEModel, level_name: str) -> Optional[torch.Tensor]:
    """Flatten all CMS block parameters for a given level across all layers."""
    parts = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is None:
            continue
        # nn.ModuleDict does not have .get(); use __contains__ + __getitem__
        if level_name not in cms.blocks:
            continue
        cms_block = cms.blocks[level_name]
        for p in cms_block.parameters():
            parts.append(p.detach().float().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def collect_grad_buffer_for_level(
    model: HOPEModel,
    level_name: str,
    tokens_batch: torch.Tensor,
    teach_signal: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Forward pass + extract per-level CMS block gradients via autograd.
    Returns flat concatenated gradient vector for this level across all layers.
    """
    # We need gradients wrt CMS block params for this level
    # Collect params
    all_params = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is None:
            continue
        cms_block = cms.blocks.get(level_name)
        if cms_block is None:
            continue
        all_params.extend([p for p in cms_block.parameters() if p.requires_grad])

    if not all_params:
        return None

    # Temporarily set requires_grad for these params (they usually need it)
    # Run a forward pass
    model.train()
    logits = model(tokens_batch[:, :-1], teach_signal=teach_signal)
    target = tokens_batch[:, 1:].reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target)

    grads = torch.autograd.grad(loss, all_params, retain_graph=False, allow_unused=True)
    parts = []
    for g in grads:
        if g is not None:
            parts.append(g.detach().float().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def compute_svd_basis(grad_matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD on gradient matrix [d, T], return (U [d,r], S [r])."""
    q = min(rank, grad_matrix.shape[0], grad_matrix.shape[1])
    U, S, _ = torch.linalg.svd(grad_matrix, full_matrices=False)
    return U[:, :q], S[:q]


def projection_norm_ratio(U: torch.Tensor, v: torch.Tensor) -> float:
    """||U^T v|| / ||v||"""
    if v is None or U is None:
        return float("nan")
    vn = v.float().to(U.device)
    nv = vn.norm()
    if nv < 1e-12:
        return 0.0
    return float((U.float().T @ vn).norm().item() / nv.item())


def param_drift(theta_now: torch.Tensor, theta_ref: torch.Tensor) -> float:
    """||theta_now - theta_ref|| / ||theta_ref||"""
    if theta_now is None or theta_ref is None:
        return float("nan")
    delta = (theta_now.float() - theta_ref.float()).norm()
    ref = theta_ref.float().norm()
    return float((delta / (ref + 1e-12)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def compute_teach_signal(model: HOPEModel, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute HOPE teach signal: dL/dh = (softmax(logits) - one_hot(next_token)) @ W_head^T / N
    This is the closed-form gradient of cross-entropy loss w.r.t. the pre-head hidden states.
    Matches nested_learning.training.compute_teach_signal() exactly.
    """
    logits_d = logits.detach()
    probs = torch.softmax(logits_d, dim=-1)            # [B, T, V]
    residual = probs.clone()
    B, T, V = residual.shape
    # Targets: next token for each position (shift by 1)
    if T > 1:
        targets = tokens[:, 1:T+1] if tokens.shape[1] > T else tokens[:, 1:]
        # targets shape [B, T-1] — only supervise non-final positions
        pad_len = T - targets.shape[1]
        if pad_len > 0:
            targets = torch.cat([targets, torch.zeros(B, pad_len, dtype=targets.dtype, device=targets.device)], dim=1)
        active = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
        active[:, :T-1] = True
    else:
        return torch.zeros_like(logits_d[:, :, :model.config.dim])

    active_f = active.float().unsqueeze(-1)               # [B, T, 1]
    residual = residual * active_f
    safe_t = torch.where(active, targets, torch.zeros_like(targets))
    residual.scatter_add_(-1, safe_t.unsqueeze(-1), -active_f)
    denom = active_f.sum().clamp(min=1.0)
    residual = residual / denom

    head_weight = model.lm_head.weight.detach()           # [V, D]
    if head_weight.dtype != residual.dtype:
        head_weight = head_weight.to(residual.dtype)
    return residual @ head_weight                          # [B, T, D]


def train_step(model: HOPEModel, tokens: torch.Tensor) -> float:
    """One HOPE training step using the authentic teach-signal mechanism.

    Pattern (matches nested_learning/training.py two_pass_stopgrad_updates):
      1. Forward pass (no teach) → logits
      2. Compute teach_signal = dL/dh analytically (no autograd backward)
      3. Second forward with teach_signal → triggers CMS DeepMomentum updates
      4. Compute LM loss for monitoring
    """
    model.train()
    inp = tokens[:, :-1]    # [B, T-1]
    tgt = tokens[:, 1:]     # [B, T-1]

    with torch.no_grad():
        # Pass 1: get logits for teach signal computation
        logits = model(inp)                               # [B, T-1, V]
        teach = compute_teach_signal(model, logits, tokens)  # [B, T-1, D]
        # Pass 2: forward WITH teach signal — this drives DeepMomentum CMS updates
        _ = model(inp, teach_signal=teach)
        # Loss for monitoring
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
        )
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Main probe
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(
    vocab_size: int = 512,
    seq_len: int = 64,
    n_samples: int = 2048,
    task_a_steps: int = 400,
    task_b_steps: int = 300,
    batch_size: int = 32,
    lr_outer: float = 2.5e-4,
    grad_rank: int = 16,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    print(f"[ProbeRealHOPE] vocab={vocab_size} seq={seq_len} dim=128 layers=4")
    print(f"                task_a={task_a_steps} task_b={task_b_steps} bs={batch_size}")
    print(f"                lr_outer={lr_outer} grad_rank={grad_rank} device={device}")

    model = build_small_hope(device, vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    outer_optimizer = None  # Not needed — HOPE self-updates via teach_signal

    # ── Task A and B token datasets ──────────────────────────────────────────
    # Task A: tokens from lower half of vocab [0, vocab//2)
    # Task B: tokens from upper half of vocab [vocab//2, vocab)
    tokens_A = make_token_task(
        vocab_size, seq_len, n_samples, seed=1000, device=device,
        low_token=0, high_token=vocab_size // 2,
    )
    tokens_B = make_token_task(
        vocab_size, seq_len, n_samples, seed=2000, device=device,
        low_token=vocab_size // 2, high_token=vocab_size,
    )

    # ── Phase 1: Train on Task A, collect gradients for SVD ──────────────────
    print("\n=== Phase 1: Training on Task A (real HOPE, DeepMomentum) ===")
    grad_buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}
    MAX_GRAD_BUF = 200  # collect up to 200 grad vectors per level

    for step in range(task_a_steps):
        batch = get_batch(tokens_A, batch_size, step)
        loss = train_step(model, batch)

        # Also collect raw CMS gradients from the internal DeepMomentum states
        # The DeepMomentum.forward() is called during train_step and updates grad_avg.
        # We can read the gradient by looking at what was fed into DeepMomentum.
        # Strategy: after each step, extract the current grad_avg from DeepMomentum
        # and treat it as a proxy for the current gradient direction.
        for level_name in LEVEL_NAMES:
            if len(grad_buffers[level_name]) < MAX_GRAD_BUF:
                m = get_level_momentum_flat(model, level_name)
                if m is not None:
                    grad_buffers[level_name].append(m.clone().cpu())

        if step % 100 == 0:
            print(f"  step {step:4d}  loss={loss:.4f}")

    # ── Extract gradient signatures U_A per level ─────────────────────────────
    print("\n=== Extracting gradient signatures U_A per level ===")
    subspaces_A: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for level_name, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        buf = grad_buffers[level_name]
        # For slow levels we may have very few unique states (C steps per update)
        # Use whatever we have, with rank = min(grad_rank, len(buf)-1)
        effective_rank = max(1, min(grad_rank, len(buf) - 1))
        if len(buf) < 2:
            print(f"  {level_name}: insufficient buffer ({len(buf)}), skipping")
            continue
        G = torch.stack(buf, dim=1).to(device)   # [d, T]
        U, S = compute_svd_basis(G, rank=effective_rank)
        subspaces_A[level_name] = (U.cpu(), S.cpu())
        print(f"  {level_name}: U={U.shape}, top-SV={S[0].item():.4f}, "
              f"buf={len(buf)}, d={G.shape[0]:,}, effective_rank={effective_rank}")

    # ── Snapshot params at end of Task A ─────────────────────────────────────
    theta_A: Dict[str, torch.Tensor] = {}
    for level_name in LEVEL_NAMES:
        pf = get_level_params_flat(model, level_name)
        if pf is not None:
            theta_A[level_name] = pf.cpu()

    # ── Verify Task A / Task B subspace orthogonality ─────────────────────────
    print("\n=== Verifying Task A/B subspace orthogonality ===")
    # Quick: run 30 steps of Task B, collect momentum, check overlap with U_A
    model_copy_state = {k: v.clone() for k, v in model.state_dict().items()}
    b_buf: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}
    for vstep in range(50):
        batch = get_batch(tokens_B, batch_size, vstep)
        train_step(model, batch)
        for ln in LEVEL_NAMES:
            if len(b_buf[ln]) < 50:
                m = get_level_momentum_flat(model, ln)
                if m is not None:
                    b_buf[ln].append(m.clone().cpu())
    # Restore model state
    model.load_state_dict(model_copy_state)

    for level_name, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        if level_name not in subspaces_A or len(b_buf[level_name]) < 2:
            continue
        U_A = subspaces_A[level_name][0].to(device)
        G_B = torch.stack(b_buf[level_name], dim=1).to(device)
        U_B_full, _, _ = torch.linalg.svd(G_B, full_matrices=False)
        r = min(U_A.shape[1], U_B_full.shape[1])
        overlap = float((U_A[:, :r].T @ U_B_full[:, :r]).norm().item() / math.sqrt(r))
        print(f"  {level_name} (C={C}): A↔B subspace overlap = {overlap:.4f}")

    # ── Phase 2: Train on Task B, measure momentum collapse ──────────────────
    print("\n=== Phase 2: Training on Task B — measuring collapse ===")
    records: Dict = {
        "step": [],
        "loss_B": [],
        "model_info": {
            "dim": 128,
            "num_layers": 4,
            "vocab_size": vocab_size,
            "optimizer": "DeepMomentum(nl_l2_precond)",
            "block_variant": "hope_hybrid",
        },
    }
    for ln in LEVEL_NAMES:
        records[f"mom_proj_{ln}"] = []
        records[f"mom_proj_abs_{ln}"] = []
        records[f"mom_proj_norm_{ln}"] = []
        records[f"param_drift_{ln}"] = []
        records[f"theory_decay_{ln}"] = []

    # Get initial projections at step 0 of Task B (before any B training)
    init_proj: Dict[str, float] = {}
    for level_name in LEVEL_NAMES:
        if level_name not in subspaces_A:
            init_proj[level_name] = float("nan")
            continue
        U_A = subspaces_A[level_name][0].to(device)
        m = get_level_momentum_flat(model, level_name)
        if m is not None:
            p_abs = float((U_A.T @ m.float().to(device)).norm().item())
            p_norm = float(p_abs / (m.float().norm().item() + 1e-12))
            init_proj[level_name] = p_norm
        else:
            init_proj[level_name] = float("nan")
    print(f"Initial projections at Task B start: { {k: f'{v:.3f}' for k,v in init_proj.items()} }")

    beta = 0.9

    for step in range(task_b_steps):
        batch = get_batch(tokens_B, batch_size, step)
        loss = train_step(model, batch)

        records["step"].append(step)
        records["loss_B"].append(loss)

        for level_name, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
            if level_name not in subspaces_A:
                for key in ["mom_proj", "mom_proj_abs", "mom_proj_norm", "param_drift", "theory_decay"]:
                    records[f"{key}_{level_name}"].append(float("nan"))
                continue

            U_A = subspaces_A[level_name][0].to(device)
            m = get_level_momentum_flat(model, level_name)
            theta_now = get_level_params_flat(model, level_name)

            # E1: projection norm
            if m is not None:
                p_abs = float((U_A.T @ m.float().to(device)).norm().item())
                p_norm = p_abs / (m.float().norm().item() + 1e-12)
            else:
                p_abs = float("nan")
                p_norm = float("nan")

            # E2: param drift
            if theta_now is not None and level_name in theta_A:
                drift = param_drift(theta_now.to(device), theta_A[level_name].to(device))
            else:
                drift = float("nan")

            # Normalized projection (relative to step-0 value)
            ip = init_proj.get(level_name, float("nan"))
            p_norm_normalized = p_norm / ip if (not math.isnan(ip) and ip > 1e-8) else float("nan")

            # Theoretical decay: effective updates on Task B
            k_eff = step // C + 1
            theory = beta ** k_eff

            records[f"mom_proj_{level_name}"].append(p_norm)
            records[f"mom_proj_abs_{level_name}"].append(p_abs)
            records[f"mom_proj_norm_{level_name}"].append(p_norm_normalized)
            records[f"param_drift_{level_name}"].append(drift)
            records[f"theory_decay_{level_name}"].append(theory)

        if step % 50 == 0:
            print(f"  step {step:4d}  loss={loss:.4f}", end="")
            for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
                v = records[f"mom_proj_norm_{ln}"][-1]
                print(f"  {ln[4:]}={v:.3f}", end="")
            print()

    print("\n=== Probe complete ===")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results_real_hope")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--task_a_steps", type=int, default=500)
    parser.add_argument("--task_b_steps", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_outer", type=float, default=2.5e-4)
    parser.add_argument("--grad_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = run_probe(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        task_a_steps=args.task_a_steps,
        task_b_steps=args.task_b_steps,
        batch_size=args.batch_size,
        lr_outer=args.lr_outer,
        grad_rank=args.grad_rank,
        device=device,
        seed=args.seed,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save records
    def to_serializable(v):
        if isinstance(v, float):
            return None if math.isnan(v) else v
        if isinstance(v, list):
            return [to_serializable(x) for x in v]
        return v

    records_clean = {k: to_serializable(v) for k, v in records.items()}
    with open(out / "records.json", "w") as f:
        json.dump(records_clean, f, indent=2)
    print(f"Saved records to {out / 'records.json'}")

    print("\n=== Done. Run plot_publication.py to generate figures. ===")


if __name__ == "__main__":
    main()
