"""
probe_claim2_ogp_cam.py  (v2 — fixed per-block HGC wiring)
============================================================
Claim 2 Probe: OGP + CAM truly protect old-knowledge subspace.

Four conditions run in sequence (one GPU at a time, or in parallel):
  1. Vanilla  — standard DeepMomentum, no HGC
  2. OGP only — HGC enabled, CAM disabled (gamma_scale=0)
  3. CAM only — HGC enabled, OGP disabled (alpha forced to 0 after build)
  4. Full HGC — HGC enabled, both OGP + CAM

Architecture:
  HOPEModel (hope_hybrid), each HOPEBlock owns its own LevelGradientMemory
  when hgc_enabled=True.  Consolidation is called per-block at task boundary.

Measured at each Task B step (per CMS level):
  E1: energy_retention    = ||U_A^T m_t||² / ||U_A^T m_0||²
  E2: principal_angle     = arccos(|u1(U_A)·m_t/||m_t|||)  (degrees)
  E3: subspace_overlap    = ||U_A^T U_t||_F / sqrt(r)
  E4: param_drift         = ||θ_t - θ_A|| / ||θ_A||
  E5: loss_B (plasticity proxy)

Usage:
  python probe_claim2_ogp_cam.py --condition vanilla  --device cuda:0 --output_dir ~/experiments/results_claim2
  python probe_claim2_ogp_cam.py --condition ogp_only --device cuda:2 --output_dir ~/experiments/results_claim2
  python probe_claim2_ogp_cam.py --condition cam_only --device cuda:3 --output_dir ~/experiments/results_claim2
  python probe_claim2_ogp_cam.py --condition full_hgc --device cuda:5 --output_dir ~/experiments/results_claim2
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

# ── Constants ─────────────────────────────────────────────────────────────────
LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = [1, 4, 32, 128]
BETA          = 0.9

CONDITIONS = {
    "vanilla":  dict(use_hgc=False, alpha_scale=1.0, gamma_scale=0.1),
    "ogp_only": dict(use_hgc=True,  alpha_scale=1.0, gamma_scale=0.0),
    "cam_only": dict(use_hgc=True,  alpha_scale=0.0, gamma_scale=0.1),
    "full_hgc": dict(use_hgc=True,  alpha_scale=1.0, gamma_scale=0.1),
}


# ── Model builder ─────────────────────────────────────────────────────────────

def build_hope(device, dim, num_layers, heads, vocab_size, seq_len, condition: str):
    """Build HOPEModel with per-block LevelGradientMemory for HGC conditions."""
    cond_cfg = CONDITIONS[condition]
    use_hgc = cond_cfg["use_hgc"]

    cfg = ModelConfig(
        vocab_size    = vocab_size,
        dim           = dim,
        num_layers    = num_layers,
        heads         = heads,
        titan_level   = LevelSpec("titan", update_period=1),
        cms_levels    = [
            LevelSpec("cms_fast",  update_period=1),
            LevelSpec("cms_mid",   update_period=4),
            LevelSpec("cms_slow",  update_period=32),
            LevelSpec("cms_ultra", update_period=128),
        ],
        block_variant = "hope_hybrid",
        # HGC settings — each HOPEBlock creates its own LevelGradientMemory
        hgc_enabled              = use_hgc,
        hgc_r_base               = 4,
        hgc_gamma_scale          = cond_cfg["gamma_scale"],
        hgc_distillation_strength= 0.0,   # disable distillation (isolate OGP+CAM)
    )
    model = HOPEModel(cfg).to(device)

    if use_hgc:
        # Apply alpha / gamma overrides per condition
        alpha_scale = cond_cfg["alpha_scale"]
        gamma_scale = cond_cfg["gamma_scale"]
        for block in model.blocks:
            gm = getattr(block, "grad_memory", None)
            if gm is None:
                continue
            # Disable distillation entirely to isolate OGP/CAM effects
            gm.distillation_strength = 0.0
            gm.distillation_records = []
            for ln in gm.alpha:
                gm.alpha[ln] = gm.alpha[ln] * alpha_scale
                gm.gamma[ln] = gm.gamma[ln] * (gamma_scale / 0.1 if gamma_scale > 0 else 0.0)
        print(f"  [HGC] condition={condition}, alpha_scale={alpha_scale}, gamma_scale={gamma_scale}")

    return model


# ── Data ──────────────────────────────────────────────────────────────────────

def make_vocab_split_batches(vocab_size, split, seq_len, batch_size, n_steps, device):
    lo = 0 if split == "A" else vocab_size // 2
    hi = vocab_size // 2 if split == "A" else vocab_size
    total = n_steps * batch_size * (seq_len + 1)
    tokens = torch.randint(lo, hi, (total,), device=device)
    batches = []
    idx = 0
    for _ in range(n_steps):
        chunk = tokens[idx: idx + batch_size * (seq_len + 1)].view(batch_size, seq_len + 1)
        batches.append(chunk)
        idx += batch_size * (seq_len + 1)
    return batches


# ── Train / probe utilities ───────────────────────────────────────────────────

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


def get_level_momentum_flat(model, level_name):
    """Concatenate momentum buffers across all blocks for a given level."""
    parts = []
    for block in model.blocks:
        lm = getattr(block, "level_manager", None)
        if lm is None:
            continue
        opt = lm.optimizers.get(level_name)
        if opt is None:
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
    """SVD on CPU with NaN guard and QR fallback."""
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
        try:
            Q, R = torch.linalg.qr(G)
            U, S, _ = torch.linalg.svd(R, full_matrices=False)
            U = Q @ U
        except Exception:
            return None, None
    return U[:, :q].cpu(), S[:q].cpu()


def subspace_energy_retention(U_A, m_t, m_0_proj_sq):
    """||U_A^T m_t||² / ||U_A^T m_0||²"""
    if m_0_proj_sq < 1e-24:
        return float("nan")
    proj_now_sq = float((U_A.T @ m_t.float()).norm().item() ** 2)
    return proj_now_sq / m_0_proj_sq


def principal_angle(U_A, m_t):
    """Minimum principal angle between U_A subspace and momentum direction (degrees)."""
    m_unit = m_t.float()
    n = m_unit.norm()
    if n < 1e-12:
        return float("nan")
    m_unit = m_unit / n
    cosine = float((U_A.T @ m_unit).norm().item())
    cosine = max(0.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def current_subspace_overlap(U_A, m_snapshots_B, rank):
    """||U_A^T U_B||_F / sqrt(r) using recent Task B momentum snapshots."""
    recent = m_snapshots_B[-min(16, len(m_snapshots_B)):]
    if len(recent) < 2:
        return float("nan")
    G_B = torch.stack(recent, dim=1).float()
    G_B = torch.nan_to_num(G_B)
    try:
        U_B, _, _ = torch.linalg.svd(G_B.cpu(), full_matrices=False)
    except Exception:
        return float("nan")
    r = min(U_A.shape[1], U_B.shape[1], rank)
    return float((U_A[:, :r].T @ U_B[:, :r]).norm() / math.sqrt(r))


def param_drift(now, ref):
    ref_n = ref.float().norm()
    if ref_n < 1e-12:
        return 0.0
    return float((now.float() - ref.float()).norm() / ref_n)


# ── Main probe ────────────────────────────────────────────────────────────────

def run_probe(condition, dim, num_layers, heads, vocab_size, seq_len,
              batch_size, task_a_steps, task_b_steps, lr_outer, grad_rank,
              eval_every, device):

    print(f"\n{'='*60}")
    print(f"  Claim 2 Probe — Condition: {condition.upper()}")
    print(f"  dim={dim}  layers={num_layers}  heads={heads}")
    print(f"  task_a={task_a_steps}  task_b={task_b_steps}  device={device}")
    print(f"{'='*60}")

    MAX_GRAD_BUF = 32 if dim >= 512 else 64

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n=== Building HOPEModel ===")
    model = build_hope(device, dim, num_layers, heads, vocab_size, seq_len, condition)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M")

    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)

    # ── Task A ────────────────────────────────────────────────────────────────
    print("\n=== Phase 1: Train on Task A ===")
    batches_A = make_vocab_split_batches(vocab_size, "A", seq_len, batch_size, task_a_steps, device)
    # We collect whole-model momentum snapshots ONLY for measuring U_A subspace externally
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
            vram = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
            print(f"  A step {step:5d}  loss={loss:.4f}  VRAM={vram:.0f}MB")

    print(f"  Task A final loss={loss:.4f}")

    # ── Consolidate per-block grad_memory ─────────────────────────────────────
    # For HGC conditions: each block's grad_memory has been accumulating
    # per-block gradients via DeepMomentumHGC.forward() throughout Task A.
    # Calling consolidate_all() extracts the SVD basis per level.
    # This happens correctly because the grad shapes match (per-block, not whole-model).
    use_hgc = CONDITIONS[condition]["use_hgc"]
    if use_hgc:
        print("  [HGC] Consolidating per-block gradient memories at task boundary...")
        n_consolidated = 0
        for block in model.blocks:
            gm = getattr(block, "grad_memory", None)
            if gm is None:
                continue
            gm.consolidate_all()
            n_consolidated += 1
        print(f"  [HGC] Consolidated {n_consolidated} blocks.")
        # Print stats from first block as sample
        for block in model.blocks:
            gm = getattr(block, "grad_memory", None)
            if gm is None:
                continue
            stats = gm.signature_stats()
            for ln, s in stats.items():
                if s["has_basis"]:
                    print(f"    [block0] {ln}: rank={s['rank']}, alpha={gm.alpha.get(ln,0):.3f}, gamma={gm.gamma.get(ln,0):.3f}, top-SV={s.get('top_singular_value',0):.4f}")
            break  # only show first block

    # ── Extract measurement subspace U_A (whole-model, for E1-E3 measurement) ──
    print("\n=== Extracting measurement subspace U_A (whole-model) ===")
    subspaces_A: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        buf = grad_buffers[ln]
        if len(buf) < 2:
            continue
        U, S = compute_svd_basis_cpu(buf, grad_rank)
        if U is None:
            continue
        subspaces_A[ln] = (U, S)
        print(f"  {ln}: U={U.shape}, top-SV={S[0]:.4f}")

    # Snapshot params & init momentum projection norms
    theta_A: Dict[str, torch.Tensor] = {}
    init_mom_proj_sq: Dict[str, float] = {}
    for ln in LEVEL_NAMES:
        pf = get_level_params_flat(model, ln)
        if pf is not None:
            theta_A[ln] = pf.cpu()
        if ln in subspaces_A:
            U_A = subspaces_A[ln][0]
            m = get_level_momentum_flat(model, ln)
            if m is not None:
                v = float((U_A.T @ m.cpu().float()).norm().item() ** 2)
                init_mom_proj_sq[ln] = v

    # Task A loss baseline
    with torch.no_grad():
        sample = batches_A[0]
        logits = model(sample[:, :-1])
        loss_A_baseline = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), sample[:, 1:].reshape(-1)
        ).item()
    print(f"\n  Task A loss baseline: {loss_A_baseline:.4f}")

    # ── Task B ────────────────────────────────────────────────────────────────
    print("\n=== Phase 2: Train on Task B — measuring protection ===")
    batches_B = make_vocab_split_batches(vocab_size, "B", seq_len, batch_size, task_b_steps, device)

    records: Dict = {
        "condition": condition,
        "step": [],
        "loss_B": [],
        "loss_A_forgetting": [],
        "loss_A_step": [],
        "model_info": {
            "dim": dim, "num_layers": num_layers, "heads": heads,
            "n_params_M": round(n_params / 1e6, 2),
            "vocab_size": vocab_size, "seq_len": seq_len,
            "task_a_steps": task_a_steps, "task_b_steps": task_b_steps,
            "use_hgc": use_hgc,
        },
    }
    for ln in LEVEL_NAMES:
        records[f"energy_retention_{ln}"] = []
        records[f"principal_angle_{ln}"]  = []
        records[f"subspace_overlap_{ln}"] = []
        records[f"param_drift_{ln}"]      = []
        records[f"theory_decay_{ln}"]     = []

    b_momentum_buf: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}

    for step in range(task_b_steps):
        loss_b = train_step(model, batches_B[step], outer_optimizer)
        records["step"].append(step)
        records["loss_B"].append(loss_b)

        if step % eval_every == 0:
            with torch.no_grad():
                sample = batches_A[step % len(batches_A)]
                logits = model(sample[:, :-1])
                loss_A_now = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), sample[:, 1:].reshape(-1)
                ).item()
            records["loss_A_forgetting"].append(loss_A_now)
            records["loss_A_step"].append(step)

        for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
            m = get_level_momentum_flat(model, ln)
            k_eff = step // C + 1
            records[f"theory_decay_{ln}"].append(BETA ** k_eff)

            if ln not in subspaces_A or m is None:
                for k in ["energy_retention", "principal_angle", "subspace_overlap", "param_drift"]:
                    records[f"{k}_{ln}"].append(None)
                continue

            U_A = subspaces_A[ln][0]
            m_cpu = m.cpu().float()

            er = subspace_energy_retention(U_A, m_cpu, init_mom_proj_sq.get(ln, 1.0))
            pa = principal_angle(U_A, m_cpu)
            b_momentum_buf[ln].append(m_cpu)
            if len(b_momentum_buf[ln]) > 32:
                b_momentum_buf[ln].pop(0)
            so = current_subspace_overlap(U_A, b_momentum_buf[ln], grad_rank)
            theta_now = get_level_params_flat(model, ln)
            drift = param_drift(theta_now.cpu(), theta_A[ln]) \
                    if (theta_now is not None and ln in theta_A) else float("nan")

            def _s(v): return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
            records[f"energy_retention_{ln}"].append(_s(er))
            records[f"principal_angle_{ln}"].append(_s(pa))
            records[f"subspace_overlap_{ln}"].append(_s(so))
            records[f"param_drift_{ln}"].append(_s(drift))

        if step % 50 == 0:
            er_fast = records["energy_retention_cms_fast"][-1]
            pa_fast = records["principal_angle_cms_fast"][-1]
            fa = records["loss_A_forgetting"][-1] if records["loss_A_forgetting"] else None
            msg = f"  B step {step:5d}  loss_B={loss_b:.4f}"
            if fa is not None:
                msg += f"  loss_A={fa:.4f}"
            if er_fast is not None:
                msg += f"  ER(fast)={er_fast:.3f}"
            if pa_fast is not None:
                msg += f"  PA(fast)={pa_fast:.1f}°"
            print(msg)

    print("\n=== Probe complete ===")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",    default="vanilla",
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--dim",          type=int,   default=256)
    parser.add_argument("--num_layers",   type=int,   default=8)
    parser.add_argument("--heads",        type=int,   default=4)
    parser.add_argument("--vocab_size",   type=int,   default=512)
    parser.add_argument("--seq_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--task_a_steps", type=int,   default=2000)
    parser.add_argument("--task_b_steps", type=int,   default=500)
    parser.add_argument("--lr_outer",     type=float, default=2e-4)
    parser.add_argument("--grad_rank",    type=int,   default=16)
    parser.add_argument("--eval_every",   type=int,   default=100)
    parser.add_argument("--output_dir",   default="./results_claim2")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Condition: {args.condition}")

    records = run_probe(
        condition    = args.condition,
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
    rec_path = out_dir / f"records_{args.condition}.json"
    with open(rec_path, "w") as f:
        json.dump(records, f, default=lambda x: None if x is None else float(x))
    print(f"Saved -> {rec_path}")
    print("=== Done. ===")


if __name__ == "__main__":
    main()
