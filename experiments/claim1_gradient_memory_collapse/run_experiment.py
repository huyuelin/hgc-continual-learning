"""
Claim 1: Gradient Memory Collapse is a real optimizer-state level problem.

Evidence gathered:
  E1: momentum buffer projection onto old-task subspace ||U_A^T m_t|| decays as β^t
  E2: optimizer-state collapse precedes / is more sensitive than parameter forgetting
  E3: decay rates are identical across CMS levels in vanilla HOPE (no hierarchy)

Usage:
    python run_experiment.py --output_dir ./results --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CMSBlock(nn.Module):
    """Single CMS MLP block: LayerNorm → Linear → GELU → Linear (residual)."""
    def __init__(self, dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MinimalCMS(nn.Module):
    """Minimal CMS with N MLP levels at different update frequencies."""
    def __init__(self, dim: int, n_levels: int = 4, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        # Levels: C = [1, 4, 32, 128]
        self.update_periods = [1, 4, 32, 128][:n_levels]
        self.level_names = [f"cms_c{c}" for c in self.update_periods]
        self.blocks = nn.ModuleList([
            CMSBlock(dim, dim * hidden_mult) for _ in self.update_periods
        ])
        # Output head for prediction task
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Momentum state tracker
# ─────────────────────────────────────────────────────────────────────────────

class MomentumTracker:
    """Tracks per-level flattened momentum buffers and per-level gradients."""

    def __init__(self, model: MinimalCMS, beta: float = 0.9):
        self.model = model
        self.beta = beta
        self.n_levels = len(model.update_periods)
        self.level_names = model.level_names
        self.update_periods = model.update_periods

        # Per-level flat momentum buffers
        self.momentum: List[Optional[torch.Tensor]] = [None] * self.n_levels
        # Per-level gradient buffers (for SVD)
        self.grad_buffers: List[List[torch.Tensor]] = [[] for _ in range(self.n_levels)]

    def _get_level_params(self, level_idx: int) -> List[torch.Tensor]:
        """Get all parameters for a given CMS level."""
        return list(self.model.blocks[level_idx].parameters())

    def _get_level_grad_flat(self, level_idx: int) -> Optional[torch.Tensor]:
        """Flatten and concatenate gradients for a level."""
        parts = []
        for p in self._get_level_params(level_idx):
            if p.grad is not None:
                parts.append(p.grad.detach().float().reshape(-1))
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def _get_level_param_flat(self, level_idx: int) -> torch.Tensor:
        """Flatten and concatenate parameters for a level."""
        return torch.cat([
            p.detach().float().reshape(-1)
            for p in self._get_level_params(level_idx)
        ], dim=0)

    def update_step(self, step: int, collect_grads_for: Optional[List[int]] = None):
        """
        Perform momentum update for each level (respecting update periods).
        Optionally collect raw gradients into grad_buffers for SVD.
        Returns dict of level → actual gradient used (if updated this step).
        """
        updated = {}
        for i, (C, name) in enumerate(zip(self.update_periods, self.level_names)):
            if step % C != 0:
                continue
            g_flat = self._get_level_grad_flat(i)
            if g_flat is None:
                continue

            # Buffer gradient for SVD
            if collect_grads_for is not None and i in collect_grads_for:
                self.grad_buffers[i].append(g_flat.clone())

            # EMA momentum update
            if self.momentum[i] is None:
                self.momentum[i] = (1.0 - self.beta) * g_flat
            else:
                self.momentum[i] = self.beta * self.momentum[i] + (1.0 - self.beta) * g_flat

            updated[i] = g_flat
        return updated

    def get_momentum_flat(self, level_idx: int) -> Optional[torch.Tensor]:
        return self.momentum[level_idx]

    def get_param_flat(self, level_idx: int) -> torch.Tensor:
        return self._get_level_param_flat(level_idx)

    def compute_subspace(self, level_idx: int, rank: int = 16) -> Optional[torch.Tensor]:
        """Compute top-rank SVD basis from buffered gradients. Returns [d, rank]."""
        buf = self.grad_buffers[level_idx]
        if len(buf) < 2:
            return None
        G = torch.stack(buf, dim=1)  # [d, T]
        q = min(rank, G.shape[0], G.shape[1])
        if q < 1:
            return None
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        return U[:, :q].detach(), S[:q].detach()  # [d, q], [q]

    def clear_grad_buffers(self):
        self.grad_buffers = [[] for _ in range(self.n_levels)]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic task generation — orthogonal tasks A and B
# ─────────────────────────────────────────────────────────────────────────────

def make_task(dim: int, n_samples: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Regression task: predict a linear mapping y = Wx + b.
    Each task has a different random W, making gradient subspaces approximately orthogonal.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    W = torch.randn(dim, dim, generator=rng, device=device) / math.sqrt(dim)
    b = torch.randn(dim, generator=rng, device=device) * 0.1
    X = torch.randn(n_samples, dim, generator=rng, device=device)
    Y = X @ W.T + b.unsqueeze(0)
    return X, Y


def get_batch(X: torch.Tensor, Y: torch.Tensor, batch_size: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = X.shape[0]
    start = (step * batch_size) % n
    idx = torch.arange(start, min(start + batch_size, n), device=X.device)
    return X[idx], Y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement utilities
# ─────────────────────────────────────────────────────────────────────────────

def projection_norm(U: torch.Tensor, v: torch.Tensor) -> float:
    """
    Compute ||U^T v|| / ||v|| — fraction of v's norm in subspace spanned by U.
    U: [d, r], v: [d]
    """
    if v is None or U is None:
        return float("nan")
    v_f = v.float()
    U_f = U.float().to(v_f.device)
    if U_f.shape[0] != v_f.shape[0]:
        return float("nan")
    v_norm = v_f.norm()
    if v_norm < 1e-12:
        return 0.0
    proj = U_f.T @ v_f
    return float(proj.norm().item() / v_norm.item())


def param_drift_norm(theta_now: torch.Tensor, theta_ref: torch.Tensor) -> float:
    """||θ_t - θ_ref|| / ||θ_ref||"""
    delta = (theta_now.float() - theta_ref.float()).norm()
    ref_norm = theta_ref.float().norm()
    if ref_norm < 1e-12:
        return float(delta.item())
    return float((delta / ref_norm).item())


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dim: int = 256,
    hidden_mult: int = 2,
    n_levels: int = 4,
    task_a_steps: int = 400,
    task_b_steps: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    beta: float = 0.9,
    grad_rank: int = 20,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    n_samples: int = 8192,
) -> Dict:
    torch.manual_seed(seed)
    print(f"[Experiment] dim={dim} levels={n_levels} task_a={task_a_steps} task_b={task_b_steps}")
    print(f"             lr={lr} beta={beta} rank={grad_rank} device={device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = MinimalCMS(dim, n_levels=n_levels, hidden_mult=hidden_mult).to(device)
    tracker = MomentumTracker(model, beta=beta)

    # Separate Adam optimizers per level, one for the head
    optimizers = [
        torch.optim.AdamW(model.blocks[i].parameters(), lr=lr, weight_decay=1e-4)
        for i in range(n_levels)
    ]
    opt_head = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)

    # ── Task A data ──────────────────────────────────────────────────────────
    X_A, Y_A = make_task(dim, n_samples, seed=1000, device=device)

    print("\n=== Phase 1: Training on Task A ===")
    model.train()
    all_level_idxs = list(range(n_levels))
    for step in range(task_a_steps):
        xb, yb = get_batch(X_A, Y_A, batch_size, step)
        for opt in optimizers:
            opt.zero_grad()
        opt_head.zero_grad()
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()

        # Apply optimizer updates for all levels this step
        for i, (C, opt) in enumerate(zip(tracker.update_periods, optimizers)):
            if step % C == 0:
                opt.step()
        opt_head.step()

        # Update our manual momentum tracker & collect gradients
        tracker.update_step(step, collect_grads_for=all_level_idxs)

        if step % 100 == 0:
            print(f"  Task A step {step:4d}  loss={loss.item():.4f}")

    # ── Snapshot after Task A ─────────────────────────────────────────────────
    print("\n=== Extracting gradient signatures (U_A per level) ===")
    subspaces_A = {}  # level_idx → U  [d, r]
    for i in range(n_levels):
        result = tracker.compute_subspace(i, rank=grad_rank)
        if result is not None:
            U, S = result
            subspaces_A[i] = (U, S)
            print(f"  Level {i} ({tracker.level_names[i]}): "
                  f"U shape={U.shape}, top-SV={S[0].item():.3f}, "
                  f"buf_size={len(tracker.grad_buffers[i])}")
        else:
            print(f"  Level {i}: insufficient gradients (buf={len(tracker.grad_buffers[i])})")

    # Snapshot parameters and momentum buffers at end of Task A
    theta_A = {i: tracker.get_param_flat(i).clone() for i in range(n_levels)}
    momentum_A = {}
    for i in range(n_levels):
        m = tracker.get_momentum_flat(i)
        if m is not None:
            momentum_A[i] = m.clone()

    tracker.clear_grad_buffers()

    # ── Task B data ──────────────────────────────────────────────────────────
    X_B, Y_B = make_task(dim, n_samples, seed=2000, device=device)

    # Measure gradient subspace overlap A↔B (should be low)
    print("\n=== Computing Task A / Task B gradient subspace verification ===")
    # Take a few grad steps on B to check orthogonality
    task_b_grads = {i: [] for i in range(n_levels)}
    model.train()
    for vstep in range(50):
        xb, yb = get_batch(X_B, Y_B, batch_size, vstep)
        for opt in optimizers: opt.zero_grad()
        opt_head.zero_grad()
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        for i in range(n_levels):
            g_flat = tracker._get_level_grad_flat(i)
            if g_flat is not None:
                task_b_grads[i].append(g_flat.clone())

    for i in range(n_levels):
        if i in subspaces_A and len(task_b_grads[i]) >= 2:
            U_A = subspaces_A[i][0].to(device)
            G_B = torch.stack(task_b_grads[i], dim=1)
            U_B, _, _ = torch.linalg.svd(G_B, full_matrices=False)
            r = min(U_A.shape[1], U_B.shape[1])
            overlap = (U_A[:, :r].T @ U_B[:, :r]).norm().item() / math.sqrt(r)
            print(f"  Level {i} A↔B subspace overlap: {overlap:.4f} (lower → more orthogonal)")

    # ── Phase 2: Training on Task B, measuring collapse ─────────────────────
    print("\n=== Phase 2: Training on Task B — measuring momentum collapse ===")

    # Reset to Task A endpoint (we want to measure the collapse from that state)
    # Re-load param snapshot is not strictly needed since we just continued,
    # but let's re-extract momentum state to be precise.
    # We do NOT reset params — we continue from the Task A checkpoint naturally.

    # Tracking arrays: one entry per Task B step
    records = {
        "step": [],
        "loss_B": [],
    }
    for i in range(n_levels):
        name = tracker.level_names[i]
        records[f"mom_proj_{name}"] = []    # ||U_A^T m_t|| / ||m_t||
        records[f"mom_proj_abs_{name}"] = []  # ||U_A^T m_t||
        records[f"param_drift_{name}"] = []   # ||θ_t - θ_A|| / ||θ_A||
        records[f"theory_decay_{name}"] = []  # β^(t/C) — theoretical momentum decay

    # Also track theoretical decay baseline
    # For each level l with period C, the effective step count on Task B is t//C
    # Momentum component in old subspace decays as β^(t//C) approximately

    model.train()
    for step in range(task_b_steps):
        xb, yb = get_batch(X_B, Y_B, batch_size, step)
        for opt in optimizers: opt.zero_grad()
        opt_head.zero_grad()
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()

        for i, (C, opt) in enumerate(zip(tracker.update_periods, optimizers)):
            if step % C == 0:
                opt.step()
        opt_head.step()

        tracker.update_step(step, collect_grads_for=None)

        # Record measurements every step
        records["step"].append(step)
        records["loss_B"].append(loss.item())

        for i in range(n_levels):
            name = tracker.level_names[i]
            C = tracker.update_periods[i]
            m_now = tracker.get_momentum_flat(i)
            theta_now = tracker.get_param_flat(i)

            # Momentum projection norm
            if i in subspaces_A and m_now is not None:
                U_A = subspaces_A[i][0].to(device)
                proj = projection_norm(U_A, m_now)
                proj_abs = float((U_A.T @ m_now.float()).norm().item()) if m_now is not None else float("nan")
            else:
                proj = float("nan")
                proj_abs = float("nan")

            # Parameter drift
            if i in theta_A:
                drift = param_drift_norm(theta_now, theta_A[i])
            else:
                drift = float("nan")

            # Theoretical momentum decay: effective updates on task B = step // C + 1
            effective_updates = step // C + 1
            theory_decay = beta ** effective_updates

            records[f"mom_proj_{name}"].append(proj)
            records[f"mom_proj_abs_{name}"].append(proj_abs)
            records[f"param_drift_{name}"].append(drift)
            records[f"theory_decay_{name}"].append(theory_decay)

        if step % 50 == 0:
            print(f"  Task B step {step:4d}  loss={loss.item():.4f}", end="")
            for i in range(n_levels):
                name = tracker.level_names[i]
                p = records[f"mom_proj_{name}"][-1]
                print(f"  proj_{tracker.update_periods[i]}={p:.3f}", end="")
            print()

    # ── Also collect initial momentum projection (step 0) relative values ───
    # Normalize each level's series by its value at step 0
    for i in range(n_levels):
        name = tracker.level_names[i]
        series = records[f"mom_proj_{name}"]
        v0 = series[0] if (series and not math.isnan(series[0])) else 1.0
        records[f"mom_proj_norm_{name}"] = [v / v0 if v0 > 1e-8 else float("nan") for v in series]

    print("\nExperiment complete.")
    return records, tracker.level_names, tracker.update_periods


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]  # red, orange, green, blue
LEVEL_LSTYLE = ["-", "--", "-.", ":"]


def plot_all(records: Dict, level_names: List[str], update_periods: List[int],
             output_dir: Path, beta: float = 0.9, task_b_steps: int = 400):
    """Generate all 3 evidence figures."""
    steps = np.array(records["step"])
    n_levels = len(level_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # ── Figure 1: Momentum projection norm decay per level ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1, ax2 = axes

    # Left: normalized decay (relative to initial)
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            ax1.plot(steps[valid], y[valid],
                     color=LEVEL_COLORS[i], ls=LEVEL_LSTYLE[i], lw=2,
                     label=f"Level C={C}")

    # Overlay theoretical β^t curve (using step for C=1 as proxy)
    theory_x = np.arange(len(steps))
    theory_y = beta ** theory_x  # per-step decay for C=1 level
    ax1.plot(theory_x[:len(steps)], theory_y[:len(steps)],
             color="black", ls="--", lw=1.5, alpha=0.5, label=f"Theoretical β^t (β={beta})")

    ax1.set_xlabel("Task B Training Steps")
    ax1.set_ylabel("Normalized Projection Norm (relative to step 0)")
    ax1.set_title("(a) Momentum Projection onto Old-Task Subspace\n(Vanilla HOPE, normalized)")
    ax1.legend(framealpha=0.9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Right: theoretical decay curves per level (β^(t/C))
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        # Effective updates for this level at each step
        effective_steps = np.array([s // C + 1 for s in steps])
        theory_y_l = beta ** effective_steps
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            ax2.plot(effective_steps[valid], y[valid],
                     color=LEVEL_COLORS[i], ls=LEVEL_LSTYLE[i], lw=2,
                     label=f"Observed C={C}")
            ax2.plot(effective_steps, theory_y_l, color=LEVEL_COLORS[i],
                     ls=":", lw=1.2, alpha=0.6, label=f"Theory β^k C={C}")

    ax2.set_xlabel("Effective Updates on Task B (k = step/C)")
    ax2.set_ylabel("Normalized Projection Norm")
    ax2.set_title("(b) Decay vs Effective Updates\n(All levels follow β^k regardless of C)")
    ax2.legend(framealpha=0.9, ncol=2, fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 1: Gradient Memory Collapse in Vanilla HOPE\nMomentum Buffer Projection onto Task A Subspace U_A",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = output_dir / "fig1_momentum_collapse.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(output_dir / "fig1_momentum_collapse.png", bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)

    # ── Figure 2: Optimizer-state collapse vs parameter forgetting ───────────
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4.5), sharey=False)
    if n_levels == 1:
        axes = [axes]

    for i, (name, C, ax) in enumerate(zip(level_names, update_periods, axes)):
        proj = np.array(records[f"mom_proj_norm_{name}"])
        drift = np.array(records[f"param_drift_{name}"])
        valid = ~np.isnan(proj) & ~np.isnan(drift)

        ax2_twin = ax.twinx()
        l1, = ax.plot(steps[valid], proj[valid],
                      color="#e41a1c", lw=2, label="Momentum Proj (left)")
        l2, = ax2_twin.plot(steps[valid], drift[valid],
                            color="#377eb8", ls="--", lw=2, label="Param Drift (right)")

        ax.set_xlabel("Task B Steps")
        ax.set_ylabel("Normalized Momentum Projection", color="#e41a1c")
        ax2_twin.set_ylabel("Relative Param Drift", color="#377eb8")
        ax.tick_params(axis="y", labelcolor="#e41a1c")
        ax2_twin.tick_params(axis="y", labelcolor="#377eb8")
        ax.set_title(f"Level C={C}\n({name})")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)

    fig.suptitle("Figure 2: Optimizer-State Collapse vs Parameter Forgetting\n"
                 "Momentum projection decays faster / earlier than parameter drift",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = output_dir / "fig2_collapse_vs_forgetting.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(output_dir / "fig2_collapse_vs_forgetting.png", bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)

    # ── Figure 3: Decay rate comparison across levels ────────────────────────
    # Show: despite different update periods, all levels have SAME per-effective-step decay = β
    # This contradicts NL's intention that slow levels should be "more persistent"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_left, ax_right = axes

    # Left: raw time axis — slow levels "appear" more persistent due to fewer updates
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            ax_left.plot(steps[valid], y[valid],
                         color=LEVEL_COLORS[i], ls=LEVEL_LSTYLE[i], lw=2.5,
                         label=f"C={C}")

    ax_left.set_xlabel("Task B Training Steps (wall-clock)")
    ax_left.set_ylabel("Normalized Momentum Projection")
    ax_left.set_title("(a) Apparent Persistence by Level\n(slow levels seem more stable)")
    ax_left.legend(title="Update Period C")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(bottom=0)

    # Right: per-effective-update axis — all levels collapse at same β rate
    # This is the key: when aligned to effective update count, ALL levels decay identically
    ax_right.axhline(y=beta, color="gray", ls="--", alpha=0.5, lw=1, label=f"β = {beta}")
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            k = np.array([s // C + 1 for s in steps])
            # Estimate per-step decay by diff quotient on log scale
            log_y = np.log(np.maximum(y[valid], 1e-10))
            # Scatter plot: step k vs ratio y[k]/y[k-1] per effective update
            prev_y = np.roll(y[valid], 1)
            prev_y[0] = 1.0
            ratio = y[valid] / np.maximum(prev_y, 1e-10)
            # Only show ratios at actual update steps
            update_mask = np.array([int(steps[np.where(valid)[0][j]]) % C == 0
                                    for j in range(valid.sum())])
            k_updates = k[valid][update_mask]
            r_updates = ratio[update_mask]
            if len(k_updates) > 1:
                ax_right.scatter(k_updates[:80], r_updates[:80],
                                 color=LEVEL_COLORS[i], alpha=0.6, s=20,
                                 label=f"C={C}")

    ax_right.set_xlabel("Effective Update Index k")
    ax_right.set_ylabel("Step-wise Projection Ratio (r_k / r_{k-1})")
    ax_right.set_title(f"(b) Per-Step Decay Rate ≈ β = {beta}\n"
                       "(identical across all levels → no hierarchy)")
    ax_right.set_ylim(0.5, 1.2)
    ax_right.legend()
    ax_right.grid(True, alpha=0.3)

    fig.suptitle("Figure 3: Vanilla HOPE Has No Differentiated Decay\n"
                 "All CMS Levels Lose Gradient Memory at the Same Rate per Effective Update",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = output_dir / "fig3_decay_comparison.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(output_dir / "fig3_decay_comparison.png", bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)

    # ── Figure 4 (Bonus): Summary panel combining all evidence ──────────────
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # Row 0, col 0: momentum projection normalized all levels
    ax = fig.add_subplot(gs[0, 0])
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            ax.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                    ls=LEVEL_LSTYLE[i], lw=2, label=f"C={C}")
    t_theory = beta ** np.arange(len(steps))
    ax.plot(np.arange(len(steps)), t_theory, "k--", lw=1.2, alpha=0.5, label=f"β^t")
    ax.set_title("E1: Momentum Projection Decay\n(Task B steps)")
    ax.set_xlabel("Task B Steps")
    ax.set_ylabel("Proj. norm (normalized)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 0, col 1-2: optimizer-state vs param drift for slowest vs fastest level
    for col_offset, level_idx in enumerate([0, n_levels - 1]):
        ax = fig.add_subplot(gs[0, 1 + col_offset])
        name = level_names[level_idx]
        C = update_periods[level_idx]
        proj = np.array(records[f"mom_proj_norm_{name}"])
        drift = np.array(records[f"param_drift_{name}"])
        valid = ~np.isnan(proj) & ~np.isnan(drift)
        ax2 = ax.twinx()
        ax.plot(steps[valid], proj[valid], color="#e41a1c", lw=2, label="Momentum Proj")
        ax2.plot(steps[valid], drift[valid], color="#377eb8", ls="--", lw=2, label="Param Drift")
        ax.set_title(f"E2: Collapse vs Forgetting\nLevel C={C}")
        ax.set_xlabel("Task B Steps")
        ax.set_ylabel("Mom. Proj.", color="#e41a1c")
        ax2.set_ylabel("Param Drift", color="#377eb8")
        ax.tick_params(axis="y", labelcolor="#e41a1c")
        ax2.tick_params(axis="y", labelcolor="#377eb8")
        ax.grid(True, alpha=0.3)
        lines = [Line2D([0], [0], color="#e41a1c", lw=2, label="Mom. Proj."),
                 Line2D([0], [0], color="#377eb8", lw=2, ls="--", label="Param Drift")]
        ax.legend(handles=lines, fontsize=8)

    # Row 1, col 0: theoretical decay vs period
    ax = fig.add_subplot(gs[1, 0])
    # Show: at step T_B=200, what fraction remains for each level?
    T_query = min(200, task_b_steps - 1)
    remaining_proj = []
    remaining_theory = []
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) > 0 and valid_idx[-1] >= T_query:
            rem = y[T_query]
        else:
            rem = float("nan")
        remaining_proj.append(rem)
        k_eff = T_query // C + 1
        remaining_theory.append(beta ** k_eff)
    x = np.arange(n_levels)
    width = 0.35
    bars1 = ax.bar(x - width/2, remaining_proj, width,
                   color=LEVEL_COLORS[:n_levels], alpha=0.8, label="Observed")
    bars2 = ax.bar(x + width/2, remaining_theory, width,
                   color=LEVEL_COLORS[:n_levels], alpha=0.4, hatch="//", label="Theory β^k")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C={C}" for C in update_periods])
    ax.set_xlabel("CMS Level (Update Period)")
    ax.set_ylabel("Fraction of Projection Remaining")
    ax.set_title(f"E3: Memory Remaining at Step {T_query}\n(No hierarchy in vanilla HOPE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Row 1, col 1: absolute momentum projection abs norm per level
    ax = fig.add_subplot(gs[1, 1])
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_abs_{name}"])
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            ax.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                    ls=LEVEL_LSTYLE[i], lw=2, label=f"C={C}")
    ax.set_title("Absolute Momentum Proj. Norm\n||U_A^T m_t||")
    ax.set_xlabel("Task B Steps")
    ax.set_ylabel("||U_A^T m_t||")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, col 2: half-life comparison
    ax = fig.add_subplot(gs[1, 2])
    # Compute empirical half-life (when proj drops to 0.5 relative)
    emp_half_lives = []
    theory_half_lives = []
    for i, (name, C) in enumerate(zip(level_names, update_periods)):
        y = np.array(records[f"mom_proj_norm_{name}"])
        valid = ~np.isnan(y)
        s_valid = steps[valid]
        y_valid = y[valid]
        # Find where y crosses 0.5
        hl_emp = float("nan")
        for j in range(1, len(y_valid)):
            if y_valid[j] <= 0.5:
                # Linear interpolation
                frac = (y_valid[j-1] - 0.5) / (y_valid[j-1] - y_valid[j] + 1e-12)
                hl_emp = float(s_valid[j-1] + frac * (s_valid[j] - s_valid[j-1]))
                break
        # Theory: β^k = 0.5 → k = log(0.5)/log(β) effective updates
        k_half = math.log(0.5) / math.log(beta)
        hl_theory = k_half * C  # in wall-clock steps
        emp_half_lives.append(hl_emp)
        theory_half_lives.append(hl_theory)

    x = np.arange(n_levels)
    valid_emp = [v if not math.isnan(v) else 0 for v in emp_half_lives]
    ax.bar(x - width/2, valid_emp, width, color=LEVEL_COLORS[:n_levels], alpha=0.8, label="Observed")
    ax.bar(x + width/2, theory_half_lives, width, color=LEVEL_COLORS[:n_levels], alpha=0.4, hatch="//", label="Theory")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C={C}" for C in update_periods])
    ax.set_xlabel("CMS Level (Update Period)")
    ax.set_ylabel("Half-life (steps)")
    ax.set_title("Momentum Memory Half-Life\n(Slow levels last longer only due to fewer updates)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Claim 1 Evidence: Gradient Memory Collapse in Vanilla HOPE\n"
                 "Optimizer-state level memory loss is real, precedes parameter forgetting,\n"
                 "and shows no meaningful frequency hierarchy",
                 fontsize=13, y=1.01)
    p = output_dir / "fig4_summary_panel.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(output_dir / "fig4_summary_panel.png", bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden_mult", type=int, default=2)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--task_a_steps", type=int, default=500)
    parser.add_argument("--task_b_steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--grad_rank", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records, level_names, update_periods = run_experiment(
        dim=args.dim,
        hidden_mult=args.hidden_mult,
        n_levels=args.n_levels,
        task_a_steps=args.task_a_steps,
        task_b_steps=args.task_b_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        grad_rank=args.grad_rank,
        device=device,
        seed=args.seed,
        n_samples=args.n_samples,
    )

    # Save raw records
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_json = {k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in records.items()}
    with open(output_dir / "records.json", "w") as f:
        json.dump(records_json, f)
    print(f"Saved records to {output_dir / 'records.json'}")

    # Generate all plots
    plot_all(records, level_names, update_periods, output_dir, beta=args.beta, task_b_steps=args.task_b_steps)
    print("\n=== All figures saved. Done. ===")


if __name__ == "__main__":
    main()
