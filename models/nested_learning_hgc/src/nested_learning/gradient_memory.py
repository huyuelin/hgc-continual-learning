"""Hierarchical Gradient Consolidation (HGC) for Nested Learning.

This module implements the core of HGC: a level-aware gradient memory system
that protects old-knowledge subspaces and enables cross-level distillation
between fast and slow CMS levels.

Key components:
  1. GradientSignature  -- low-rank SVD basis of accumulated gradients per level
  2. LevelGradientMemory -- manages signatures, projection, and consolidation
  3. CrossLevelDistiller -- distills fast-level gradient information into slow levels

Theoretical grounding (Nested Learning perspective):
  - Momentum = associative memory compressing gradients (NL paper Eq. 10)
  - Catastrophic forgetting = catastrophic overwrite of gradient memory
  - Protection = preserve the principal subspace of old gradients
  - Distillation = transfer fast-level knowledge to slow-level (sleep consolidation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .levels import LevelSpec


# ---------------------------------------------------------------------------
# 1.  Gradient Signature: low-rank representation of a task's gradient subspace
# ---------------------------------------------------------------------------

@dataclass
class GradientSignature:
    """Per-level gradient signature capturing the principal directions of past updates.

    Attributes:
        basis:  orthonormal columns spanning old-knowledge subspace  [d, r]
        importance:  singular values for importance-weighted projection  [r]
        grad_buffer:  FIFO buffer of flattened gradient vectors awaiting SVD
        max_buffer_size:  capacity of the gradient buffer before oldest evicted
        target_rank:  desired rank of the signature
    """

    basis: Optional[torch.Tensor] = None
    importance: Optional[torch.Tensor] = None
    grad_buffer: list = field(default_factory=list)
    max_buffer_size: int = 512
    target_rank: int = 0


# ---------------------------------------------------------------------------
# 2.  Cross-Level Distillation Record
# ---------------------------------------------------------------------------

@dataclass
class DistillationRecord:
    """Stores fast-level gradient statistics for later distillation to slow levels.

    The fast levels see many more updates and thus have richer gradient information.
    We periodically "distill" this into slow levels by injecting a regularization
    term that biases slow-level updates toward directions the fast level found useful.
    """

    source_level: str
    target_level: str
    # Running covariance of fast-level gradients (low-rank approximation)
    fast_basis: Optional[torch.Tensor] = None   # [d, r_fast]
    fast_values: Optional[torch.Tensor] = None   # [r_fast]
    # Accumulated fast-level gradient mean for direction bias
    fast_grad_ema: Optional[torch.Tensor] = None
    ema_decay: float = 0.99
    accumulation_count: int = 0


# ---------------------------------------------------------------------------
# 3.  Level Gradient Memory: the main HGC controller
# ---------------------------------------------------------------------------

class LevelGradientMemory:
    """Manages gradient signatures and cross-level distillation across CMS levels.

    This is the central HGC controller.  It is instantiated once per HOPEBlock
    (or once per model if level names are globally unique) and is wired into the
    DeepMomentum optimizer via the ``project_gradient`` and ``consolidation_boost``
    hooks.

    Design principles:
      - *Level-aware*: protection strength scales with update period
      - *Incremental*: signatures are updated incrementally via merging, not
        recomputed from scratch
      - *Distillation-augmented*: fast levels teach slow levels which directions
        matter, enabling slow levels to consolidate even before task boundaries
    """

    def __init__(
        self,
        level_specs: Sequence[LevelSpec],
        *,
        r_base: int = 4,
        alpha_mode: str = "log_period",
        gamma_scale: float = 0.1,
        distillation_pairs: Sequence[tuple[str, str]] | None = None,
        distillation_strength: float = 0.05,
        buffer_size: int = 512,
    ) -> None:
        """
        Args:
            level_specs: the CMS level specifications (name + update_period)
            r_base: base rank for signature extraction; actual rank = r_base * log2(period)
            alpha_mode: how to compute per-level projection strength
                "log_period" (default): alpha = 1 - 1/log2(period+1)
                "linear": alpha = 1 - 1/period
            gamma_scale: multiplier for the consolidation boost strength
            distillation_pairs: list of (fast_level, slow_level) pairs for cross-level
                distillation; if None, auto-generated from adjacent level pairs
            distillation_strength: lambda for distillation regularization
            buffer_size: FIFO gradient buffer capacity per level
        """
        self.level_specs = {spec.name: spec for spec in level_specs}
        self.signatures: Dict[str, GradientSignature] = {}
        self.alpha: Dict[str, float] = {}
        self.gamma: Dict[str, float] = {}
        self._target_rank: Dict[str, int] = {}

        sorted_specs = sorted(level_specs, key=lambda s: s.update_period)

        for spec in sorted_specs:
            period = spec.update_period
            rank = r_base * max(1, int(math.log2(max(period, 2))))
            if period <= 1:
                rank = 0  # fast level: no self-protection needed

            if alpha_mode == "log_period":
                alpha = 1.0 - 1.0 / math.log2(max(period, 2) + 1)
            else:
                alpha = 1.0 - 1.0 / max(period, 1)

            self.signatures[spec.name] = GradientSignature(
                max_buffer_size=buffer_size,
                target_rank=rank,
            )
            self.alpha[spec.name] = max(0.0, min(1.0, alpha))
            self.gamma[spec.name] = gamma_scale * self.alpha[spec.name]
            self._target_rank[spec.name] = rank

        # Cross-level distillation
        self.distillation_records: list[DistillationRecord] = []
        self.distillation_strength = distillation_strength
        if distillation_pairs is None and len(sorted_specs) >= 2:
            # Auto-generate: each fast level distills to next slower level
            for i in range(len(sorted_specs) - 1):
                self.distillation_records.append(
                    DistillationRecord(
                        source_level=sorted_specs[i].name,
                        target_level=sorted_specs[i + 1].name,
                    )
                )
        elif distillation_pairs is not None:
            for src, tgt in distillation_pairs:
                self.distillation_records.append(
                    DistillationRecord(source_level=src, target_level=tgt)
                )

    # ----- Gradient Accumulation -----

    def accumulate_gradient(self, level: str, grad: torch.Tensor) -> None:
        """Buffer a gradient vector for later signature extraction.

        Also updates cross-level distillation records for any record where
        this level is the source.
        """
        sig = self.signatures.get(level)
        if sig is None:
            return

        g_flat = grad.detach().reshape(-1).float()

        # Buffer for SVD (only for levels that need protection)
        if sig.target_rank > 0:
            # Guard: ensure all buffered gradients have consistent size
            if sig.grad_buffer and sig.grad_buffer[0].shape != g_flat.shape:
                sig.grad_buffer.clear()  # Reset buffer on size change
            sig.grad_buffer.append(g_flat)
            if len(sig.grad_buffer) > sig.max_buffer_size:
                sig.grad_buffer.pop(0)

        # Update distillation EMA for records where this level is source
        for rec in self.distillation_records:
            if rec.source_level != level:
                continue
            if rec.fast_grad_ema is None:
                rec.fast_grad_ema = torch.zeros_like(g_flat)
            # Guard against size mismatch in per-block training
            if rec.fast_grad_ema.shape != g_flat.shape:
                continue
            rec.fast_grad_ema.mul_(rec.ema_decay).add_(g_flat, alpha=1 - rec.ema_decay)
            rec.accumulation_count += 1

    # ----- Consolidation (called at task/segment boundaries) -----

    def consolidate(self, level: str) -> None:
        """Extract/update gradient signature via incremental SVD."""
        sig = self.signatures.get(level)
        if sig is None:
            return
        rank = sig.target_rank
        if rank == 0 or len(sig.grad_buffer) < 2:
            return

        G = torch.stack(sig.grad_buffer, dim=1)  # [d, T]

        # Merge with existing basis (incremental SVD)
        if sig.basis is not None:
            old_weighted = sig.basis * sig.importance.unsqueeze(0)
            G = torch.cat([old_weighted.to(G.device), G], dim=1)

        # Randomized SVD for efficiency
        q = min(rank, G.shape[1], G.shape[0])
        if q < 1:
            return
        try:
            U, S, _ = torch.svd_lowrank(G, q=q)
        except Exception:
            return

        sig.basis = U[:, :q].detach()
        sig.importance = S[:q].detach()
        sig.grad_buffer.clear()

    def consolidate_all(self) -> None:
        """Consolidate all levels (typically at task boundary)."""
        for name in self.signatures:
            self.consolidate(name)
        # Also extract distillation bases from accumulated fast-level info
        self._update_distillation_bases()

    def _update_distillation_bases(self) -> None:
        """Update the low-rank bases for cross-level distillation."""
        for rec in self.distillation_records:
            src_sig = self.signatures.get(rec.source_level)
            if src_sig is None or src_sig.basis is None:
                continue
            rec.fast_basis = src_sig.basis.clone()
            rec.fast_values = src_sig.importance.clone()

    # ----- Gradient Projection (OGP) -----

    def project_gradient(self, level: str, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient to preserve old-knowledge subspace.

        Applies soft orthogonal projection:
            g_new = g - alpha * U @ U^T @ g

        where U is the old-knowledge basis and alpha is level-dependent.
        """
        sig = self.signatures.get(level)
        if sig is None or sig.basis is None:
            return grad

        alpha = self.alpha.get(level, 0.0)
        if alpha < 1e-8:
            return grad

        original_shape = grad.shape
        g_flat = grad.reshape(-1).float()
        U = sig.basis.to(g_flat.device, dtype=g_flat.dtype)

        if U.shape[0] != g_flat.shape[0]:
            return grad  # shape mismatch guard

        # Importance-weighted projection: weight directions by their singular values
        if sig.importance is not None:
            # Normalize importance to [0, 1] range for weighting
            imp = sig.importance.to(g_flat.device, dtype=g_flat.dtype)
            imp_normalized = imp / (imp.max() + 1e-8)
            # Weighted projection: directions with larger singular values get more protection
            proj_coeffs = U.t() @ g_flat        # [r]
            weighted_coeffs = proj_coeffs * imp_normalized  # importance-weighted
            projection = U @ weighted_coeffs     # [d]
        else:
            projection = U @ (U.t() @ g_flat)

        g_new = g_flat - alpha * projection
        return g_new.reshape(original_shape).to(grad.dtype)

    # ----- Consolidation Boost (CAM) -----

    def consolidation_boost(self, level: str, momentum: torch.Tensor) -> torch.Tensor:
        """Boost momentum components aligned with old knowledge (prevents exponential decay).

        Returns an additive correction term for the momentum buffer.
        """
        sig = self.signatures.get(level)
        if sig is None or sig.basis is None:
            return torch.zeros_like(momentum)

        gamma = self.gamma.get(level, 0.0)
        if gamma < 1e-8:
            return torch.zeros_like(momentum)

        original_shape = momentum.shape
        m_flat = momentum.reshape(-1).float()
        U = sig.basis.to(m_flat.device, dtype=m_flat.dtype)

        if U.shape[0] != m_flat.shape[0]:
            return torch.zeros_like(momentum)

        boost = gamma * (U @ (U.t() @ m_flat))
        return boost.reshape(original_shape).to(momentum.dtype)

    # ----- Cross-Level Distillation Signal -----

    def distillation_signal(self, target_level: str, grad: torch.Tensor) -> torch.Tensor:
        """Compute distillation regularization for a slow level's gradient.

        The fast level has seen many more updates and accumulated richer gradient
        statistics.  We bias the slow level's update direction toward the subspace
        the fast level found important.

        Concretely:  nudge_t = lambda * U_fast @ (U_fast^T @ g_slow)
        This gently pulls g_slow toward the fast level's principal gradient subspace.
        """
        if self.distillation_strength < 1e-8:
            return torch.zeros_like(grad)

        total_nudge = torch.zeros_like(grad).reshape(-1).float()
        original_shape = grad.shape
        g_flat = grad.reshape(-1).float()

        for rec in self.distillation_records:
            if rec.target_level != target_level:
                continue
            if rec.fast_basis is None:
                continue

            U_fast = rec.fast_basis.to(g_flat.device, dtype=g_flat.dtype)
            if U_fast.shape[0] != g_flat.shape[0]:
                continue

            # Project slow grad onto fast subspace => direction that fast level cares about
            proj_onto_fast = U_fast @ (U_fast.t() @ g_flat)
            total_nudge.add_(proj_onto_fast)

        return (self.distillation_strength * total_nudge).reshape(original_shape).to(grad.dtype)

    # ----- Diagnostics -----

    def subspace_overlap(self, level_a: str, level_b: str) -> float:
        """Compute subspace overlap between two levels' signatures.

        overlap = ||U_a^T @ U_b||_F / sqrt(min(r_a, r_b))
        """
        sig_a = self.signatures.get(level_a)
        sig_b = self.signatures.get(level_b)
        if sig_a is None or sig_b is None:
            return 0.0
        if sig_a.basis is None or sig_b.basis is None:
            return 0.0
        U_a = sig_a.basis.float()
        U_b = sig_b.basis.float().to(U_a.device)
        if U_a.shape[0] != U_b.shape[0]:
            return 0.0
        cross = U_a.t() @ U_b
        r_min = min(U_a.shape[1], U_b.shape[1])
        return float(cross.norm().item() / math.sqrt(max(r_min, 1)))

    def signature_stats(self) -> Dict[str, dict]:
        """Return diagnostic statistics for all levels."""
        stats = {}
        for name, sig in self.signatures.items():
            entry = {
                "rank": sig.target_rank,
                "has_basis": sig.basis is not None,
                "buffer_size": len(sig.grad_buffer),
                "alpha": self.alpha.get(name, 0.0),
                "gamma": self.gamma.get(name, 0.0),
            }
            if sig.basis is not None and sig.importance is not None:
                entry["top_singular_value"] = float(sig.importance[0].item())
                entry["effective_rank"] = float(
                    (sig.importance / sig.importance.sum()).pow(2).sum().reciprocal().item()
                )
            stats[name] = entry
        return stats
