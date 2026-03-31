from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the NL paper."""

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.state: dict[str, DeepMomentumState] = {}
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state.clear()

    def _precondition(self, grad: torch.Tensor, state: DeepMomentumState) -> torch.Tensor:
        if state.sq_avg is None or state.sq_avg.shape != grad.shape:
            state.sq_avg = torch.zeros_like(grad)
        state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {
            "ctx_norm": 0.0,
            "proj_norm": 0.0,
            "proj_skipped": 0.0,
        }
        if context is None:
            return grad, metrics
        ctx = context
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = ctx_norm.item()

        if ctx_norm > 0:
            if grad.ndim == 0 or grad.shape[-1] != ctx.shape[-1]:
                metrics["proj_skipped"] = 1.0
                return grad, metrics
            unit = ctx / (ctx_norm + self.eps)
            # Project grad orthogonal to context (rank-1 projector).
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = torch.norm(update).item()
            return update, metrics
        return grad, metrics

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
    ) -> torch.Tensor:
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return state.grad_avg


class DeepMomentumHGC(DeepMomentum):
    """DeepMomentum extended with Hierarchical Gradient Consolidation.

    Three additional operations per step:
      1. Accumulate gradient into the level's gradient memory buffer
      2. OGP: project gradient to preserve old-knowledge subspace
      3. CAM: boost momentum components aligned with old knowledge
      4. Distillation: inject cross-level teaching signal for slow levels
    """

    def __init__(
        self,
        *,
        grad_memory=None,
        level_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.grad_memory = grad_memory      # LevelGradientMemory instance
        self.level_name = level_name

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
    ) -> torch.Tensor:
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}

        # --- Standard update computation (identical to parent) ---
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)

        # --- HGC: Hierarchical Gradient Consolidation ---
        if self.grad_memory is not None and self.level_name:
            # Step 1: Accumulate gradient for future signature extraction
            self.grad_memory.accumulate_gradient(self.level_name, update)

            # Step 2: OGP -- orthogonal projection to protect old knowledge
            update = self.grad_memory.project_gradient(self.level_name, update)

            # Step 3: Cross-level distillation -- fast teaches slow
            distill_nudge = self.grad_memory.distillation_signal(
                self.level_name, update
            )
            update = update + distill_nudge

            # Step 4: Standard EMA update
            state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)

            # Step 5: CAM -- consolidation boost to prevent exponential decay
            boost = self.grad_memory.consolidation_boost(
                self.level_name, state.grad_avg
            )
            state.grad_avg.add_(boost)
        else:
            # Fallback to vanilla momentum
            state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)

        return state.grad_avg
