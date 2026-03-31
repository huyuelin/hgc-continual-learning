from __future__ import annotations

from typing import Any, Dict

from .deep import DeepMomentum, DeepMomentumHGC


def build_optimizer(
    config: Dict[str, Any],
    *,
    grad_memory=None,
    level_name: str = "",
) -> DeepMomentum:
    """Build an optimizer instance from config dict.

    Supports:
      - "deep_momentum": original DeepMomentum (NL paper)
      - "deep_momentum_hgc": DeepMomentum + Hierarchical Gradient Consolidation
    """
    opt_type = config.get("type", "deep_momentum").lower()
    params = config.get("params", {})

    if opt_type == "deep_momentum":
        return DeepMomentum(**params)
    elif opt_type == "deep_momentum_hgc":
        return DeepMomentumHGC(
            grad_memory=grad_memory,
            level_name=level_name,
            **params,
        )
    else:
        raise ValueError(f"Unsupported optimizer type {opt_type}")
