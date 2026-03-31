import torch

from nested_learning.levels import LevelSpec
from nested_learning.optim.manager import LevelConfig, LevelOptimizerManager


def _manager() -> LevelOptimizerManager:
    spec = LevelSpec(name="cms_fast", update_period=1, optimizer_key="default")
    cfg = LevelConfig(
        specs=(spec,),
        optimizer_configs={"default": {"type": "deep_momentum", "params": {"variant": "basic"}}},
        default_lr=0.1,
    )
    return LevelOptimizerManager(cfg)


def test_apply_grads_differentiable_preserves_gradient_path() -> None:
    mgr = _manager()
    base = torch.randn(4, requires_grad=True)
    loss = (base**2).sum()
    (grad,) = torch.autograd.grad(loss, (base,), create_graph=True)
    updated, _ = mgr.apply_grads(
        "cms_fast",
        {"w": base},
        {"w": grad},
        force=True,
        differentiable=True,
    )
    downstream = (updated["w"] ** 2).sum()
    downstream.backward()
    assert base.grad is not None
    assert float(base.grad.abs().sum().item()) > 0.0


def test_apply_grads_nondifferentiable_breaks_gradient_path() -> None:
    mgr = _manager()
    base = torch.randn(4, requires_grad=True)
    loss = (base**2).sum()
    (grad,) = torch.autograd.grad(loss, (base,), create_graph=True)
    updated, _ = mgr.apply_grads(
        "cms_fast",
        {"w": base},
        {"w": grad},
        force=True,
        differentiable=False,
    )
    assert updated["w"].requires_grad is False
