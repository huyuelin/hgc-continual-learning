import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _compute_layer_teach_signals


def _build_attention_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_attention",
        cms_flush_partial_at_end=True,
    )
    return HOPEModel(cfg).train()


def _two_chunk_grad_norm(*, differentiable_updates: bool) -> float:
    torch.manual_seed(0)
    model = _build_attention_model()
    state = model.init_fast_state()
    tokens = torch.randint(0, 64, (1, 6))
    chunk1 = tokens[:, :3]
    chunk2 = tokens[:, 3:]

    logits1, _pre, block_outputs = model.forward_with_block_outputs(
        chunk1,
        fast_state=state,
    )
    targets1 = torch.cat([chunk1[:, 1:], chunk2[:, :1]], dim=1)
    loss1 = F.cross_entropy(
        logits1[:, : targets1.size(1), :].reshape(-1, logits1.size(-1)),
        targets1.reshape(-1),
    )
    teach_signals = _compute_layer_teach_signals(
        loss1,
        block_outputs,
        detach=not differentiable_updates,
        create_graph=differentiable_updates,
    )

    _ = model(
        chunk1,
        teach_signals=teach_signals,
        fast_state=state,
        finalize_updates=False,
        differentiable_updates=differentiable_updates,
    )

    logits2 = model(chunk2, fast_state=state)
    loss2 = F.cross_entropy(
        logits2[:, :-1].reshape(-1, logits2.size(-1)),
        chunk2[:, 1:].reshape(-1),
    )
    grad = torch.autograd.grad(loss2, block_outputs[0], allow_unused=True)[0]
    if grad is None:
        return 0.0
    return float(grad.detach().norm().item())


def test_boundary_state_grad_mode_propagates_across_write_path() -> None:
    assert _two_chunk_grad_norm(differentiable_updates=True) > 0.0


def test_stopgrad_mode_blocks_boundary_state_grad_path() -> None:
    assert _two_chunk_grad_norm(differentiable_updates=False) == 0.0
