import torch

from nested_learning.eval_state import (
    forward_with_eval_state,
    init_eval_streaming_state,
    parse_eval_state_mode,
)
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def _transformer_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=2,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="transformer",
    )
    return HOPEModel(cfg).eval()


def test_parse_eval_state_mode_variants() -> None:
    assert parse_eval_state_mode("reset_per_sample") is False
    assert parse_eval_state_mode("isolated") is False
    assert parse_eval_state_mode("carry_across_samples") is True
    assert parse_eval_state_mode("carry") is True


def test_forward_with_eval_state_attention_cache_continuity() -> None:
    torch.manual_seed(0)
    model = _transformer_model()
    tokens = torch.randint(0, 32, (1, 7))
    with torch.no_grad():
        full = model(tokens)
        state = init_eval_streaming_state(
            model,
            use_fast_state=False,
            use_attention_cache=True,
        )
        logits_a, state = forward_with_eval_state(model, tokens[:, :3], state=state)
        logits_b, state = forward_with_eval_state(model, tokens[:, 3:], state=state)
        stitched = torch.cat([logits_a, logits_b], dim=1)
    assert state is not None
    assert state.attention_cache is not None
    assert torch.allclose(full, stitched, atol=1e-5, rtol=1e-5)


def test_forward_with_eval_state_none_state_passthrough() -> None:
    model = _transformer_model()
    tokens = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        logits, state = forward_with_eval_state(model, tokens, state=None)
        expected = model(tokens)
    assert state is None
    assert torch.allclose(logits, expected)
