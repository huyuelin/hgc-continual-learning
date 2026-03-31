import pytest
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def _metric(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _build_attention_model(*, flush_partial: bool) -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="fast", update_period=4),),
        block_variant="hope_attention",
        cms_flush_partial_at_end=flush_partial,
    )
    return HOPEModel(cfg).eval()


def _build_attention_model_with_period(*, update_period: int) -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="fast", update_period=update_period),),
        block_variant="hope_attention",
        cms_flush_partial_at_end=False,
    )
    return HOPEModel(cfg).eval()


def test_model_streaming_cadence_matches_single_call_counts() -> None:
    torch.manual_seed(0)
    model = _build_attention_model(flush_partial=False)
    key_prefix = "layer0.cms.fast"

    full_state = model.init_fast_state()
    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 4)),
            teach_signal=torch.ones(1, 4, 8),
            fast_state=full_state,
            finalize_updates=False,
        )
    full_metrics = model.pop_update_metrics()

    split_state = model.init_fast_state()
    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 2)),
            teach_signal=torch.ones(1, 2, 8),
            fast_state=split_state,
            finalize_updates=False,
        )
    _ = model.pop_update_metrics()
    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 2)),
            teach_signal=torch.ones(1, 2, 8),
            fast_state=split_state,
            finalize_updates=False,
        )
    split_metrics = model.pop_update_metrics()

    assert _metric(full_metrics, f"{key_prefix}.updates_applied") == 1.0
    assert _metric(split_metrics, f"{key_prefix}.updates_applied") == 1.0
    assert _metric(full_metrics, f"{key_prefix}.chunk_tokens") == 4.0
    assert _metric(split_metrics, f"{key_prefix}.chunk_tokens") == 4.0


@pytest.mark.parametrize("update_period", [2, 4, 8])
def test_model_streaming_cadence_matches_for_multiple_periods(update_period: int) -> None:
    torch.manual_seed(0)
    model = _build_attention_model_with_period(update_period=update_period)
    key_prefix = "layer0.cms.fast"
    total_tokens = update_period * 2
    full_tokens = torch.randint(0, 32, (1, total_tokens))
    full_teach = torch.ones(1, total_tokens, 8)

    full_state = model.init_fast_state()
    with torch.no_grad():
        _ = model(
            full_tokens,
            teach_signal=full_teach,
            fast_state=full_state,
            finalize_updates=False,
        )
    full_metrics = model.pop_update_metrics()

    split_state = model.init_fast_state()
    for _ in range(2):
        with torch.no_grad():
            _ = model(
                torch.randint(0, 32, (1, update_period)),
                teach_signal=torch.ones(1, update_period, 8),
                fast_state=split_state,
                finalize_updates=False,
            )
    split_metrics = model.pop_update_metrics()

    assert _metric(full_metrics, f"{key_prefix}.updates_applied") == 2.0
    assert _metric(split_metrics, f"{key_prefix}.updates_applied") == 1.0
    assert _metric(full_metrics, f"{key_prefix}.chunk_tokens") == float(total_tokens)
    assert _metric(split_metrics, f"{key_prefix}.chunk_tokens") == float(update_period)


def test_model_finalize_flushes_partial_once() -> None:
    torch.manual_seed(0)
    model = _build_attention_model(flush_partial=True)
    state = model.init_fast_state()
    key_prefix = "layer0.cms.fast"

    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 3)),
            teach_signal=torch.ones(1, 3, 8),
            fast_state=state,
            finalize_updates=False,
        )
    first = model.pop_update_metrics()
    assert _metric(first, f"{key_prefix}.updates_applied") == 0.0
    assert _metric(first, f"{key_prefix}.pending_tokens") == 3.0

    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 3)),
            teach_signal=torch.ones(1, 3, 8),
            fast_state=state,
            finalize_updates=False,
        )
    second = model.pop_update_metrics()
    assert _metric(second, f"{key_prefix}.updates_applied") == 1.0
    assert _metric(second, f"{key_prefix}.chunk_tokens") == 4.0
    assert _metric(second, f"{key_prefix}.pending_tokens") == 2.0

    with torch.no_grad():
        _ = model(
            torch.randint(0, 32, (1, 1)),
            teach_signal=torch.ones(1, 1, 8),
            fast_state=state,
            finalize_updates=True,
        )
    final = model.pop_update_metrics()
    assert _metric(final, f"{key_prefix}.updates_applied") == 1.0
    assert _metric(final, f"{key_prefix}.tokens_flushed") == 3.0
    assert _metric(final, f"{key_prefix}.pending_tokens") == 0.0


def test_slow_cms_level_does_not_starve_under_segmented_calls() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(
            LevelSpec(name="fast", update_period=2),
            LevelSpec(name="slow", update_period=8),
        ),
        block_variant="hope_attention",
        cms_flush_partial_at_end=False,
    )
    model = HOPEModel(cfg).eval()
    state = model.init_fast_state()
    saw_slow_update = False
    for _ in range(4):
        with torch.no_grad():
            _ = model(
                torch.randint(0, 32, (1, 2)),
                teach_signal=torch.ones(1, 2, 8),
                fast_state=state,
                finalize_updates=False,
            )
        metrics = model.pop_update_metrics()
        if _metric(metrics, "layer0.cms.slow.updates_applied") > 0:
            saw_slow_update = True
    assert saw_slow_update
