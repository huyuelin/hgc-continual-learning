import pytest
import torch

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import (
    HOPEAttentionBlock,
    HOPEAttentionBlockConfig,
    HOPEBlock,
    HOPEBlockConfig,
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
)
from nested_learning.levels import LevelSpec


def _build_variant(variant: str, *, flush_partial: bool):
    cms_levels = (LevelSpec(name="fast", update_period=4),)
    if variant == "attention":
        cfg = HOPEAttentionBlockConfig(
            dim=8,
            heads=1,
            cms_levels=cms_levels,
            cms_online_updates=True,
            cms_flush_partial_at_end=flush_partial,
            cms_chunk_reduction="sum",
        )
        block = HOPEAttentionBlock(cfg)
        state = build_block_fast_state(
            titan_module=None,
            cms_blocks=dict(block.cms.blocks.items()),
            specs=cfg.cms_levels,
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
        return block, state
    if variant == "selfmod":
        cfg = HOPESelfModBlockConfig(
            dim=8,
            cms_levels=cms_levels,
            cms_online_updates=True,
            cms_flush_partial_at_end=flush_partial,
            cms_chunk_reduction="sum",
            selfmod_chunk_size=1,
            selfmod_chunk_size_memory=4,
        )
        block = HOPESelfModBlock(cfg)
        state = build_block_fast_state(
            titan_module=None,
            cms_blocks=dict(block.cms.blocks.items()),
            selfmod_module=block.selfmod,
            specs=cfg.cms_levels,
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
        return block, state
    if variant == "hybrid":
        cfg = HOPEBlockConfig(
            dim=8,
            heads=1,
            titan_level=LevelSpec(name="titan", update_period=1),
            cms_levels=cms_levels,
            cms_online_updates=True,
            cms_flush_partial_at_end=flush_partial,
            cms_chunk_reduction="sum",
        )
        block = HOPEBlock(cfg)
        state = build_block_fast_state(
            titan_module=block.titan_memory,
            cms_blocks=dict(block.cms.blocks.items()),
            specs=(cfg.titan_level, *cfg.cms_levels),
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
        return block, state
    raise ValueError(f"unknown variant {variant}")


@pytest.mark.parametrize("variant", ["attention", "selfmod", "hybrid"])
def test_cms_fast_state_buffers_persist_across_calls(variant: str) -> None:
    torch.manual_seed(0)
    block, state = _build_variant(variant, flush_partial=False)
    x = torch.randn(1, 2, 8)
    teach = torch.ones(1, 2, 8)

    _ = block(x, teach_signal=teach, fast_state=state, finalize_updates=False)
    first_stats = block.pop_update_stats()
    assert first_stats["cms.fast"]["updates_applied"] == 0.0
    assert first_stats["cms.fast"]["pending_tokens"] == 2.0

    _ = block(x, teach_signal=teach, fast_state=state, finalize_updates=False)
    payload = block.pop_update_stats()["cms.fast"]
    assert payload["gate_hit"] == 1.0
    assert payload["chunk_tokens"] == 4.0


@pytest.mark.parametrize("variant", ["attention", "selfmod", "hybrid"])
def test_cms_fast_state_flushes_only_on_finalize(variant: str) -> None:
    torch.manual_seed(0)
    block, state = _build_variant(variant, flush_partial=True)
    x3 = torch.randn(1, 3, 8)
    x1 = torch.randn(1, 1, 8)
    teach3 = torch.ones(1, 3, 8)
    teach1 = torch.ones(1, 1, 8)

    _ = block(x3, teach_signal=teach3, fast_state=state, finalize_updates=False)
    first_stats = block.pop_update_stats()
    assert first_stats["cms.fast"]["updates_applied"] == 0.0
    assert first_stats["cms.fast"]["pending_tokens"] == 3.0

    _ = block(x3, teach_signal=teach3, fast_state=state, finalize_updates=False)
    payload_mid = block.pop_update_stats()["cms.fast"]
    assert payload_mid["gate_hit"] == 1.0
    assert payload_mid["chunk_tokens"] == 4.0

    _ = block(x1, teach_signal=teach1, fast_state=state, finalize_updates=True)
    payload_final = block.pop_update_stats()["cms.fast"]
    assert payload_final["gate_hit"] == 1.0
    assert payload_final["chunk_tokens"] == 3.0
