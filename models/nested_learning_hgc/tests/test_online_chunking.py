import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import (
    _compute_layer_teach_signals,
    _iter_online_boundary_chunks,
    _iter_online_token_chunks,
)


def test_online_chunking_carries_boundary_overlap_and_token_pairs() -> None:
    tokens = torch.arange(10).view(1, 10)
    chunks = list(_iter_online_token_chunks(tokens, chunk_size=4))

    lengths = [chunk.size(1) for chunk, _ in chunks]
    finals = [is_final for _, is_final in chunks]
    pairs = sum(chunk.size(1) - 1 for chunk, _ in chunks)

    assert lengths == [4, 5, 3]
    assert finals == [False, False, True]
    assert pairs == tokens.size(1) - 1


def test_online_chunking_supports_chunk_size_one() -> None:
    tokens = torch.arange(5).view(1, 5)
    chunks = list(_iter_online_token_chunks(tokens, chunk_size=1))
    # First chunk has length 1 (no CE pairs), remaining chunks have overlap length 2.
    lengths = [chunk.size(1) for chunk, _ in chunks]
    assert lengths[0] == 1
    assert all(length == 2 for length in lengths[1:])
    assert sum(chunk.size(1) - 1 for chunk, _ in chunks) == tokens.size(1) - 1


def test_online_chunking_chunk_size_one_preserves_train_loop_token_accounting() -> None:
    tokens = torch.arange(9).view(1, 9)
    total_pairs = 0
    for chunk, _finalize in _iter_online_token_chunks(tokens, chunk_size=1):
        if chunk.size(1) <= 1:
            continue
        total_pairs += chunk.size(1) - 1
    assert total_pairs == tokens.size(1) - 1


def test_online_boundary_chunks_emit_next_tokens_and_exact_target_count() -> None:
    tokens = torch.arange(10).view(1, 10)
    chunks = list(_iter_online_boundary_chunks(tokens, chunk_size=4))
    lengths = [chunk.size(1) for chunk, _next, _final in chunks]
    next_tokens = [None if nxt is None else int(nxt[0].item()) for _chunk, nxt, _final in chunks]
    finals = [is_final for _chunk, _next, is_final in chunks]
    target_count = sum(chunk.size(1) - 1 + (0 if nxt is None else 1) for chunk, nxt, _ in chunks)
    assert lengths == [4, 4, 2]
    assert next_tokens == [4, 8, None]
    assert finals == [False, False, True]
    assert target_count == tokens.size(1) - 1


def _supervised_targets_overlap(tokens: torch.Tensor, chunk_size: int) -> list[int]:
    targets: list[int] = []
    for chunk, _ in _iter_online_token_chunks(tokens, chunk_size=chunk_size):
        if chunk.size(1) <= 1:
            continue
        targets.extend(chunk[:, 1:].reshape(-1).tolist())
    return targets


def _supervised_targets_boundary(tokens: torch.Tensor, chunk_size: int) -> list[int]:
    targets: list[int] = []
    for chunk, next_tokens, _ in _iter_online_boundary_chunks(tokens, chunk_size=chunk_size):
        if chunk.size(1) > 1:
            targets.extend(chunk[:, 1:].reshape(-1).tolist())
        if next_tokens is not None:
            targets.extend(next_tokens.reshape(-1).tolist())
    return targets


def _build_transformer_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=2,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="transformer",
    )
    return HOPEModel(cfg).eval()


def test_online_boundary_chunked_loss_matches_monolithic_with_attention_cache() -> None:
    torch.manual_seed(0)
    model = _build_transformer_model()
    tokens = torch.randint(0, 64, (1, 11))
    with torch.no_grad():
        full_logits = model(tokens)
        full_loss = F.cross_entropy(
            full_logits[:, :-1].reshape(-1, full_logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )

        cache = model.init_attention_cache()
        total_pairs = 0
        total_loss = 0.0
        for chunk, next_tokens, _ in _iter_online_boundary_chunks(tokens, chunk_size=4):
            logits, cache = model(
                chunk,
                attention_cache=cache,
                return_attention_cache=True,
            )
            if next_tokens is None:
                targets = chunk[:, 1:]
            else:
                targets = torch.cat([chunk[:, 1:], next_tokens.unsqueeze(1)], dim=1)
            if targets.numel() == 0:
                continue
            loss = F.cross_entropy(
                logits[:, : targets.size(1), :].reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_pairs += targets.size(1)
            total_loss += float(loss.item()) * targets.size(1)
        chunked_loss = total_loss / max(total_pairs, 1)
    assert total_pairs == tokens.size(1) - 1
    assert torch.isclose(torch.tensor(chunked_loss), full_loss, atol=1e-6, rtol=1e-6)


def test_online_target_coverage_property_randomized() -> None:
    torch.manual_seed(0)
    for seq_len in range(2, 25):
        tokens = torch.arange(seq_len).view(1, seq_len)
        expected = list(range(1, seq_len))
        for chunk_size in range(1, seq_len + 1):
            overlap_targets = _supervised_targets_overlap(tokens, chunk_size)
            boundary_targets = _supervised_targets_boundary(tokens, chunk_size)
            assert sorted(overlap_targets) == expected
            assert sorted(boundary_targets) == expected


def test_chunk_schedule_permutations_preserve_supervision_set() -> None:
    tokens = torch.arange(17).view(1, 17)
    baseline_overlap = sorted(_supervised_targets_overlap(tokens, chunk_size=3))
    baseline_boundary = sorted(_supervised_targets_boundary(tokens, chunk_size=3))
    for chunk_size in (1, 2, 4, 5, 8, 16):
        assert (
            sorted(_supervised_targets_overlap(tokens, chunk_size=chunk_size))
            == baseline_overlap
        )
        assert (
            sorted(_supervised_targets_boundary(tokens, chunk_size=chunk_size))
            == baseline_boundary
        )


def test_per_layer_teach_with_boundary_chunks_runs_update_path() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    state = model.init_fast_state()
    cache = model.init_attention_cache()

    chunk, next_tokens, finalize_updates = next(_iter_online_boundary_chunks(tokens, chunk_size=4))
    logits, _pre, block_outputs, cache = model.forward_with_block_outputs(
        chunk,
        fast_state=state,
        attention_cache=cache,
        return_attention_cache=True,
    )
    assert next_tokens is not None
    targets = torch.cat([chunk[:, 1:], next_tokens.unsqueeze(1)], dim=1)
    loss = F.cross_entropy(
        logits[:, : targets.size(1), :].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    teach_signals = _compute_layer_teach_signals(loss, block_outputs)
    loss.backward()
    with torch.no_grad():
        _ = model(
            chunk,
            teach_signals=teach_signals,
            fast_state=state,
            attention_cache=cache,
            finalize_updates=finalize_updates,
        )
    metrics = model.pop_update_metrics()
    assert "layer0.cms.cms_fast.gate_hit" in metrics
