import pytest
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.transformer import TransformerBlock, TransformerBlockConfig


def _build_transformer_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=2,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="transformer",
        qk_l2_norm=False,
    )
    model = HOPEModel(cfg).eval()
    return model


def test_attention_cache_chunked_logits_match_full_logits() -> None:
    torch.manual_seed(0)
    model = _build_transformer_model()
    tokens = torch.randint(0, 64, (1, 11))
    with torch.no_grad():
        full = model(tokens)
        cache = model.init_attention_cache()
        pieces = []
        for start, end in ((0, 3), (3, 7), (7, 11)):
            chunk = tokens[:, start:end]
            logits_chunk, cache = model(
                chunk,
                attention_cache=cache,
                return_attention_cache=True,
            )
            pieces.append(logits_chunk)
        stitched = torch.cat(pieces, dim=1)
    assert torch.allclose(full, stitched, atol=1e-5, rtol=1e-5)


def test_attention_cache_reset_changes_continuation_state() -> None:
    torch.manual_seed(1)
    model = _build_transformer_model()
    tokens = torch.randint(0, 64, (1, 8))
    prefix = tokens[:, :4]
    suffix = tokens[:, 4:]
    with torch.no_grad():
        cache = model.init_attention_cache()
        _, cache = model(prefix, attention_cache=cache, return_attention_cache=True)
        carried, _ = model(suffix, attention_cache=cache, return_attention_cache=True)
        fresh = model(suffix)
    # Carrying cache should generally differ from a fresh-only suffix pass.
    assert not torch.allclose(carried, fresh)


def test_transformer_block_rejects_kv_cache_with_local_conv() -> None:
    block = TransformerBlock(
        TransformerBlockConfig(
            dim=16,
            heads=4,
            local_conv_window=4,
        )
    )
    x = torch.randn(1, 3, 16)
    # Build a minimal cache tensor matching [B, H, T, D].
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    from nested_learning.fast_state import AttentionKVCache

    with pytest.raises(RuntimeError, match="local_conv_window"):
        block(x, attention_cache=AttentionKVCache(key=k, value=v))
