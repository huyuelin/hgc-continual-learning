import pytest
from omegaconf import OmegaConf

from nested_learning.training import _validate_distributed_config


def test_fail_if_paper_faithful_disabled_blocks_ddp_per_layer_teach() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": True,
                "online_updates": False,
            }
        }
    )
    with pytest.raises(RuntimeError, match="per_layer_teach_signal"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_blocks_ddp_online_updates() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": False,
                "online_updates": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_updates"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_allows_single_process() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": True,
                "online_updates": True,
            }
        }
    )
    _validate_distributed_config(cfg, distributed=False)


def test_strict_streaming_contract_blocks_ddp_online_features() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "strict_streaming_contract": True,
                "fail_if_paper_faithful_disabled": False,
                "per_layer_teach_signal": True,
                "online_updates": False,
            }
        }
    )
    with pytest.raises(RuntimeError, match="strict_streaming_contract"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_blocks_ddp_boundary_targets() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": False,
                "online_updates": False,
                "online_boundary_targets": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_boundary_targets"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_blocks_ddp_attention_cache_carry() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": False,
                "online_updates": False,
                "online_carry_attention_cache": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_carry_attention_cache"):
        _validate_distributed_config(cfg, distributed=True)
