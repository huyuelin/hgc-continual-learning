from __future__ import annotations

import torch
from omegaconf import OmegaConf

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _log_run_features


class _CaptureLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[dict[str, object], int]] = []

    def log(self, data: dict[str, object], step: int) -> None:
        self.entries.append((data, step))


def _tiny_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_attention",
    )
    return HOPEModel(cfg)


def _tiny_cfg(algorithm_mode: str) -> object:
    return OmegaConf.create(
        {
            "train": {
                "mixed_precision": {"enabled": False, "dtype": "bf16"},
                "compile": {"enable": False, "mode": "default"},
                "strict_streaming_contract": True,
                "online_updates": True,
                "online_boundary_targets": True,
                "online_carry_attention_cache": True,
                "use_fast_state": True,
                "algorithm_mode": algorithm_mode,
            },
            "optim": {"param_policy": "all"},
        }
    )


def test_run_features_reports_stopgrad_mode_flag() -> None:
    model = _tiny_model()
    cfg = _tiny_cfg("two_pass_stopgrad_updates")
    logger = _CaptureLogger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _log_run_features(logger, model, cfg, optimizer, torch.device("cpu"))
    payload, step = logger.entries[-1]
    assert step == -1
    assert payload["train.algorithm_mode"] == "two_pass_stopgrad_updates"
    assert payload["train.backprop_through_online_writes"] is False


def test_run_features_reports_boundary_state_mode_flag() -> None:
    model = _tiny_model()
    cfg = _tiny_cfg("boundary_state_grad_through_write")
    logger = _CaptureLogger()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _log_run_features(logger, model, cfg, optimizer, torch.device("cpu"))
    payload, _ = logger.entries[-1]
    assert payload["train.algorithm_mode"] == "boundary_state_grad_through_write"
    assert payload["train.backprop_through_online_writes"] is True
