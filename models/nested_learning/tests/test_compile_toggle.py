from __future__ import annotations

import torch
from omegaconf import OmegaConf

from nested_learning.training import run_training_loop


def _tiny_compile_cfg():
    return OmegaConf.create(
        {
            "model": {
                "type": "hope",
                "vocab_size": 64,
                "dim": 16,
                "num_layers": 1,
                "heads": 2,
                "block_variant": "hope_attention",
                "titan_level": {"name": "titan", "update_period": 1},
                "cms_levels": [{"name": "cms_fast", "update_period": 1}],
            },
            "data": {
                "source": "synthetic",
                "vocab_size": 64,
                "seq_len": 8,
                "dataset_size": 8,
                "batch_size": 1,
                "num_workers": 0,
            },
            "optim": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "fused": False,
                "param_policy": "all",
            },
            "train": {
                "steps": 1,
                "log_interval": 1,
                "seed": 0,
                "deterministic": True,
                "algorithm_mode": "two_pass_stopgrad_updates",
                "per_layer_teach_signal": True,
                "online_updates": True,
                "online_chunk_size": 2,
                "online_boundary_targets": True,
                "online_carry_attention_cache": True,
                "use_fast_state": True,
                "strict_streaming_contract": False,
                "fail_if_paper_faithful_disabled": False,
                "mixed_precision": {"enabled": False, "dtype": "bf16"},
                "compile": {"enable": True, "strict": False},
                "checkpoint": {"enable": False},
            },
            "logging": {"enabled": False},
        }
    )


def test_compile_toggle_smoke_does_not_crash() -> None:
    cfg = _tiny_compile_cfg()
    metrics = run_training_loop(cfg, device=torch.device("cpu"), distributed=False)
    assert "loss" in metrics
