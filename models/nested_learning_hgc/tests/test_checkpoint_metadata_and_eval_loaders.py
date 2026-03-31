from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nested_learning.training import (
    build_model_from_cfg,
    verify_checkpoint_integrity,
    write_checkpoint_metadata,
)


def _tiny_cfg():
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
            "train": {
                "algorithm_mode": "boundary_state_grad_through_write",
                "online_updates": True,
                "online_boundary_targets": True,
                "online_carry_attention_cache": True,
                "use_fast_state": True,
            },
            "data": {
                "tokenizer_path": None,
            },
        }
    )


def _load_script_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_metadata_includes_algorithm_and_online_flags(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    ckpt_path = tmp_path / "step_000001.pt"
    torch.save({"model": {}}, ckpt_path)
    write_checkpoint_metadata(cfg, ckpt_path, step=1)

    metadata_path = ckpt_path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["algorithm_mode"] == "boundary_state_grad_through_write"
    assert metadata["online_updates"] is True
    assert metadata["online_boundary_targets"] is True
    assert metadata["online_carry_attention_cache"] is True
    assert metadata["use_fast_state"] is True
    verified = verify_checkpoint_integrity(ckpt_path)
    assert verified["algorithm_mode"] == "boundary_state_grad_through_write"


def test_eval_loaders_accept_boundary_state_checkpoint(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    model = build_model_from_cfg(cfg.model)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))

    root = Path(__file__).resolve().parents[1]
    script_paths = (
        root / "scripts/eval/zeroshot.py",
        root / "scripts/eval/niah.py",
        root / "scripts/eval/passkey.py",
        root / "scripts/eval/pg19_perplexity.py",
    )
    for script_path in script_paths:
        module = _load_script_module(script_path)
        loaded = module.load_model(config_path, ckpt_path, torch.device("cpu"))
        tokens = torch.randint(0, 64, (1, 6))
        logits = loaded(tokens)
        assert logits.shape == (1, 6, 64)
