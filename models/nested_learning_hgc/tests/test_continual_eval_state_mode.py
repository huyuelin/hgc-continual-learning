import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig
from nested_learning.model import HOPEModel, ModelConfig


def _load_evaluate_segment():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "eval" / "continual.py"
    spec = importlib.util.spec_from_file_location("tests.continual_eval_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_segment


evaluate_segment = _load_evaluate_segment()


class _TokenDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [torch.randint(0, 32, (12,)) for _ in range(6)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def _build_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="transformer",
    )
    return HOPEModel(cfg)


def test_continual_eval_state_modes_run_without_errors() -> None:
    torch.manual_seed(0)
    model = _build_model()
    dataset = _TokenDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    mem_cfg = MemorizeConfig(enabled=False)

    base_reset, mem_reset, _stats_reset = evaluate_segment(
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
        memorize_cfg=mem_cfg,
        eval_state_mode="reset_per_sample",
        eval_use_fast_state=False,
        eval_use_attention_cache=True,
    )
    base_carry, mem_carry, _stats_carry = evaluate_segment(
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
        memorize_cfg=mem_cfg,
        eval_state_mode="carry_across_samples",
        eval_use_fast_state=False,
        eval_use_attention_cache=True,
    )

    assert torch.isfinite(torch.tensor(base_reset))
    assert torch.isfinite(torch.tensor(mem_reset))
    assert torch.isfinite(torch.tensor(base_carry))
    assert torch.isfinite(torch.tensor(mem_carry))
