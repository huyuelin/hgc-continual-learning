import random

import numpy as np
import torch

from nested_learning.training import _seed_everything


def test_seed_everything_reproducible_python_numpy_torch() -> None:
    _seed_everything(1234, deterministic=False)
    a = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )
    _seed_everything(1234, deterministic=False)
    b = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )
    assert a == b


def test_seed_everything_toggles_deterministic_algorithms() -> None:
    prev_flag = torch.are_deterministic_algorithms_enabled()
    has_cudnn = hasattr(torch.backends, "cudnn")
    prev_benchmark = None
    prev_deterministic = None
    if has_cudnn:
        prev_benchmark = bool(torch.backends.cudnn.benchmark)  # type: ignore[attr-defined]
        prev_deterministic = bool(torch.backends.cudnn.deterministic)  # type: ignore[attr-defined]
    try:
        _seed_everything(1, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        _seed_everything(1, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(prev_flag)
        if has_cudnn and prev_benchmark is not None and prev_deterministic is not None:
            torch.backends.cudnn.benchmark = prev_benchmark  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = prev_deterministic  # type: ignore[attr-defined]
