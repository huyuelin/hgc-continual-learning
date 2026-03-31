import pytest
from omegaconf import OmegaConf

from nested_learning.training import _validate_fast_state_batch_semantics


def test_fast_state_batch_semantics_raises_when_strict() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 2},
        }
    )
    with pytest.raises(RuntimeError, match="fast state"):
        _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_batch1() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 1},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_warns_with_structured_payload_when_not_strict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "strict_streaming_contract": False},
            "data": {"batch_size": 3},
        }
    )
    _validate_fast_state_batch_semantics(cfg)
    captured = capsys.readouterr()
    assert "warning_code" in captured.out
    assert "shared_fast_state_batch" in captured.out
