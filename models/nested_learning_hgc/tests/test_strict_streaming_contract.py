import pytest
from omegaconf import OmegaConf

from nested_learning.training import (
    _check_online_supervised_pairs,
    _resolve_algorithm_mode,
    _validate_algorithm_mode_constraints,
    _validate_online_chunking_constraints,
    _validate_online_update_fast_state_semantics,
    _validate_paper_auditing_variant,
)


def test_strict_streaming_contract_rejects_non_paper_variant() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"strict_streaming_contract": True},
            "model": {"block_variant": "hope_hybrid"},
        }
    )
    with pytest.raises(RuntimeError, match="paper-defined HOPE variant"):
        _validate_paper_auditing_variant(cfg)


def test_non_strict_streaming_contract_warns_for_non_paper_variant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = OmegaConf.create(
        {
            "train": {"strict_streaming_contract": False},
            "model": {"block_variant": "hope_hybrid"},
        }
    )
    _validate_paper_auditing_variant(cfg)
    out = capsys.readouterr().out
    assert "warning_code" in out
    assert "non_paper_variant" in out


def test_strict_streaming_contract_allows_paper_defined_variants() -> None:
    for variant in ("hope_attention", "hope_selfmod"):
        cfg = OmegaConf.create(
            {
                "train": {"strict_streaming_contract": True},
                "model": {"block_variant": variant},
            }
        )
        _validate_paper_auditing_variant(cfg)


def test_online_updates_without_fast_state_warns_when_not_strict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "online_updates": True,
                "use_fast_state": False,
                "strict_streaming_contract": False,
                "fail_if_paper_faithful_disabled": False,
            }
        }
    )
    _validate_online_update_fast_state_semantics(cfg)
    out = capsys.readouterr().out
    assert "warning_code" in out
    assert "online_updates_without_fast_state" in out


def test_online_updates_without_fast_state_fails_in_strict_mode() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "online_updates": True,
                "use_fast_state": False,
                "strict_streaming_contract": True,
                "fail_if_paper_faithful_disabled": False,
            }
        }
    )
    with pytest.raises(RuntimeError, match="train.online_updates=true"):
        _validate_online_update_fast_state_semantics(cfg)


def test_online_supervised_pairs_mismatch_warns_when_not_strict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _check_online_supervised_pairs(strict=False, observed_pairs=3, seq_len=8)
    out = capsys.readouterr().out
    assert "online_supervision_mismatch" in out


def test_online_supervised_pairs_mismatch_fails_in_strict_mode() -> None:
    with pytest.raises(RuntimeError, match="online chunk supervision mismatch"):
        _check_online_supervised_pairs(strict=True, observed_pairs=3, seq_len=8)


def test_algorithm_mode_defaults_to_two_pass_stopgrad_updates() -> None:
    cfg = OmegaConf.create({"train": {}})
    assert _resolve_algorithm_mode(cfg) == "two_pass_stopgrad_updates"


def test_algorithm_mode_rejects_unknown_values() -> None:
    cfg = OmegaConf.create({"train": {"algorithm_mode": "unknown"}})
    with pytest.raises(RuntimeError, match="Unsupported train.algorithm_mode"):
        _resolve_algorithm_mode(cfg)


def test_algorithm_mode_accepts_boundary_state_mode_name() -> None:
    cfg = OmegaConf.create({"train": {"algorithm_mode": "boundary_state_grad_through_write"}})
    assert _resolve_algorithm_mode(cfg) == "boundary_state_grad_through_write"


def test_boundary_state_mode_requires_online_per_layer_and_fast_state() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "algorithm_mode": "boundary_state_grad_through_write",
                "online_updates": False,
                "per_layer_teach_signal": False,
                "use_fast_state": False,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_updates=true"):
        _validate_algorithm_mode_constraints(
            cfg,
            algorithm_mode="boundary_state_grad_through_write",
            distributed=False,
        )


def test_boundary_state_mode_rejects_distributed() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "algorithm_mode": "boundary_state_grad_through_write",
                "online_updates": True,
                "per_layer_teach_signal": True,
                "use_fast_state": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="not supported in DDP"):
        _validate_algorithm_mode_constraints(
            cfg,
            algorithm_mode="boundary_state_grad_through_write",
            distributed=True,
        )


def test_boundary_state_mode_emits_experimental_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "algorithm_mode": "boundary_state_grad_through_write",
                "online_updates": True,
                "per_layer_teach_signal": True,
                "use_fast_state": True,
                "online_boundary_targets": True,
                "online_carry_attention_cache": True,
            }
        }
    )
    _validate_algorithm_mode_constraints(
        cfg,
        algorithm_mode="boundary_state_grad_through_write",
        distributed=False,
    )
    out = capsys.readouterr().out
    assert "experimental_boundary_state_mode" in out


def test_online_cache_requires_boundary_targets() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "online_updates": True,
                "online_boundary_targets": False,
                "online_carry_attention_cache": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_boundary_targets=true"):
        _validate_online_chunking_constraints(cfg)


def test_online_cache_requires_online_updates() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "online_updates": False,
                "online_boundary_targets": True,
                "online_carry_attention_cache": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_updates=true"):
        _validate_online_chunking_constraints(cfg)
