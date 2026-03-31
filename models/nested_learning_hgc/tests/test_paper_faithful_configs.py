from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from nested_learning.training import build_model_from_cfg


def _compose_config(name: str, overrides: list[str] | None = None):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=name, overrides=overrides or [])


def test_pilot_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_paper_faithful")
    assert cfg.model.block_variant == "hope_attention"
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.strict_streaming_contract is True
    assert cfg.train.online_updates is True
    assert cfg.train.online_boundary_targets is True
    assert cfg.train.online_carry_attention_cache is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.train.algorithm_mode == "two_pass_stopgrad_updates"
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)


def test_pilot_selfmod_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_selfmod_paper_faithful")
    assert cfg.model.block_variant == "hope_selfmod"
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.model.self_mod_use_skip is False
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.strict_streaming_contract is True
    assert cfg.train.online_updates is True
    assert cfg.train.online_boundary_targets is True
    assert cfg.train.online_carry_attention_cache is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)


def test_paper_faithful_variants_are_explicitly_paper_defined() -> None:
    attention_cfg = _compose_config("pilot_paper_faithful")
    selfmod_cfg = _compose_config("pilot_selfmod_paper_faithful")
    allowed = {"hope_attention", "hope_selfmod"}
    assert attention_cfg.model.block_variant in allowed
    assert selfmod_cfg.model.block_variant in allowed


def test_pilot_paper_faithful_override_to_boundary_state_mode_applies() -> None:
    cfg = _compose_config(
        "pilot_paper_faithful",
        overrides=["train.algorithm_mode=boundary_state_grad_through_write"],
    )
    assert cfg.train.algorithm_mode == "boundary_state_grad_through_write"


def test_pilot_paper_faithful_never_implicitly_falls_back_to_stopgrad() -> None:
    cfg = _compose_config(
        "pilot_paper_faithful",
        overrides=["train.algorithm_mode=boundary_state_grad_through_write"],
    )
    assert cfg.train.algorithm_mode != "two_pass_stopgrad_updates"
