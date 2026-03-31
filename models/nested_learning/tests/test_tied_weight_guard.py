import pytest
import torch
from omegaconf import OmegaConf

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _validate_tied_lm_head_for_paper_auditing


def _tiny_hope_model() -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=32,
        dim=8,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_attention",
    )
    return HOPEModel(cfg)


def test_paper_auditing_guard_accepts_tied_weights() -> None:
    model = _tiny_hope_model()
    cfg = OmegaConf.create({"train": {"strict_streaming_contract": True}})
    _validate_tied_lm_head_for_paper_auditing(cfg, model)


def test_paper_auditing_guard_rejects_untied_weights() -> None:
    model = _tiny_hope_model()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    cfg = OmegaConf.create({"train": {"strict_streaming_contract": True}})
    with pytest.raises(RuntimeError, match="requires tied LM head"):
        _validate_tied_lm_head_for_paper_auditing(cfg, model)
