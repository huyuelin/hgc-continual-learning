import torch

from nested_learning.training import _compute_surprise_override


def _entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    ent = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
    return float(ent.item())


def test_logit_entropy_surprise_uses_boundary_target_step_when_present() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 13)
    tokens = torch.randint(0, 13, (1, 4))
    value = _compute_surprise_override(
        "logit_entropy",
        logits=logits,
        tokens=tokens,
        loss=torch.tensor(1.0),
        next_tokens=torch.randint(0, 13, (1,)),
    )
    assert value is not None
    assert abs(value - _entropy(logits[:, :4])) < 1e-8


def test_logit_entropy_surprise_default_excludes_last_unsupervised_step() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 11)
    tokens = torch.randint(0, 11, (1, 5))
    value = _compute_surprise_override(
        "logit_entropy",
        logits=logits,
        tokens=tokens,
        loss=torch.tensor(1.0),
    )
    assert value is not None
    assert abs(value - _entropy(logits[:, :4])) < 1e-8


def test_logit_entropy_surprise_returns_none_when_no_supervised_steps() -> None:
    logits = torch.randn(1, 1, 7)
    tokens = torch.randint(0, 7, (1, 1))
    value = _compute_surprise_override(
        "logit_entropy",
        logits=logits,
        tokens=tokens,
        loss=torch.tensor(1.0),
    )
    assert value is None
