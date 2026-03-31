import torch

from nested_learning.optim.m3 import M3


def test_m3_updates_and_slow_momentum() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = M3(
        [param],
        lr=0.1,
        beta1=0.9,
        beta2=0.9,
        beta3=0.5,
        alpha=1.0,
        ns_steps=1,
        slow_chunk=2,
        eps=1e-6,
    )
    param.grad = torch.ones_like(param)
    opt.step()
    first = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    assert not torch.allclose(first, param)
    assert torch.any(state["o2"] != 0)


def test_m3_step_matches_reference_denominator_for_first_update() -> None:
    # With ns_steps=0 and 1D params, orthogonalization is identity.
    # This pins the exact first-step denominator/scaling behavior.
    param = torch.nn.Parameter(torch.tensor([2.0]))
    grad = torch.tensor([3.0])
    opt = M3(
        [param],
        lr=0.1,
        beta1=0.5,
        beta2=0.25,
        beta3=0.0,
        alpha=0.0,
        ns_steps=0,
        slow_chunk=100,
        eps=1e-6,
        weight_decay=0.0,
    )
    param.grad = grad.clone()
    opt.step()

    m1 = 0.5 * grad
    v = 0.25 * grad * grad
    expected_update = m1 / (torch.sqrt(v) + 1e-6)
    expected_param = torch.tensor([2.0]) - 0.1 * expected_update
    assert torch.allclose(param.detach(), expected_param, atol=1e-6, rtol=1e-6)


def test_m3_two_steps_match_closed_form_without_slow_momentum() -> None:
    # 1D + ns_steps=0 makes orthogonalization identity, so we can pin exact numerics.
    param = torch.nn.Parameter(torch.tensor([1.5]))
    grad = torch.tensor([2.0])
    lr = 0.05
    beta1 = 0.2
    beta2 = 0.3
    eps = 1e-6
    opt = M3(
        [param],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        beta3=0.0,
        alpha=0.0,
        ns_steps=0,
        slow_chunk=1000,
        eps=eps,
        weight_decay=0.0,
    )
    param.grad = grad.clone()
    opt.step()
    param.grad = grad.clone()
    opt.step()

    g = grad
    m1_1 = beta1 * g
    v_1 = beta2 * g * g
    p1 = torch.tensor([1.5]) - lr * (m1_1 / (torch.sqrt(v_1) + eps))

    m1_2 = m1_1 + beta1 * g
    v_2 = v_1 + beta2 * g * g
    p2 = p1 - lr * (m1_2 / (torch.sqrt(v_2) + eps))
    assert torch.allclose(param.detach(), p2, atol=1e-6, rtol=1e-6)


def test_m3_weight_decay_is_included_in_reference_step() -> None:
    param = torch.nn.Parameter(torch.tensor([2.0]))
    grad = torch.tensor([3.0])
    lr = 0.1
    wd = 0.4
    eps = 1e-6
    opt = M3(
        [param],
        lr=lr,
        beta1=0.5,
        beta2=0.25,
        beta3=0.0,
        alpha=0.0,
        ns_steps=0,
        slow_chunk=100,
        eps=eps,
        weight_decay=wd,
    )
    param.grad = grad.clone()
    opt.step()

    g_eff = grad + wd * torch.tensor([2.0])
    m1 = 0.5 * g_eff
    v = 0.25 * g_eff * g_eff
    expected = torch.tensor([2.0]) - lr * (m1 / (torch.sqrt(v) + eps))
    assert torch.allclose(param.detach(), expected, atol=1e-6, rtol=1e-6)


def test_m3_slow_buffer_resets_and_o2_updates_on_chunk_boundary() -> None:
    param = torch.nn.Parameter(torch.ones(2))
    opt = M3(
        [param],
        lr=0.01,
        beta1=0.0,
        beta2=0.0,
        beta3=0.5,
        alpha=1.0,
        ns_steps=0,
        slow_chunk=2,
        eps=1e-6,
    )
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    assert torch.allclose(state["slow_buffer"], torch.ones_like(param))
    assert torch.all(state["o2"] == 0)

    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    # Boundary step consumes accumulated slow buffer into m2/o2 and clears it.
    assert torch.allclose(state["slow_buffer"], torch.zeros_like(param))
    assert torch.any(state["o2"] != 0)
