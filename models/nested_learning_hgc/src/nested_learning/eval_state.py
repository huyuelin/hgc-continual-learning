from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EvalStreamingState:
    fast_state: object | None = None
    attention_cache: object | None = None


def parse_eval_state_mode(mode: str) -> bool:
    """
    Returns True when eval state should be carried across samples.
    """
    normalized = str(mode).strip().lower()
    if normalized in {"reset", "reset_per_sample", "isolated"}:
        return False
    if normalized in {"carry", "carry_across_samples", "stream"}:
        return True
    raise ValueError(
        "Unsupported eval_state_mode={!r}; expected one of "
        "['reset_per_sample', 'carry_across_samples']".format(mode)
    )


def init_eval_streaming_state(
    model,
    *,
    use_fast_state: bool,
    use_attention_cache: bool,
) -> EvalStreamingState:
    state = EvalStreamingState()
    if use_fast_state:
        init_fast_state = getattr(model, "init_fast_state", None)
        if not callable(init_fast_state):
            raise RuntimeError(
                "Requested fast-state eval mode, but model.init_fast_state() is missing"
            )
        state.fast_state = init_fast_state()
    if use_attention_cache:
        init_attention_cache = getattr(model, "init_attention_cache", None)
        if not callable(init_attention_cache):
            raise RuntimeError(
                "Requested attention-cache eval mode, but model.init_attention_cache() is missing"
            )
        state.attention_cache = init_attention_cache()
    return state


def forward_with_eval_state(
    model,
    tokens: torch.Tensor,
    *,
    state: EvalStreamingState | None = None,
) -> tuple[torch.Tensor, EvalStreamingState | None]:
    if state is None:
        return model(tokens), None
    if state.attention_cache is not None:
        logits, next_cache = model(
            tokens,
            fast_state=state.fast_state,
            attention_cache=state.attention_cache,
            return_attention_cache=True,
        )
        state.attention_cache = next_cache
        return logits, state
    if state.fast_state is not None:
        return model(tokens, fast_state=state.fast_state), state
    return model(tokens), state
