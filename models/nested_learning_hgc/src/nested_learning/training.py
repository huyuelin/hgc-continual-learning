from __future__ import annotations

import base64
import json
import os
import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterator, Protocol, Tuple, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from .data import (
    MixtureShardDataset,
    ShardSourceConfig,
    SyntheticTextConfig,
    SyntheticTextDataset,
    TokenShardDataset,
    collate_batch,
)
from .levels import LevelSpec
from .logging_utils import BaseLogger, NullLogger, init_logger
from .model import HOPEModel, ModelConfig
from .optim.m3 import M3
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def unwrap_config(cfg: DictConfig) -> DictConfig:
    """Hydra can wrap grouped configs (e.g., hope/pilot) under the group name."""
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cast(DictConfig, cfg.hope)
    if "ablations" in cfg:
        return cast(DictConfig, cfg.ablations)
    return cfg


def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    model_type = model_cfg.get("type", "hope")
    optimizer_cfg: Dict[str, dict] = {}
    if "optimizers" in model_cfg:
        optimizer_cfg = cast(
            Dict[str, dict],
            OmegaConf.to_container(model_cfg.optimizers, resolve=True),
        )
    teach_scale = model_cfg.get("teach_scale", 1.0)
    teach_clip = model_cfg.get("teach_clip", 0.0)
    teach_schedule: Dict[str, float] = {}
    if "teach_schedule" in model_cfg:
        teach_schedule = cast(
            Dict[str, float],
            OmegaConf.to_container(model_cfg.teach_schedule, resolve=True),
        )
    qk_l2_norm = bool(model_cfg.get("qk_l2_norm", False))
    local_conv_window_raw = model_cfg.get("local_conv_window")
    local_conv_window = None if local_conv_window_raw is None else int(local_conv_window_raw)
    surprise_threshold_raw = model_cfg.get("surprise_threshold")
    surprise_threshold = (
        None if surprise_threshold_raw is None else float(surprise_threshold_raw)
    )
    surprise_metric = str(model_cfg.get("surprise_metric", "l2"))
    cms_use_layernorm = bool(model_cfg.get("cms_use_layernorm", True))
    if model_type == "titan":
        titan_spec = LevelSpec(**model_cfg.titan_level)
        titan_cfg = TitanOnlyModelConfig(
            vocab_size=model_cfg.vocab_size,
            dim=model_cfg.dim,
            num_layers=model_cfg.num_layers,
            heads=model_cfg.heads,
            titan_level=titan_spec,
            optimizers=optimizer_cfg,
            teach_scale=teach_scale,
            teach_clip=teach_clip,
            teach_schedule=teach_schedule,
            qk_l2_norm=qk_l2_norm,
            local_conv_window=local_conv_window,
            surprise_threshold=surprise_threshold,
            surprise_metric=surprise_metric,
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
            self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        )
        return TitanOnlyModel(titan_cfg)
    titan_spec = LevelSpec(**model_cfg.titan_level)
    cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]
    self_mod_chunk_size_memory_raw = model_cfg.get("self_mod_chunk_size_memory")
    self_mod_chunk_size_memory = (
        None if self_mod_chunk_size_memory_raw is None else int(self_mod_chunk_size_memory_raw)
    )
    self_mod_local_conv_window_raw = model_cfg.get("self_mod_local_conv_window", 4)
    self_mod_local_conv_window = (
        None if self_mod_local_conv_window_raw is None else int(self_mod_local_conv_window_raw)
    )
    hope_cfg = ModelConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        titan_level=titan_spec,
        cms_levels=cms_specs,
        cms_flush_partial_at_end=bool(model_cfg.get("cms_flush_partial_at_end", False)),
        cms_use_layernorm=cms_use_layernorm,
        optimizers=optimizer_cfg,
        teach_scale=teach_scale,
        teach_clip=teach_clip,
        teach_schedule=teach_schedule,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        surprise_threshold=surprise_threshold,
        surprise_metric=surprise_metric,
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        qk_l2_norm=qk_l2_norm,
        local_conv_window=local_conv_window,
        self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
        self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        self_mod_chunk_size=int(model_cfg.get("self_mod_chunk_size", 1)),
        self_mod_chunk_size_memory=self_mod_chunk_size_memory,
        self_mod_objective=str(model_cfg.get("self_mod_objective", "l2")),
        self_mod_stopgrad_vhat=bool(model_cfg.get("self_mod_stopgrad_vhat", True)),
        self_mod_use_rank1_precond=bool(model_cfg.get("self_mod_use_rank1_precond", True)),
        self_mod_use_alpha=bool(model_cfg.get("self_mod_use_alpha", True)),
        self_mod_use_skip=bool(model_cfg.get("self_mod_use_skip", True)),
        self_mod_momentum=float(model_cfg.get("self_mod_momentum", 0.0)),
        self_mod_adaptive_q=bool(model_cfg.get("self_mod_adaptive_q", False)),
        self_mod_local_conv_window=self_mod_local_conv_window,
        transformer_mlp_hidden_multiplier=int(
            model_cfg.get("transformer_mlp_hidden_multiplier", 4)
        ),
        transformer_activation=str(model_cfg.get("transformer_activation", "gelu")),
        block_variant=str(model_cfg.get("block_variant", "hope_hybrid")),
    )
    return HOPEModel(hope_cfg)


def build_dataloader(
    data_cfg: DictConfig,
    *,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    seed: int | None = None,
) -> Tuple[DataLoader, DistributedSampler | None]:
    dataset = _build_dataset(data_cfg)
    use_sampler = distributed and not isinstance(dataset, IterableDataset)
    if use_sampler:
        assert dist_ctx is not None
        sampler: DistributedSampler | None = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    if isinstance(dataset, IterableDataset):
        shuffle = False
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _make_worker_init_fn(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return dataloader, sampler


def _build_dataset(data_cfg: DictConfig):
    source = data_cfg.source
    if source == "synthetic":
        synth_cfg = SyntheticTextConfig(
            vocab_size=data_cfg.vocab_size,
            seq_len=data_cfg.seq_len,
            dataset_size=data_cfg.dataset_size,
        )
        return SyntheticTextDataset(synth_cfg)
    if source == "shards":
        shard_dir = data_cfg.shards_dir
        return TokenShardDataset(shard_dir)
    if source == "mixture":
        mixture_cfg = data_cfg.mixture
        sources = [
            ShardSourceConfig(
                name=entry.name,
                shards_dir=entry.shards_dir,
                weight=entry.weight,
            )
            for entry in mixture_cfg.sources
        ]
        samples_per_epoch = mixture_cfg.samples_per_epoch
        seed = mixture_cfg.get("seed", 0)
        return MixtureShardDataset(
            sources,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    msg = f"Unsupported data source {source}"
    raise ValueError(msg)


def compute_teach_signal(
    model: "_HasLMHead",
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    next_tokens: torch.Tensor | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Approximate dL/dh where h is the hidden state before the LM head.

    This matches the gradient of mean next-token CE.

    By default this corresponds to CE(logits[:, :-1], tokens[:, 1:]).
    If `next_tokens` is provided, the final logit position is also supervised
    against that boundary target (used for chunked streaming boundaries).

    If ignore_index is provided, targets equal to ignore_index are masked out and
    the mean reduction denominator becomes the number of active targets (matching
    PyTorch CE semantics).
    """
    logits_detached = logits.detach()
    probs = torch.softmax(logits_detached, dim=-1)
    residual = probs.clone()
    batch_size, seq_len, _ = residual.shape

    targets = torch.zeros(
        batch_size,
        seq_len,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    active = torch.zeros(
        batch_size,
        seq_len,
        device=tokens.device,
        dtype=torch.bool,
    )
    if seq_len > 1:
        targets[:, :-1] = tokens[:, 1:]
        active[:, :-1] = True
    if next_tokens is not None:
        if next_tokens.ndim == 2 and next_tokens.size(1) == 1:
            next_targets = next_tokens[:, 0]
        elif next_tokens.ndim == 1:
            next_targets = next_tokens
        else:
            raise ValueError("next_tokens must have shape [B] or [B, 1]")
        if next_targets.size(0) != batch_size:
            raise ValueError("next_tokens batch dimension must match tokens batch dimension")
        targets[:, -1] = next_targets.to(device=tokens.device, dtype=tokens.dtype)
        active[:, -1] = True
    if ignore_index is not None:
        active = active & (targets != ignore_index)

    active_f = active.to(dtype=residual.dtype)
    residual.mul_(active_f.unsqueeze(-1))
    safe_targets = torch.where(active, targets, torch.zeros_like(targets))
    src = -active_f.unsqueeze(-1)
    residual.scatter_add_(-1, safe_targets.unsqueeze(-1), src)
    denom: torch.Tensor = active_f.sum().clamp(min=1.0)
    residual = residual / denom

    head_weight = model.lm_head.weight.detach()
    if head_weight.dtype != residual.dtype:
        head_weight = head_weight.to(dtype=residual.dtype)
    grad = residual @ head_weight
    return grad


def _compute_layer_teach_signals(
    loss: torch.Tensor,
    block_outputs: list[torch.Tensor],
    *,
    detach: bool = True,
    create_graph: bool = False,
) -> list[torch.Tensor]:
    grads = torch.autograd.grad(
        loss,
        block_outputs,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=False,
    )
    if detach:
        return [g.detach() for g in grads]
    return list(grads)


def _compute_surprise_override(
    metric: str,
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss: torch.Tensor,
    next_tokens: torch.Tensor | None = None,
) -> float | None:
    normalized = str(metric).strip().lower()
    if normalized == "loss":
        return float(loss.detach().item())
    if normalized == "logit_entropy":
        supervised_steps = int(tokens.size(1) - 1 + (0 if next_tokens is None else 1))
        if supervised_steps <= 0:
            return None
        logits_detached = logits[:, :supervised_steps].detach().float()
        probs = torch.softmax(logits_detached, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
        return float(entropy.item())
    return None


def _infer_online_chunk_size(model: HOPEModel) -> int | None:
    min_period: int | None = None
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        cfg = getattr(block, "config", None)
        levels = getattr(cfg, "cms_levels", None)
        if not levels:
            continue
        for spec in levels:
            period = int(spec.update_period)
            if period <= 0:
                continue
            min_period = period if min_period is None else min(min_period, period)
    return min_period


def _iter_online_token_chunks(
    tokens: torch.Tensor, *, chunk_size: int
) -> Iterator[tuple[torch.Tensor, bool]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    seq_len = tokens.size(1)
    for core_start in range(0, seq_len, chunk_size):
        core_end = min(core_start + chunk_size, seq_len)
        if core_end <= core_start:
            continue
        # Carry one-token overlap so chunk boundaries still include next-token supervision.
        chunk_start = core_start - 1 if core_start > 0 else core_start
        chunk_tokens = tokens[:, chunk_start:core_end]
        finalize_updates = core_end >= seq_len
        yield chunk_tokens, finalize_updates


def _iter_online_boundary_chunks(
    tokens: torch.Tensor, *, chunk_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, bool]]:
    """
    Yield non-overlapping chunks plus the boundary target token for chunk end.

    This enables exact boundary supervision without one-token overlap.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    seq_len = tokens.size(1)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        if end <= start:
            continue
        next_tokens = None
        if end < seq_len:
            next_tokens = tokens[:, end]
        finalize_updates = end >= seq_len
        yield tokens[:, start:end], next_tokens, finalize_updates


class _HasLMHead(Protocol):
    lm_head: torch.nn.Linear


def _checksum_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_save_checkpoint(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    step_offset: int = 0,
) -> None:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False):
        return
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        return
    save_interval = ckpt_cfg.get("save_interval", total_steps)
    save_last = ckpt_cfg.get("save_last", True)
    is_last_step = (step + 1) >= total_steps
    should_save = ((step + 1) % max(1, save_interval) == 0) or (save_last and is_last_step)
    if not should_save:
        return
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/default"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = step + 1 + int(step_offset)
    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step + 1,
        "config": resolved_cfg,
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)
    write_checkpoint_metadata(cfg, ckpt_path, global_step)
    prefix = "[checkpoint]"
    if distributed and dist_ctx is not None:
        prefix = f"[checkpoint rank={dist_ctx.rank}]"
    print(f"{prefix} saved {ckpt_path} (global_step={global_step})")


def _validate_distributed_config(cfg: DictConfig, distributed: bool) -> None:
    if not distributed:
        return
    strict = bool(cfg.train.get("strict_streaming_contract", False))
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    fail_hard = strict or fail_if_faithful_disabled
    if not fail_hard:
        return
    if bool(cfg.train.get("per_layer_teach_signal", False)):
        raise RuntimeError(
            "train.per_layer_teach_signal=true is not supported under DDP in this repo. "
            "Set train.strict_streaming_contract=false and "
            "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )
    if bool(cfg.train.get("online_updates", False)):
        raise RuntimeError(
            "train.online_updates=true is not supported under DDP in this repo. "
            "Set train.strict_streaming_contract=false and "
            "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )
    if bool(cfg.train.get("online_boundary_targets", False)):
        raise RuntimeError(
            "train.online_boundary_targets=true is not supported under DDP in this repo. "
            "Set train.strict_streaming_contract=false and "
            "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )
    if bool(cfg.train.get("online_carry_attention_cache", False)):
        raise RuntimeError(
            "train.online_carry_attention_cache=true is not supported under DDP in this repo. "
            "Set train.strict_streaming_contract=false and "
            "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )


def _emit_streaming_warning(
    *,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {"warning_code": code, "message": message}
    if details:
        payload["details"] = details
    print(f"[train.warn] {json.dumps(payload, sort_keys=True)}")


def _validate_paper_auditing_variant(cfg: DictConfig) -> None:
    strict = bool(cfg.train.get("strict_streaming_contract", False))
    block_variant = str(cfg.model.get("block_variant", "")).strip().lower()
    if not block_variant:
        return
    allowed = {"hope_attention", "hope_selfmod"}
    if block_variant in allowed:
        return
    msg = (
        "strict streaming contract expects a paper-defined HOPE variant "
        f"({sorted(allowed)}), got model.block_variant={block_variant!r}"
    )
    if strict:
        raise RuntimeError(msg)
    _emit_streaming_warning(
        code="non_paper_variant",
        message=msg,
        details={"block_variant": block_variant},
    )


def _validate_tied_lm_head_for_paper_auditing(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> None:
    strict = bool(cfg.train.get("strict_streaming_contract", False))
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    if not (strict or fail_if_faithful_disabled):
        return
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(model, "embed", None)
    if lm_head is None or embed is None:
        return
    lm_weight = getattr(lm_head, "weight", None)
    emb_weight = getattr(embed, "weight", None)
    if lm_weight is None or emb_weight is None:
        return
    if lm_weight.data_ptr() == emb_weight.data_ptr():
        return
    raise RuntimeError(
        "paper-auditing mode requires tied LM head and embedding weights "
        "(lm_head.weight must alias embed.weight)."
    )


def _validate_fast_state_batch_semantics(cfg: DictConfig) -> None:
    if not bool(cfg.train.get("use_fast_state", False)):
        return
    data_cfg = cfg.get("data")
    if data_cfg is None:
        return
    batch_size_raw = data_cfg.get("batch_size", 1)
    try:
        batch_size = int(batch_size_raw)
    except (TypeError, ValueError):
        return
    if batch_size <= 1:
        return
    msg = (
        "train.use_fast_state=true currently shares CMS/TITAN fast state across the batch. "
        "For strict per-context semantics, set data.batch_size=1."
    )
    strict = bool(cfg.train.get("strict_streaming_contract", False))
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    if strict or fail_if_faithful_disabled:
        raise RuntimeError(msg)
    _emit_streaming_warning(
        code="shared_fast_state_batch",
        message=msg,
        details={"batch_size": batch_size},
    )


def _validate_online_update_fast_state_semantics(cfg: DictConfig) -> None:
    train_cfg = cfg.get("train")
    if train_cfg is None:
        return
    online_updates = bool(train_cfg.get("online_updates", False))
    use_fast_state = bool(train_cfg.get("use_fast_state", False))
    if not online_updates or use_fast_state:
        return
    msg = (
        "train.online_updates=true with train.use_fast_state=false applies online writes "
        "directly to base parameters within each step. This can make gradients across chunks "
        "harder to interpret. Use train.use_fast_state=true for paper-faithful runs."
    )
    strict = bool(train_cfg.get("strict_streaming_contract", False))
    fail_if_faithful_disabled = bool(train_cfg.get("fail_if_paper_faithful_disabled", False))
    if strict or fail_if_faithful_disabled:
        raise RuntimeError(msg)
    _emit_streaming_warning(
        code="online_updates_without_fast_state",
        message=msg,
        details={"online_updates": True, "use_fast_state": False},
    )


def _resolve_algorithm_mode(cfg: DictConfig) -> str:
    mode = str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates")).strip()
    allowed = {"two_pass_stopgrad_updates", "boundary_state_grad_through_write"}
    if mode not in allowed:
        raise RuntimeError(f"Unsupported train.algorithm_mode={mode!r}; allowed={sorted(allowed)}")
    return mode


def _validate_algorithm_mode_constraints(
    cfg: DictConfig,
    *,
    algorithm_mode: str,
    distributed: bool,
) -> None:
    if algorithm_mode != "boundary_state_grad_through_write":
        return
    if distributed:
        raise RuntimeError(
            "train.algorithm_mode='boundary_state_grad_through_write' is not supported in DDP."
        )
    if not bool(cfg.train.get("online_updates", False)):
        raise RuntimeError(
            "train.algorithm_mode='boundary_state_grad_through_write' requires "
            "train.online_updates=true."
        )
    if not bool(cfg.train.get("per_layer_teach_signal", False)):
        raise RuntimeError(
            "train.algorithm_mode='boundary_state_grad_through_write' requires "
            "train.per_layer_teach_signal=true."
        )
    if not bool(cfg.train.get("use_fast_state", False)):
        raise RuntimeError(
            "train.algorithm_mode='boundary_state_grad_through_write' requires "
            "train.use_fast_state=true."
        )
    if bool(cfg.train.get("online_carry_attention_cache", False)) and not bool(
        cfg.train.get("online_boundary_targets", False)
    ):
        raise RuntimeError(
            "online_carry_attention_cache=true requires train.online_boundary_targets=true "
            "(non-overlap chunking)."
        )
    _emit_streaming_warning(
        code="experimental_boundary_state_mode",
        message=(
            "train.algorithm_mode='boundary_state_grad_through_write' is an experimental "
            "single-process path for mechanism probing and may use more memory."
        ),
        details={"algorithm_mode": algorithm_mode},
    )


def _validate_online_chunking_constraints(cfg: DictConfig) -> None:
    online_updates = bool(cfg.train.get("online_updates", False))
    online_boundary_targets = bool(cfg.train.get("online_boundary_targets", False))
    online_carry_attention_cache = bool(cfg.train.get("online_carry_attention_cache", False))
    if online_carry_attention_cache and not online_updates:
        raise RuntimeError("online_carry_attention_cache=true requires train.online_updates=true")
    if online_carry_attention_cache and not online_boundary_targets:
        raise RuntimeError(
            "online_carry_attention_cache=true requires train.online_boundary_targets=true "
            "(non-overlap chunking)."
        )


def _check_online_supervised_pairs(
    *,
    strict: bool,
    observed_pairs: int,
    seq_len: int,
) -> None:
    expected_pairs = max(int(seq_len) - 1, 0)
    if observed_pairs == expected_pairs:
        return
    msg = (
        "online chunk supervision mismatch: observed pair coverage does not match sequence length "
        f"(observed_pairs={observed_pairs}, expected_pairs={expected_pairs})"
    )
    if strict:
        raise RuntimeError(msg)
    _emit_streaming_warning(
        code="online_supervision_mismatch",
        message=msg,
        details={"observed_pairs": observed_pairs, "expected_pairs": expected_pairs},
    )


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
    dist_ctx: DistributedContext | None = None,
) -> Dict[str, float]:
    algorithm_mode = _resolve_algorithm_mode(cfg)
    _validate_algorithm_mode_constraints(
        cfg,
        algorithm_mode=algorithm_mode,
        distributed=distributed,
    )
    _validate_online_chunking_constraints(cfg)
    _validate_distributed_config(cfg, distributed)
    _validate_paper_auditing_variant(cfg)
    _validate_fast_state_batch_semantics(cfg)
    _validate_online_update_fast_state_semantics(cfg)
    model = build_model_from_cfg(cfg.model).to(device)
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model = _maybe_compile_model(model, cfg.train.get("compile"))
    if distributed:
        assert dist_ctx is not None
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[idx],
                output_device=idx,
                find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )
        base_model = model.module
    else:
        base_model = model

    _validate_tied_lm_head_for_paper_auditing(cfg, base_model)

    seed_offset = 0
    if train_seed is not None and dist_ctx is not None:
        seed_offset = dist_ctx.rank
    dataloader_seed = None if train_seed is None else int(train_seed) + seed_offset
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=distributed,
        dist_ctx=dist_ctx,
        seed=dataloader_seed,
    )
    optimizer = _build_optimizer(base_model, cfg, device=device)
    autocast_factory = _make_autocast_factory(device, cfg.train.get("mixed_precision"))
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        logger = NullLogger()
    _log_run_features(logger, base_model, cfg, optimizer, device)
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 1)
    per_layer_teach = bool(cfg.train.get("per_layer_teach_signal", False))
    online_updates = bool(cfg.train.get("online_updates", False))
    online_chunk_size = int(cfg.train.get("online_chunk_size", 0) or 0)
    online_boundary_targets = bool(cfg.train.get("online_boundary_targets", False))
    online_carry_attention_cache = bool(cfg.train.get("online_carry_attention_cache", False))
    use_fast_state = bool(cfg.train.get("use_fast_state", False))
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    strict_streaming = bool(cfg.train.get("strict_streaming_contract", False))
    if distributed and per_layer_teach:
        msg = "per_layer_teach_signal disabled under DDP (uses base model methods)"
        if fail_if_faithful_disabled or strict_streaming:
            raise RuntimeError(
                f"{msg}. Set train.strict_streaming_contract=false and "
                "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
                "or run single-process training."
            )
        _emit_streaming_warning(
            code="ddp_disables_per_layer_teach",
            message=msg,
            details={"distributed": True},
        )
        per_layer_teach = False
    if distributed and online_updates:
        msg = "online_updates disabled under DDP (uses base model methods)"
        if fail_if_faithful_disabled or strict_streaming:
            raise RuntimeError(
                f"{msg}. Set train.strict_streaming_contract=false and "
                "train.fail_if_paper_faithful_disabled=false to allow the fallback, "
                "or run single-process training."
            )
        _emit_streaming_warning(
            code="ddp_disables_online_updates",
            message=msg,
            details={"distributed": True},
        )
        online_updates = False
    if online_boundary_targets and not online_updates:
        msg = "online_boundary_targets=true requires train.online_updates=true"
        if fail_if_faithful_disabled or strict_streaming:
            raise RuntimeError(msg)
        _emit_streaming_warning(
            code="boundary_targets_without_online_updates",
            message=msg,
        )
        online_boundary_targets = False
    if online_carry_attention_cache and not online_updates:
        raise RuntimeError("online_carry_attention_cache=true requires train.online_updates=true")
    if online_carry_attention_cache and not online_boundary_targets:
        raise RuntimeError(
            "online_carry_attention_cache=true requires train.online_boundary_targets=true "
            "(non-overlap chunking)."
        )
    step_iter = iter(dataloader)
    epoch = 0
    metrics: Dict[str, float] = {}
    surprise_metric_getter = getattr(base_model, "get_surprise_metric", None)
    surprise_metric = (
        str(surprise_metric_getter()).strip().lower()
        if callable(surprise_metric_getter)
        else str(cfg.model.get("surprise_metric", "l2")).strip().lower()
    )
    for step in range(steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(device)
        fast_state = None
        if use_fast_state:
            init_fn = getattr(base_model, "init_fast_state", None)
            if not callable(init_fn):
                raise ValueError("train.use_fast_state=true requires model.init_fast_state()")
            fast_state = init_fn()
        _apply_teach_schedule(base_model, cfg, step)
        update_metrics: Dict[str, float] = {}
        if online_updates and hasattr(base_model, "forward_with_block_outputs"):
            total_loss = 0.0
            total_tokens = 0
            teach_signal_norm = 0.0
            optimizer.zero_grad()
            chunk_size = online_chunk_size
            if chunk_size <= 0:
                inferred = _infer_online_chunk_size(base_model)
                chunk_size = inferred if inferred is not None else tokens.size(1)
            if chunk_size < 1:
                print(f"[train] online_chunk_size={chunk_size} is too small; clamping to 1")
                chunk_size = 1
            attention_cache = None
            if online_carry_attention_cache:
                init_attention_cache = getattr(base_model, "init_attention_cache", None)
                if not callable(init_attention_cache):
                    raise RuntimeError(
                        "online_carry_attention_cache=true requires model.init_attention_cache()"
                    )
                attention_cache = init_attention_cache()

            chunk_iter: Iterator[tuple[torch.Tensor, torch.Tensor | None, bool]]
            if online_boundary_targets:
                chunk_iter = _iter_online_boundary_chunks(tokens, chunk_size=chunk_size)
            else:
                chunk_iter = (
                    (chunk, None, finalize_updates)
                    for chunk, finalize_updates in _iter_online_token_chunks(
                        tokens, chunk_size=chunk_size
                    )
                )
            for chunk_tokens, next_tokens, finalize_updates in chunk_iter:
                target_count = chunk_tokens.size(1) - 1 + (0 if next_tokens is None else 1)
                if target_count <= 0:
                    continue
                chunk_attention_cache = attention_cache
                with autocast_factory():
                    if attention_cache is not None:
                        logits, _pre, block_outputs, attention_cache = (
                            base_model.forward_with_block_outputs(
                                chunk_tokens,
                                fast_state=fast_state,
                                attention_cache=chunk_attention_cache,
                                return_attention_cache=True,
                            )
                        )
                    else:
                        logits, _pre, block_outputs = (
                            base_model.forward_with_block_outputs(
                                chunk_tokens,
                                fast_state=fast_state,
                            )
                            if fast_state is not None
                            else base_model.forward_with_block_outputs(chunk_tokens)
                        )
                    if next_tokens is None:
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            chunk_tokens[:, 1:].reshape(-1),
                        )
                    else:
                        boundary_targets = torch.cat(
                            [chunk_tokens[:, 1:], next_tokens.unsqueeze(1)],
                            dim=1,
                        )
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, : boundary_targets.size(1), :].reshape(-1, logits.size(-1)),
                            boundary_targets.reshape(-1),
                        )
                surprise_override = _compute_surprise_override(
                    surprise_metric,
                    logits=logits,
                    tokens=chunk_tokens,
                    loss=loss,
                    next_tokens=next_tokens,
                )
                if per_layer_teach:
                    differentiable_updates = algorithm_mode == "boundary_state_grad_through_write"
                    teach_signals = _compute_layer_teach_signals(
                        loss,
                        block_outputs,
                        detach=not differentiable_updates,
                        create_graph=differentiable_updates,
                    )
                    mean_teach_norm = torch.stack(
                        [sig.detach().norm(dim=-1).mean() for sig in teach_signals]
                    ).mean()
                    teach_signal_norm += float(
                        mean_teach_norm
                    ) * target_count
                else:
                    teach_signal = compute_teach_signal(
                        base_model,
                        logits,
                        chunk_tokens,
                        next_tokens=next_tokens,
                    )
                    teach_signal_norm += teach_signal.norm(dim=-1).mean().item() * target_count
                differentiable_updates = algorithm_mode == "boundary_state_grad_through_write"
                # Boundary-state mode keeps a cross-chunk differentiable write path.
                # Retain the graph so later chunks can backprop through earlier writes.
                loss.backward(retain_graph=differentiable_updates)
                if differentiable_updates:
                    if per_layer_teach:
                        base_model(
                            chunk_tokens,
                            teach_signals=teach_signals,
                            surprise_value=surprise_override,
                            fast_state=fast_state,
                            finalize_updates=finalize_updates,
                            attention_cache=chunk_attention_cache,
                            differentiable_updates=True,
                        )
                    else:
                        base_model(
                            chunk_tokens,
                            teach_signal=teach_signal,
                            surprise_value=surprise_override,
                            fast_state=fast_state,
                            finalize_updates=finalize_updates,
                            attention_cache=chunk_attention_cache,
                            differentiable_updates=True,
                        )
                    if hasattr(base_model, "pop_update_metrics"):
                        update_metrics = base_model.pop_update_metrics()
                else:
                    with torch.no_grad():
                        if per_layer_teach:
                            base_model(
                                chunk_tokens,
                                teach_signals=teach_signals,
                                surprise_value=surprise_override,
                                fast_state=fast_state,
                                finalize_updates=finalize_updates,
                                attention_cache=chunk_attention_cache,
                                differentiable_updates=False,
                            )
                        else:
                            base_model(
                                chunk_tokens,
                                teach_signal=teach_signal,
                                surprise_value=surprise_override,
                                fast_state=fast_state,
                                finalize_updates=finalize_updates,
                                attention_cache=chunk_attention_cache,
                                differentiable_updates=False,
                            )
                        if hasattr(base_model, "pop_update_metrics"):
                            update_metrics = base_model.pop_update_metrics()
                total_loss += loss.item() * target_count
                total_tokens += target_count
            _check_online_supervised_pairs(
                strict=strict_streaming,
                observed_pairs=total_tokens,
                seq_len=int(tokens.size(1)),
            )
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            loss = torch.tensor(total_loss / max(total_tokens, 1), device=device)
            teach_signal_norm = teach_signal_norm / max(total_tokens, 1)
        else:
            with autocast_factory():
                if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                    logits, _pre, block_outputs = (
                        base_model.forward_with_block_outputs(tokens, fast_state=fast_state)
                        if fast_state is not None
                        else base_model.forward_with_block_outputs(tokens)
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        tokens[:, 1:].reshape(-1),
                    )
                else:
                    if fast_state is not None:
                        logits = model(tokens, fast_state=fast_state)
                    else:
                        logits = model(tokens)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        tokens[:, 1:].reshape(-1),
                    )
            surprise_override = _compute_surprise_override(
                surprise_metric,
                logits=logits,
                tokens=tokens,
                loss=loss,
                next_tokens=None,
            )
            optimizer.zero_grad()
            if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                teach_signals = _compute_layer_teach_signals(loss, block_outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                    teach_signal_norm = float(
                        torch.stack([sig.norm(dim=-1).mean() for sig in teach_signals]).mean()
                    )
                    base_model(
                        tokens,
                        teach_signals=teach_signals,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                else:
                    teach_signal = compute_teach_signal(base_model, logits, tokens)
                    teach_signal_norm = teach_signal.norm(dim=-1).mean().item()
                    base_model(
                        tokens,
                        teach_signal=teach_signal,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                if hasattr(base_model, "pop_update_metrics"):
                    update_metrics = base_model.pop_update_metrics()
        if step % log_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            metrics_payload = {
                "loss": loss.item(),
                "ppl": ppl,
                "teach_signal_norm": teach_signal_norm,
            }
            metrics_payload.update(update_metrics)
            logger.log(metrics_payload, step=step)
            if (not distributed) or (dist_ctx and dist_ctx.rank == 0):
                print(
                    f"[train] step={step} loss={loss.item():.4f} "
                    f"ppl={ppl:.2f} teach_norm={teach_signal_norm:.4f}"
                )
            metrics = metrics_payload
        maybe_save_checkpoint(
            cfg,
            base_model,
            optimizer,
            step=step,
            total_steps=steps,
            distributed=distributed,
            dist_ctx=dist_ctx,
            step_offset=int(cfg.train.get("step_offset", 0) or 0),
        )
    logger.finish()
    return metrics


def _apply_teach_schedule(model: HOPEModel, cfg: DictConfig, step: int) -> None:
    schedule = cfg.model.get("teach_schedule")
    base_scale = cfg.model.get("teach_scale", 1.0)
    scale = base_scale
    if schedule:
        warmup = schedule.get("warmup_steps", 0)
        if warmup and warmup > 0:
            scale *= min(1.0, (step + 1) / warmup)
        decay_start = schedule.get("decay_start")
        decay_duration = schedule.get("decay_duration")
        if (
            decay_start is not None
            and decay_duration
            and decay_duration > 0
            and (step + 1) > decay_start
        ):
            progress = min(1.0, (step + 1 - decay_start) / decay_duration)
            scale *= max(0.0, 1.0 - progress)
    model.set_teach_runtime(scale=scale)


def _maybe_compile_model(model: torch.nn.Module, compile_cfg: dict | None) -> torch.nn.Module:
    if not compile_cfg or not compile_cfg.get("enable", False):
        return model
    kwargs = {}
    if "mode" in compile_cfg:
        kwargs["mode"] = compile_cfg["mode"]
    if "backend" in compile_cfg:
        kwargs["backend"] = compile_cfg["backend"]
    try:
        return cast(torch.nn.Module, torch.compile(model, **kwargs))  # type: ignore[attr-defined]
    except Exception as err:  # pragma: no cover - compile is optional
        if compile_cfg.get("strict", False):
            raise
        print(f"[compile] fallback to eager due to: {err}")
        return model


def _make_autocast_factory(device: torch.device, mp_cfg: dict | None):
    if not mp_cfg or not mp_cfg.get("enabled", False):
        return lambda: nullcontext()
    dtype = _resolve_autocast_dtype(mp_cfg.get("dtype", "bf16"))
    device_type = device.type
    if device_type not in {"cuda", "cpu", "mps"}:
        device_type = "cpu"

    def factory():
        try:
            return torch.autocast(device_type=device_type, dtype=dtype)
        except Exception as err:  # pragma: no cover - device/dtype support varies by backend
            print(f"[autocast] disabled for device_type={device_type} dtype={dtype}: {err}")
            return nullcontext()

    return factory


def _resolve_autocast_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    msg = f"Unsupported autocast dtype {name}"
    raise ValueError(msg)


def _build_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, device: torch.device
) -> torch.optim.Optimizer:
    optimizer_cfg_raw = cfg.get("optim")
    if isinstance(optimizer_cfg_raw, DictConfig):
        optimizer_cfg = optimizer_cfg_raw
    else:
        optimizer_cfg = cast(DictConfig, OmegaConf.create(optimizer_cfg_raw or {}))
    param_policy_raw = optimizer_cfg.get("param_policy")
    if param_policy_raw is None:
        outer_updates_memory_modules = optimizer_cfg.get("outer_updates_memory_modules")
        if outer_updates_memory_modules is None:
            param_policy = "all"
        else:
            param_policy = "all" if bool(outer_updates_memory_modules) else "exclude_memory"
    else:
        param_policy = str(param_policy_raw).strip().lower()
    named_params = _select_outer_named_parameters(model, param_policy)
    if not named_params:
        raise ValueError(
            f"No trainable parameters selected for optim.param_policy={param_policy!r}. "
            "Check freeze_backbone, requires_grad flags, or adjust the policy."
        )
    optim_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optim_type == "muon":
        return _build_muon_optimizer(
            model,
            optimizer_cfg,
            device=device,
            named_params=named_params,
            param_policy=param_policy,
        )
    if optim_type == "m3":
        return _build_m3_optimizer(
            model,
            optimizer_cfg,
            device=device,
            named_params=named_params,
            param_policy=param_policy,
        )
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    kwargs = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
    if fused:
        kwargs["fused"] = True
    params = [param for _, param in named_params]
    return torch.optim.AdamW(params, **kwargs)


def _build_muon_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
    *,
    device: torch.device,
    named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    param_policy: str | None = None,
):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    momentum = optimizer_cfg.get("momentum", 0.95)
    ns_coefficients = optimizer_cfg.get("ns_coefficients")
    ns_steps = optimizer_cfg.get("ns_steps")
    eps = optimizer_cfg.get("eps", 1e-7)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    source = named_params if named_params is not None else model.named_parameters()
    for name, param in source:
        if not param.requires_grad:
            continue
        if _is_muon_candidate(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    muon_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "eps": eps,
    }
    if ns_coefficients is not None:
        muon_kwargs["ns_coefficients"] = tuple(ns_coefficients)
    if ns_steps is not None:
        muon_kwargs["ns_steps"] = int(ns_steps)
    muon_opt = torch.optim.Muon(muon_params, **muon_kwargs) if muon_params else None  # type: ignore[attr-defined]
    adamw_kwargs = {
        "lr": lr,
        "betas": optimizer_cfg.get("betas", (0.9, 0.999)),
        "weight_decay": weight_decay,
    }
    if fused:
        adamw_kwargs["fused"] = True
    adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    muon_elems = int(sum(p.numel() for p in muon_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))
    return _HybridOptimizer(
        muon_opt,
        adamw_opt,
        muon_elems,
        adamw_elems,
        primary_name="muon",
        param_policy=param_policy,
    )


def _build_m3_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
    *,
    device: torch.device,
    named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    param_policy: str | None = None,
):
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    beta1 = optimizer_cfg.get("beta1", 0.9)
    beta2 = optimizer_cfg.get("beta2", 0.999)
    beta3 = optimizer_cfg.get("beta3", 0.9)
    alpha = optimizer_cfg.get("alpha", 1.0)
    ns_steps = int(optimizer_cfg.get("ns_steps", 3))
    slow_chunk = int(optimizer_cfg.get("slow_chunk", 100))
    eps = optimizer_cfg.get("eps", 1e-8)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)

    m3_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    source = named_params if named_params is not None else model.named_parameters()
    for name, param in source:
        if not param.requires_grad:
            continue
        if _is_muon_candidate(name, param):
            m3_params.append(param)
        else:
            adamw_params.append(param)
    m3_opt = (
        M3(
            m3_params,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        if m3_params
        else None
    )
    adamw_kwargs = {
        "lr": lr,
        "betas": optimizer_cfg.get("betas", (0.9, 0.999)),
        "weight_decay": weight_decay,
    }
    if fused:
        adamw_kwargs["fused"] = True
    adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    m3_elems = int(sum(p.numel() for p in m3_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))
    return _HybridOptimizer(
        m3_opt,
        adamw_opt,
        m3_elems,
        adamw_elems,
        primary_name="m3",
        param_policy=param_policy,
    )


def _select_outer_named_parameters(
    model: torch.nn.Module, param_policy: str
) -> list[tuple[str, torch.nn.Parameter]]:
    policy = str(param_policy).strip().lower()
    trainable: list[tuple[str, torch.nn.Parameter]] = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    if policy in {"all", "full"}:
        return trainable
    if policy in {"exclude_memory", "no_memory"}:
        return [(name, param) for name, param in trainable if not _is_memory_param_name(name)]
    if policy in {"only_memory", "memory_only"}:
        return [(name, param) for name, param in trainable if _is_memory_param_name(name)]
    raise ValueError(
        f"Unsupported optim.param_policy={param_policy!r}. "
        "Expected one of ['all', 'exclude_memory', 'only_memory']."
    )


def _is_memory_param_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in (".cms.", ".titan_memory.", ".selfmod."))


def _is_muon_candidate(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    lowered = name.lower()
    if "norm" in lowered or "embed" in lowered:
        return False
    return True


class _HybridOptimizer:
    def __init__(
        self,
        primary_opt: torch.optim.Optimizer | None,
        secondary_opt: torch.optim.Optimizer | None,
        primary_param_elems: int,
        secondary_param_elems: int,
        *,
        primary_name: str = "muon",
        param_policy: str | None = None,
    ):
        self.primary_opt = primary_opt
        self.secondary_opt = secondary_opt
        self.primary_param_elems = primary_param_elems
        self.secondary_param_elems = secondary_param_elems
        self.primary_name = primary_name
        self.param_policy = param_policy

    def zero_grad(self) -> None:
        if self.primary_opt:
            self.primary_opt.zero_grad()
        if self.secondary_opt:
            self.secondary_opt.zero_grad()

    def step(self) -> None:
        if self.primary_opt:
            self.primary_opt.step()
        if self.secondary_opt:
            self.secondary_opt.step()

    def state_dict(self) -> dict:
        return {
            self.primary_name: self.primary_opt.state_dict() if self.primary_opt else None,
            "adamw": self.secondary_opt.state_dict() if self.secondary_opt else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.primary_opt and state.get(self.primary_name) is not None:
            self.primary_opt.load_state_dict(state[self.primary_name])
        if self.secondary_opt and state.get("adamw") is not None:
            self.secondary_opt.load_state_dict(state["adamw"])

    @property
    def param_groups(self):
        groups = []
        if self.primary_opt:
            groups.extend(self.primary_opt.param_groups)
        if self.secondary_opt:
            groups.extend(self.secondary_opt.param_groups)
        return groups

    def get_param_split(self) -> dict[str, int]:
        return {
            self.primary_name: self.primary_param_elems,
            "adamw": self.secondary_param_elems,
        }


def _log_run_features(
    logger: BaseLogger,
    model: torch.nn.Module,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    mp_cfg = cfg.train.get("mixed_precision", {})
    compile_cfg = cfg.train.get("compile", {})
    algorithm_mode = str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates"))
    features: dict[str, object] = {
        "train.mixed_precision_enabled": bool(mp_cfg.get("enabled", False)),
        "train.mixed_precision_dtype": str(mp_cfg.get("dtype", "bf16")),
        "train.compile_enabled": bool(compile_cfg.get("enable", False)),
        "train.compile_mode": str(compile_cfg.get("mode", "default")) if compile_cfg else "default",
        "train.strict_streaming_contract": bool(cfg.train.get("strict_streaming_contract", False)),
        "train.online_updates": bool(cfg.train.get("online_updates", False)),
        "train.online_boundary_targets": bool(cfg.train.get("online_boundary_targets", False)),
        "train.online_carry_attention_cache": bool(
            cfg.train.get("online_carry_attention_cache", False)
        ),
        "train.use_fast_state": bool(cfg.train.get("use_fast_state", False)),
        "train.algorithm_mode": algorithm_mode,
        "train.backprop_through_online_writes": algorithm_mode
        == "boundary_state_grad_through_write",
        "attention.flash_enabled": _detect_flash_attention(model),
        "device": device.type,
    }
    optimizer_cfg_raw = cfg.get("optim")
    if isinstance(optimizer_cfg_raw, DictConfig):
        optimizer_cfg = optimizer_cfg_raw
    else:
        optimizer_cfg = cast(DictConfig, OmegaConf.create(optimizer_cfg_raw or {}))
    param_policy_raw = optimizer_cfg.get("param_policy")
    if param_policy_raw is None:
        outer_updates_memory_modules = optimizer_cfg.get("outer_updates_memory_modules")
        if outer_updates_memory_modules is None:
            param_policy = "all"
        else:
            param_policy = "all" if bool(outer_updates_memory_modules) else "exclude_memory"
    else:
        param_policy = str(param_policy_raw).strip().lower()
    try:
        selected = _select_outer_named_parameters(model, param_policy)
        total_elems = int(sum(param.numel() for _, param in selected))
        memory_elems = int(
            sum(param.numel() for name, param in selected if _is_memory_param_name(name))
        )
        features["optim.param_policy"] = param_policy
        features["optim.param_policy_param_elems"] = total_elems
        features["optim.param_policy_memory_param_elems"] = memory_elems
        features["optim.param_policy_non_memory_param_elems"] = total_elems - memory_elems
    except Exception as err:  # pragma: no cover - purely diagnostic
        features["optim.param_policy"] = param_policy
        features["optim.param_policy_error"] = str(err)
    split_fn = getattr(optimizer, "get_param_split", None)
    if callable(split_fn):
        split = split_fn()
        for key, value in split.items():
            features[f"optim.{key}_param_elems"] = int(value)
    logger.log(features, step=-1)
    print(f"[train] run_features {features}")


def _detect_flash_attention(model: torch.nn.Module) -> bool:
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        attn = getattr(block, "attn", None)
        config = getattr(attn, "config", None)
        if config is not None and hasattr(config, "use_flash"):
            return bool(config.use_flash)
    return False


def write_checkpoint_metadata(cfg: DictConfig, ckpt_path: Path, step: int) -> None:
    config_yaml = OmegaConf.to_yaml(cfg)
    config_path = ckpt_path.with_suffix(".yaml")
    config_path.write_text(config_yaml)
    config_hash = sha256(config_yaml.encode("utf-8")).hexdigest()
    ckpt_hash = _checksum_path(str(ckpt_path))
    sha_path = ckpt_path.with_suffix(".sha256")
    if ckpt_hash:
        sha_path.write_text(f"{ckpt_hash}  {ckpt_path.name}\n")
    tokenizer_path = cfg.data.get("tokenizer_path") if hasattr(cfg, "data") else None
    metadata = {
        "step": step,
        "checkpoint_sha256": ckpt_hash,
        "config_sha256": config_hash,
        "tokenizer_hash": _checksum_path(tokenizer_path) if tokenizer_path else None,
        "config_path": str(config_path),
        "algorithm_mode": str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates")),
        "online_updates": bool(cfg.train.get("online_updates", False)),
        "online_boundary_targets": bool(cfg.train.get("online_boundary_targets", False)),
        "online_carry_attention_cache": bool(
            cfg.train.get("online_carry_attention_cache", False)
        ),
        "use_fast_state": bool(cfg.train.get("use_fast_state", False)),
        "rng_states": _capture_rng_states(),
    }
    ckpt_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))


def verify_checkpoint_integrity(ckpt_path: Path) -> Dict[str, object]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    meta_path = ckpt_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file {meta_path} missing")
    metadata = json.loads(meta_path.read_text())
    computed_sha = _checksum_path(str(ckpt_path))
    recorded_sha = metadata.get("checkpoint_sha256")
    if recorded_sha and computed_sha and recorded_sha != computed_sha:
        raise ValueError(
            f"Checkpoint SHA mismatch: recorded {recorded_sha} vs computed {computed_sha}"
        )
    sha_file = ckpt_path.with_suffix(".sha256")
    if sha_file.exists() and computed_sha:
        recorded_line = sha_file.read_text().strip().split()
        if recorded_line:
            recorded = recorded_line[0]
            if recorded != computed_sha:
                raise ValueError(f".sha256 mismatch: {recorded} vs {computed_sha}")
    config_path = ckpt_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} missing")
    config_hash = sha256(config_path.read_text().encode("utf-8")).hexdigest()
    recorded_cfg_hash = metadata.get("config_sha256")
    if recorded_cfg_hash and recorded_cfg_hash != config_hash:
        raise ValueError(
            f"Config SHA mismatch: recorded {recorded_cfg_hash} vs computed {config_hash}"
        )
    if "rng_states" not in metadata:
        raise ValueError("Metadata missing rng_states")
    return metadata


def _capture_rng_states() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "python": _encode_pickle(random.getstate()),
        "numpy": _encode_pickle(np.random.get_state()),
        "torch": _tensor_state_to_hex(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [
            _tensor_state_to_hex(state) for state in torch.cuda.get_rng_state_all()
        ]  # type: ignore[attr-defined]
    return payload


def _encode_pickle(obj: object) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _tensor_state_to_hex(state: torch.Tensor) -> str:
    return state.cpu().numpy().tobytes().hex()


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]


def _make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn
