#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import typer
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nested_learning.data import TokenShardDataset, collate_batch
from nested_learning.device import resolve_device
from nested_learning.eval_state import (
    EvalStreamingState,
    forward_with_eval_state,
    init_eval_streaming_state,
    parse_eval_state_mode,
)
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_tokens,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Continual learning evaluation harness.")


def load_segments(yaml_path: Path) -> List[Dict[str, str]]:
    payload = yaml.safe_load(yaml_path.read_text())
    return payload.get("segments", [])


def evaluate_segment(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    memorize_cfg: MemorizeConfig,
    *,
    eval_state_mode: str,
    eval_use_fast_state: bool,
    eval_use_attention_cache: bool,
) -> tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss_base = 0.0
    total_loss_mem = 0.0
    total_tokens = 0
    batches = 0
    path_stats: Dict[str, float] = defaultdict(float)
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    carry_eval_state = parse_eval_state_mode(eval_state_mode)
    eval_state_base: EvalStreamingState | None = None
    eval_state_mem: EvalStreamingState | None = None
    if eval_use_fast_state or eval_use_attention_cache:
        eval_state_base = init_eval_streaming_state(
            model,
            use_fast_state=eval_use_fast_state,
            use_attention_cache=eval_use_attention_cache,
        )
        eval_state_mem = init_eval_streaming_state(
            model,
            use_fast_state=eval_use_fast_state,
            use_attention_cache=eval_use_attention_cache,
        )
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)
    for batch in dataloader:
        tokens = batch.to(device)
        if (eval_use_fast_state or eval_use_attention_cache) and not carry_eval_state:
            eval_state_base = init_eval_streaming_state(
                model,
                use_fast_state=eval_use_fast_state,
                use_attention_cache=eval_use_attention_cache,
            )
            eval_state_mem = init_eval_streaming_state(
                model,
                use_fast_state=eval_use_fast_state,
                use_attention_cache=eval_use_attention_cache,
            )
        with torch.no_grad():
            logits, eval_state_base = forward_with_eval_state(
                model,
                tokens,
                state=eval_state_base,
            )
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_loss_base += loss.item()
        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_tokens(model, tokens, memorize_cfg, fast_state=fast_state)
            else:
                stats = memorize_tokens(model, tokens, memorize_cfg)
            for key, value in stats.items():
                path_stats[key] += value
            with torch.no_grad():
                if eval_use_fast_state or eval_use_attention_cache:
                    logits_mem, eval_state_mem = forward_with_eval_state(
                        model,
                        tokens,
                        state=eval_state_mem,
                    )
                else:
                    logits_mem = (
                        model(tokens, fast_state=fast_state)
                        if memorize_cfg.use_fast_state
                        else model(tokens)
                    )
                loss_mem = torch.nn.functional.cross_entropy(
                    logits_mem[:, :-1].reshape(-1, logits_mem.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction="sum",
                )
            total_loss_mem += loss_mem.item()
            if (not memorize_cfg.use_fast_state) and memorize_cfg.reset and base_state is not None:
                restore_state_dict(model, base_state)
        else:
            total_loss_mem += loss.item()
        total_tokens += tokens[:, 1:].numel()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    base_ce = total_loss_base / total_tokens if total_tokens > 0 else float("nan")
    mem_ce = total_loss_mem / total_tokens if total_tokens > 0 else float("nan")
    return base_ce, mem_ce, path_stats


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config for HOPE."),
    checkpoints: List[Path] = typer.Option(
        ..., help="Ordered list of checkpoints (chronological)."
    ),
    segments_yaml: Path = typer.Option(..., help="YAML describing shard directories per segment."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path (unused for now)."),
    batch_size: int = typer.Option(4, help="Batch size for evaluation."),
    max_batches: int = typer.Option(50, help="Max batches per segment (0 = entire dataset)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/continual_results.json")),
    memorize: bool = typer.Option(False, help="Enable memorization while evaluating segments."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per batch."),
    memorize_no_reset: bool = typer.Option(True, help="Keep memory between segments by default."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm needed to memorize a batch."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for default behavior."
        ),
    ),
    eval_state_mode: str = typer.Option(
        "reset_per_sample",
        help="Streaming eval state mode: 'reset_per_sample' or 'carry_across_samples'.",
    ),
    eval_use_fast_state: bool = typer.Option(
        False,
        help="Use model fast state during inference scoring (independent from memorization state).",
    ),
    eval_use_attention_cache: bool = typer.Option(
        False,
        help="Use attention KV cache during inference scoring.",
    ),
) -> None:
    segments = load_segments(segments_yaml)
    if not segments:
        raise typer.BadParameter("No segments found in YAML.")

    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    device_obj = resolve_device(device)
    results = []

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=False,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    for step_idx, ckpt_path in enumerate(checkpoints):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_cfg(cfg.model)
        state_dict = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                "[continual] Warning: state_dict mismatch "
                f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
            )
        model = model.to(device_obj)

        segment_losses = {}
        baseline_losses = {}
        segment_stats = {}
        for segment in segments:
            name = segment["name"]
            shards_dir = Path(segment["shards_dir"])
            dataset = TokenShardDataset(shards_dir)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
            )
            base_loss, mem_loss, stats = evaluate_segment(
                model,
                loader,
                device_obj,
                None if max_batches <= 0 else max_batches,
                memorize_cfg,
                eval_state_mode=eval_state_mode,
                eval_use_fast_state=eval_use_fast_state,
                eval_use_attention_cache=eval_use_attention_cache,
            )
            baseline_losses[name] = base_loss
            segment_losses[name] = mem_loss
            if stats:
                segment_stats[name] = stats

        entry = {"checkpoint": str(ckpt_path), "segment_losses": segment_losses}
        if memorize_cfg.enabled:
            entry["segment_baseline_losses"] = baseline_losses
            entry["segment_memorize_delta"] = {
                name: baseline_losses[name] - segment_losses[name] for name in segment_losses
            }
            if segment_stats:
                entry["memorize_stats"] = segment_stats
        entry["eval_state_mode"] = eval_state_mode
        entry["eval_use_fast_state"] = bool(eval_use_fast_state)
        entry["eval_use_attention_cache"] = bool(eval_use_attention_cache)
        results.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Continual] Saved results to {output}")


if __name__ == "__main__":
    app()
