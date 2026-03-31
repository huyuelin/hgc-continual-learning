#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(
    add_completion=False,
    help="Aggregate implementation compliance checks into a machine-readable report.",
)


def _load_resolved_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model_cfg = cfg.get("model")
    needs_compose = bool(
        cfg.get("defaults") is not None
        and model_cfg is not None
        and model_cfg.get("titan_level") is None
    )
    if not needs_compose:
        return cfg
    config_dir = config_path.resolve().parent
    config_name = config_path.stem
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)
    return unwrap_config(composed)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _append(results: list[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail))


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra config path."),
    output: Path = typer.Option(
        Path("eval/compliance_report.json"),
        help="Path to write the compliance JSON.",
    ),
    require_strict: bool = typer.Option(
        False,
        help="Require train.strict_streaming_contract=true.",
    ),
    cadence_report: Path | None = typer.Option(
        None,
        help="Optional JSON emitted by scripts/checks/verify_update_cadence.py.",
    ),
) -> None:
    cfg = _load_resolved_config(config)
    results: list[CheckResult] = []

    strict = bool(cfg.train.get("strict_streaming_contract", False))
    algorithm_mode = str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates")).strip()
    allowed_algorithm_modes = {"two_pass_stopgrad_updates", "boundary_state_grad_through_write"}
    if require_strict:
        _append(
            results,
            "strict_streaming_contract_enabled",
            strict,
            f"train.strict_streaming_contract={strict}",
        )
    else:
        _append(
            results,
            "strict_streaming_contract_observed",
            True,
            f"train.strict_streaming_contract={strict}",
        )
    _append(
        results,
        "algorithm_mode_supported",
        algorithm_mode in allowed_algorithm_modes,
        f"train.algorithm_mode={algorithm_mode!r}; allowed={sorted(allowed_algorithm_modes)}",
    )

    block_variant = str(cfg.model.get("block_variant", "hope_hybrid")).strip().lower()
    allowed_variants = {"hope_attention", "hope_selfmod"}
    if strict:
        _append(
            results,
            "strict_variant_is_paper_defined",
            block_variant in allowed_variants,
            f"model.block_variant={block_variant!r}; allowed={sorted(allowed_variants)}",
        )
    else:
        _append(
            results,
            "variant_recorded",
            True,
            f"model.block_variant={block_variant!r}",
        )

    model = build_model_from_cfg(cfg.model)
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(model, "embed", None)
    tied = False
    if lm_head is not None and embed is not None:
        lm_weight = getattr(lm_head, "weight", None)
        emb_weight = getattr(embed, "weight", None)
        if lm_weight is not None and emb_weight is not None:
            tied = lm_weight.data_ptr() == emb_weight.data_ptr()
    _append(
        results,
        "lm_head_embed_tied",
        tied,
        "lm_head.weight aliases embed.weight" if tied else "weights are not tied",
    )

    online_updates = bool(cfg.train.get("online_updates", False))
    boundary = bool(cfg.train.get("online_boundary_targets", False))
    carry_attn = bool(cfg.train.get("online_carry_attention_cache", False))
    per_layer_teach = bool(cfg.train.get("per_layer_teach_signal", False))
    use_fast_state = bool(cfg.train.get("use_fast_state", False))
    if algorithm_mode == "boundary_state_grad_through_write":
        boundary_mode_ok = online_updates and per_layer_teach and use_fast_state
        _append(
            results,
            "boundary_algorithm_mode_constraints",
            boundary_mode_ok,
            (
                "boundary mode requires online_updates, per_layer_teach_signal, and use_fast_state "
                f"(online_updates={online_updates}, per_layer_teach_signal={per_layer_teach}, "
                f"use_fast_state={use_fast_state})"
            ),
        )
    _append(
        results,
        "boundary_requires_online_updates",
        (not boundary) or online_updates,
        f"online_updates={online_updates}, online_boundary_targets={boundary}",
    )
    _append(
        results,
        "carry_attention_requires_boundary",
        (not carry_attn) or boundary,
        (
            "online_carry_attention_cache requires online_boundary_targets"
            f" (carry={carry_attn}, boundary={boundary})"
        ),
    )

    batch_size = int(cfg.data.get("batch_size", 1))
    fast_state_semantics_ok = (not use_fast_state) or batch_size <= 1
    _append(
        results,
        "fast_state_batch_semantics",
        fast_state_semantics_ok,
        f"use_fast_state={use_fast_state}, data.batch_size={batch_size}",
    )

    cadence_payload: dict[str, Any] | None = None
    if cadence_report is not None:
        cadence_payload = json.loads(cadence_report.read_text())
        cadence_ok = bool(cadence_payload.get("ok", False))
        _append(
            results,
            "cadence_report_ok",
            cadence_ok,
            f"cadence_report={cadence_report}",
        )

    summary = {
        "config": str(config),
        "overall_ok": all(item.ok for item in results),
        "checks": [asdict(item) for item in results],
        "cadence_report": cadence_payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["overall_ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
