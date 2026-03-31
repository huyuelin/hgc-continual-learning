from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _run_report(config_path: Path, output_path: Path, repo_root: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/checks/compliance_report.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(output_path.read_text())


def test_compliance_report_includes_algorithm_mode_checks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = _run_report(
        config_path=repo_root / "configs/pilot.yaml",
        output_path=tmp_path / "report.json",
        repo_root=repo_root,
    )
    checks = {item["name"] for item in report["checks"]}
    assert "algorithm_mode_supported" in checks


def test_compliance_report_validates_boundary_mode_constraints(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo_root / "configs/pilot.yaml")
    cfg.train.algorithm_mode = "boundary_state_grad_through_write"
    cfg.train.per_layer_teach_signal = True
    cfg.train.online_updates = True
    cfg.train.online_boundary_targets = True
    cfg.train.online_carry_attention_cache = True
    cfg.train.use_fast_state = True
    cfg.data.batch_size = 1
    tmp_cfg = tmp_path / "boundary_config.yaml"
    tmp_cfg.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    report = _run_report(
        config_path=tmp_cfg,
        output_path=tmp_path / "boundary_report.json",
        repo_root=repo_root,
    )
    by_name = {item["name"]: item for item in report["checks"]}
    assert by_name["algorithm_mode_supported"]["ok"] is True
    assert by_name["boundary_algorithm_mode_constraints"]["ok"] is True
