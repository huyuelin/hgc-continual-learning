import importlib.util
from pathlib import Path

from typer.testing import CliRunner


def _load_eval_script(name: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


niah = _load_eval_script("niah")
zeroshot = _load_eval_script("zeroshot")


def test_zeroshot_rejects_carry_eval_state_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(
        zeroshot.app,
        [
            "--config",
            "dummy.yaml",
            "--checkpoint",
            "dummy.pt",
            "--tokenizer-path",
            "dummy.model",
            "--list-tasks",
            "--eval-state-mode",
            "carry_across_samples",
        ],
    )
    assert result.exit_code != 0
    assert "reset_per_sample" in result.stdout + result.stderr


def test_zeroshot_allows_reset_eval_state_mode_for_task_listing() -> None:
    runner = CliRunner()
    result = runner.invoke(
        zeroshot.app,
        [
            "--config",
            "dummy.yaml",
            "--checkpoint",
            "dummy.pt",
            "--tokenizer-path",
            "dummy.model",
            "--list-tasks",
            "--eval-state-mode",
            "reset_per_sample",
        ],
    )
    assert result.exit_code == 0
    assert "Available tasks:" in result.stdout


def test_niah_rejects_carry_eval_state_mode_before_loading_inputs(tmp_path: Path) -> None:
    runner = CliRunner()
    config = tmp_path / "cfg.yaml"
    ckpt = tmp_path / "ckpt.pt"
    tok = tmp_path / "tok.model"
    config.write_text("model: {}\ntrain: {}\ndata: {}\n")
    ckpt.write_bytes(b"")
    tok.write_text("")
    result = runner.invoke(
        niah.app,
        [
            "--config",
            str(config),
            "--checkpoint",
            str(ckpt),
            "--tokenizer-path",
            str(tok),
            "--context-lengths",
            "32",
            "--samples-per-length",
            "1",
            "--device",
            "cpu",
            "--eval-state-mode",
            "carry_across_samples",
        ],
    )
    assert result.exit_code != 0
    assert "reset_per_sample" in result.stdout + result.stderr
