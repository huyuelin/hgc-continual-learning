from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_package_script_includes_train_flags_and_excludes_raw_data(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_src = repo_root / "scripts/package_pilot_release.sh"
    script_dst = tmp_path / "scripts/package_pilot_release.sh"
    script_dst.parent.mkdir(parents=True)
    script_dst.write_text(script_src.read_text(), encoding="utf-8")

    # Minimal repo layout expected by package script.
    (tmp_path / "artifacts/checkpoints/pilot").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/secret.txt").write_text("do-not-copy", encoding="utf-8")
    (tmp_path / "configs/pilot.yaml").write_text("model: {}\ntrain: {}\n", encoding="utf-8")

    ckpt = tmp_path / "artifacts/checkpoints/pilot/step_000001.pt"
    ckpt.write_bytes(b"dummy-checkpoint")
    (tmp_path / "artifacts/checkpoints/pilot/step_000001.meta.json").write_text(
        json.dumps(
            {
                "algorithm_mode": "two_pass_stopgrad_updates",
                "online_updates": True,
                "online_boundary_targets": False,
                "online_carry_attention_cache": False,
                "use_fast_state": False,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts/checkpoints/pilot/step_000001.yaml").write_text(
        "model: {}\ntrain: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "artifacts/checkpoints/pilot/step_000001.sha256").write_text(
        "abc  step_000001.pt\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(script_dst)], cwd=tmp_path, check=True)

    manifest = (tmp_path / "artifacts/pilot_release/MANIFEST.txt").read_text(encoding="utf-8")
    assert "HOPE Train Flags:" in manifest
    assert "algorithm_mode='two_pass_stopgrad_updates'" in manifest
    assert not (tmp_path / "artifacts/pilot_release/secret.txt").exists()
