from __future__ import annotations

import subprocess
from pathlib import Path


def test_data_scripts_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", "scripts/checks/check_data_script_help.sh"],
        cwd=repo_root,
        check=True,
    )
