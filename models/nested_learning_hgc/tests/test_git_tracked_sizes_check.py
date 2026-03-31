from __future__ import annotations

import subprocess
from pathlib import Path


def test_git_tracked_sizes_check_passes_repo_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", "scripts/checks/check_git_tracked_sizes.sh"],
        cwd=repo_root,
        check=True,
    )
