import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, *args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    """Run ``python -m <module> <args>`` from the repository root, the way the run_*.ps1 launchers do."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"exit code {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
