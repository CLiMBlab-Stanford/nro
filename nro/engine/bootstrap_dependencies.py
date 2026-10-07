"""Enter the isolated installer environment before importing nro."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

UV_VERSION = "0.8.22"
PYYAML_VERSION = "6.0.3"


def _ready(python: Path, uv: Path) -> bool:
    """Return whether the bootstrap interpreter has its pinned tools."""
    if not python.is_file() or not uv.is_file():
        return False
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m; import yaml; "
            f"assert m.version('uv') == {UV_VERSION!r}; "
            f"assert m.version('PyYAML') == {PYYAML_VERSION!r}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def enter(root: Path, script: Path, arguments: list[str]) -> None:
    """Create the bootstrap environment when needed and re-execute the installer in it."""
    environment = root / ".nro-bootstrap"
    python = environment / "bin/python"
    uv = environment / "bin/uv"
    if Path(sys.prefix).resolve() == environment.resolve():
        if not _ready(python, uv):
            raise RuntimeError(
                "The nro bootstrap environment is incomplete; rerun without --offline"
            )
        return
    if not _ready(python, uv):
        if "--offline" in arguments:
            raise RuntimeError("Offline setup needs an existing bootstrap environment")
        if not python.is_file():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"uv=={UV_VERSION}",
                f"PyYAML=={PYYAML_VERSION}",
            ],
            check=True,
        )
    os.execv(str(python), [str(python), str(script), *arguments])
