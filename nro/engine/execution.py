"""Execution-environment primitives used by scientific modules."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterable
from itertools import count
from pathlib import Path
from typing import Any, Callable, Sequence

from nro.engine.io import atomic_copy_file
from nro.orchestration.runner_graph import Step

_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def allocated_cpus() -> int:
    """Return the CPU count assigned to the current worker process."""
    for variable in ("NRO_ALLOCATED_CPUS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            cpus = int(value)
        except ValueError as error:
            raise ValueError(f"{variable} must be a positive integer") from error
        if cpus < 1:
            raise ValueError(f"{variable} must be a positive integer")
        return cpus
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def thread_environment(cpus: int | None = None) -> dict[str, str]:
    """Limit common numerical libraries to the worker's CPU allocation."""
    thread_count = str(allocated_cpus() if cpus is None else max(1, int(cpus)))
    return {variable: thread_count for variable in _THREAD_ENVIRONMENT_VARIABLES}


def new_step_counter(start: int = 1) -> Callable[[], int]:
    """Return an independent monotonically increasing step-number source."""
    return count(start).__next__


def create_copy_file_step(
    *,
    src: Path,
    dst: Path,
    force: bool,
    step_name: str,
    validate: Callable[[], tuple[bool, str]] | None = None,
    parameters: object | None = None,
) -> Step:
    """Create an atomic file-copy step."""
    return Step.python(
        name=step_name,
        outputs=(dst,),
        inputs=(src,),
        force=force,
        action=lambda: atomic_copy_file(src, dst),
        validate=validate,
        parameters=parameters,
    )


def runner_path_exists(
    runner: Any,
    env: dict[str, str],
    path: Path,
) -> bool:
    """Test path existence in the runner's host or container namespace."""
    path = Path(path)
    if not runner.using_container():
        return path.exists()
    marker = "__NRO_EXISTS__"
    output = (
        runner.run_child(
            [
                "bash",
                "-lc",
                f"if [ -e {shlex.quote(str(path))} ]; then printf {marker}; fi",
            ],
            env=env,
            capture_stdout=True,
        )
        or ""
    )
    return marker in strip_ansi(output)


def resolve_runner_command(
    runner: Any,
    env: dict[str, str],
    names: Sequence[str],
) -> str | None:
    """Resolve the first command name within a runner's active environment."""
    marker = "__NRO_CMD__="
    probes = " || ".join(
        f"(cmd=$(command -v {shlex.quote(name)} 2>/dev/null) && printf '{marker}%s' \"$cmd\")"
        for name in names
    )
    output = (
        runner.run_child(
            ["bash", "-c", probes + " || true"],
            env=env,
            capture_stdout=True,
        )
        or ""
    )
    for line in strip_ansi(output).splitlines():
        marker_index = line.find(marker)
        if marker_index >= 0:
            return line[marker_index + len(marker) :].strip()
    return None


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove terminal control sequences from captured command output."""
    return _ANSI_ESCAPE.sub("", text)


def ensure_directory(path: Path) -> None:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def require_existing_path(path: Path | None, description: str) -> None:
    """Reject an absent required input with a user-facing error."""
    if path is None:
        raise SystemExit(f"Missing required input: {description}")
    if not path.exists():
        raise SystemExit(f"Missing required file for {description}: {path}")


def neuroimaging_environment(*, subjects_dir: Path) -> dict[str, str]:
    """Build the common FSL, ITK, and FreeSurfer process environment."""
    environment = {
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "SUBJECTS_DIR": str(subjects_dir),
        **thread_environment(),
    }
    from nro.configuration.site import settings

    site, _ = settings()
    environment["FS_LICENSE"] = site["license"]
    return environment


def collect_bind_directories(paths: Iterable[Path | None]) -> list[str]:
    """Return minimal existing host directories covering the given paths.

    Output paths often do not exist when a module constructs its runner. Bind
    their nearest existing ancestor rather than handing a nonexistent mount
    source to the container engine. Input files continue to contribute their
    containing directories.
    """
    directories: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError(f"Container bind paths must be absolute host paths: {path}")
        # Do not resolve symlinks here. A host-visible symlink may deliberately
        # target an installation path that exists only inside the container.
        absolute = candidate.absolute()
        directory = absolute.parent if absolute.is_file() else absolute
        while not directory.is_dir() and directory.parent != directory:
            directory = directory.parent
        if not directory.is_dir():
            raise FileNotFoundError(f"No existing directory can anchor container bind for {path}")
        if directory == Path(directory.anchor):
            raise ValueError(
                f"Refusing to bind the host filesystem root while resolving container path: {path}"
            )
        directories.add(directory)

    minimal: list[Path] = []
    for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
        if any(directory == parent or directory.is_relative_to(parent) for parent in minimal):
            continue
        minimal.append(directory)
    return sorted(str(directory) for directory in minimal)
