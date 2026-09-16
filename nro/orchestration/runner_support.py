"""Container policy and small serialization helpers used by the runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from nro.engine.io import atomic_write_text


def shlex_quote(value: str) -> str:
    """Quote one shell token for readable command logging."""
    if not value:
        return "''"
    if all(character.isalnum() or character in "._/+-=:" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_bind_spec(bind_spec: str) -> tuple[Path, str, str | None] | None:
    """Parse a container bind into its host source, target, and options."""
    fields = bind_spec.split(":")
    if len(fields) == 1:
        source = destination = fields[0]
        options = None
    elif len(fields) in {2, 3}:
        source, destination = fields[:2]
        options = fields[2] if len(fields) == 3 else None
    else:
        return None
    if not source or not destination:
        return None
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        return None
    return source_path, str(destination_path), options


@dataclass(frozen=True)
class ContainerSpec:
    """Container image and execution policy for one runner.

    The image is a host path; bind mappings, clean environment, container home,
    and inner setup determine how host commands execute inside it.
    """

    image: Path
    engine: str = "singularity"
    cleanenv: bool = True
    extra_binds: Tuple[str, ...] = ()
    home_dir: Optional[Path] = None
    inner_setup: str = ""


def write_completion_breadcrumb(path: Path, text: str = "complete\n") -> Path:
    """Atomically mark a directory-producing or compound step complete."""
    atomic_write_text(path, text)
    return path
