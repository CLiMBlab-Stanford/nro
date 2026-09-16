"""Shared container configuration for scientific modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nro.engine.paths import resolve_cwd_path
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import ContainerSpec


def _get(value: Any, name: str) -> Any:
    return value[name] if isinstance(value, dict) else getattr(value, name)


@dataclass(frozen=True)
class ContainerSettings:
    """Resolved image and runtime settings for one module container."""

    image: Path | None
    engine: str
    cleanenv: bool
    binds: tuple[str, ...]
    home: Path | None
    inner_setup: str
    disabled: bool

    @classmethod
    def from_config(cls, value: Any) -> "ContainerSettings":
        """Decode the common nested container configuration."""
        image = _get(value, "image")
        home = _get(value, "home")
        return cls(
            image=None if image is None else Path(str(image)),
            engine=str(_get(value, "engine")),
            cleanenv=bool(_get(value, "cleanenv")),
            binds=tuple(str(item) for item in _get(value, "bind")),
            home=None if home is None else Path(str(home)),
            inner_setup=str(_get(value, "inner_setup") or ""),
            disabled=bool(_get(value, "no_container")),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ContainerSettings":
        """Decode arguments added by :func:`add_container_arguments`."""
        return cls(
            image=args.container,
            engine=str(args.container_engine),
            cleanenv=not bool(args.container_no_cleanenv),
            binds=tuple(str(item) for item in (args.container_bind or ())),
            home=args.container_home,
            inner_setup=str(args.container_inner_setup or ""),
            disabled=bool(args.no_container),
        )


def add_container_arguments(parser: argparse.ArgumentParser, config: Any) -> None:
    """Add the common internal module arguments for one container block."""
    settings = ContainerSettings.from_config(config)
    parser.add_argument("--container", type=Path, default=settings.image)
    parser.add_argument("--no-container", action="store_true", default=settings.disabled)
    parser.add_argument("--container-engine", default=settings.engine)
    parser.add_argument(
        "--container-no-cleanenv",
        action="store_true",
        default=not settings.cleanenv,
    )
    parser.add_argument("--container-bind", action="append", default=list(settings.binds))
    parser.add_argument("--container-home", type=Path, default=settings.home)
    parser.add_argument("--container-inner-setup", default=settings.inner_setup)


def build_container(
    settings: ContainerSettings,
    *,
    work_directory: Path,
    execution_context: ExecutionContext | None,
) -> ContainerSpec | None:
    """Build a runner container and place its home in the private output tree."""
    if settings.disabled:
        return None
    if settings.image is None:
        raise ValueError("Container image is required unless container execution is disabled")
    image = resolve_cwd_path(settings.image) or settings.image
    home = (
        resolve_cwd_path(settings.home)
        if settings.home is not None
        else work_directory / "_qunex_home"
    )
    container = ContainerSpec(
        image=Path(image),
        engine=settings.engine,
        cleanenv=settings.cleanenv,
        extra_binds=settings.binds,
        home_dir=home,
        inner_setup=settings.inner_setup,
    )
    return bind_container_to_execution(container, execution_context)


def bind_container_to_execution(
    container: ContainerSpec | None,
    execution_context: ExecutionContext | None,
) -> ContainerSpec | None:
    """Route a container home through the authorized private output tree."""
    if container is None or execution_context is None or container.home_dir is None:
        return container
    return replace(
        container,
        home_dir=execution_context.output_path(container.home_dir, private=True),
    )
