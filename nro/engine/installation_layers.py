"""Build immutable dependency and application layers for shared installations."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from nro.engine.io import atomic_write_json

if TYPE_CHECKING:
    from nro.orchestration.source_snapshots import SourceSnapshot

DEPENDENCY_PROTOCOL = 1


def _dependency_lock_identity(path: Path) -> str:
    """Hash the resolved lock while excluding the release-only nro version."""
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"Cannot read the dependency lock: {path}") from error
    projects = [
        package
        for package in lock.get("package", ())
        if package.get("name") == "nro" and package.get("source") == {"editable": "."}
    ]
    if len(projects) != 1:
        raise RuntimeError("uv.lock must contain exactly one editable nro project")
    projects[0].pop("version", None)
    return hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dependency_spec(root: Path, options: dict, *, uv_version: str) -> dict:
    """Describe every input that can change a shared dependency environment."""
    lock = root / "uv.lock"
    if not lock.is_file() or lock.is_symlink():
        raise RuntimeError("Shared installation requires a regular uv.lock file")
    return {
        "protocol": DEPENDENCY_PROTOCOL,
        "lock_identity": _dependency_lock_identity(lock),
        "python": "3.12",
        "platform": sys.platform,
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "uv": uv_version,
        "extras": sorted(
            name
            for name, enabled in (
                ("oslom", options["with_oslom"]),
                ("bidsify", options["with_bidsify"]),
                ("lesion", options["with_lesion"]),
                ("marss", options["with_marss"]),
            )
            if enabled
        ),
        "dev": bool(options["dev"]),
    }


def _dependency_key(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dependency_marker(environment: Path) -> Path:
    return environment / ".nro-dependencies.json"


def _installed_packages(environment: Path) -> list[list[str]]:
    """Return a stable inventory of distributions visible to the environment."""
    program = (
        "import importlib.metadata as m, json; "
        "print(json.dumps(sorted(["
        "[(d.metadata.get('Name') or '').lower(), d.version] "
        "for d in m.distributions()])))"
    )
    result = subprocess.run(
        [str(environment / "bin/python"), "-I", "-B", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        inventory = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"Cannot inventory shared dependency environment: {environment}"
        ) from error
    if not isinstance(inventory, list):
        raise RuntimeError(f"Invalid package inventory for {environment}")
    return inventory


def _validate_dependency_environment(environment: Path, spec: dict, *, checkout: Path) -> None:
    """Reject an incomplete, redirected, or differently resolved dependency layer."""
    marker = _dependency_marker(environment)
    python = environment / "bin/python"
    if environment.is_symlink() or not environment.is_dir() or not python.is_file():
        raise RuntimeError(f"Invalid shared dependency environment: {environment}")
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid shared dependency marker: {marker}") from error
    if recorded.get("spec") != spec:
        raise RuntimeError(f"Shared dependency identity changed: {environment}")
    if recorded.get("packages") != _installed_packages(environment):
        raise RuntimeError(f"Shared dependency packages changed: {environment}")
    try:
        owner = (environment / ".nro-checkout").read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(
            f"Shared dependency checkout marker is missing: {environment}"
        ) from error
    if owner != str(checkout):
        raise RuntimeError(f"Shared dependency environment belongs to another checkout: {owner}")


def _sync_command(uv: Path, options: dict) -> list[str]:
    command = [str(uv), "sync", "--frozen", "--python", "3.12", "--no-install-project"]
    for extra in ("oslom", "bidsify", "lesion", "marss"):
        if options[f"with_{extra}"]:
            command += ["--extra", extra]
    if not options["dev"]:
        command += ["--no-dev"]
    return command


def _adopt_existing_environment(
    environment: Path,
    root: Path,
    uv: Path,
    options: dict,
    spec: dict,
    *,
    env: dict,
) -> bool:
    """Adopt a legacy shared environment when uv verifies its dependencies."""
    if (
        environment.is_symlink()
        or not (environment / "bin/python").is_file()
        or _dependency_marker(environment).exists()
    ):
        return False
    try:
        owner = (environment / ".nro-checkout").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if owner != str(root):
        return False
    command = [*_sync_command(uv, options), "--check", "--inexact"]
    check_env = {**env, "UV_PROJECT_ENVIRONMENT": str(environment)}
    try:
        subprocess.run(command, cwd=root, env=check_env, check=True)
    except subprocess.CalledProcessError:
        return False
    atomic_write_json(
        _dependency_marker(environment),
        {"spec": spec, "packages": _installed_packages(environment)},
        sort_keys=True,
    )
    return True


def prepare_shared_dependencies(
    root: Path,
    uv: Path,
    options: dict,
    *,
    uv_version: str,
    env: dict,
    offline: bool,
    previous_environment: Path | None = None,
) -> tuple[Path, str]:
    """Create or reuse the dependency environment selected by installation inputs."""
    root = root.resolve()
    spec = _dependency_spec(root, options, uv_version=uv_version)
    key = _dependency_key(spec)
    parent = root / ".nro-environments"
    parent.mkdir(parents=True, exist_ok=True)
    environment = parent / f"dependencies-{key}"
    if environment.exists():
        subprocess.run([str(uv), "lock", "--check"], cwd=root, env=env, check=True)
        _validate_dependency_environment(environment, spec, checkout=root)
        print(f"Reusing shared dependency environment {environment}", flush=True)
        return environment, key
    if previous_environment is not None:
        previous_environment = previous_environment.resolve()
        if _dependency_marker(previous_environment).is_file():
            try:
                _validate_dependency_environment(previous_environment, spec, checkout=root)
            except RuntimeError:
                pass
            else:
                subprocess.run([str(uv), "lock", "--check"], cwd=root, env=env, check=True)
                print(f"Reusing active shared environment {previous_environment}", flush=True)
                return previous_environment, key
        elif _adopt_existing_environment(previous_environment, root, uv, options, spec, env=env):
            print(f"Reusing active shared environment {previous_environment}", flush=True)
            return previous_environment, key
    stage = parent / f".dependencies-{key}.tmp-{uuid.uuid4().hex}"
    command = _sync_command(uv, options)
    if offline:
        command += ["--offline"]
    build_env = {**env, "UV_PROJECT_ENVIRONMENT": str(stage)}
    try:
        subprocess.run(command, cwd=root, env=build_env, check=True)
        (stage / ".nro-checkout").write_text(f"{root}\n", encoding="utf-8")
        atomic_write_json(
            _dependency_marker(stage),
            {"spec": spec, "packages": _installed_packages(stage)},
            sort_keys=True,
        )
        stage.rename(environment)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    _validate_dependency_environment(environment, spec, checkout=root)
    return environment, key


def capture_shared_application(root: Path) -> SourceSnapshot:
    """Publish release source separately from its reusable dependencies."""
    from nro.orchestration.source_snapshots import SourceStore

    root = root.resolve()
    return SourceStore(root / ".nro-environments" / "applications").capture(root)


def prune_shared_environments(
    root: Path, active: Path, *, active_application: Path | None = None
) -> None:
    """Remove inactive dependency and application layers after a successful cutover."""
    parent = root / ".nro-environments"
    if not parent.is_dir() or active.parent != parent:
        return
    for path in (*parent.glob("candidate-*"), *parent.glob("dependencies-*")):
        if path != active and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
    applications = parent / "applications"
    if applications.is_dir() and active_application is not None:
        for path in applications.iterdir():
            if (
                path != active_application
                and path.name != "publish.lock"
                and path.name != "publish.recovery-lock"
                and path.is_dir()
                and not path.is_symlink()
            ):
                shutil.rmtree(path, ignore_errors=True)
