"""Build and activate the installer environment from pinned managed tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

PINS_PATH = Path(__file__).with_name("bootstrap.pins")
TRANSITION = ".nro-bootstrap-transition.json"


def read_pins(path: Path = PINS_PATH) -> dict[str, str]:
    """Read the shell-compatible bootstrap pin file."""
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid bootstrap pin at {path}:{number}")
        if key in pins:
            raise ValueError(f"Duplicate bootstrap pin {key} at {path}:{number}")
        pins[key] = value
    required = {"UV_VERSION", "PYTHON_VERSION", "PYYAML_VERSION", "PYYAML_SHA256"}
    missing = required - pins.keys()
    if missing:
        raise ValueError(f"Missing bootstrap pin(s): {', '.join(sorted(missing))}")
    hexadecimal = set("0123456789abcdef")
    for key, value in pins.items():
        if key.endswith("SHA256") and (len(value) != 64 or set(value) - hexadecimal):
            raise ValueError(f"Invalid SHA-256 bootstrap pin: {key}")
        if key.startswith("UV_ARCHIVE_"):
            checksum = "UV_SHA256_" + key.removeprefix("UV_ARCHIVE_")
            if checksum not in pins:
                raise ValueError(f"Missing bootstrap pin: {checksum}")
    return pins


PINS = read_pins()
UV_VERSION = PINS["UV_VERSION"]
PYTHON_VERSION = PINS["PYTHON_VERSION"]
PYYAML_VERSION = PINS["PYYAML_VERSION"]


def pin_digest(path: Path = PINS_PATH) -> str:
    """Return the identity shared by a bootstrap environment and its pins."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_target(root: Path, path: Path = PINS_PATH) -> Path:
    """Return the in-place venv selected by the complete bootstrap pin set."""
    return root / ".nro-bootstraps" / pin_digest(path)


def _ready(python: Path, *, root: Path, pins: dict[str, str] | None = None) -> bool:
    """Return whether a bootstrap uses the pinned uv-managed interpreter and package."""
    pins = pins or PINS
    if not python.is_file():
        return False
    program = (
        "import importlib.metadata as m,json,sys; "
        "print(json.dumps({'version':list(sys.version_info[:3]),"
        "'base':sys._base_executable,'yaml':m.version('PyYAML'),"
        "'uv':next((d.version for d in m.distributions() "
        "if (d.metadata.get('Name') or '').lower()=='uv'),None)}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return False
    try:
        record = json.loads(result.stdout)
        expected = [int(part) for part in pins["PYTHON_VERSION"].split(".")]
        base = Path(record["base"]).resolve()
        managed = (root / ".nro-python").resolve()
        return (
            record["version"] == expected
            and record["yaml"] == pins["PYYAML_VERSION"]
            and record["uv"] is None
            and base.is_relative_to(managed)
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def prepare_target(
    root: Path,
    uv: Path,
    *,
    offline: bool,
    pins_path: Path = PINS_PATH,
) -> Path:
    """Build the content-addressed bootstrap venv in its final location."""
    pins = read_pins(pins_path)
    target = bootstrap_target(root, pins_path)
    python = target / "bin/python"
    if _ready(python, root=root, pins=pins):
        return target
    active = root / ".nro-bootstrap"
    if active.is_symlink() and active.resolve() == target.resolve():
        raise RuntimeError("The active nro bootstrap is incomplete; maintenance cannot replace it")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(uv),
        "venv",
        str(target),
        "--python",
        pins["PYTHON_VERSION"],
        "--managed-python",
    ]
    if offline:
        command.append("--offline")
    requirements: Path | None = None
    try:
        subprocess.run(command, check=True)
        requirement = f"PyYAML=={pins['PYYAML_VERSION']} --hash=sha256:{pins['PYYAML_SHA256']}\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as stream:
            requirements = Path(stream.name)
            stream.write(requirement)
        command = [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "--requirements",
            str(requirements),
        ]
        if offline:
            command.append("--offline")
        subprocess.run(command, check=True)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        if requirements is not None:
            requirements.unlink(missing_ok=True)
    if not _ready(python, root=root, pins=pins):
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError("The managed nro bootstrap did not pass interpreter validation")
    for path in (target, *target.rglob("*")):
        try:
            mode = path.stat().st_mode | 0o040
            if path.is_dir() or os.access(path, os.X_OK):
                mode |= 0o010
            path.chmod(mode)
        except OSError:
            pass
    return target


def _write_journal(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_target(root: Path, target: Path) -> Path:
    return target.relative_to(root)


def _publish_symlink(active: Path, target: Path) -> None:
    temporary = active.with_name(f".{active.name}.tmp-{uuid.uuid4().hex}")
    temporary.symlink_to(_relative_target(active.parent, target))
    temporary.replace(active)
    descriptor = os.open(active.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def activate_target(root: Path, target: Path) -> None:
    """Activate a prepared bootstrap and recover an interrupted prior cutover."""
    active = root / ".nro-bootstrap"
    journal_path = root / TRANSITION
    pins = read_pins()
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else None
    if not _ready(target / "bin/python", root=root, pins=pins):
        backup = root / journal["backup"] if journal and journal.get("backup") else None
        if backup is not None and backup.exists() and (not active.exists() or active.is_symlink()):
            active.unlink(missing_ok=True)
            backup.rename(active)
            journal_path.unlink(missing_ok=True)
            raise RuntimeError("Rolled back an incomplete nro bootstrap transition")
        raise RuntimeError(f"Cannot activate incomplete nro bootstrap: {target}")
    if journal is not None:
        recorded_target = root / journal["target"]
        backup = root / journal["backup"] if journal.get("backup") else None
        if recorded_target != target:
            raise RuntimeError("A different nro bootstrap transition is incomplete")
        if active.is_symlink() and active.resolve() == target.resolve():
            if backup is not None and backup.exists():
                shutil.rmtree(backup)
            journal_path.unlink()
            return
        if not active.exists() and backup is not None and backup.exists():
            _publish_symlink(active, target)
            shutil.rmtree(backup)
            journal_path.unlink()
            return
    if active.is_symlink() and active.resolve() == target.resolve():
        return
    if active.exists() and not active.is_symlink():
        backup = root / f".nro-bootstrap.previous-{uuid.uuid4().hex}"
        journal = {
            "target": str(_relative_target(root, target)),
            "backup": str(_relative_target(root, backup)),
            "phase": "prepared",
        }
        _write_journal(journal_path, journal)
        active.rename(backup)
        journal["phase"] = "renamed"
        _write_journal(journal_path, journal)
        try:
            _publish_symlink(active, target)
            if not _ready(active / "bin/python", root=root, pins=pins):
                raise RuntimeError("The activated nro bootstrap failed validation")
        except BaseException:
            active.unlink(missing_ok=True)
            backup.rename(active)
            journal_path.unlink(missing_ok=True)
            raise
        shutil.rmtree(backup)
        journal_path.unlink()
        return
    if active.is_symlink():
        previous = os.readlink(active)
        journal = {
            "target": str(_relative_target(root, target)),
            "previous": previous,
            "phase": "prepared",
        }
        _write_journal(journal_path, journal)
        try:
            _publish_symlink(active, target)
            if not _ready(active / "bin/python", root=root, pins=pins):
                raise RuntimeError("The activated nro bootstrap failed validation")
        except BaseException:
            rollback = active.with_name(f".{active.name}.rollback-{uuid.uuid4().hex}")
            rollback.symlink_to(previous)
            rollback.replace(active)
            journal_path.unlink(missing_ok=True)
            raise
        journal_path.unlink()
        return
    _publish_symlink(active, target)


def main(argv: list[str] | None = None) -> None:
    """Prepare a bootstrap target for the POSIX shell entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--activate", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    target = prepare_target(root, arguments.uv, offline=arguments.offline)
    if arguments.activate:
        activate_target(root, target)


if __name__ == "__main__":
    main()
