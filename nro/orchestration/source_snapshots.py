"""Capture executable package contents without changing the development checkout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nro.orchestration.registry import RegistryLock, ensure_shared_directory
from nro.orchestration.source_launcher import verify_source


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files(root: Path) -> tuple[Path, ...]:
    package = root / "nro"
    if package.is_symlink() or not (package / "__init__.py").is_file():
        raise ValueError("Expected an nro source package")
    files = []
    for parent, directories, names in os.walk(package, followlinks=False):
        for name in list(directories):
            path = Path(parent) / name
            if name.startswith(".") or name == "__pycache__":
                directories.remove(name)
            elif path.is_symlink():
                raise ValueError(f"Source snapshots cannot contain symlinks: {path}")
        for name in names:
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Source snapshots require regular files: {path}")
            files.append(path.relative_to(root))
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Source snapshots cannot contain symlinks: {path}")
        if path.is_file():
            files.append(Path(name))
    return tuple(sorted(files))


def _entry(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "executable": bool(path.stat().st_mode & 0o111)}


def source_fingerprint(checkout: Path) -> str:
    """Hash executable source without creating a snapshot or recording Git metadata."""
    return _digest({str(path): _entry(checkout / path) for path in _files(checkout)})


@dataclass(frozen=True)
class SourceSnapshot:
    """A source tree addressed by content, independent of Git branch and version.

    Package data and executable bits are included. Python environments,
    container images, site settings, and definition stores are separate inputs;
    a source snapshot alone does not pin those resources.
    """

    root: Path
    digest: str

    def verify(self) -> None:
        """Reject missing, added, modified, or symlinked source files."""
        verify_source(self.root, self.digest)

    def command(self, command: Sequence[str], *, site: Path | None = None) -> tuple[str, ...]:
        """Bind a Python -m nro command to this source tree, retaining its arguments.

        The launcher verifies source and, when supplied, the resolved site file
        before running nro. The caller must separately select the Python environment.
        """
        if len(command) < 3 or command[1] != "-m" or not command[2].startswith("nro."):
            raise ValueError("Expected a Python -m nro command")
        launcher = self.root / "nro/orchestration/source_launcher.py"
        if not launcher.is_file():
            raise ValueError("Source snapshot has no execution launcher")
        site_path = str(site.resolve()) if site is not None else "-"
        site_digest = _entry(site)["sha256"] if site is not None else "-"
        return (
            str(command[0]),
            str(launcher),
            self.digest,
            site_path,
            site_digest,
            *map(str, command[2:]),
        )


class SourceStore:
    """Publish verified, content-addressed copies of nro's executable source.

    Capture includes non-hidden regular files under nro, including uncommitted
    code and package resources, plus pyproject.toml and uv.lock when present.
    It excludes checkout metadata, environments, documentation, and caches.
    Existing snapshots are checked and reused, never overwritten or repaired.
    """

    def __init__(self, root: Path) -> None:
        """Select a snapshot store without reading or creating it."""
        self.root = Path(root).expanduser().resolve()

    def capture(self, checkout: Path) -> SourceSnapshot:
        """Copy source, detect edits during capture, and atomically publish it.

        Source files are made read-only to prevent accidental in-place edits.
        Verification detects later replacement; this is not an OS sandbox.
        No Git refs or source files are changed. Concurrent identical captures
        reuse the same directory under the store's publication lock.
        """
        checkout = Path(checkout).expanduser().resolve()
        if self.root.is_relative_to(checkout / "nro"):
            raise ValueError("Source store cannot be inside the captured package")
        paths = _files(checkout)
        ensure_shared_directory(self.root)
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=self.root) as temporary:
            stage = Path(temporary) / "source"
            stage.mkdir()
            manifest = {}
            for relative in paths:
                source = checkout / relative
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target, follow_symlinks=False)
                if target.is_symlink():
                    raise ValueError("Source changed to a symlink during capture")
                target.chmod(0o555 if source.stat().st_mode & 0o111 else 0o444)
                manifest[str(relative)] = _entry(target)
            if paths != _files(checkout) or any(
                manifest[str(path)] != _entry(checkout / path) for path in paths
            ):
                raise ValueError("Source changed during capture; retry after edits finish")
            digest = _digest(manifest)
            (stage / "source.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
            (stage / "source.json").chmod(0o444)
            snapshot = SourceSnapshot(self.root / digest, digest)
            with RegistryLock(self.root / "publish.lock", self.root / "publish.recovery-lock"):
                if snapshot.root.exists():
                    snapshot.verify()
                else:
                    stage.rename(snapshot.root)
            return snapshot
