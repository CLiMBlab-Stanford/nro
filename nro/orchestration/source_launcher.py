"""Internal script entry point for executing a verified source snapshot.

Run by absolute filename so no installed nro package is imported before the
snapshot is verified. This file uses only Python's standard library.
"""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def verify_source(root: Path, expected_digest: str) -> None:
    """Check source content and reject extra files, symlinks, or bytecode caches."""
    if root.resolve() != root or (root / "source.json").is_symlink():
        raise ValueError("Source snapshot cannot be a symlink")
    manifest = json.loads((root / "source.json").read_text())
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != expected_digest:
        raise ValueError("Source snapshot manifest changed")
    for name, entry in manifest.items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.resolve() != path
            or not path.is_file()
        ):
            raise ValueError("Invalid source snapshot file")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"] or bool(path.stat().st_mode & 0o111) != entry["executable"]:
            raise ValueError(f"Source snapshot file changed: {name}")
    # An extra Python file could override a module without changing listed files.
    actual_files = set()
    for parent, directories, files in os.walk(root / "nro", followlinks=False):
        for directory in list(directories):
            path = Path(parent) / directory
            if directory == "__pycache__":
                raise ValueError("Source snapshot contains a bytecode cache")
            if directory.startswith("."):
                directories.remove(directory)
            elif path.is_symlink():
                raise ValueError("Source snapshot directory changed to a symlink")
        if any(name.endswith((".pyc", ".pyo")) for name in files):
            raise ValueError("Source snapshot contains a bytecode cache")
        actual_files.update(
            str((Path(parent) / name).relative_to(root))
            for name in files
            if not name.startswith(".")
        )
    actual_files.update(name for name in ("pyproject.toml", "uv.lock") if (root / name).exists())
    if actual_files != set(manifest):
        raise ValueError("Source snapshot file list changed")


def main() -> None:
    """Verify the expected source digest, then execute the requested nro module."""
    if len(sys.argv) < 5 or not sys.argv[4].startswith("nro."):
        raise SystemExit("Expected source digest, site path and digest, and nro module name")
    root = Path(__file__).resolve().parents[2]
    try:
        verify_source(root, sys.argv[1])
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    site_path, site_digest = sys.argv[2:4]
    if site_path != "-":
        site = Path(site_path)
        if site.resolve() != site or not site.is_file():
            raise SystemExit("Invalid execution site file")
        if hashlib.sha256(site.read_bytes()).hexdigest() != site_digest:
            raise SystemExit("Execution site settings changed")
    module = sys.argv[4]
    sys.argv = [module, *sys.argv[5:]]
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.chdir(root)
    if site_path != "-":
        from nro.configuration.site import ENVIRONMENT_KEYS

        for key in ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ["NRO_SITE_CONFIG"] = site_path
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
