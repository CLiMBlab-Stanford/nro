"""Capture source and resolved site settings when creating derivative demand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nro.configuration.site import definitions_root, settings, validate_setting
from nro.engine.io import atomic_write_text
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory
from nro.orchestration.source_snapshots import SourceSnapshot, SourceStore


def capture_site(root: Path, values: dict) -> Path:
    """Store resolved site settings by content without overwriting existing records."""
    for key, value in values.items():
        validate_setting(key, value)
    text = "".join(f"{key} = {json.dumps(value)}\n" for key, value in sorted(values.items()))
    digest = hashlib.sha256(text.encode()).hexdigest()
    ensure_shared_directory(root)
    path = root / f"{digest}.toml"
    with RegistryLock(root / "publish.lock", root / "publish.recovery-lock"):
        if path.exists():
            if path.is_symlink() or path.read_text() != text:
                raise ValueError("Execution site settings changed")
        else:
            atomic_write_text(path, text, mode=0o444, durable=True)
    return path


def capture_execution(
    control: Path,
    bids_root: Path,
    *,
    expected_source: str | None = None,
    site_values: dict | None = None,
) -> tuple[SourceSnapshot, Path]:
    """Pin nro source and site paths for demand, excluding them from freshness.

    This does not copy Python environments, images, or definitions. The resolved
    module configuration remains separately pinned by the work registry.
    """
    paths = ControlPaths(control)
    paths.require_current_layout()
    source = SourceStore(paths.implementations).capture(
        Path(__file__).resolve().parents[2], expected_digest=expected_source
    )
    if expected_source is not None and source.digest != expected_source:
        raise ValueError("Source changed during planning; retry the request after edits finish")
    values = dict(site_values) if site_values is not None else settings()[0]
    if site_values is None:
        values["definitions"] = str(definitions_root())
    values.update(bids=str(bids_root.resolve()), registry=str(control.resolve()))
    site = capture_site(paths.execution_sites, values)
    return source, site
