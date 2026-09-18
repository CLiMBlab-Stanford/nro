"""Inspect and validate generated registry schemas during development."""

from __future__ import annotations

import difflib
import json
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from nro.orchestration.migrations.core import RegistrySchema, schema_fingerprint


def families() -> Mapping[str, RegistrySchema]:
    """Return the registry families without making the migration core import them."""
    from nro.orchestration.branch_registry import SCHEMA_DEFINITION as scientific
    from nro.orchestration.registry_schema import SCHEMA as scheduler

    return {"scheduler": scheduler, "scientific": scientific}


def _baseline_manifest() -> dict:
    resource = files("nro.orchestration.migrations").joinpath("baselines.json")
    return json.loads(resource.read_text())


def validate_families() -> None:
    """Validate immutable baselines, contiguous chains, and every generated schema."""
    manifest = _baseline_manifest()
    if set(manifest) != set(families()):
        raise ValueError("Registry baseline manifest does not match the defined families")
    for name, schema in families().items():
        baseline = manifest[name]
        if baseline != {
            "version": schema.baseline_version,
            "fingerprint": schema_fingerprint(schema.baseline_sql),
        }:
            raise ValueError(
                f"The immutable {name} baseline changed. Add a migration instead of "
                "editing the baseline."
            )
        database = schema.build()
        try:
            schema.validate(database)
        finally:
            database.close()


def require_unchanged_baselines(root: Path, base_ref: str) -> None:
    """Reject a PR that rewrites a baseline or an existing migration."""
    relative = "nro/orchestration/migrations/baselines.json"
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{base_ref}:{relative}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return  # The comparison target predates the migration apparatus.
    current = (root / relative).read_text()
    if current != result.stdout:
        raise ValueError(
            "Registry migration baselines cannot change in an ordinary PR; "
            "advance them through the explicit baseline-retirement process"
        )
    migration_roots = (
        "nro/orchestration/migrations/scheduler",
        "nro/orchestration/migrations/scientific",
    )
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-status", base_ref, "--", *migration_roots],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rewritten = [line for line in changed if line and not line.startswith("A\t")]
    if rewritten:
        raise ValueError(
            "Published registry migrations are immutable; add the next destination file: "
            + ", ".join(rewritten)
        )


def render(name: str, *, version: int | None = None) -> str:
    """Render one generated registry schema at a supported version."""
    try:
        schema = families()[name]
    except KeyError as error:
        raise ValueError(f"Unknown registry family: {name}") from error
    return schema.sql(version=version)


def diff(name: str, old: int, new: int) -> str:
    """Return a unified diff between two generated schema versions."""
    before = render(name, version=old).splitlines(keepends=True)
    after = render(name, version=new).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(before, after, fromfile=f"{name}-v{old}", tofile=f"{name}-v{new}")
    )
