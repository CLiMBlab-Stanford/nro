"""Rehearse scheduler installation maintenance in an isolated control store."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from nro.configuration.definition_migrations import (
    SCHEMA_VERSION as DEFINITIONS_SCHEMA_VERSION,
)
from nro.configuration.definition_migrations import (
    store_schema,
)
from nro.engine.site_setup import save_settings
from nro.orchestration.branch_registry import SCHEMA_VERSION as SCIENTIFIC_SCHEMA_VERSION
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import SCHEMA_VERSION
from nro.orchestration.scheduler_implementation import implementation_path
from nro.orchestration.source_snapshots import SourceStore


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _default_baseline(checkout: Path) -> str:
    """Select the newest release reachable from the candidate checkout."""
    tags = _git(checkout, "tag", "--merged", "HEAD", "--sort=-version:refname", "--list", "v*")
    if not tags:
        raise RuntimeError("No release tag is reachable from this checkout; pass a baseline ref")
    return tags.splitlines()[0]


def _copy_candidate(candidate: Path, target: Path) -> None:
    """Replace executable snapshot inputs while retaining the synthetic installation state."""
    package = target / "nro"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(
        candidate / "nro",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for name in ("pyproject.toml", "uv.lock"):
        source = candidate / name
        destination = target / name
        if source.is_file():
            shutil.copyfile(source, destination)
        else:
            destination.unlink(missing_ok=True)


def _initialize_control(
    checkout: Path,
    site: Path,
    control: Path,
    bids: Path,
    python: Path,
) -> None:
    """Initialize isolated control state from the candidate checkout."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from nro.orchestration.branch_store import BranchStore\n"
        "from nro.orchestration.registry import Registry\n"
        "control, bids, checkout = map(Path, sys.argv[1:])\n"
        "registry = Registry.for_project('', bids_root=bids, registry_path=control)\n"
        "registry.initialize()\n"
        "branches = BranchStore(control)\n"
        "state = branches.initialize()\n"
        "branches.authorize_checkout('main', checkout, revision=state.revision)\n"
    )
    environment = {
        **os.environ,
        "NRO_SITE_CONFIG": str(site),
        "PYTHONPATH": str(checkout),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in (
        "NRO_EXECUTION_SOURCE_DIGEST",
        "NRO_EXECUTION_SOURCE_ROOT",
        "NRO_PROCESS_ROLE",
        "NRO_SCHEDULER_MAINTENANCE",
    ):
        environment.pop(key, None)
    subprocess.run(
        [str(python), "-B", "-c", code, str(control), str(bids), str(checkout)],
        cwd=checkout,
        env=environment,
        check=True,
    )


def _initialize_definitions(
    checkout: Path,
    site: Path,
    definitions: Path,
    python: Path,
) -> None:
    """Create a definitions store with the baseline release implementation."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from nro.configuration.definitions import create_store\n"
        "create_store(Path(sys.argv[1]))\n"
    )
    environment = {
        **os.environ,
        "NRO_SITE_CONFIG": str(site),
        "PYTHONPATH": str(checkout),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [str(python), "-B", "-c", code, str(definitions)],
        cwd=checkout,
        env=environment,
        check=True,
    )


def _prepare_pool(
    checkout: Path,
    site: Path,
    control: Path,
    bids: Path,
    definitions: Path,
    python: Path,
) -> None:
    """Run candidate definitions and registry maintenance from the shared checkout."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from nro.configuration.definitions import ensure_store\n"
        "from nro.engine.shared_installation import prepare_pool\n"
        "from nro.engine.site_setup import migrate_site_configuration\n"
        "from nro.orchestration.registry import Registry\n"
        "control, bids, checkout, definitions, site = map(Path, sys.argv[1:])\n"
        "migrate_site_configuration(site)\n"
        "ensure_store(definitions)\n"
        "migrate_site_configuration(site)\n"
        "registry = Registry.for_project('', bids_root=bids, registry_path=control, "
        "installation_maintenance=True)\n"
        "result = prepare_pool(registry, checkout=checkout, confirm=lambda _: 'stop', "
        "poll_interval=0.01, report_interval=60.0, update_schema=True)\n"
        "assert result['done']\n"
    )
    environment = {
        **os.environ,
        "NRO_SITE_CONFIG": str(site),
        "PYTHONPATH": str(checkout),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in (
        "NRO_EXECUTION_SOURCE_DIGEST",
        "NRO_EXECUTION_SOURCE_ROOT",
        "NRO_PROCESS_ROLE",
        "NRO_SCHEDULER_MAINTENANCE",
    ):
        environment.pop(key, None)
    subprocess.run(
        [
            str(python),
            "-B",
            "-c",
            code,
            str(control),
            str(bids),
            str(checkout),
            str(definitions),
            str(site),
        ],
        cwd=checkout,
        env=environment,
        check=True,
    )


def rehearse(checkout: Path, *, baseline: str | None = None) -> dict:
    """Exercise an old-to-new maintenance transition without using shared site state.

    The baseline supplies the active source snapshot. The candidate working tree then
    replaces it at the same checkout path, matching what happens after ``git pull``.
    The candidate must coordinate maintenance through its real source launcher and
    one-shot scheduler process. All registry, BIDS, and installation files live in a
    temporary directory.
    """
    checkout = Path(checkout).expanduser().resolve()
    if not (checkout / "nro/__init__.py").is_file() or not (checkout / ".git").exists():
        raise ValueError("Upgrade rehearsal requires an nro Git checkout")
    baseline = baseline or _default_baseline(checkout)
    _git(checkout, "rev-parse", "--verify", f"{baseline}^{{commit}}")
    managed_python = checkout / ".venv/bin/python"
    python = managed_python if managed_python.is_file() else Path(sys.executable)
    # The rehearsal is local and process-scoped. Keep it on the node-local
    # filesystem so delayed directory metadata on shared storage cannot race
    # cleanup after the one-shot scheduler exits.
    with tempfile.TemporaryDirectory(prefix="nro-upgrade-rehearsal-", dir="/tmp") as temporary_name:
        temporary = Path(temporary_name)
        synthetic = temporary / "shared"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(checkout), str(synthetic)],
            check=True,
        )
        _git(synthetic, "checkout", "--quiet", "-B", "main", baseline)

        bids = temporary / "BIDS"
        work = temporary / "WORK"
        development = temporary / "NRO_DEV"
        definitions = temporary / "definitions"
        control = temporary / "control"
        for path in (bids, work, development, definitions):
            if path != definitions:
                path.mkdir()
        site = temporary / "site.toml"
        save_settings(
            site,
            {
                "bids": str(bids),
                "work": str(work),
                "development": str(development),
                "definitions": str(definitions),
                "registry": str(control),
                "binds": [],
            },
        )
        _initialize_definitions(synthetic, site, definitions, python)
        environment = python.parent.parent
        release = {
            "version": "0.0.0",
            "commit": _git(synthetic, "rev-parse", "HEAD"),
            "tree": _git(synthetic, "rev-parse", "HEAD^{tree}"),
            "registry_id": "upgrade-rehearsal",
        }
        installation = {
            "mode": "shared",
            "checkout": str(synthetic),
            "environment": str(environment),
            "site": str(site),
            "ready": True,
            "release": release,
        }
        (synthetic / ".nro-installation.json").write_text(json.dumps(installation, indent=2) + "\n")
        source_store = SourceStore(ControlPaths(control).implementations)
        baseline_source = source_store.capture(synthetic)
        binding = {
            "protocol": 1,
            "checkout": str(synthetic),
            "python": str(python),
            "site": str(site),
            "release": release,
            "source_digest": baseline_source.digest,
        }
        binding_path = implementation_path(control)
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")

        _initialize_control(synthetic, site, control, bids, python)
        database_path = ControlPaths(control).database
        scientific_path = ControlPaths(control).branch("main") / "registry.sqlite3"

        _copy_candidate(checkout, synthetic)
        _prepare_pool(synthetic, site, control, bids, definitions, python)
        with sqlite3.connect(database_path) as database:
            rebuilt_schema = int(database.execute("PRAGMA user_version").fetchone()[0])
        if rebuilt_schema != SCHEMA_VERSION:
            raise RuntimeError("Rehearsal did not produce the candidate scheduler schema")
        with sqlite3.connect(scientific_path) as database:
            scientific_schema = int(database.execute("PRAGMA user_version").fetchone()[0])
        if scientific_schema != SCIENTIFIC_SCHEMA_VERSION:
            raise RuntimeError("Rehearsal did not produce the candidate scientific schema")
        definitions_schema = store_schema(definitions)
        if definitions_schema != DEFINITIONS_SCHEMA_VERSION:
            raise RuntimeError("Rehearsal did not produce the candidate definitions schema")
        if json.loads(implementation_path(control).read_text()) != binding:
            raise RuntimeError("Maintenance changed the active implementation before publication")
        return {
            "baseline": baseline,
            "baseline_source": baseline_source.digest,
            "candidate_source": source_store.capture(synthetic).digest,
            "definitions_schema": definitions_schema,
        }


def main(argv: list[str] | None = None) -> None:
    """Run the isolated rehearsal and report the two source identities."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--baseline")
    args = parser.parse_args(argv)
    result = rehearse(args.checkout, baseline=args.baseline)
    print(f"Upgrade rehearsal passed: {result['baseline']}")
    print(f"  active source:    {result['baseline_source']}")
    print(f"  candidate source: {result['candidate_source']}")


if __name__ == "__main__":
    main()
