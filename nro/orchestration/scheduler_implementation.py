"""Select the site's orchestration installation independently of job source."""

from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar
from pathlib import Path

from nro.configuration.site import installation_record, settings
from nro.engine.io import atomic_write_json, atomic_write_text
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.execution_pins import capture_site
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.source_snapshots import SourceSnapshot, SourceStore, source_fingerprint

# The installed scheduler and worker shell need only this stable site protocol.
# Scientific subprocesses receive a separate, complete execution-site snapshot
# captured with their own source version.
_COORDINATION_SITE_KEYS = frozenset(
    {
        "definitions",
        "bids",
        "work",
        "development",
        "registry",
        "images",
        "gradient_coefficients",
        "templates",
        "workbench",
        "oslom",
        "pycicada",
        "license",
        "runtime",
        "partition",
        "viewing_partition",
        "account",
        "flywheel_server",
        "flywheel_project",
        "binds",
    }
)

_CENTRAL_SOURCE_VERIFIED: ContextVar[bool] = ContextVar(
    "nro_central_source_verified", default=False
)


def central_source_verified() -> bool:
    """Return whether this process verified the active central implementation."""
    return _CENTRAL_SOURCE_VERIFIED.get()


def _coordination_site(values: dict) -> dict:
    """Exclude version-specific scientific resources from scheduler settings."""
    return {key: values[key] for key in _COORDINATION_SITE_KEYS if key in values}


def implementation_path(control: Path) -> Path:
    """Locate the explicit central installation binding, not a code archive."""
    return ControlPaths(control).scheduler / "implementation.json"


def activate(
    registry,
    checkout: Path,
    *,
    installation_maintenance: bool = False,
    installation: dict | None = None,
    installation_path: Path | None = None,
) -> dict:
    """Designate a recorded main installation while site execution is quiescent.

    Installation maintenance may preserve inactive demand behind its global
    barrier. This changes neither Git refs nor the user's command launcher.
    Future workers use this interpreter and a per-submission source snapshot.
    """
    from nro.orchestration.execution_cache import _busy, cache_lock

    checkout = Path(checkout).expanduser().resolve()
    installation = dict(installation or installation_record(checkout))
    if installation.get("mode") != "shared" or not installation.get("ready"):
        raise ValueError("Scheduler activation requires a ready shared installation")
    python = Path(installation["environment"]) / "bin/python"
    site = Path(installation["site"])
    values = settings(path=site)[0]
    if (
        Path(values["registry"]).resolve() != registry.paths.control.resolve()
        or Path(values["bids"]).resolve() != registry.paths.bids_root.resolve()
    ):
        raise ValueError("Scheduler installation belongs to another site")
    if not python.is_file():
        raise ValueError("Scheduler interpreter is unavailable")
    if installation_path is not None and not installation_maintenance:
        raise ValueError("Installation publication requires installation maintenance")
    with cache_lock(registry.paths.control):
        release = ReleaseStore(BranchStore(registry.paths.control)).require_approved(checkout)
        path = implementation_path(registry.paths.control)
        installation_path = Path(installation_path) if installation_path is not None else None
        previous_binding = path.read_bytes() if path.is_file() else None
        previous_installation = (
            installation_path.read_bytes()
            if installation_path is not None and installation_path.is_file()
            else None
        )
        try:
            with registry.connection(write=installation_maintenance) as db:
                if installation_maintenance:
                    owner = db.execute(
                        "SELECT value FROM metadata WHERE key='installation_checkout'"
                    ).fetchone()
                    if owner is None or owner["value"] != str(checkout):
                        raise ValueError(
                            "Scheduler activation does not own installation maintenance"
                        )
                busy = _busy(
                    registry,
                    db,
                    preserve_demand=installation_maintenance,
                    allowed_maintenance="installation" if installation_maintenance else None,
                )
                if busy:
                    raise ValueError(f"Cannot activate the scheduler during {busy}")
                source = SourceStore(ControlPaths(registry.paths.control).implementations).capture(
                    checkout
                )
                if installation.get("application") is not None:
                    application = SourceSnapshot(
                        Path(installation["application"]),
                        installation.get("application_digest", ""),
                    )
                    application.verify_manifest()
                    if application.digest != source.digest:
                        raise ValueError(
                            "Shared application layer does not match the approved release source"
                        )
                record = dict(
                    protocol=1,
                    checkout=str(checkout),
                    python=str(python),
                    site=str(site),
                    release=release,
                    source_digest=source.digest,
                )
                if path.is_symlink():
                    raise ValueError("Scheduler binding cannot be a symlink")
                atomic_write_json(path, record, mode=0o664, durable=True)
                if installation_path is not None:
                    atomic_write_json(installation_path, installation, mode=0o644, durable=True)
                if installation_maintenance:
                    db.execute(
                        "DELETE FROM metadata WHERE "
                        "(key='maintenance_mode' AND value='installation') "
                        "OR key IN ('installation_checkout','installation_action')"
                    )
        except BaseException:
            for target, previous, mode in (
                (path, previous_binding, 0o664),
                (installation_path, previous_installation, 0o644),
            ):
                if target is None:
                    continue
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_write_text(target, previous.decode("utf-8"), mode=mode, durable=True)
            raise
    return record


def capture_worker_implementation(control: Path, bids_root: Path, *, check_checkout: bool = True):
    """Capture the designated orchestration source, interpreter, and resolved site.

    Call while holding the execution-cache publication lock. Activation has
    already published the content-addressed source snapshot, so routine calls
    reuse it without rescanning or copying the checkout. The source launcher
    verifies the snapshot before executing it. The caller's development
    checkout is never a substitute.
    """
    path = implementation_path(control)
    if not path.exists():
        if installation_record().get("mode") == "branch":
            raise ValueError(
                "No central scheduler is active; activate an approved main installation"
            )
        from nro.orchestration.execution_pins import capture_execution

        source, site = capture_execution(control, bids_root)
        return source, site, Path(sys.executable)
    if path.is_symlink():
        raise ValueError("Scheduler binding cannot be a symlink")
    record = json.loads(path.read_text())
    if (
        set(record) != {"protocol", "checkout", "python", "site", "release", "source_digest"}
        or record["protocol"] != 1
    ):
        raise ValueError("Unsupported scheduler installation binding")
    digest = record["source_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Scheduler installation has an invalid source digest")
    for key in ("checkout", "python", "site"):
        if not isinstance(record[key], str) or not Path(record[key]).is_absolute():
            raise ValueError("Scheduler installation paths must be absolute")
    checkout = Path(record["checkout"])
    installed = installation_record(checkout)
    if (
        installed.get("mode") != "shared"
        or not installed.get("ready")
        or str(Path(installed["environment"]) / "bin/python") != record["python"]
        or installed["site"] != record["site"]
        or not Path(record["python"]).is_file()
        or installed.get("release") != record["release"]
    ):
        raise ValueError("The designated scheduler installation changed or is unavailable")
    releases = ReleaseStore(BranchStore(control))
    if (
        releases.require_recorded(checkout, record["release"], check_head=check_checkout)
        != record["release"]
    ):
        raise ValueError(
            "Scheduler release changed; activate it explicitly after draining the pool"
        )
    values = settings(path=Path(record["site"]))[0]
    if (
        Path(values["registry"]).resolve() != Path(control).resolve()
        or Path(values["bids"]).resolve() != Path(bids_root).resolve()
    ):
        raise ValueError("The designated scheduler site changed")
    paths = ControlPaths(control)
    source = SourceSnapshot(paths.implementations / digest, digest)
    if not source.root.is_dir():
        raise ValueError("The designated scheduler source snapshot is unavailable")
    if not (source.root / "nro/orchestration/source_launcher.py").is_file():
        raise ValueError("The designated scheduler source snapshot is incomplete")
    site = capture_site(paths.execution_sites, _coordination_site(values))
    return source, site, Path(record["python"])


def capture_maintenance_implementation(control: Path, bids_root: Path, checkout: Path):
    """Capture a tagged shared checkout while it upgrades or resumes its installation."""
    path = implementation_path(control)
    record = json.loads(path.read_text())
    checkout = checkout.resolve()
    if checkout != Path(record.get("checkout", "")).resolve():
        raise ValueError("Only the active shared checkout may maintain its installation")
    installed = installation_record(checkout)
    if (
        installed.get("mode") != "shared"
        or installed.get("site") != record.get("site")
        or str(Path(installed.get("environment", "")) / "bin/python") != record.get("python")
    ):
        raise ValueError("The active shared installation is unavailable")
    values = settings(path=Path(record["site"]))[0]
    if (
        Path(values["registry"]).resolve() != Path(control).resolve()
        or Path(values["bids"]).resolve() != Path(bids_root).resolve()
    ):
        raise ValueError("The shared installation belongs to another site")
    source = SourceStore(ControlPaths(control).implementations).capture(checkout)
    site = capture_site(ControlPaths(control).execution_sites, values)
    python = Path(record["python"])
    if not python.is_file():
        raise ValueError("The active shared interpreter is unavailable")
    return source, site, python


def validate_worker_script(control: Path, script: Path) -> None:
    """Reject replacement or expansion scripts from another orchestration source."""
    import shlex

    from nro.orchestration.execution_cache import validate_script_cache

    validate_script_cache(script)
    path = implementation_path(control)
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError("Scheduler binding cannot be a symlink")
    record = json.loads(path.read_text())
    commands = [
        shlex.split(line[5:])
        for line in script.read_text().splitlines()
        if line.startswith("exec ")
    ]
    if len(commands) != 1:
        raise ValueError("Expected one pinned worker command")
    command = commands[0]
    if (
        len(command) < 6
        or command[0] != record["python"]
        or Path(command[1]).name != "source_launcher.py"
        or command[2] != record["source_digest"]
        or command[5] != "nro.orchestration.worker"
    ):
        raise ValueError("Worker script does not use the active central implementation")


def require_worker_source(control: Path) -> None:
    """Reject direct worker startup with a different source or Python environment."""
    path = implementation_path(control)
    if not path.exists():
        return
    record = json.loads(path.read_text())
    launched_root = os.environ.get("NRO_EXECUTION_SOURCE_ROOT")
    launched_digest = os.environ.get("NRO_EXECUTION_SOURCE_DIGEST")
    package_root = Path(__file__).resolve().parents[2]
    if (
        os.environ.get("NRO_PROCESS_ROLE") == "scheduler"
        and os.environ.get("NRO_SCHEDULER_MAINTENANCE") == "1"
        and launched_root == str(package_root)
        and launched_digest == source_fingerprint(package_root)
    ):
        _CENTRAL_SOURCE_VERIFIED.set(True)
        return
    if launched_root is not None or launched_digest is not None:
        expected_root = ControlPaths(control).implementations / record["source_digest"]
        if (
            launched_digest == record["source_digest"]
            and launched_root == str(expected_root)
            and package_root == expected_root
            and str(Path(sys.executable)) == record["python"]
        ):
            _CENTRAL_SOURCE_VERIFIED.set(True)
            return
        raise ValueError("Worker did not start from the active central implementation")
    if (
        path.is_symlink()
        or str(Path(sys.executable)) != record["python"]
        or source_fingerprint(package_root) != record["source_digest"]
    ):
        raise ValueError("Start workers through nro run using the active central implementation")
    _CENTRAL_SOURCE_VERIFIED.set(True)


def run_local_worker(
    registry,
    *,
    memory_gb: int,
    drain_seconds: float,
    cpus: int | None = None,
    poll_interval: float = 5.0,
    stdout=None,
    wait: bool = True,
    worker_id: str | None = None,
):
    """Run the designated worker in a separate foreground process."""
    import subprocess

    from nro.engine.execution import allocated_cpus, thread_environment
    from nro.orchestration.execution_cache import cache_lock

    with cache_lock(registry.paths.control):
        source, site, python = capture_worker_implementation(
            registry.paths.control, registry.paths.bids_root
        )
        worker_id = worker_id or f"local-{os.getpid()}-{os.urandom(4).hex()}"
        command = source.command(
            (
                str(python),
                "-m",
                "nro.orchestration.worker",
                "--bids-root",
                str(registry.paths.bids_root),
                "--resource-class",
                "large",
                "--memory-gb",
                str(memory_gb),
                "--idle-timeout",
                "1",
                "--poll-interval",
                str(poll_interval),
                "--drain-seconds",
                str(drain_seconds),
                "--worker-id",
                worker_id,
            ),
            site=site,
        )
        cpu_count = allocated_cpus() if cpus is None else max(1, int(cpus))
        environment = {
            **os.environ,
            "NRO_PROCESS_ROLE": "worker",
            "NRO_ALLOCATED_CPUS": str(cpu_count),
            **thread_environment(cpu_count),
        }
        environment.pop("SLURM_JOB_ID", None)
        process = subprocess.Popen(command, stdout=stdout, env=environment)
        process.nro_worker_id = worker_id
    if not wait:
        return process
    try:
        if process.wait():
            raise RuntimeError("Local scheduler worker exited unsuccessfully")
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    return process
