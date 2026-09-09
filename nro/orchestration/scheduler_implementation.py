"""Select the site's orchestration installation independently of job source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nro.configuration.site import installation_record, settings
from nro.engine.io import atomic_write_json
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.execution_pins import capture_site
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.source_snapshots import SourceStore, source_fingerprint


def implementation_path(control: Path) -> Path:
    """Locate the explicit central installation binding, not a code archive."""
    return ControlPaths(control).scheduler / "implementation.json"


def activate(registry, checkout: Path) -> dict:
    """Designate an approved main installation while all site work is quiescent.

    This changes neither Git refs nor the user's command launcher. Future workers
    use this installation's interpreter and a per-submission source snapshot.
    """
    from nro.orchestration.execution_cache import _busy, cache_lock

    checkout = Path(checkout).expanduser().resolve()
    installation = installation_record(checkout)
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
    with cache_lock(registry.paths.control):
        release = ReleaseStore(BranchStore(registry.paths.control)).require_approved(checkout)
        with registry.connection() as db:
            busy = _busy(registry, db)
            if busy:
                raise ValueError(f"Cannot activate the scheduler during {busy}")
            record = dict(
                protocol=1,
                checkout=str(checkout),
                python=str(python),
                site=str(site),
                release=release,
                source_digest=source_fingerprint(checkout),
            )
            path = implementation_path(registry.paths.control)
            if path.is_symlink():
                raise ValueError("Scheduler binding cannot be a symlink")
            atomic_write_json(path, record, mode=0o664, durable=True)
    return record


def capture_worker_implementation(control: Path, bids_root: Path):
    """Capture the designated orchestration source, interpreter, and resolved site.

    Call while holding the execution-cache publication lock. Once a site has a
    binding, an unavailable or modified installation is an error; the caller's
    development checkout is never a substitute.
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
    ):
        raise ValueError("The designated scheduler installation changed or is unavailable")
    releases = ReleaseStore(BranchStore(control))
    if releases.require_approved(checkout) != record["release"]:
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
    source = SourceStore(paths.implementations).capture(checkout)
    if source.digest != record["source_digest"]:
        raise ValueError("The designated scheduler source changed")
    site = capture_site(paths.execution_sites, values)
    if releases.require_approved(checkout) != record["release"]:
        raise ValueError("Scheduler source changed during capture")
    return source, site, Path(record["python"])


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
    if (
        path.is_symlink()
        or str(Path(sys.executable)) != record["python"]
        or source_fingerprint(Path(__file__).resolve().parents[2]) != record["source_digest"]
    ):
        raise ValueError("Start workers through nro run using the active central implementation")


def run_local_worker(
    registry,
    *,
    memory_gb: int,
    drain_seconds: float,
    poll_interval: float = 5.0,
    stdout=None,
) -> None:
    """Run the designated worker in a separate foreground process."""
    import subprocess

    from nro.orchestration.execution_cache import cache_lock

    with cache_lock(registry.paths.control):
        source, site, python = capture_worker_implementation(
            registry.paths.control, registry.paths.bids_root
        )
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
            ),
            site=site,
        )
        process = subprocess.Popen(command, stdout=stdout)
    try:
        if process.wait():
            raise RuntimeError("Local scheduler worker exited unsuccessfully")
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
