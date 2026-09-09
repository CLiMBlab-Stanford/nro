"""Pin reviewed ingestion work to its source and publication namespace."""

import fcntl
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from nro.configuration.site import CHECKOUT, installation_record, settings
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_cache import cache_lock, service_lease
from nro.orchestration.execution_pins import capture_execution
from nro.orchestration.source_snapshots import SourceSnapshot


def launch_review(argv: list[str]) -> None:
    """Run the interactive reviewer from a capture without changing the default CLI."""
    values = settings()[0]
    control = Path(values["registry"])
    topology = BranchStore(control).read().topology
    name = topology.require_checkout(CHECKOUT)
    record = installation_record()
    if record and not record.get("ready"):
        raise ValueError("Finish installation before starting ingestion")
    release = None
    if name == "main":
        from nro.orchestration.releases import ReleaseStore

        release = ReleaseStore(BranchStore(control)).require_approved(CHECKOUT)
    with ExitStack() as stack:
        descriptors = []
        if record:
            lock = stack.enter_context((CHECKOUT / ".nro-install.lock").open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Installation maintenance is in progress") from error
            current = installation_record()
            if (
                not current.get("ready")
                or str(Path(current["environment"]) / "bin/python") != sys.executable
            ):
                raise ValueError(
                    "Use this checkout’s installed Python environment after installation completes"
                )
            descriptors.append(lock.fileno())
        lease = stack.enter_context(service_lease(control))
        with cache_lock(control):
            source, site = capture_execution(control, Path(values["bids"]))
            pin = dict(
                branch=name,
                registry_id=topology.records[name].registry_id,
                checkout=str(CHECKOUT),
                source_root=str(source.root),
                source_digest=source.digest,
                site=str(site),
                python=sys.executable,
                release=release,
                command_prefix=list(
                    source.command((sys.executable, "-m", "nro.bidsify"), site=site)
                ),
            )
            command = source.command(
                (sys.executable, "-m", "nro.bin.bidsify", *argv, "--execution", json.dumps(pin)),
                site=site,
            )
        try:
            result = subprocess.run(command, pass_fds=(*descriptors, lease[1]))
        except KeyboardInterrupt:
            return
        if result.returncode:
            raise SystemExit(result.returncode)


def validate_execution(pin: dict, *, current_source: bool = True) -> BranchPaths:
    """Check source, registration, and exact pinned site before using a namespace."""
    source = SourceSnapshot(Path(pin["source_root"]), pin["source_digest"])
    source.verify()
    if current_source and source.root != CHECKOUT:
        raise ValueError("Ingestion must run from its captured implementation")
    values = settings(path=Path(pin["site"]))[0]
    branches = BranchStore(Path(values["registry"]))
    topology = branches.read().topology
    name = topology.require_checkout(Path(pin["checkout"]))
    if name != pin["branch"] or topology.records[name].registry_id != pin["registry_id"]:
        raise ValueError("Ingestion branch registration changed")
    expected = source.command((pin["python"], "-m", "nro.bidsify"), site=Path(pin["site"]))
    if tuple(pin["command_prefix"]) != expected:
        raise ValueError("Ingestion execution or site settings changed")
    if name == "main":
        from nro.orchestration.releases import ReleaseStore

        if ReleaseStore(branches).require_approved(Path(pin["checkout"])) != pin["release"]:
            raise ValueError("Production ingestion lacks its approved main release")
    return BranchPaths(name, *(Path(values[key]) for key in ("bids", "work", "development")))


def stage_command(record: dict, registry) -> tuple[str, ...]:
    """Build the private stage invocation from an admitted execution pin."""
    pin = record["execution"]
    paths = validate_execution(pin, current_source=False)
    if paths.branch != record["branch"] or paths.bids != registry.paths.bids_root.resolve():
        raise ValueError("Ingestion execution belongs to another namespace")
    return (
        *pin["command_prefix"],
        "--request",
        record["id"],
        "--bids-root",
        str(paths.bids),
        "--control",
        str(registry.paths.control),
        "--execution",
        json.dumps(pin),
    )
