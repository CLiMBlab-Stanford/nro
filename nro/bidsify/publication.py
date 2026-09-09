"""Hash-bound approval and session publication without partial BIDS directories."""

import ctypes
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from nro.engine.io import atomic_write_text
from nro.orchestration.branches import BranchPaths

from .errors import BidsificationError
from .identity import identity_issues
from .paths import secure_directory


def file_hash(path: Path) -> str:
    """Hash a regular file without following a symbolic link."""
    if path.is_symlink() or not path.is_file():
        raise BidsificationError(f"Publication requires a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    """Describe all files below one session; reject symbolic links and special files."""
    if root.is_symlink():
        raise BidsificationError("Publication root must not be a symlink")
    result = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise BidsificationError("Publication tree contains a symbolic link")
            if not path.is_dir():
                result[str(path.relative_to(root))] = file_hash(path)
    return result


def session_paths(
    record: dict, registry, *, branch_paths: BranchPaths | None = None
) -> tuple[Path, Path]:
    """Return session directories, rejecting requests whose BIDS labels are unresolved."""
    unresolved = identity_issues(record)
    if unresolved:
        raise BidsificationError("; ".join(unresolved))
    from .store import IngestionStore

    store = IngestionStore(registry, branch_paths=branch_paths)
    store.require_record(record)
    relative = Path(f"sub-{record['participant']}") / f"ses-{record['session']}"
    target = store.project_root(record["project"]) / relative
    if target.resolve() != target:
        raise BidsificationError("BIDS destination cannot be redirected through a symlink")
    return (Path(record["config"]["staging"]) / record["id"] / "bids" / relative, target)


def approval_snapshot(record: dict, registry, *, branch_paths: BranchPaths | None = None) -> dict:
    """Bind approval to both new output bytes and the current replacement target."""
    staged, target = session_paths(record, registry, branch_paths=branch_paths)
    outputs = inventory(staged)
    if not outputs:
        raise BidsificationError("No staged outputs to publish")
    if outputs != record.get("output_hashes"):
        raise BidsificationError(
            "Outputs changed since validation; rerun conversion and validation before approval"
        )
    existing = inventory(target)
    if target.exists() and not record["replace"]:
        raise BidsificationError(
            "Destination already exists; explicit re-bidsification is required"
        )
    return {
        "outputs": outputs,
        "existing": existing,
        "target_exists": target.exists(),
        "target": str(target),
    }


def _exchange(first: Path, second: Path) -> None:
    """Atomically exchange existing directories on Linux; never use a two-rename fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise BidsificationError(
            "Atomic directory exchange is unavailable; existing BIDS was not changed"
        )
    if rename(-100, os.fsencode(first), -100, os.fsencode(second), 2):
        error = ctypes.get_errno()
        raise OSError(error, "Atomic directory exchange failed; existing BIDS was not changed")


def publish(record: dict, registry, *, branch_paths: BranchPaths | None = None) -> dict:
    """Publish only approved bytes, preserving a receipt through interruption and repair.

    Copy to a hidden sibling for a same-filesystem atomic rename. Replacements
    in production require no active nro derivative attempts in the destination
    project. Debug publication cannot change shared raw BIDS.
    External readers must be stopped by the operator before approval.
    """
    from .store import IngestionStore

    detached = bool(getattr(registry, "detached_ingestion", False))
    from nro.orchestration.scheduler_implementation import implementation_path

    if implementation_path(registry.paths.control).exists() and not detached:
        from .execution import validate_execution

        pin = record.get("execution")
        if not pin:
            raise BidsificationError(
                "Publication requires an admitted execution; reopen nro bidsify"
            )
        approved_paths = validate_execution(pin)
        if branch_paths != approved_paths:
            raise BidsificationError("Publication paths differ from the approved execution")
    store = IngestionStore(registry, branch_paths=branch_paths)
    if detached and store.branch == "main":
        raise BidsificationError("Production publication requires the central scheduler registry")
    staged, target = session_paths(record, registry, branch_paths=branch_paths)
    approval = record.get("approval")
    if not approval:
        raise BidsificationError("Publication has not been approved")
    receipt = store.root / "receipts" / f"{record['id']}.json"
    secure_directory(target.parent)
    temporary = target.parent / f".nro-publish-{record['id']}"
    if temporary.is_symlink():
        raise BidsificationError("Unsafe publication temporary path")
    if receipt.exists() and inventory(target) == approval["outputs"]:
        if temporary.exists():
            if inventory(temporary) != approval["existing"]:
                raise BidsificationError(
                    "Interrupted publication has an unexpected backup; operator review is required"
                )
            shutil.rmtree(temporary)
        return {"published_path": str(target)}
    if temporary.exists():
        # A prior atomic exchange may have succeeded before its receipt write.
        if (
            inventory(target) == approval["outputs"]
            and inventory(temporary) == approval["existing"]
        ):
            secure_directory(receipt.parent)
            atomic_write_text(
                receipt, json.dumps({"target": str(target), **approval}), mode=0o660, durable=True
            )
            shutil.rmtree(temporary)
            return {"published_path": str(target)}
        shutil.rmtree(temporary)
    if approval_snapshot(record, registry, branch_paths=branch_paths) != approval:
        raise BidsificationError("Staged outputs or destination changed after approval")
    shutil.copytree(staged, temporary)
    if inventory(temporary) != approval["outputs"]:
        raise BidsificationError("Publication copy verification failed")

    @contextmanager
    def publication_guard():
        if detached:
            from nro.orchestration.registry import RegistryLock

            with RegistryLock(
                store.root / "publication.lock",
                store.root / "publication.recovery-lock",
            ):
                yield None
        else:
            with registry.connection(write=True) as db:
                yield db

    with publication_guard() as db:
        current = store.get(record["id"])
        if (
            current["state"] != "running"
            or current["worker"] != record["worker"]
            or current["approval"] != approval
        ):
            raise BidsificationError("Publication ownership or approval changed")
        busy = (
            None
            if db is None
            else db.execute(
                "SELECT 1 FROM attempts a JOIN instances i ON i.id=a.instance_id WHERE i.project=? AND a.state IN ('queued','running','cancel_requested') LIMIT 1",
                (record["project"],),
            ).fetchone()
        )
        if busy and store.branch == "main":
            raise BidsificationError(
                "Destination project has active derivative attempts; stop them before publication"
            )
        if approval_snapshot(record, registry, branch_paths=branch_paths) != approval:
            raise BidsificationError("Publication inputs changed while copying")
        project = store.project_root(record["project"])
        description = project / "dataset_description.json"
        if description.exists():
            metadata = json.loads(description.read_text())
            if metadata.get("DatasetType", "raw") != "raw":
                raise BidsificationError("Destination is not a raw BIDS dataset")
        else:
            atomic_write_text(
                description,
                json.dumps(
                    {"Name": record["project"], "BIDSVersion": "1.11.0", "DatasetType": "raw"},
                    indent=2,
                ),
            )
        secure_directory(receipt.parent)
        # The prepared receipt is also the recovery journal. Output hashes
        # establish whether the single atomic filesystem operation completed.
        atomic_write_text(
            receipt, json.dumps({"target": str(target), **approval}), mode=0o660, durable=True
        )
        if target.exists():
            _exchange(temporary, target)
        else:
            temporary.rename(target)
    if temporary.exists():
        shutil.rmtree(temporary)
    return {"published_path": str(target)}
