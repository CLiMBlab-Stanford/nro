"""Publish verified development artifacts under an accepting ancestor's authority."""

import hashlib
import json
import os
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.engine.io import atomic_write_json
from nro.orchestration.branch_purge import token
from nro.orchestration.branch_reconciliation import candidates_locked
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.manifests import (
    MANIFEST_VERSION,
    _public_derivative_completion,
    assess_registry,
    inventory,
)
from nro.orchestration.registry import utcnow


def _rows(db):
    return {
        row["id"]: dict(row)
        for row in db.execute("""SELECT i.*,c.config_fingerprint
        FROM instances i JOIN configuration_lineages c ON c.id=i.configuration_lineage_id""")
    }


def _authority(registry, checkout, source, pr, attest):
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    target = topology.require_checkout(checkout)
    if not attest or not isinstance(pr, str) or not pr.strip():
        raise ValueError(
            "Promotion requires an accepted-PR reference and explicit merge attestation"
        )
    if source == target or target not in topology.ancestors(source):
        raise ValueError("Only an ancestor checkout can accept development artifacts")
    release = None
    if target == "main":
        from nro.orchestration.releases import ReleaseStore

        release = ReleaseStore(branches).require_approved(checkout)
    return topology, target, release


def preview(
    registry, *, checkout: Path, source: str, requests: list[str], pr: str, attest: bool
) -> dict:
    """Compare source outputs with freshly compiled target requests, without demand.

    Matching target artifacts are retained. Missing source dependencies reject
    the transfer before any output is changed. Source data remain in place.
    """
    topology, target, release = _authority(registry, checkout, source, pr, attest)
    owner = topology.records[target].registry_id
    assess_registry(registry, compiled=True)
    with registry.connection() as db:
        ids = set()
        for request in requests:
            record = db.execute(
                "SELECT registry_id FROM request_owners WHERE request_id=?", (request,)
            ).fetchone()
            if record is None or record[0] != owner:
                raise ValueError("Promotion request belongs to another branch")
            ids.update(
                row[0]
                for row in db.execute(
                    "SELECT instance_id FROM request_artifacts WHERE request_id=?", (request,)
                )
            )
        rows = _rows(db)
        execution = {
            row["instance_id"]: dict(row) for row in db.execute("SELECT * FROM instance_execution")
        }
        candidates = [
            candidate
            for project in {rows[key]["project"] for key in ids}
            for candidate in candidates_locked(db, project)
        ]
        contracts = {
            row["instance_id"]: json.loads(row["scientific_contract_json"])
            for row in db.execute("SELECT * FROM branch_instances WHERE registry_id=?", (owner,))
        }
        items = []
        for instance_id in sorted(ids):
            row = rows[instance_id]
            matches = [
                candidate
                for candidate in candidates
                if candidate.branch == source
                and fingerprint(candidate.contract) == fingerprint(contracts[instance_id])
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous matching source artifacts")
            if row["artifact_state"] == "fresh":
                items.append(
                    dict(
                        target=instance_id,
                        source=matches[0].evidence["instance_id"] if matches else None,
                        action="keep",
                        root=row["output_root"],
                        target_token=token(row, execution.get(instance_id, {}).get("context_json")),
                    )
                )
                continue
            if not matches:
                raise ValueError(
                    f"No fresh, scientifically equivalent {source} artifact for {row['instance_key']}"
                )
            candidate = matches[0]
            producer = rows[candidate.evidence["instance_id"]]
            metadata = execution.get(instance_id)
            if metadata is None or metadata["registry_id"] != owner:
                raise ValueError("Promotion cannot replace inherited outputs")
            completion, reason = _public_derivative_completion(producer, registry, compiled=True)
            if completion is None:
                raise ValueError(reason)
            source_root, destination = Path(producer["output_root"]), Path(row["output_root"])
            files = [str(path.relative_to(source_root)) for path in completion.outputs]
            conflict = any((destination / member).exists() for member in files)
            items.append(
                dict(
                    target=instance_id,
                    source=producer["id"],
                    action="replace" if conflict else "copy",
                    root=str(destination),
                    source_root=str(source_root),
                    files=files,
                    source_token=token(
                        producer, execution.get(producer["id"], {}).get("context_json")
                    ),
                    target_token=token(row, metadata["context_json"]),
                )
            )
    return dict(
        source=source, target=target, release=release, pr=pr.strip(), requests=requests, items=items
    )


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relocate(path: Path, mappings: list[tuple[str, str]], forbidden: str) -> None:
    """Rewrite textual paths; reject opaque files containing development references."""
    # Most image data contain no paths. Scan in chunks before considering text
    # decoding, so large NIfTI and CIFTI files need no extra in-memory copy.
    needle = forbidden.encode()
    found = False
    with path.open("rb") as stream:
        tail = b""
        while chunk := stream.read(1024 * 1024):
            data = tail + chunk
            if needle in data:
                found = True
                break
            tail = data[-len(needle) :]
    if not found:
        return
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(
            f"Large file contains development references and cannot be relocated: {path.name}"
        )
    try:
        value = path.read_text()
    except UnicodeError as error:
        raise ValueError(f"Binary file contains development references: {path.name}") from error
    for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        value = value.replace(old, new)
    if forbidden in value:
        raise ValueError(f"Unresolved development reference in {path.name}")
    from nro.engine.io import atomic_write_text

    atomic_write_text(path, value)


def _rollback_publication(item: dict) -> None:
    """Restore files described by a promotion journal in reverse order."""
    for entry in reversed(item.get("publication", [])):
        destination = Path(entry["destination"])
        backup = Path(entry["backup"])
        if entry["had_destination"]:
            if backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        elif destination.exists():
            source = Path(entry["source"]) if entry.get("source") else None
            if entry.get("state") != "prepared" or source is None or not source.exists():
                destination.unlink()


def _committed(registry, event: str, target: int) -> bool:
    """Return whether the scheduler committed this promotion event."""
    with registry.connection() as db:
        row = db.execute(
            """SELECT i.artifact_state,e.provenance_json FROM instances i
            JOIN instance_execution e ON e.instance_id=i.id WHERE i.id=?""",
            (target,),
        ).fetchone()
    if row is None or row["artifact_state"] != "fresh":
        return False
    provenance = json.loads(row["provenance_json"])
    return provenance.get("promotion", {}).get("event") == event


def _recover_transfer(registry, journal: Path, record: dict, contexts: dict) -> str:
    """Finish ownership metadata or roll back an interrupted transfer."""
    from nro.orchestration.ownership import write_instance_ownership

    for item in record.get("items", []):
        if not item.get("staged"):
            continue
        if item["target"] not in contexts:
            raise ValueError("Promotion journal refers to an unavailable target instance")
        stage = Path(item["staged"])
        contexts[item["target"]].require_output(stage)
        if (
            stage.parent != Path(item["root"])
            or stage.name != f".nro-promotion-{record['event']}-{item['target']}"
        ):
            raise ValueError("Promotion journal contains an invalid staging path")
        if _committed(registry, record["event"], item["target"]):
            write_instance_ownership(registry, item["target"])
        else:
            _rollback_publication(item)
        if stage.exists():
            shutil.rmtree(stage)
    record["state"] = (
        "complete"
        if all(
            item.get("action") == "keep" or _committed(registry, record["event"], item["target"])
            for item in record.get("items", [])
        )
        else "rolled_back"
    )
    record["recovered_at"] = utcnow()
    atomic_write_json(journal, record, durable=True)
    return record["state"]


def publish(registry, *, checkout: Path, report: dict, replace: bool, attest: bool) -> dict:
    """Serialize target-side transfers and recover abandoned staging directories."""
    from nro.orchestration.execution_context import ExecutionContext
    from nro.orchestration.registry import RegistryLock

    topology, target, _ = _authority(registry, checkout, report["source"], report["pr"], attest)
    root = ControlPaths(registry.paths.control).branch(target)
    with RegistryLock(root / "promotion.lock", root / "promotion.recovery-lock"):
        with registry.connection() as db:
            contexts = {
                row["instance_id"]: ExecutionContext.from_dict(json.loads(row["context_json"]))
                for row in db.execute(
                    "SELECT * FROM instance_execution WHERE registry_id=?",
                    (topology.records[target].registry_id,),
                )
            }
        journals = ControlPaths(registry.paths.control).promotions
        recovered = None
        for path in journals.glob("*.json"):
            old = json.loads(path.read_text())
            if old.get("target") != target or old.get("state") in {"complete", "rolled_back"}:
                continue
            state = _recover_transfer(registry, path, old, contexts)
            if state == "complete" and all(
                old.get(key) == report.get(key) for key in ("source", "target", "pr", "requests")
            ):
                recovered = {
                    "promoted": sum(item.get("action") != "keep" for item in old["items"]),
                    "retained": sum(item.get("action") == "keep" for item in old["items"]),
                    "receipt": str(path),
                }
        if recovered is not None:
            return recovered
        return _publish(registry, checkout=checkout, report=report, replace=replace, attest=attest)


def _publish(registry, *, checkout: Path, report: dict, replace: bool, attest: bool) -> dict:
    """Stage copies, fence readers, and publish manifests last; retain source outputs.

    A failed or interrupted transfer never records success. Retrying retains
    completed target artifacts and verifies remaining files again. A durable
    journal records every attempted transfer and its original provenance.
    """
    current = preview(
        registry,
        checkout=checkout,
        source=report["source"],
        requests=report["requests"],
        pr=report["pr"],
        attest=attest,
    )
    if current != report:
        raise ValueError("Promotion selection changed; review a new report")
    working = deepcopy(report)
    copies = [item for item in working["items"] if item["action"] != "keep"]
    if not replace and any(item["action"] == "replace" for item in copies):
        raise ValueError("Existing incompatible outputs require --replace and confirmation")
    if not copies:
        return {"promoted": 0, "retained": len(report["items"])}
    from nro.orchestration.execution_context import ExecutionContext
    from nro.orchestration.ownership import write_instance_ownership

    with registry.connection() as db:
        rows = _rows(db)
        execution = {
            row["instance_id"]: dict(row) for row in db.execute("SELECT * FROM instance_execution")
        }
    context = ExecutionContext.from_dict(json.loads(execution[copies[0]["target"]]["context_json"]))
    mappings = [
        (rows[item["source"]]["output_root"], rows[item["target"]]["output_root"])
        for item in report["items"]
        if item["source"] is not None
    ]
    event = uuid.uuid4().hex
    journal = ControlPaths(registry.paths.control).promotions / f"{event}.json"
    record = dict(working, event=event, state="staging", created_at=utcnow(), completed=[])
    atomic_write_json(journal, record, durable=True)
    stages = []
    cleanup_stages = True
    try:
        for item in copies:
            destination = Path(item["root"])
            owner = ExecutionContext.from_dict(
                json.loads(execution[item["target"]]["context_json"])
            )
            owner.require_output(destination)
            destination.mkdir(parents=True, exist_ok=True)
            stage = destination / f".nro-promotion-{event}-{item['target']}"
            item["staged"] = str(stage)
            atomic_write_json(journal, record, durable=True)
            stage.mkdir()
            stages.append(stage)
            item["source_records"] = inventory(
                Path(item["source_root"]) / member for member in item["files"]
            )
            for member in item["files"]:
                original, copied = Path(item["source_root"]) / member, stage / member
                owner.require_output(destination / member)
                if any(
                    Path(raw).resolve() == (destination / member).resolve()
                    for key, other in rows.items()
                    if key != item["target"]
                    for raw in json.loads(other["expected_outputs_json"])
                ):
                    raise ValueError("Promotion would overwrite another registered artifact")
                if not original.resolve().is_relative_to(Path(item["source_root"]).resolve()):
                    raise ValueError("Promotion source contains an external symlink")
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
                if _digest(original) != _digest(copied):
                    raise ValueError("Source changed during promotion copy")
                from nro.orchestration.branches import BranchPaths

                source_paths = BranchPaths(
                    report["source"],
                    context.paths.bids,
                    context.paths.work,
                    context.paths.development,
                )
                _relocate(
                    copied,
                    mappings,
                    str(source_paths.output_project(rows[item["source"]]["project"]).parent.parent),
                )
            for raw in json.loads(rows[item["source"]]["input_paths_json"]):
                original = Path(raw)
                mapped = str(original)
                for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
                    if original.is_relative_to(Path(old)):
                        mapped = str(Path(new) / original.relative_to(Path(old)))
                        break
                if mapped == raw:
                    continue
                if any(Path(mapped).is_relative_to(Path(other["root"])) for other in copies):
                    continue
                if not Path(mapped).is_file() or _digest(original) != _digest(Path(mapped)):
                    raise ValueError(
                        "An existing target input differs from the source input; promote an equivalent dependency closure"
                    )
        with registry.artifact_mutation(item["target"] for item in copies):
            with registry.connection(write=True) as db:
                for kept in (item for item in report["items"] if item["action"] == "keep"):
                    row = dict(
                        db.execute(
                            "SELECT * FROM instances WHERE id=?", (kept["target"],)
                        ).fetchone()
                    )
                    metadata = db.execute(
                        "SELECT context_json FROM instance_execution WHERE instance_id=?",
                        (kept["target"],),
                    ).fetchone()
                    if (
                        row["artifact_state"] != "fresh"
                        or token(row, metadata[0] if metadata else None) != kept["target_token"]
                    ):
                        raise ValueError("A retained target dependency changed during staging")
                for item in copies:
                    for role in ("source", "target"):
                        row = dict(
                            db.execute(
                                "SELECT * FROM instances WHERE id=?", (item[role],)
                            ).fetchone()
                        )
                        metadata = db.execute(
                            "SELECT context_json FROM instance_execution WHERE instance_id=?",
                            (item[role],),
                        ).fetchone()
                        if token(row, metadata[0] if metadata else None) != item[role + "_token"]:
                            raise ValueError(
                                "Promotion generation or contract changed during staging"
                            )
                        if role == "source" and row["artifact_state"] != "fresh":
                            raise ValueError("Promotion source became stale during staging")
                    if (
                        inventory(Path(item["source_root"]) / member for member in item["files"])
                        != item["source_records"]
                    ):
                        raise ValueError("Source files changed during staging")
                pending = {item["target"]: item for item in copies}
                while pending:
                    ready = [
                        key
                        for key in pending
                        if not any(
                            row[0] in pending
                            for row in db.execute(
                                "SELECT upstream_instance_id FROM instance_dependencies WHERE instance_id=?",
                                (key,),
                            )
                        )
                    ]
                    if not ready:
                        raise ValueError("Promotion dependency closure is cyclic")
                    for key in ready:
                        item = pending.pop(key)
                        row = rows[key]
                        parents = [
                            dict(parent)
                            for parent in db.execute(
                                """SELECT i.id,i.current_generation,i.manifest_path,i.artifact_state
                            FROM instance_dependencies d JOIN instances i ON i.id=d.upstream_instance_id WHERE d.instance_id=?""",
                                (key,),
                            )
                        ]
                        if any(parent["artifact_state"] != "fresh" for parent in parents):
                            raise ValueError("Promotion requires fresh target-side dependencies")
                        outputs = []
                        for member in item["files"]:
                            destination = Path(item["root"]) / member
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            backup = Path(item["staged"]) / ".rollback" / member
                            entry = {
                                "destination": str(destination),
                                "backup": str(backup),
                                "source": str(Path(item["staged"]) / member),
                                "had_destination": destination.exists(),
                                "state": "prepared",
                            }
                            item.setdefault("publication", []).append(entry)
                            atomic_write_json(journal, record, durable=True)
                            if entry["had_destination"]:
                                backup.parent.mkdir(parents=True, exist_ok=True)
                                os.replace(destination, backup)
                                entry["state"] = "backed_up"
                                atomic_write_json(journal, record, durable=True)
                            os.replace(Path(item["staged"]) / member, destination)
                            entry["state"] = "published"
                            atomic_write_json(journal, record, durable=True)
                            outputs.append(destination)
                        original = Path(rows[item["source"]]["manifest_path"])
                        original_manifest = (
                            json.loads(original.read_text()) if original.is_file() else {}
                        )
                        provenance = dict(
                            original_manifest.get("implementation")
                            or json.loads(
                                execution.get(item["source"], {}).get("provenance_json", "{}")
                            )
                        )
                        provenance["promotion"] = dict(
                            event=event,
                            from_branch=report["source"],
                            to_branch=report["target"],
                            pr=report["pr"],
                            accepting_release=report["release"],
                            source_generation=rows[item["source"]]["current_generation"],
                        )
                        manifest = dict(
                            manifest_version=MANIFEST_VERSION,
                            instance_id=key,
                            instance_key=row["instance_key"],
                            module=row["module"],
                            project=row["project"],
                            participant=row["participant"],
                            entities=json.loads(row["entities_json"]),
                            revision_fingerprint=row["revision_fingerprint"],
                            artifact_fingerprint=row["artifact_fingerprint"],
                            artifact_contract=json.loads(row["artifact_contract_json"]),
                            generation=row["current_generation"] + 1,
                            configuration={"fingerprint": row["config_fingerprint"]},
                            implementation=provenance,
                            inputs=inventory(json.loads(row["input_paths_json"])),
                            public_outputs=inventory(outputs),
                            private_artifacts=[],
                            upstream=[
                                dict(
                                    instance_id=parent["id"],
                                    generation=parent["current_generation"],
                                    manifest=parent["manifest_path"],
                                )
                                for parent in parents
                            ],
                            completed_at=utcnow(),
                        )
                        atomic_write_json(Path(row["manifest_path"]), manifest, durable=True)
                        db.execute(
                            "UPDATE instances SET artifact_state='fresh',artifact_reason='Promoted with target contract validation',current_generation=?,updated_at=? WHERE id=?",
                            (manifest["generation"], utcnow(), key),
                        )
                        db.execute(
                            "UPDATE instance_execution SET provenance_json=? WHERE instance_id=?",
                            (json.dumps(provenance), key),
                        )
                        db.executemany(
                            "UPDATE instance_dependencies SET required_generation=? WHERE instance_id=? AND upstream_instance_id=?",
                            [
                                (parent["current_generation"], key, parent["id"])
                                for parent in parents
                            ],
                        )
                        record["completed"].append(key)
            for item in copies:
                write_instance_ownership(registry, item["target"])
        record.update(state="complete", completed_at=utcnow())
        atomic_write_json(journal, record, durable=True)
        return {
            "promoted": len(copies),
            "retained": len(report["items"]) - len(copies),
            "receipt": str(journal),
        }
    except BaseException:
        record["state"] = "interrupted"
        try:
            committed = [_committed(registry, event, item["target"]) for item in copies]
            if any(committed):
                if not all(committed):
                    raise RuntimeError(
                        "Promotion transaction committed only part of its target closure"
                    )
                cleanup_stages = False
            else:
                for item in reversed(copies):
                    _rollback_publication(item)
                record["state"] = "rolled_back"
            atomic_write_json(journal, record, durable=True)
        except BaseException as rollback_error:
            cleanup_stages = False
            record["recovery_error"] = f"{type(rollback_error).__name__}: {rollback_error}"
            atomic_write_json(journal, record, durable=True)
        raise
    finally:
        if cleanup_stages:
            for stage in stages:
                shutil.rmtree(stage, ignore_errors=True)
