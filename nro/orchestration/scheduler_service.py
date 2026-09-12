"""Accept compiled requests in the centrally selected orchestration process."""

import fcntl
import json
import sys
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.artifact_resolution import scientific_contracts
from nro.orchestration.branch_admission import _admit_resolved
from nro.orchestration.branch_reconciliation import candidates_locked, resolve_payload
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import decode_spec
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.source_snapshots import SourceSnapshot


def admit(
    registry,
    payload: dict,
    *,
    checkout: Path,
    site_values: dict,
    assess: bool = True,
    source_verified: bool = False,
) -> str:
    """Keep an installed environment out of maintenance until demand is published."""
    record_path = checkout / ".nro-installation.json"
    if not record_path.exists():
        return _admit(
            registry,
            payload,
            checkout=checkout,
            site_values=site_values,
            assess=assess,
            source_verified=source_verified,
        )
    lock = checkout / ".nro-install.lock"
    if not lock.is_file():
        raise ValueError("Installed checkout lacks its maintenance lock; rerun installation")
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("The submitting installation is undergoing maintenance") from error
        record = json.loads(record_path.read_text())
        if record.get("checkout") != str(checkout) or not record.get("ready"):
            raise ValueError("The submitting installation is not ready")
        if str(Path(record["environment"]) / "bin/python") != payload["python"]:
            raise ValueError("Request interpreter does not match the installed environment")
        return _admit(
            registry,
            payload,
            checkout=checkout,
            site_values=site_values,
            assess=assess,
            source_verified=source_verified,
        )


def admit_many(registry, entries: list[dict], *, checkout: Path, site_values: dict) -> list[str]:
    """Admit one invocation's requests after one source check and registry assessment."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("Admission batch must contain at least one request")
    projects = []
    sources = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"project", "payload"}:
            raise ValueError("Invalid admission batch entry")
        project, payload = entry["project"], entry["payload"]
        if not isinstance(project, str) or not project or payload.get("project") != project:
            raise ValueError("Admission batch project does not match its payload")
        projects.append(project)
        descriptor = payload.get("source")
        if not isinstance(descriptor, dict) or set(descriptor) != {"root", "digest"}:
            raise ValueError("Admission batch has an invalid source")
        sources[(descriptor["root"], descriptor["digest"])] = SourceSnapshot(
            Path(descriptor["root"]), descriptor["digest"]
        )
    for source in sources.values():
        source.verify_manifest()
    from nro.orchestration.manifests import assess_registry

    assess_registry(registry, projects=tuple(dict.fromkeys(projects)), compiled=True)
    from nro.orchestration.registry import Registry

    return [
        admit(
            Registry.for_project(
                entry["project"],
                bids_root=registry.paths.bids_root,
                registry_path=registry.paths.control,
            ),
            entry["payload"],
            checkout=checkout,
            site_values=site_values,
            assess=False,
            source_verified=True,
        )
        for entry in entries
    ]


def _admit(
    registry,
    payload: dict,
    *,
    checkout: Path,
    site_values: dict,
    assess: bool = True,
    source_verified: bool = False,
) -> str:
    """Validate and admit a detached graph without opening its scientific registry.

    The transport holds the cache publication lock until this call returns.
    Scientific revisions reject delayed requests from another checkout of the
    same branch. Central registration authorizes the checkout, while the source
    digest binds the request to the captured implementation.
    """
    branches = BranchStore(registry.paths.control)
    source = SourceSnapshot(Path(payload["source"]["root"]), payload["source"]["digest"])
    if not source_verified:
        source.verify()
    if type(payload.get("demand", True)) is not bool:
        raise ValueError("Demand must be a boolean")
    if type(payload["concurrency"]) is not int or payload["concurrency"] < 1:
        raise ValueError("Concurrency must be a positive integer")
    if not Path(payload["python"]).is_absolute() or not Path(payload["python"]).is_file():
        raise ValueError("The job interpreter is unavailable")
    context = ExecutionContext.from_dict(payload["context"])
    if any(
        getattr(context.paths, key) != Path(site_values[key]).resolve()
        for key in ("bids", "work", "development")
    ):
        raise ValueError("Compiled request paths differ from the central site")
    if context.project != registry.paths.project:
        raise ValueError("Compiled request belongs to another project")
    specs = tuple(decode_spec(value) for value in payload["specifications"])
    contracts = payload.get("contracts")
    if contracts is None:
        contracts = scientific_contracts(specs)
    elif (
        not isinstance(contracts, dict)
        or set(contracts) != {spec.key for spec in specs}
        or not all(isinstance(value, dict) for value in contracts.values())
    ):
        raise ValueError("Scientific contracts must cover the complete request")
    if set(payload["revisions"]) != set(contracts):
        raise ValueError("Scientific revisions must cover the complete request")
    if assess:
        from nro.orchestration.manifests import assess_registry

        assess_registry(registry, projects=(context.project,), compiled=True)
    if payload["branch"] == "main":
        from nro.orchestration.scheduler_implementation import implementation_path

        active = json.loads(implementation_path(registry.paths.control).read_text())
        if (
            Path(str(active.get("checkout", ""))).resolve() != checkout.resolve()
            or active.get("release") != payload["release"]
            or active.get("source_digest") != source.digest
        ):
            raise ValueError("Main request does not match its approved release")
    with branches._lock():
        topology = branches.read().topology
        name = topology.registered_checkout(checkout)
        if (
            name != payload["branch"]
            or topology.records[name].registry_id != payload["registry_id"]
        ):
            raise ValueError("Request ownership does not match the submitting checkout")
        with registry.connection(write=True) as db:
            for key, contract in contracts.items():
                revision = payload["revisions"][key]
                if type(revision) is not int or revision < 1:
                    raise ValueError("Scientific revisions must be positive integers")
                digest = fingerprint(contract)
                row = db.execute(
                    "SELECT revision,fingerprint FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                    (payload["registry_id"], key),
                ).fetchone()
                if row and (
                    revision < row["revision"]
                    or (revision == row["revision"] and digest != row["fingerprint"])
                ):
                    raise ValueError(
                        "A newer scientific request was admitted; refresh this checkout before retrying"
                    )
                db.execute(
                    """INSERT INTO compiled_revisions VALUES (?,?,?,?)
                    ON CONFLICT(registry_id,logical_key) DO UPDATE SET revision=excluded.revision,
                    fingerprint=excluded.fingerprint""",
                    (payload["registry_id"], key, revision, digest),
                )
            plan = resolve_payload(topology, payload, candidates_locked(db, context.project))
            return _admit_resolved(registry, db, plan, payload)


def supply(registry, request_ids: list[str], options: dict, *, checkout: Path) -> dict:
    """Supply central workers only for requests owned by the authorized checkout."""
    branches = BranchStore(registry.paths.control)
    name = branches.read().topology.registered_checkout(checkout)
    owner = branches.read().topology.records[name].registry_id
    with registry.connection() as db:
        for request_id in request_ids:
            row = db.execute(
                "SELECT registry_id FROM request_owners WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] != owner:
                raise ValueError("Worker supply request belongs to another branch")
    registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    from nro.orchestration.scheduler_implementation import run_local_worker
    from nro.orchestration.submission import _submit_workers, _write_worker_script

    submitted = []
    if options["local"]:
        run_local_worker(
            registry,
            memory_gb=options["memory"],
            drain_seconds=options["drain_minutes"] * 60,
            poll_interval=options.get("worker_poll_interval", 5.0),
            stdout=sys.stderr,
        )
    elif not options["no_submit"]:
        tier, scripts = options["memory"], {}
        while True:
            scripts[tier] = _write_worker_script(
                registry,
                bids_root=registry.paths.bids_root,
                partition=options["partition"],
                account=options["account"],
                hours=options["time"],
                memory_gb=tier,
                cpus=options["cpus"],
                idle_timeout=options["worker_idle_timeout"],
                drain_seconds=options["drain_minutes"] * 60,
            )
            if tier >= options["max_memory"]:
                break
            tier = min(options["max_memory"], tier * 2)
        submitted = _submit_workers(
            registry, request_ids[0], scripts[options["memory"]], options["memory"]
        )
    return {"submitted_workers": submitted}


def status(registry, *, checkout: Path, mode: str) -> dict:
    """Report this branch's registered selections, retaining upstream error details."""
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    name = topology.registered_checkout(checkout)
    owner = topology.records[name].registry_id
    from nro.bidsify.store import IngestionStore

    ingestion = [
        {
            key: row[key]
            for key in (
                "id",
                "server",
                "project",
                "participant",
                "session",
                "state",
                "stage",
                "issues",
            )
        }
        for row in IngestionStore(registry, branch=name).rows()
    ]
    if not registry.existing_database_path().is_file():
        return {"rows": [], "visible_ids": [], "ingestion": ingestion, "dependencies": []}
    from nro.orchestration.manifests import assess_registry, preview_registry

    if mode == "verify":
        assess_registry(registry, compiled=True)
    elif mode not in {"cached", "preview"}:
        raise ValueError("Unknown status mode")
    states = preview_registry(registry, compiled=True) if mode == "preview" else None
    rows = registry.instance_status_snapshot(read_only=True, artifact_states=states)
    with registry.connection() as db:
        visible = {
            row[0]
            for row in db.execute(
                "SELECT instance_id FROM branch_instances WHERE registry_id=?", (owner,)
            )
        }
        scientific = {
            row["instance_id"]: (row["logical_key"], row["revision"])
            for row in db.execute(
                """
            SELECT b.instance_id,b.logical_key,r.revision FROM branch_instances b
            JOIN compiled_revisions r ON r.registry_id=b.registry_id AND r.logical_key=b.logical_key
            WHERE b.registry_id=?""",
                (owner,),
            )
        }
        if name == "main":
            visible.update(
                row[0]
                for row in db.execute(
                    """SELECT i.id FROM instances i
                    LEFT JOIN instance_execution e ON e.instance_id=i.id
                    WHERE e.instance_id IS NULL AND i.artifact_state!='missing'"""
                )
            )
        workflows = {}
        for row in db.execute(
            """SELECT ri.instance_id,w.workflow_id FROM request_artifacts ri
            JOIN requests r ON r.id=ri.request_id JOIN request_owners o ON o.request_id=r.id
            JOIN workflow_revisions w ON w.id=r.workflow_revision_id WHERE o.registry_id=?""",
            (owner,),
        ):
            workflows.setdefault(row[0], set()).add(row[1].removeprefix(owner + ":"))
    for row in rows:
        if row["id"] in visible:
            logical_key, revision = scientific.get(row["id"], (row["instance_key"], None))
            row["logical_key"] = logical_key
            row["scientific_revision"] = revision
        row["workflow_ids"] = ",".join(sorted(workflows.get(row["id"], ()))) or (
            row.get("workflow_ids", "") if name == "main" else ""
        )
    return {
        "rows": rows,
        "visible_ids": sorted(visible),
        "ingestion": ingestion,
        "dependencies": registry.instance_dependencies(read_only=True),
    }


def stop(registry, *, checkout: Path, selection: dict) -> dict:
    """Cancel selected branch demand without cancelling another branch's requests."""
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    name = topology.registered_checkout(checkout)
    return registry.request_cancellation(
        **selection, branch_registry_id=topology.records[name].registry_id
    )


def logs(registry, *, checkout: Path, selection: dict, instance_level: bool) -> dict:
    """Resolve logs of selected branch artifacts or the workers that executed them."""
    from nro.engine.cli import matches_instance_selectors

    name = BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    report = status(registry, checkout=checkout, mode="cached")
    visible = set(report["visible_ids"])
    requested_modules = set(selection["modules"])
    bidsify_selected = "bidsify" in requested_modules
    scientific_modules = requested_modules - {"bidsify"}
    scientific_selected = not requested_modules or bool(scientific_modules)
    selected = [
        row
        for row in report["rows"]
        if scientific_selected
        and row["id"] in visible
        and all(
            not selection[key] or row[field] in selection[key]
            for key, field in (
                ("projects", "project"),
                ("participants", "participant"),
            )
        )
        and (not scientific_modules or row["module"] in scientific_modules)
        and (
            not selection["workflows"]
            or set(selection["workflows"]).intersection(row["workflow_ids"].split(","))
        )
        and matches_instance_selectors(json.loads(row["entities_json"]), selection["selectors"])
    ]
    if instance_level:
        paths = [row["log_path"] for row in selected if row.get("log_path")]
    else:
        ids = {row["id"] for row in selected}
        with registry.connection() as db:
            paths = [
                str(registry.paths.workers / f"slurm-{row['slurm_job_id']}.log")
                for row in db.execute("""SELECT a.instance_id,w.slurm_job_id FROM attempts a
                    JOIN workers w ON w.id=a.worker_id WHERE w.slurm_job_id IS NOT NULL""")
                if row["instance_id"] in ids
            ]
    if bidsify_selected and not selection["workflows"] and set(selection["selectors"]) <= {"ses"}:
        from nro.bidsify.store import IngestionStore

        sessions = selection["selectors"].get("ses", ())
        store = IngestionStore(registry, branch=name)
        paths.extend(
            str(store.root / f"{row['id']}.log")
            for row in report["ingestion"]
            if (
                not selection.get("ingestion_projects")
                or row["project"] in selection["ingestion_projects"]
            )
            and (not selection["participants"] or row["participant"] in selection["participants"])
            and (not sessions or row["session"] in sessions)
        )
    return {"paths": sorted(set(paths))}


def pool_operation(registry, *, checkout: Path, operation: str, concurrency=None) -> dict:
    """Apply explicit lab-wide pool controls from an authorized checkout."""
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    if operation == "concurrency":
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Concurrency must be a positive integer")
        return {"updated_requests": registry.set_active_concurrency(concurrency)}
    if operation != "stop_workers":
        raise ValueError("Unknown pool operation")
    from nro.orchestration.worker_control import cancel_worker_allocations

    shutdown = registry.request_worker_shutdown()
    stopped, failures = cancel_worker_allocations(registry, shutdown)
    return dict(shutdown, stopped_jobs=stopped, failures=failures)


def require_environment_idle(registry, *, checkout: Path, environment: Path) -> dict:
    """Reject maintenance while any demanded recipe or attempt uses this environment."""
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    if not environment.is_absolute():
        raise ValueError("Environment path must be absolute")
    with registry.connection() as db:
        commands = [
            row[0]
            for row in db.execute("""SELECT DISTINCT i.command_json FROM instances i
            JOIN request_instances ri ON ri.instance_id=i.id JOIN requests r ON r.id=ri.request_id
            WHERE r.state='active' AND ri.demand_state='active'
            UNION SELECT COALESCE(e.command_json,i.command_json) FROM attempts a
            JOIN instances i ON i.id=a.instance_id LEFT JOIN attempt_execution e ON e.attempt_id=a.id
            WHERE a.state IN ('queued','running','cancel_requested')""")
        ]
        if any(
            Path(json.loads(command)[0]).absolute().is_relative_to(environment.absolute())
            for command in commands
        ):
            raise ValueError(
                "Outstanding work uses this environment; stop its demand and attempts before maintenance"
            )
        from nro.bidsify.index import IngestionIndex

        if any(
            row["state"] in {"queued", "running"}
            and row.get("execution")
            and Path(row["execution"]["python"]).absolute().is_relative_to(environment.absolute())
            for row in IngestionIndex(registry).execution_records()
        ):
            raise ValueError(
                "Outstanding ingestion uses this environment; stop it before maintenance"
            )
    return {"idle": True}


def main() -> None:
    """Exchange one JSON request and response over standard input and output."""
    from nro.configuration.site import settings
    from nro.orchestration.registry import Registry
    from nro.orchestration.scheduler_implementation import require_worker_source

    try:
        message = json.load(sys.stdin)
        values = settings()[0]
        require_worker_source(Path(values["registry"]))
        registry = Registry.for_project(
            message.get("project", ""), bids_root=values["bids"], registry_path=values["registry"]
        )
        if message["operation"] == "admit":
            request_id = admit(
                registry, message["payload"], checkout=Path(message["checkout"]), site_values=values
            )
            result = {"request_id": request_id}
        elif message["operation"] == "admit_many":
            result = {
                "request_ids": admit_many(
                    registry,
                    message["entries"],
                    checkout=Path(message["checkout"]),
                    site_values=values,
                )
            }
        elif message["operation"] == "supply":
            result = supply(
                registry,
                message["request_ids"],
                message["options"],
                checkout=Path(message["checkout"]),
            )
        elif message["operation"] == "status":
            result = status(registry, checkout=Path(message["checkout"]), mode=message["mode"])
        elif message["operation"] == "stop":
            result = stop(
                registry, checkout=Path(message["checkout"]), selection=message["selection"]
            )
        elif message["operation"] == "logs":
            result = logs(
                registry,
                checkout=Path(message["checkout"]),
                selection=message["selection"],
                instance_level=message["instance_level"],
            )
        elif message["operation"] in {"concurrency", "stop_workers"}:
            result = pool_operation(
                registry,
                checkout=Path(message["checkout"]),
                operation=message["operation"],
                concurrency=message.get("concurrency"),
            )
        elif message["operation"] == "environment_idle":
            result = require_environment_idle(
                registry,
                checkout=Path(message["checkout"]),
                environment=Path(message["environment"]),
            )
        elif message["operation"] == "purge_snapshot":
            from nro.orchestration.branch_purge import snapshot

            result = snapshot(registry, checkout=Path(message["checkout"]), site_values=values)
        elif message["operation"] == "purge":
            from nro.orchestration.branch_purge import purge

            result = purge(
                registry,
                checkout=Path(message["checkout"]),
                site_values=values,
                plan=message["plan"],
                logs_only=message["logs_only"],
                dry_run=message["dry_run"],
            )
        elif message["operation"] == "cache":
            BranchStore(registry.paths.control).read().topology.registered_checkout(
                Path(message["checkout"])
            )
            from nro.orchestration.execution_cache import collect_cache

            collection = collect_cache(
                registry,
                dry_run=message["dry_run"],
                service=message["service"],
                approved=None
                if message["approved"] is None
                else tuple(map(Path, message["approved"])),
            )
            result = dict(
                paths=list(map(str, collection.paths)),
                retained=list(map(str, collection.retained)),
                reason=collection.reason,
            )
        elif message["operation"] == "repair_prepare":
            from nro.orchestration.branch_repair import prepare

            result = prepare(
                registry,
                checkout=Path(message["checkout"]),
                reservation=message.get("reservation"),
                allow_stop=message.get("allow_stop", False),
            )
        elif message["operation"] == "repair_finish":
            from nro.orchestration.branch_repair import finish

            result = finish(
                registry, checkout=Path(message["checkout"]), reservation=message["reservation"]
            )
        elif message["operation"] == "promotion_preview":
            from nro.orchestration.promotion import preview

            result = preview(
                registry,
                checkout=Path(message["checkout"]),
                source=message["source"],
                requests=message["requests"],
                pr=message["pr"],
                attest=message["attest"],
            )
        elif message["operation"] == "promotion_publish":
            from nro.orchestration.promotion import publish

            result = publish(
                registry,
                checkout=Path(message["checkout"]),
                report=message["report"],
                replace=message["replace"],
                attest=message["attest"],
            )
        elif message["operation"] == "publish":
            topology = BranchStore(registry.paths.control).read().topology
            name = topology.registered_checkout(Path(message["checkout"]))
            with registry.connection() as db:
                owner = db.execute(
                    "SELECT registry_id FROM request_owners WHERE request_id=?",
                    (message["request"],),
                ).fetchone()
                if owner is None or owner[0] != topology.records[name].registry_id:
                    raise ValueError("Publication request belongs to another branch")
            destination = Path(message["destination"]).expanduser().resolve()
            if any(
                destination.is_relative_to(Path(values[key]).resolve())
                for key in ("bids", "work", "development", "registry")
            ):
                raise ValueError(
                    "Standalone publication must be outside managed data and control stores"
                )
            from nro.orchestration.publish import publish

            result = {
                "destination": str(
                    publish(
                        registry,
                        request_id=message["request"],
                        destination=destination,
                        validate=message["validate"],
                        compiled=True,
                    )
                )
            }
        elif message["operation"] == "branch_update":
            from nro.orchestration.branch_operations import update

            result = update(
                registry,
                checkout=Path(message["checkout"]),
                branch=message["branch"],
                action=message["action"],
                revision=message["revision"],
                parent=message.get("parent"),
            )
        else:
            raise ValueError("Unsupported scheduler operation")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1) from error
    print(json.dumps({"result": result}))


if __name__ == "__main__":
    main()
