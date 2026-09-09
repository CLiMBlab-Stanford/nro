"""Validate branch-selected deletion paths against central ownership and readers."""

import json
from dataclasses import replace
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.purge_paths import (
    _purge_attempt_logs,
    _purge_inactive_worker_logs,
    _remove_path,
)
from nro.orchestration.registry import Registry, utcnow


def token(row: dict, context_json: str | None) -> str:
    """Bind confirmation to the selected generation, storage paths, and contract."""
    return fingerprint(
        {
            key: row[key]
            for key in (
                "artifact_fingerprint",
                "current_generation",
                "output_root",
                "output_prefix",
                "manifest_path",
            )
        }
        | {"context": context_json}
    )


def snapshot(registry, *, checkout: Path, site_values: dict) -> dict:
    """Return only owned instances; inherited selections are not deletion targets."""
    from nro.orchestration.scheduler_service import status

    topology = BranchStore(registry.paths.control).read().topology
    name = topology.require_checkout(checkout)
    owner = topology.records[name].registry_id
    report = status(registry, checkout=checkout, mode="cached")
    paths = BranchPaths(name, *(Path(site_values[key]) for key in ("bids", "work", "development")))
    if not registry.existing_database_path().is_file():
        return {"rows": [], "branch": name}
    with registry.connection() as db:
        metadata = {
            row["instance_id"]: dict(row) for row in db.execute("SELECT * FROM instance_execution")
        }
    rows = []
    for row in report["rows"]:
        item = metadata.get(row["id"])
        if (item is None and name != "main") or (item is not None and item["registry_id"] != owner):
            continue
        encoded = item["context_json"] if item else None
        context = (
            ExecutionContext.from_dict(json.loads(encoded))
            if encoded
            else ExecutionContext(paths, row["project"], row["instance_key"], ())
        )
        rows.append(dict(row, execution_context=context.as_dict(), purge_token=token(row, encoded)))
    return {"rows": rows, "branch": name}


def purge(
    registry, *, checkout: Path, site_values: dict, plan: list[dict], logs_only: bool, dry_run: bool
) -> dict:
    """Delete confirmed owned paths after fencing all affected readers.

    The branch computes paths using its own module definitions. Central code
    checks ownership, generation, overlap, and shutdown before removing them.
    It never interprets a branch's module catalog.
    """
    view = snapshot(registry, checkout=checkout, site_values=site_values)
    owned = {row["id"]: row for row in view["rows"]}
    ids = {item["id"] for item in plan}
    if len(ids) != len(plan) or not ids <= owned.keys():
        raise ValueError("Purge includes duplicate, inherited, or foreign instances")
    for item in plan:
        if item["token"] != owned[item["id"]]["purge_token"]:
            raise ValueError("Purge targets changed; generate and confirm a new report")
        if logs_only and (item["public"] or item["private"]):
            raise ValueError("Log-only purge cannot delete derivative paths")

    def validate(db):
        protected = [
            Path(path)
            for row in db.execute("SELECT id,expected_outputs_json FROM instances")
            if row["id"] not in ids
            for path in json.loads(row["expected_outputs_json"])
        ]
        for item in plan:
            row = dict(db.execute("SELECT * FROM instances WHERE id=?", (item["id"],)).fetchone())
            execution = db.execute(
                "SELECT context_json FROM instance_execution WHERE instance_id=?", (item["id"],)
            ).fetchone()
            if token(row, execution[0] if execution else None) != item["token"]:
                raise ValueError("Purge targets changed; generate and confirm a new report")
            context = ExecutionContext.from_dict(owned[item["id"]]["execution_context"])
            for raw in (*item["public"], *item["private"]):
                path = Path(raw)
                if not path.is_absolute() or ".." in path.parts:
                    raise ValueError("Purge paths must be normalized and absolute")
                if path != Path(row["manifest_path"]):
                    context.require_output(path)
                elif path.resolve() != path:
                    raise ValueError("Manifest path is redirected")
                if any(other.resolve().is_relative_to(path.resolve()) for other in protected):
                    raise ValueError(
                        "Purge would remove another registered instance; narrow the paths or expand the selection"
                    )

    with registry.connection() as db:
        validate(db)
    counts = {"instances": len(ids), "derivative_paths": 0, "work_paths": 0}
    if not logs_only:
        from contextlib import nullcontext

        with nullcontext() if dry_run else registry.artifact_mutation(ids):
            with registry.connection(write=not dry_run) as db:
                validate(db)
                for kind, counter in (("public", "derivative_paths"), ("private", "work_paths")):
                    paths = sorted(
                        {Path(raw) for item in plan for raw in item[kind]},
                        key=lambda path: len(path.parts),
                    )
                    for path in paths:
                        counts[counter] += int(_remove_path(path, dry_run=dry_run))
                if not dry_run:
                    db.executemany(
                        "UPDATE instances SET artifact_state='missing',artifact_reason='Purged by user',updated_at=? WHERE id=?",
                        [(utcnow(), instance_id) for instance_id in ids],
                    )
    from nro.orchestration.control_paths import ControlPaths

    scoped = Registry(
        replace(
            registry.paths,
            events=ControlPaths(registry.paths.control).branch(view["branch"]) / "events",
        )
    )
    counts["attempt_logs"] = _purge_attempt_logs(scoped, instance_ids=ids, dry_run=dry_run)
    counts["worker_logs"] = _purge_inactive_worker_logs(registry, dry_run=dry_run)
    return counts
