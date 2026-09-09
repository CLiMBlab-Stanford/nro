"""Apply branch topology changes together with their scheduler effects."""

from pathlib import Path

from nro.orchestration import dependency_state
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import utcnow


def update(
    registry,
    *,
    checkout: Path,
    branch: str,
    action: str,
    revision: str,
    parent: str | None = None,
) -> dict:
    """Change one owned branch and immediately reconcile affected work."""
    if action not in {"reparent", "retire"}:
        raise ValueError("Unknown branch update")
    store = BranchStore(registry.paths.control)
    own_cancelled = 0
    with store._lock():
        store._finish_pending()
        before = store.read()
        if before.revision != revision:
            raise ValueError("Branch registrations changed; read the current tree before retrying")
        before.topology.require_branch_checkout(branch, checkout)
        if action == "reparent":
            if parent is None:
                raise ValueError("Reparenting requires a parent branch")
            after = before.topology.reparent(branch, parent)
        else:
            after = before.topology.retire(branch)
            owner = before.topology.records[branch].registry_id
            now = utcnow()
            with registry.connection(write=True) as db:
                requests = tuple(
                    row[0]
                    for row in db.execute(
                        "SELECT request_id FROM request_owners WHERE registry_id=?", (owner,)
                    )
                )
                if requests:
                    placeholders = ",".join("?" for _ in requests)
                    db.execute(
                        f"UPDATE requests SET state='cancelled',updated_at=? "
                        f"WHERE id IN ({placeholders}) AND state='active'",
                        (now, *requests),
                    )
                    db.execute(
                        f"UPDATE request_instances SET demand_state='cancelled' "
                        f"WHERE request_id IN ({placeholders})",
                        requests,
                    )
                instances = tuple(
                    row[0]
                    for row in db.execute(
                        "SELECT instance_id FROM branch_instances WHERE registry_id=?", (owner,)
                    )
                )
                if instances:
                    placeholders = ",".join("?" for _ in instances)
                    result = db.execute(
                        f"UPDATE attempts SET state='cancel_requested',error_type='BranchRetired',"
                        f"error_message='Owning branch retired',completed_at=NULL "
                        f"WHERE instance_id IN ({placeholders}) AND state IN ('queued','running')",
                        instances,
                    )
                    own_cancelled = result.rowcount
                    dependency_state.invalidate(
                        db,
                        instances,
                        now=now,
                        reason=f"Upstream branch {branch} retired",
                    )
        published = store._publish(before, after)

    from nro.orchestration.branch_reconciliation import reconcile_branch_requests

    rebound = reconcile_branch_requests(registry)
    cancelled = registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    return {
        "branch": branch,
        "parent": published.topology.records[branch].parent,
        "retired": published.topology.records[branch].retired,
        "revision": published.revision,
        "reconciled_requests": rebound,
        "cancelled_attempts": own_cancelled + len(cancelled),
    }
