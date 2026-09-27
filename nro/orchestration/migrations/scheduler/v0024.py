"""Index current-state lookups from work-item identities."""

from nro.orchestration.migrations import CreateIndex, Migration

MIGRATION = Migration(
    destination=24,
    summary="Index scheduler current-state projections and targeted mutations",
    compact=True,
    operations=(
        CreateIndex("attempt_work_item_history", "attempts", ("work_item_id", "id")),
        CreateIndex(
            "request_work_item_demand",
            "request_work_items",
            ("work_item_id", "demand_state", "request_id"),
        ),
        CreateIndex(
            "resource_step_work_item_history",
            "resource_step_tasks",
            ("work_item_id", "generation", "revision_fingerprint", "id"),
        ),
        CreateIndex(
            "dependency_upstream_consumers",
            "work_item_dependencies",
            ("upstream_work_item_id", "work_item_id"),
        ),
        CreateIndex(
            "branch_work_item_identity",
            "branch_work_items",
            ("registry_id", "work_item_id"),
        ),
        CreateIndex("artifact_work_item", "artifacts", ("work_item_id",)),
        CreateIndex("worker_state", "workers", ("state",)),
        CreateIndex("worker_updated_at", "workers", ("updated_at",)),
        CreateIndex(
            "scheduler_submission_state",
            "scheduler_submissions",
            ("state", "resource_class", "memory_gb"),
        ),
    ),
)
