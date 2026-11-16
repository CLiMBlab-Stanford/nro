"""Index attempt references and work-item freshness for scheduler maintenance."""

from nro.orchestration.migrations import CreateIndex, Migration

MIGRATION = Migration(
    destination=27,
    summary="Index attempt references and work-item freshness for scheduler maintenance",
    operations=(
        CreateIndex("artifact_attempt", "artifacts", ("attempt_id",)),
        CreateIndex("resource_step_attempt", "resource_step_tasks", ("attempt_id",)),
        CreateIndex("work_item_artifact_state", "work_items", ("artifact_state",)),
    ),
)
