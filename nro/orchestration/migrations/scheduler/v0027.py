"""Index artifact and resource-step references to their attempts."""

from nro.orchestration.migrations import CreateIndex, Migration

MIGRATION = Migration(
    destination=27,
    summary="Index artifact and resource-step attempt references for history pruning",
    operations=(
        CreateIndex("artifact_attempt", "artifacts", ("attempt_id",)),
        CreateIndex("resource_step_attempt", "resource_step_tasks", ("attempt_id",)),
    ),
)
