"""Add durable resource-step tasks for intra-work-item worker handoffs."""

from nro.orchestration.migrations import (
    Column,
    CreateIndex,
    CreateTable,
    ForeignKey,
    Migration,
)

MIGRATION = Migration(
    destination=22,
    summary="Add resource-step tasks for CPU/GPU runner handoffs",
    operations=(
        CreateTable(
            "resource_step_tasks",
            columns=(
                Column("id", "INTEGER", nullable=False),
                Column("work_item_id", "INTEGER", nullable=False),
                Column("step_id", "TEXT", nullable=False),
                Column("resource_class", "TEXT", nullable=False),
                Column("memory_gb", "INTEGER", nullable=False, default=32),
                Column("generation", "INTEGER", nullable=False),
                Column("revision_fingerprint", "TEXT", nullable=False),
                Column("state", "TEXT", nullable=False),
                Column("worker_id", "TEXT"),
                Column("attempt_id", "INTEGER"),
                Column("error_type", "TEXT"),
                Column("error_message", "TEXT"),
                Column("created_at", "TEXT", nullable=False),
                Column("updated_at", "TEXT", nullable=False),
                Column("completed_at", "TEXT"),
            ),
            primary_key=("id",),
            foreign_keys=(
                ForeignKey("work_item_id", "work_items", on_delete="CASCADE"),
                ForeignKey("worker_id", "workers", on_delete="SET NULL"),
                ForeignKey("attempt_id", "attempts", on_delete="SET NULL"),
            ),
            unique=(("work_item_id", "step_id", "generation", "revision_fingerprint"),),
        ),
        CreateIndex(
            "resource_step_task_state",
            "resource_step_tasks",
            ("state", "resource_class", "memory_gb"),
        ),
    ),
)
