"""Store retryable scheduler requests in the central registry."""

from nro.orchestration.migrations import Column, CreateIndex, CreateTable, Migration

MIGRATION = Migration(
    destination=23,
    summary="Replace filesystem request journals with scheduler registry records",
    operations=(
        CreateTable(
            "scheduler_requests",
            columns=(
                Column("id", "TEXT", nullable=False),
                Column("kind", "TEXT", nullable=False),
                Column("record_fingerprint", "TEXT", nullable=False),
                Column("record_json", "TEXT", nullable=False),
                Column("state", "TEXT", nullable=False),
                Column("response_json", "TEXT"),
                Column("created_at", "TEXT", nullable=False),
                Column("updated_at", "TEXT", nullable=False),
            ),
            primary_key=("id",),
        ),
        CreateIndex(
            "scheduler_request_state",
            "scheduler_requests",
            ("state", "updated_at"),
        ),
    ),
)
