"""Strengthen reusable source-file signatures with change times."""

from nro.orchestration.migrations import AddColumn, Column, Migration

MIGRATION = Migration(
    destination=8,
    operations=(
        AddColumn(
            "planning_files",
            Column("ctime_ns", "INTEGER", nullable=False, default=0),
        ),
    ),
    summary="Detect same-size source edits even when modification times are preserved",
)
