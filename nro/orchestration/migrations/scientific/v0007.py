"""Add reconstructible caches for compiled participant plans and source manifests."""

from nro.orchestration.migrations import Column, CreateIndex, CreateTable, Migration

MIGRATION = Migration(
    destination=7,
    operations=(
        CreateTable(
            "planning_cache",
            (
                Column("cache_key", "TEXT", nullable=False),
                Column("scope_key", "TEXT", nullable=False),
                Column("payload", "BLOB", nullable=False),
                Column("updated_at", "TEXT", nullable=False),
            ),
            primary_key=("cache_key",),
        ),
        CreateIndex("planning_cache_scope", "planning_cache", ("scope_key",)),
        CreateTable(
            "planning_files",
            (
                Column("path", "TEXT", nullable=False),
                Column("size", "INTEGER", nullable=False),
                Column("mtime_ns", "INTEGER", nullable=False),
                Column("kind", "TEXT", nullable=False),
                Column("digest", "TEXT", nullable=False),
            ),
            primary_key=("path",),
        ),
    ),
    summary="Cache compiled participant plans and reusable source-file checksums",
)
