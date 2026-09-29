"""Enable bounded reclamation after current-state retention removes old rows."""

from nro.orchestration.migrations import Migration

MIGRATION = Migration(
    destination=25,
    summary="Enable incremental page reclamation for the current-state registry",
    operations=(),
    compact=True,
    incremental_vacuum=True,
)
