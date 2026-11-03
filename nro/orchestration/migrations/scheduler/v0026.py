"""Allow requests that no workflow revision owns, such as study analyses."""

from nro.orchestration.migrations import DropNotNull, Migration

MIGRATION = Migration(
    destination=26,
    summary="Allow requests without a workflow revision for study analyses",
    operations=(DropNotNull("requests", "workflow_revision_id"),),
)
