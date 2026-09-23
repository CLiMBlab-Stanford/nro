"""Maintain scheduler-owned execution state independently of worker polling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nro.orchestration.registry import Registry


def refresh_scheduler_state(registry: Registry) -> int:
    """Refresh active execution state once for the shared worker pool.

    The scheduler calls this function on a global cadence. Workers only claim
    assignments and report results, so a batch of new workers cannot repeat the
    same filesystem assessment or allocation reconciliation.
    """
    from nro.orchestration.assessment import AssessmentConflict
    from nro.orchestration.branch_reconciliation import reconcile_branch_requests
    from nro.orchestration.manifests import assess_registry

    registry.reconcile_scheduler_submissions()
    registry.recover_orphaned_attempts()
    cancelled: list[dict] = []
    if registry.reserve_artifact_assessment():
        try:
            demanded = registry.demanded_work_item_ids()
            if demanded:
                try:
                    assess_registry(registry, work_item_ids=demanded, compiled=True)
                except AssessmentConflict:
                    pass
            reconcile_branch_requests(registry)
            cancelled = registry.cancel_attempts_with_stale_upstreams()
            registry.reconcile_requests()
        finally:
            registry.finish_artifact_assessment()
    return len(cancelled)
