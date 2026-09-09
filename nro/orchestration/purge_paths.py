"""Remove eligible logs and individually authorized filesystem paths."""

import shutil
import subprocess
from pathlib import Path

from nro.orchestration.registry import Registry


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if dry_run:
        return True
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)
    return True


def _purge_attempt_logs(
    registry: Registry,
    *,
    instance_ids: set[int],
    dry_run: bool,
) -> int:
    """Remove terminal attempt logs belonging to the selected instances."""
    if not instance_ids:
        return 0
    attempt_count = 0
    with registry.connection() as db:
        terminal_attempts = db.execute(
            """SELECT id, instance_id, log_path FROM attempts
               WHERE state NOT IN ('queued', 'running', 'cancel_requested')"""
        ).fetchall()
        active_instance_logs = {
            str(row["log_path"])
            for row in db.execute(
                """SELECT DISTINCT log_path FROM attempts
                   WHERE state IN ('queued', 'running', 'cancel_requested')
                     AND log_path IS NOT NULL"""
            )
        }

    deleted_attempt_ids = []
    for attempt in terminal_attempts:
        if int(attempt["instance_id"]) not in instance_ids:
            continue
        raw_path = str(attempt["log_path"] or "").strip()
        # Sequential attempts deliberately share one current instance log. Never
        # let bare log cleanup remove it while a newer attempt is active.
        if not raw_path or raw_path in active_instance_logs:
            continue
        path = Path(raw_path)
        if _is_within(path, registry.paths.events) and _remove_path(path, dry_run=dry_run):
            attempt_count += 1
            deleted_attempt_ids.append(int(attempt["id"]))
    if deleted_attempt_ids and not dry_run:
        with registry.connection(write=True) as db:
            placeholders = ",".join("?" for _ in deleted_attempt_ids)
            db.execute(
                f"""UPDATE attempts SET log_path=NULL
                    WHERE id IN ({placeholders})
                      AND state NOT IN ('queued', 'running', 'cancel_requested')""",
                deleted_attempt_ids,
            )
    return attempt_count


def _purge_inactive_worker_logs(registry: Registry, *, dry_run: bool) -> int:
    """Remove worker logs only when their Slurm jobs are known to be inactive."""
    worker_count = 0
    with registry.connection() as db:
        active_job_ids = {
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM scheduler_submissions
                   WHERE state IN ('prepared', 'submitted', 'running')
                     AND slurm_job_id IS NOT NULL"""
            )
        }
        active_job_ids.update(
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM workers
                   WHERE state IN ('idle', 'running', 'draining')
                     AND slurm_job_id IS NOT NULL"""
            )
        )
        terminal_job_ids = {
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM scheduler_submissions
                   WHERE state NOT IN ('prepared', 'submitted', 'running')
                     AND slurm_job_id IS NOT NULL"""
            )
        }

    for path in registry.paths.workers.glob("slurm-*.log"):
        job_id = path.stem.removeprefix("slurm-")
        if job_id in active_job_ids:
            continue
        if job_id not in terminal_job_ids and _slurm_job_may_be_active(job_id):
            continue
        worker_count += int(_remove_path(path, dry_run=dry_run))

    return worker_count


def _slurm_job_may_be_active(job_id: str) -> bool:
    """Protect an untracked worker log unless Slurm confirms it is absent."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())
