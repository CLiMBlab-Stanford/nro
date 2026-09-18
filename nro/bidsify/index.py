"""Account for every registered ingestion namespace inside the controller."""

from nro.orchestration.branch_store import BranchStore

from .store import ACTIVE, IngestionStore


class IngestionIndex:
    """Aggregate runtime accounting without merging session identities or review leases.

    Namespace reads use the atomically published branch catalog. The scheduler
    controller serializes scheduling and recovery decisions.
    Retired branches still count until their running stages have stopped.
    """

    def __init__(self, registry):
        """Bind existing production and branch records without initializing stores."""
        self.registry = registry

    def stores(self) -> tuple[IngestionStore, ...]:
        """Return isolated stores, including retained state in retired branches."""
        catalog = BranchStore(self.registry.paths.control)
        branches = catalog.read().topology.records if catalog.path.exists() else {}
        return (
            IngestionStore(self.registry),
            *(
                IngestionStore(self.registry, branch=name)
                for name in sorted(branches)
                if name != "main"
            ),
        )

    def rows(self) -> list[dict]:
        """Read all namespaces for global lifecycle checks, without deduplicating them."""
        return [row for store in self.stores() for row in store.rows()]

    def execution_records(self) -> list[dict]:
        """Retain running pins and queues whose branches can still execute."""
        catalog = BranchStore(self.registry.paths.control)
        records = catalog.read().topology.records if catalog.path.exists() else {}
        return [
            row
            for row in self.rows()
            if row["state"] in {"running", "cancel_requested"}
            or (
                row["state"] == "queued"
                and (
                    row["branch"] == "main"
                    or (row["branch"] in records and not records[row["branch"]].retired)
                )
            )
        ]

    def summary(self, memory_gb: int | None = None) -> tuple[int, int, int]:
        """Count admitted queues and running stages across every namespace."""
        active = ready = limit = 0
        for store in self.stores():
            count, runnable, requested_limit = store.summary(memory_gb)
            active += count
            ready += runnable
            limit = max(limit, requested_limit)
        return active, ready, limit

    def claim(self, worker: str, memory_gb: int) -> dict | None:
        """Claim from one namespace after the controller applies capacity checks."""
        active = next(
            (
                row
                for row in self.rows()
                if row["state"] in {"running", "cancel_requested"} and row["worker"] == worker
            ),
            None,
        )
        if active is not None:
            return active
        for store in self.stores():
            record = store.claim(worker, memory_gb)
            if record is not None:
                return record
        return None

    def set_concurrency_locked(self, concurrency: int) -> int:
        """Update active requests in every namespace during a controller operation."""
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Concurrency must be a positive integer")
        changed = 0
        for store in self.stores():
            with store._lock():
                for row in store.rows():
                    if row["state"] in ACTIVE:
                        row["config"]["concurrency"] = concurrency
                        store.write_locked(row)
                        changed += 1
        return changed

    def recover_locked(self, dead_workers: set[str]) -> int:
        """Release stages only for workers whose shutdown has been confirmed."""
        recovered = 0
        for store in self.stores():
            with store._lock():
                recovered += store.recover_locked(dead_workers)
        return recovered

    def clear_publication_barriers_locked(self, db) -> None:
        """Remove project barriers whose publication stage no longer runs."""
        active = {
            row["id"]
            for row in self.rows()
            if row["state"] in {"running", "cancel_requested"} and row["stage"] == "publish"
        }
        rows = db.execute(
            "SELECT key,value FROM metadata WHERE key LIKE 'bids_publication:%'"
        ).fetchall()
        db.executemany(
            "DELETE FROM metadata WHERE key=? AND value=?",
            [(row["key"], row["value"]) for row in rows if row["value"] not in active],
        )
