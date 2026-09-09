"""Account for every registered ingestion namespace under the shared worker lock."""

from nro.orchestration.branch_store import BranchStore

from .store import ACTIVE, IngestionStore


class IngestionIndex:
    """Aggregate runtime accounting without merging session identities or review leases.

    Namespace reads use the atomically published branch catalog. Callers hold
    the global registry lock when making scheduling or recovery decisions.
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
            if row["state"] == "running"
            or row["state"] == "queued"
            and (
                row["branch"] == "main"
                or row["branch"] in records
                and not records[row["branch"]].retired
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
        """Claim from one namespace under the existing global capacity checks."""
        for store in self.stores():
            record = store.claim(worker, memory_gb)
            if record is not None:
                return record
        return None

    def set_concurrency_locked(self, concurrency: int) -> int:
        """Update active requests in every namespace under the caller's scheduler lock."""
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Concurrency must be a positive integer")
        changed = 0
        for store in self.stores():
            for row in store.rows():
                if row["state"] in ACTIVE:
                    row["config"]["concurrency"] = concurrency
                    store.write_locked(row)
                    changed += 1
        return changed

    def recover_locked(self, dead_workers: set[str]) -> int:
        """Release stages only for workers whose shutdown has been confirmed."""
        return sum(store.recover_locked(dead_workers) for store in self.stores())
