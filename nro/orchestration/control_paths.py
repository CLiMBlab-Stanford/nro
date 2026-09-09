"""Resolve private site state without creating directories or selecting a branch."""

from dataclasses import dataclass
from pathlib import Path

from nro.orchestration.branches import branch_id


@dataclass(frozen=True)
class ControlPaths:
    """One site-wide root, shared services, and branch-owned scientific state."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if self.root.parent == self.root:
            raise ValueError("The filesystem root cannot be a private-control directory")

    @property
    def shared(self) -> Path:
        """Return the directory for state shared across branches."""
        return self.root / "shared"

    @property
    def scheduler(self) -> Path:
        """Return the sole scheduling authority's directory."""
        return self.shared / "scheduler"

    @property
    def database(self) -> Path:
        """Return the shared scheduler database."""
        return self.scheduler / "registry.sqlite3"

    @property
    def catalog(self) -> Path:
        """Return the shared branch catalog."""
        return self.shared / "branches.json"

    @property
    def ingestion(self) -> Path:
        """Return site-wide ingestion state and publication receipts."""
        return self.shared / "ingestion"

    @property
    def cache(self) -> Path:
        """Return execution data whose lifetime is shared across branches."""
        return self.shared / "cache"

    @property
    def implementations(self) -> Path:
        """Return the content-addressed source cache."""
        return self.cache / "implementations"

    @property
    def execution_sites(self) -> Path:
        """Return immutable resolved site settings for attempts."""
        return self.cache / "execution-sites"

    @property
    def promotions(self) -> Path:
        """Return journals for transfers between branch output namespaces."""
        return self.shared / "promotions"

    @property
    def cutover_journal(self) -> Path:
        """Return the external journal guarding an interrupted layout publication."""
        return self.root.with_name(self.root.name + ".cutover.json")

    def branch(self, name: str) -> Path:
        """Return private state for a validated, collision-free branch ID."""
        encoded = name if name in ("main", "dev") else branch_id(name)
        path = self.root / "branches" / encoded
        if path.resolve() != path:
            raise ValueError(f"Branch state cannot be redirected through a symlink: {path}")
        return path

    def require_current_layout(self) -> None:
        """Reject an unmoved store instead of creating a second empty authority.

        This is a diagnostic guard, not an alternate-path reader or migration.
        Live layout conversion requires a separate, quiescent maintenance step.
        """
        if self.cutover_journal.exists():
            raise ValueError(
                "Private-state cutover is incomplete; use nro cutover --resume or --rollback"
            )
        if (self.root / "registry.sqlite3").exists() or (
            self.root / "branches/registrations.json"
        ).exists():
            raise ValueError(
                f"Private state at {self.root} uses the previous layout. "
                "Stop or drain work and run nro cutover; "
                "nro will not create a parallel registry or move live state automatically."
            )
        for path in (
            self.shared,
            self.scheduler,
            self.database,
            self.catalog,
            self.cache,
            self.implementations,
            self.execution_sites,
            self.ingestion,
            self.promotions,
            self.root / "branches",
        ):
            if path.resolve() != path:
                raise ValueError(f"Private state cannot be redirected through a symlink: {path}")
