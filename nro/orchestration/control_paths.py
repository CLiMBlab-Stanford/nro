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
    def service(self) -> Path:
        """Return the ephemeral scheduler service's durable exchange directory."""
        return self.scheduler / "service"

    @property
    def service_progress(self) -> Path:
        """Return transient progress records for long scheduler operations."""
        return self.service / "progress"

    @property
    def service_active(self) -> Path:
        """Return the active controller record."""
        return self.service / "active.json"

    @property
    def service_launch(self) -> Path:
        """Return the atomic controller launch claim."""
        return self.service / "launch"

    @property
    def service_snapshot(self) -> Path:
        """Return the atomic cached scheduler read model."""
        return self.service / "status.json"

    @property
    def service_shutdown(self) -> Path:
        """Return the record that keeps an intentionally stopped controller down."""
        return self.service / "shutdown.json"

    @property
    def implementation_record(self) -> Path:
        """Return the explicit central installation binding."""
        return self.scheduler / "implementation.json"

    @property
    def artifact_mutation_lock(self) -> Path:
        """Return the lock that serializes changes to published artifacts."""
        return self.scheduler / "artifact-mutation.lock"

    @property
    def artifact_mutation_recovery_lock(self) -> Path:
        """Return the recovery lock paired with `artifact_mutation_lock`."""
        return self.scheduler / "artifact-mutation.recovery-lock"

    @property
    def catalog(self) -> Path:
        """Return the shared branch catalog."""
        return self.shared / "branches.json"

    @property
    def ingestion(self) -> Path:
        """Return site-wide ingestion state and publication receipts."""
        return self.shared / "ingestion"

    def ingestion_for(self, branch: str) -> Path:
        """Return ingestion state for one branch; `main` uses the site-wide directory."""
        return self.ingestion if branch == "main" else self.branch(branch) / "ingestion"

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
    def maintenance_preparations(self) -> Path:
        """Return retained migration and project-rename preparations."""
        return self.shared / "maintenance-preparations"

    @property
    def maintenance_transactions(self) -> Path:
        """Return backups kept by shared scheduler repair."""
        return self.shared / "maintenance"

    @property
    def project_renames(self) -> Path:
        """Return project-rename journals."""
        return self.shared / "project-renames"

    @property
    def provenance_migrations(self) -> Path:
        """Return provenance-migration journals."""
        return self.shared / "provenance-migrations"

    @property
    def freshening(self) -> Path:
        """Return freshening journals."""
        return self.shared / "freshening"

    def branch(self, name: str) -> Path:
        """Return private state for a validated, collision-free branch ID."""
        encoded = name if name in ("main", "dev") else branch_id(name)
        path = self.root / "branches" / encoded
        if path.resolve() != path:
            raise ValueError(f"Branch state cannot be redirected through a symlink: {path}")
        return path

    def require_current_layout(self) -> None:
        """Reject the unsupported flat store instead of creating a second authority."""
        if (self.root / "registry.sqlite3").exists() or (
            self.root / "branches/registrations.json"
        ).exists():
            raise ValueError(
                f"Private state at {self.root} uses the previous layout. "
                "This layout predates the supported registry baseline; archive or remove "
                "it before initializing current private state."
            )
        for path in (
            self.shared,
            self.scheduler,
            self.database,
            self.service,
            self.service_progress,
            self.catalog,
            self.cache,
            self.implementations,
            self.execution_sites,
            self.ingestion,
            self.promotions,
            self.maintenance_preparations,
            self.root / "branches",
        ):
            if path.resolve() != path:
                raise ValueError(f"Private state cannot be redirected through a symlink: {path}")
