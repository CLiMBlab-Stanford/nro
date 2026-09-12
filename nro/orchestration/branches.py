"""Branch identity, inheritance, and write boundaries for development isolation.

These records do not activate branch execution. The scheduler must attach a
registered context to each request before using its paths or permissions.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import quote


@lru_cache(maxsize=256)
def _validated_branch_id(name: str) -> str:
    """Validate and encode one branch name for this process."""
    result = subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{name}"], capture_output=True, text=True
    )
    if result.returncode:
        raise ValueError(f"Invalid Git branch name: {name!r}")
    encoded = quote(name, safe="")
    if len(encoded.encode()) > 200:
        raise ValueError("Encoded branch name exceeds the supported directory length")
    return encoded


def branch_id(name: str) -> str:
    """Encode a valid Git branch name as a collision-free directory component."""
    if not isinstance(name, str) or not name or name.startswith("-") or name == "HEAD":
        raise ValueError("Expected a named Git branch")
    return _validated_branch_id(name)


def checkout_identity(checkout: Path) -> tuple[Path, str, str]:
    """Read a checkout's root, named branch, and commit; reject detached HEAD."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)
        if result.returncode:
            raise ValueError("Expected a Git checkout on a named branch")
        return result.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    name = git("symbolic-ref", "--short", "HEAD")
    branch_id(name)
    return root, name, git("rev-parse", "HEAD")


def repository_identity(checkout: Path) -> str | None:
    """Identify the Git repository used by a checkout.

    Prefer the configured origin so separate clones can attach to one branch.
    Repositories without an origin use their common Git directory, which still
    permits ordinary worktrees. None is returned only when a mocked or vanished
    checkout cannot supply repository metadata.
    """
    checkout = Path(checkout).expanduser().resolve()
    remote = subprocess.run(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if remote is not None and remote.returncode == 0 and remote.stdout.strip():
        return "origin:" + remote.stdout.strip()
    common = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if common is not None and common.returncode == 0 and common.stdout.strip():
        return "git:" + str(Path(common.stdout.strip()).resolve())
    return None


@dataclass(frozen=True)
class BranchRecord:
    """A branch's registered parent and authorized checkouts.

    An empty checkout list reserves the name but grants no execution authority.
    Retired records retain their name and artifacts; they cannot accept work.
    """

    name: str
    parent: str | None
    checkouts: tuple[Path, ...] = ()
    retired: bool = False
    registry_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        branch_id(self.name)
        if not isinstance(self.registry_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", self.registry_id
        ):
            raise ValueError("Invalid branch registry identity")
        if self.parent is not None:
            branch_id(self.parent)
        object.__setattr__(
            self,
            "checkouts",
            tuple(dict.fromkeys(Path(path).expanduser().resolve() for path in self.checkouts)),
        )


class BranchTopology:
    """Validate and query an explicit inheritance tree rooted at main and dev.

    Git merges do not mutate this tree. Reparenting returns a new topology so
    contexts already held by a request retain their chosen ancestry.
    """

    def __init__(self, records: Mapping[str, BranchRecord]) -> None:
        """Copy records and reject collisions, cycles, or feature branches outside dev."""
        records = dict(records)
        if any(name != record.name for name, record in records.items()):
            raise ValueError("Branch registration keys must match their names")
        if "main" not in records or "dev" not in records:
            raise ValueError("Branch topology must reserve main and dev")
        if records["main"].parent is not None or records["dev"].parent != "main":
            raise ValueError("The inheritance spine must be main -> dev")
        if records["main"].retired or records["dev"].retired:
            raise ValueError("Main and dev cannot be retired")
        roots: dict[Path, str] = {}
        if len({record.registry_id for record in records.values()}) != len(records):
            raise ValueError("Each branch must have its own registry identity")
        for record in records.values():
            for root in record.checkouts:
                if root in roots and roots[root] != record.name:
                    raise ValueError("A checkout cannot be registered to two branches")
                roots[root] = record.name
        self.records = MappingProxyType(records)
        for name in records:
            chain = self.ancestors(name)
            if chain[-1] != "main" or (name not in {"main", "dev"} and "dev" not in chain):
                raise ValueError("Feature branches must descend from dev")
            if not records[name].retired and any(records[parent].retired for parent in chain[1:]):
                raise ValueError("Reparent active children before retiring their parent")

    @classmethod
    def reserved(cls) -> BranchTopology:
        """Reserve the production and development spine without authorizing checkouts."""
        return cls({"main": BranchRecord("main", None), "dev": BranchRecord("dev", "main")})

    def ancestors(self, name: str, *, inherit: bool = True) -> tuple[str, ...]:
        """Return nearest-first candidates; disabling inheritance keeps only the owner."""
        chain = []
        current: str | None = name
        while current is not None:
            if current in chain:
                raise ValueError("Branch inheritance cannot contain a cycle")
            if current not in self.records:
                raise ValueError(f"Unregistered branch: {current}")
            chain.append(current)
            current = self.records[current].parent
        return tuple(chain if inherit else chain[:1])

    def register(self, record: BranchRecord) -> BranchTopology:
        """Add an unused name; retained or retired names cannot be registered again."""
        if record.name in self.records:
            raise ValueError(f"Branch name is already reserved: {record.name}")
        return BranchTopology({**self.records, record.name: record})

    def authorize_checkout(self, name: str, checkout: Path) -> BranchTopology:
        """Bind an existing named checkout to its registration without switching Git refs."""
        root, actual_name, _commit = checkout_identity(checkout)
        if actual_name != name:
            raise ValueError("Checkout branch does not match its registration")
        record = self.records[name]
        if record.retired:
            raise ValueError("Retired branches cannot authorize checkouts")
        repository = repository_identity(root)
        related = [
            repository_identity(path)
            for candidate in (name, *self.ancestors(name)[1:])
            for path in self.records[candidate].checkouts
        ]
        known = {value for value in related if value is not None}
        if repository is not None and known and repository not in known:
            raise ValueError("Checkout belongs to a different Git repository")
        return BranchTopology(
            {**self.records, name: replace(record, checkouts=(*record.checkouts, root))}
        )

    def reparent(self, name: str, parent: str) -> BranchTopology:
        """Return a validated parent change without moving artifacts or existing contexts."""
        if name in {"main", "dev"}:
            raise ValueError("Cannot reparent the main/dev spine")
        return BranchTopology({**self.records, name: replace(self.records[name], parent=parent)})

    def retire(self, name: str) -> BranchTopology:
        """Retain a name but reject further work; active children must be reparented first."""
        return BranchTopology({**self.records, name: replace(self.records[name], retired=True)})

    def require_checkout(self, checkout: Path) -> str:
        """Verify both the current Git branch and the registered checkout path."""
        root, name, _commit = checkout_identity(checkout)
        record = self.records.get(name)
        if record is None or record.retired or root not in record.checkouts:
            raise ValueError("Checkout is not authorized for this branch")
        return name

    def registered_checkout(self, checkout: Path) -> str:
        """Resolve an active checkout whose Git identity was checked at registration."""
        root = Path(checkout).expanduser().resolve()
        matches = [
            name
            for name, record in self.records.items()
            if not record.retired and root in record.checkouts
        ]
        if len(matches) != 1:
            raise ValueError("Checkout is not authorized for an active branch")
        return matches[0]

    def require_branch_checkout(self, name: str, checkout: Path) -> None:
        """Require an attached checkout of the branch being changed."""
        if self.require_checkout(checkout) != name:
            raise ValueError(f"Change branch {name} from one of its attached checkouts")


@dataclass(frozen=True)
class BranchPaths:
    """Separate shared raw BIDS from branch-owned public and private outputs.

    Path checks prevent mistakes in normal nro operations. They are not an OS
    sandbox for code running with write access to other branches' files.
    """

    branch: str
    bids: Path
    work: Path
    development: Path

    def __post_init__(self) -> None:
        branch_id(self.branch)
        for field_name in ("bids", "work", "development"):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)).expanduser().resolve(),
            )
        roots = (self.bids, self.work, self.development)
        if any(
            a.is_relative_to(b) or b.is_relative_to(a)
            for index, a in enumerate(roots)
            for b in roots[index + 1 :]
        ):
            raise ValueError("Raw BIDS, work, and development roots must not overlap")

    @staticmethod
    def _project(project: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project):
            raise ValueError("Invalid project identifier")
        return project

    def source_project(self, project: str) -> Path:
        """Return shared raw data; development trees never supply source subjects."""
        return self.bids / self._project(project)

    def output_project(self, project: str) -> Path:
        """Return the owner-specific project root without creating directories."""
        root = (
            self.bids
            if self.branch == "main"
            else self.development / branch_id(self.branch) / "BIDS"
        )
        return root / self._project(project)

    def private_project(self, project: str) -> Path:
        """Return the owner-specific work root without creating directories."""
        root = (
            self.work
            if self.branch == "main"
            else self.development / branch_id(self.branch) / "WORK"
        )
        return root / self._project(project)

    def require_output(self, path: Path, project: str, *, private: bool = False) -> Path:
        """Reject writes outside the branch's derivative root, including symlink escapes.

        The caller must verify checkout authority separately. This check is for
        individual operations and does not protect against concurrent symlink
        replacement or arbitrary code running under the same Unix identity.
        """
        root = (
            self.private_project(project) if private else self.output_project(project)
        ) / "derivatives"
        resolved = Path(path).expanduser().resolve()
        if root.resolve() != root or not resolved.is_relative_to(root):
            raise ValueError("Output is outside the branch-owned derivative root")
        return resolved
