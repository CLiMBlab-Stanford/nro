"""Record maintainer approval of clean main commits without changing Git refs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from nro.engine.io import atomic_write_json
from nro.orchestration.branch_store import BranchStore
from nro.versioning import parse_release_version, require_release_advance


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Cannot verify release Git state: " + " ".join(args[:2]))
    return result.stdout.strip()


def _source_at(checkout: Path, ref: str) -> tuple[str, str, str]:
    """Read source identity and package version at a Git revision."""
    commit = _git(checkout, "rev-parse", f"{ref}^{{commit}}")
    tree = _git(checkout, "rev-parse", f"{commit}^{{tree}}")
    metadata = tomllib.loads(_git(checkout, "show", f"{commit}:pyproject.toml"))
    version = metadata.get("project", {}).get("version")
    parse_release_version(version)
    return commit, tree, version


def _source(checkout: Path) -> tuple[str, str, str]:
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Main release approval requires a clean checkout")
    return _source_at(checkout, "HEAD")


class ReleaseStore:
    """Serialize release attestations beside the shared branch catalog.

    Attestations are explicit maintainer statements, not hosting-service
    verification or cryptographic signatures. Approval neither creates a Git
    tag nor deploys code. Release versions do not enter scientific contracts.
    """

    def __init__(self, branches: BranchStore):
        """Bind the site's approval ledger without creating or changing records."""
        self.branches = branches
        self.path = branches.root / "releases.json"

    def _read(self) -> list[dict]:
        if self.path.resolve() != self.path:
            raise ValueError("Release records cannot be redirected through a symlink")
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "releases"}
            or type(data["schema"]) is not int
            or data["schema"] != 1
        ):
            raise ValueError("Unsupported release record format")
        rows = data["releases"]
        if not isinstance(rows, list):
            raise ValueError("Invalid release history")
        previous = None
        commits = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "version",
                "commit",
                "tree",
                "registry_id",
                "pr",
                "attested_by",
                "uid",
                "approved_at",
            }:
                raise ValueError("Invalid release record")
            require_release_advance(previous, row["version"])
            bootstrap = previous is None and row["version"] == "0.0.1" and row["pr"] is None
            if (
                type(row["uid"]) is not int
                or row["uid"] < 0
                or not isinstance(row["registry_id"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", row["registry_id"])
                or any(
                    not isinstance(row[key], str) or not row[key].strip()
                    for key in ("attested_by", "approved_at")
                )
                or (not bootstrap and (not isinstance(row["pr"], str) or not row["pr"].strip()))
            ):
                raise ValueError("Invalid release attestation identity")
            datetime.fromisoformat(row["approved_at"])
            if any(
                not isinstance(row[key], str)
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row[key])
                for key in ("commit", "tree")
            ):
                raise ValueError("Invalid release source identity")
            if row["commit"] in commits:
                raise ValueError("A commit cannot have multiple main release versions")
            commits.add(row["commit"])
            previous = row["version"]
        return rows

    def history(self) -> tuple[dict, ...]:
        """Read approved versions without creating a store or contacting Git hosting."""
        return tuple(self._read())

    def approve(
        self,
        checkout: Path,
        version: str,
        *,
        pr: str | None = None,
        attest_merged: bool = False,
        bootstrap: bool = False,
    ) -> dict:
        """Approve a registered clean main commit after human attestation.

        Normal approval affirms that the referenced PR was approved and merged.
        Bootstrap approval records the tagged initial 0.0.1 commit that created
        main and is unavailable after any release exists. It may be called from a
        later main release when v0.0.1 remains in that branch's ancestry. Git
        configuration supplies the maintainer's stated name and email; the record
        also retains the executing Unix identity. Subsequent commits must descend
        from the prior approved commit and advance by at least one patch version.
        """
        if bootstrap:
            if version != "0.0.1":
                raise ValueError("Bootstrap approval is limited to release 0.0.1")
            if pr is not None or attest_merged:
                raise ValueError("Bootstrap approval does not accept PR attestation options")
        else:
            if attest_merged is not True:
                raise ValueError("Explicit attestation of PR approval and merge is required")
            if not isinstance(pr, str) or not pr.strip() or any(ord(c) < 32 for c in pr):
                raise ValueError("Supply the approved and merged PR reference")
        checkout = Path(checkout).expanduser().resolve()
        with self.branches._lock():
            snapshot = self.branches.read()
            if snapshot.topology.require_checkout(checkout) != "main":
                raise ValueError("Only an authorized main checkout can approve a release")
            current_source = _source(checkout)
            if bootstrap:
                commit, tree, packaged_version = _source_at(checkout, "refs/tags/v0.0.1")
                _git(checkout, "merge-base", "--is-ancestor", commit, current_source[0])
            else:
                commit, tree, packaged_version = current_source
            if packaged_version != version:
                raise ValueError("Release version must match the committed pyproject.toml")
            name = _git(checkout, "config", "user.name")
            email = _git(checkout, "config", "user.email")
            if not name or not email:
                raise ValueError("Configure the human maintainer Git name and email first")
            rows = self._read()
            if bootstrap and rows:
                raise ValueError("Bootstrap approval is unavailable after the first release")
            if any(
                row["registry_id"] != snapshot.topology.records["main"].registry_id for row in rows
            ):
                raise ValueError("Release history belongs to a different main registration")
            require_release_advance(rows[-1]["version"] if rows else None, version)
            if rows:
                _git(checkout, "merge-base", "--is-ancestor", rows[-1]["commit"], commit)
            row = dict(
                version=version,
                commit=commit,
                tree=tree,
                registry_id=snapshot.topology.records["main"].registry_id,
                pr=None if bootstrap else pr.strip(),
                attested_by=f"{name} <{email}>",
                uid=os.getuid(),
                approved_at=datetime.now(timezone.utc).isoformat(),
            )
            if _source(checkout) != current_source or (
                bootstrap
                and _source_at(checkout, "refs/tags/v0.0.1") != (commit, tree, packaged_version)
            ):
                raise ValueError("Main source changed during approval")
            snapshot.topology.require_checkout(checkout)
            atomic_write_json(
                self.path, {"schema": 1, "releases": [*rows, row]}, mode=0o664, durable=True
            )
            return row

    def require_approved(self, checkout: Path) -> dict:
        """Reject dirty, unregistered, or unapproved main source before production use."""
        checkout = Path(checkout).expanduser().resolve()
        with self.branches._lock():
            snapshot = self.branches.read()
            if snapshot.topology.require_checkout(checkout) != "main":
                raise ValueError("Production release source must be an authorized main checkout")
            commit, tree, version = _source(checkout)
            rows = self._read()
            matches = [
                row
                for row in rows
                if (row["commit"], row["tree"], row["version"]) == (commit, tree, version)
                and row["registry_id"] == snapshot.topology.records["main"].registry_id
            ]
            if len(matches) != 1:
                raise ValueError("Main source has no matching release attestation")
            return matches[0]
