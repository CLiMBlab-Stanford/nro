"""Explicit storage bindings carried by a scheduled scientific attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.orchestration.branches import BranchPaths


@dataclass(frozen=True)
class InputBinding:
    """Bind one logical producer root and filename prefix to its selected artifact."""

    branch: str
    key: str
    generation: int | None
    logical_root: Path
    physical_root: Path
    prefix: str | None

    def resolve(self, path: Path) -> Path | None:
        """Resolve a member of this artifact, returning None for unrelated paths."""
        path = Path(path).expanduser().absolute()
        root = self.logical_root.expanduser().absolute()
        if ".." in path.parts or ".." in root.parts:
            raise ValueError("Artifact bindings require normalized absolute paths")
        if not path.is_relative_to(root):
            return None
        if self.prefix and not (
            path.name == self.prefix or path.name.startswith(self.prefix + "_")
        ):
            return None
        return self.physical_root / path.relative_to(root)


@dataclass(frozen=True)
class ExecutionContext:
    """The owner and fixed input selections of one attempt.

    Paths are supplied by the scheduler after branch authorization. They do not
    consult Git or choose a different input when files change during execution.
    """

    paths: BranchPaths
    project: str
    instance_key: str
    inputs: tuple[InputBinding, ...]

    def require_output(self, path: Path) -> Path:
        """Validate an already resolved public or private destination without remapping it."""
        for private in (False, True):
            try:
                return self.paths.require_output(path, self.project, private=private)
            except ValueError:
                pass
        raise ValueError(f"Output is outside the branch-owned derivative roots: {path}")

    def input_path(self, logical_path: Path) -> Path:
        """Resolve a selected derivative member or retain an exact raw source path."""
        logical_path = Path(logical_path).expanduser().absolute()
        if ".." in logical_path.parts:
            raise ValueError("Artifact inputs require normalized absolute paths")
        source = self.paths.source_project(self.project)
        matches = []
        for binding in self.inputs:
            value = binding.resolve(logical_path)
            if value is None:
                continue
            if not logical_path.is_relative_to(source / "derivatives"):
                raise ValueError("Artifact bindings cannot redirect raw scientific inputs")
            producer = BranchPaths(
                binding.branch, self.paths.bids, self.paths.work, self.paths.development
            )
            producer.require_output(value, self.project)
            matches.append(value)
        if len(matches) > 1:
            raise ValueError(f"Ambiguous artifact input binding: {logical_path}")
        if matches:
            return matches[0]
        path = Path(logical_path).expanduser().absolute()
        if (
            path.is_relative_to(source / "derivatives")
            or path.is_relative_to(self.paths.development)
            or path.resolve().is_relative_to(self.paths.development)
            or path.is_relative_to(self.paths.work)
        ):
            raise ValueError(f"Derivative input was not selected for this attempt: {path}")
        return path

    def output_path(self, logical_path: Path, *, private: bool = False) -> Path:
        """Place an output in its owner's tree and reject raw or foreign destinations."""
        logical = (
            self.paths.work / self.project if private else self.paths.source_project(self.project)
        ) / "derivatives"
        path = Path(logical_path).expanduser().absolute()
        if not path.is_relative_to(logical):
            return self.paths.require_output(path, self.project, private=private)
        owner = (
            self.paths.private_project(self.project)
            if private
            else self.paths.output_project(self.project)
        ) / "derivatives"
        return self.paths.require_output(
            owner / path.relative_to(logical), self.project, private=private
        )

    def as_dict(self) -> dict:
        """Serialize exact attempt bindings without adding them to scientific identity."""
        return {
            "branch": self.paths.branch,
            "bids": str(self.paths.bids),
            "work": str(self.paths.work),
            "development": str(self.paths.development),
            "project": self.project,
            "instance_key": self.instance_key,
            "inputs": [
                dict(
                    branch=value.branch,
                    key=value.key,
                    generation=value.generation,
                    logical_root=str(value.logical_root),
                    physical_root=str(value.physical_root),
                    prefix=value.prefix,
                )
                for value in self.inputs
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> ExecutionContext:
        """Decode an attempt context; authority must be checked before publication."""
        if not isinstance(value, Mapping) or set(value) != {
            "branch",
            "bids",
            "work",
            "development",
            "project",
            "instance_key",
            "inputs",
        }:
            raise ValueError("Invalid execution context fields")
        if any(
            not isinstance(value[key], str) or not value[key]
            for key in ("branch", "bids", "work", "development", "project", "instance_key")
        ):
            raise ValueError("Invalid execution context identity")
        if not isinstance(value["inputs"], list):
            raise ValueError("Invalid execution input bindings")
        for item in value["inputs"]:
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "branch",
                    "key",
                    "generation",
                    "logical_root",
                    "physical_root",
                    "prefix",
                }
                or (
                    item["generation"] is not None
                    and (type(item["generation"]) is not int or item["generation"] < 0)
                )
                or any(
                    not isinstance(item[key], str) or not item[key]
                    for key in ("branch", "key", "logical_root", "physical_root")
                )
                or (item["prefix"] is not None and not isinstance(item["prefix"], str))
            ):
                raise ValueError("Invalid execution input binding")
        return cls(
            BranchPaths(
                str(value["branch"]),
                Path(value["bids"]),
                Path(value["work"]),
                Path(value["development"]),
            ),
            str(value["project"]),
            str(value["instance_key"]),
            tuple(
                InputBinding(
                    str(item["branch"]),
                    str(item["key"]),
                    item["generation"],
                    Path(item["logical_root"]),
                    Path(item["physical_root"]),
                    item["prefix"],
                )
                for item in value["inputs"]
            ),
        )
