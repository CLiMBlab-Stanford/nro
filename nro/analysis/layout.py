"""Locate study repositories, read their committed files, and describe studies."""

from __future__ import annotations

import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from nro.analysis import API_VERSION
from nro.analysis.manifest import Manifest, ManifestError, check_entry, parse_manifest
from nro.engine.paths import study_code_root, study_results_root, study_work_root

WORKTREE = "worktree"
_STUDY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class StudyError(ValueError):
    """A study or analysis cannot be located or read."""


def check_study_name(study: str) -> str:
    """Validate a study name used as a directory under both study roots."""
    if not _STUDY_NAME.match(study):
        raise StudyError(
            f"Study name {study!r} must start with a letter or digit and use letters, digits, "
            "hyphens, and underscores"
        )
    return study


def split_analysis_name(value: str) -> tuple[str, str]:
    """Split ``STUDY/ANALYSIS`` into its parts."""
    study, separator, analysis = value.partition("/")
    if not separator or not analysis or "/" in analysis:
        raise StudyError(f"Name analyses as STUDY/ANALYSIS, not {value!r}")
    return check_study_name(study), analysis


def _git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        capture_output=True,
        text=text,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        raise StudyError(f"git {' '.join(arguments)} failed in {repository}: {stderr.strip()}")
    return result.stdout


@dataclass(frozen=True)
class StudySource:
    """Read study files from one commit, or from the working tree.

    ``commit`` is ``WORKTREE`` for uncommitted reads, which are allowed only for
    ``nro study check --worktree`` and ``nro study try``.
    """

    study: str
    repository: Path
    commit: str

    @classmethod
    def at(cls, study: str, ref: str | None = "main") -> "StudySource":
        """Resolve a study repository and a ref, or ``None`` for the working tree."""
        repository = study_code_root(check_study_name(study))
        if not (repository / ".git").exists():
            raise StudyError(f"No study repository at {repository}; run nro study init {study}")
        if ref is None:
            return cls(study, repository, WORKTREE)
        commit = str(_git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}")).strip()
        return cls(study, repository, commit)

    @property
    def is_worktree(self) -> bool:
        """Return whether reads come from uncommitted files."""
        return self.commit == WORKTREE

    def read_text(self, relative: str) -> str | None:
        """Return one file's text, or ``None`` when it does not exist."""
        if self.is_worktree:
            path = self.repository / relative
            return path.read_text(encoding="utf-8") if path.is_file() else None
        result = subprocess.run(
            ("git", "-C", str(self.repository), "show", f"{self.commit}:{relative}"),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None

    def analyses(self) -> tuple[str, ...]:
        """Return the analysis directories that contain a manifest."""
        if self.is_worktree:
            root = self.repository / "analyses"
            names = [path.parent.name for path in root.glob("*/analysis.yml")]
        else:
            listing = str(
                _git(self.repository, "ls-tree", "-r", "--name-only", self.commit, "--", "analyses")
            )
            names = [
                parts[1]
                for parts in (line.split("/") for line in listing.splitlines())
                if len(parts) == 3 and parts[2] == "analysis.yml"
            ]
        return tuple(sorted(set(names)))

    def tree_hash(self, relative: str) -> str | None:
        """Return the Git tree hash of a directory, or ``None`` when it is absent.

        Working-tree reads have no stable hash; they return ``"worktree"`` so that
        their signatures never match a committed result.
        """
        if self.is_worktree:
            return WORKTREE if (self.repository / relative).exists() else None
        result = subprocess.run(
            ("git", "-C", str(self.repository), "rev-parse", f"{self.commit}:{relative}"),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def snapshot(self) -> Path:
        """Return a directory holding the study files that execution reads.

        A commit is extracted once with ``git archive`` below the study work root,
        so later commits cannot change a running or finished work item.
        """
        if self.is_worktree:
            return self.repository
        destination = study_work_root(self.study) / "code" / self.commit
        if (destination / ".nro-complete").is_file():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
            archive = Path(staging) / "code.tar"
            archive.write_bytes(
                _git(self.repository, "archive", "--format=tar", self.commit, text=False)
            )
            extracted = Path(staging) / "code"
            extracted.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter="data")
            (extracted / ".nro-complete").write_text(self.commit + "\n", encoding="utf-8")
            try:
                extracted.rename(destination)
            except OSError:
                if not (destination / ".nro-complete").is_file():
                    raise
        return destination

    def last_commit(self, relative: str) -> str | None:
        """Return the short hash and date of the last commit touching a path."""
        if self.is_worktree:
            return None
        value = str(
            _git(self.repository, "log", "-1", "--format=%h %cs", self.commit, "--", relative)
        ).strip()
        return value or None


@dataclass(frozen=True)
class StudyConfig:
    """Validated ``study.yml``."""

    name: str
    projects: tuple[str, ...]
    api: int
    container: str | None


def parse_study_config(text: str | None, *, study: str) -> StudyConfig:
    """Validate the text of ``study.yml``."""
    if text is None:
        raise StudyError(f"Study {study} has no study.yml")
    value = yaml.safe_load(text)
    if not isinstance(value, dict) or set(value) - {"name", "projects", "api", "container"}:
        raise StudyError("study.yml accepts name, projects, api, and container")
    if value.get("name") != study:
        raise StudyError(f"study.yml name {value.get('name')!r} does not match directory {study!r}")
    projects = value.get("projects") or []
    if not isinstance(projects, list) or not all(
        isinstance(item, str) and item for item in projects
    ):
        raise StudyError("study.yml projects must be a list of project names")
    if len(set(projects)) != len(projects):
        raise StudyError("study.yml lists a project more than once")
    api = value.get("api", API_VERSION)
    if api != API_VERSION:
        raise StudyError(
            f"study.yml requires nro.analysis API {api}; this nro provides {API_VERSION}"
        )
    container = value.get("container")
    if container is not None and (not isinstance(container, str) or not container):
        raise StudyError("study.yml container must be an image path")
    return StudyConfig(study, tuple(projects), int(api), container)


@dataclass(frozen=True)
class LoadedAnalysis:
    """A validated analysis read from one study source."""

    source: StudySource
    study: StudyConfig
    manifest: Manifest

    @property
    def name(self) -> str:
        """Return ``STUDY/ANALYSIS``."""
        return f"{self.study.name}/{self.manifest.analysis}"

    @property
    def container(self) -> str | None:
        """Return the manifest image, then the study image."""
        return self.manifest.container or self.study.container

    @property
    def results_root(self) -> Path:
        """Return the analysis results directory."""
        return study_results_root(self.study.name, self.manifest.analysis)


def load_analysis(source: StudySource, analysis: str) -> LoadedAnalysis:
    """Read and validate one analysis, including its entry function, without running it."""
    study = parse_study_config(source.read_text("study.yml"), study=source.study)
    text = source.read_text(f"analyses/{analysis}/analysis.yml")
    if text is None:
        where = "the working tree" if source.is_worktree else f"commit {source.commit[:12]}"
        raise StudyError(f"Analysis {source.study}/{analysis} has no analysis.yml in {where}")
    manifest = parse_manifest(text, analysis=analysis)
    code = source.read_text(f"analyses/{analysis}/analysis.py")
    if code is None:
        raise ManifestError(f"Analysis {source.study}/{analysis} has no analysis.py")
    check_entry(code, manifest.entry)
    return LoadedAnalysis(source, study, manifest)


def studies() -> tuple[str, ...]:
    """Return the names of study repositories under the study code root."""
    from nro.site.configuration import settings

    root = Path(settings()[0]["study_code"])
    if not root.is_dir():
        return ()
    return tuple(sorted(path.name for path in root.iterdir() if (path / ".git").exists()))
