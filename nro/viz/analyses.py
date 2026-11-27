"""Find study analyses that ship a browser viewer and serve their files.

An analysis declares a viewer with ``viz: DIR/PAGE.html`` in its manifest.
``nro viz`` lists every such analysis on its study's ``main`` branch that has
published outputs. The viewer's directory is served from the extracted commit,
and the analysis's published output slots are served by name. Clients never
supply filesystem paths, so nothing outside those two places can be read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nro.analysis.layout import LoadedAnalysis, StudySource, load_analysis, studies
from nro.analysis.planning import PROVENANCE_SUFFIX

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


@dataclass(frozen=True)
class AnalysisViewer:
    """One analysis with a viewer page, read from one study commit."""

    analysis: LoadedAnalysis

    @property
    def name(self) -> str:
        """Return ``STUDY/ANALYSIS``."""
        return self.analysis.name

    @property
    def page(self) -> PurePosixPath:
        """Return the viewer page path relative to the viewer directory."""
        assert self.analysis.manifest.viz is not None
        return PurePosixPath(self.analysis.manifest.viz).relative_to(self.directory)

    @property
    def directory(self) -> PurePosixPath:
        """Return the viewer directory relative to the analysis directory."""
        assert self.analysis.manifest.viz is not None
        return PurePosixPath(self.analysis.manifest.viz).parent

    def public(self) -> dict[str, object]:
        """Describe the viewer for the browser."""
        return {
            "name": self.name,
            "study": self.analysis.study.name,
            "analysis": self.analysis.manifest.analysis,
            "scope": self.analysis.manifest.scope,
            "commit": self.analysis.source.commit,
            "url": f"/analyses/{self.name}/{self.page}",
        }

    def viewer_file(self, relative: str) -> Path:
        """Return a file inside the viewer directory of the extracted commit.

        Raises PermissionError for a path that leaves the directory and
        FileNotFoundError for a missing file.
        """
        base = (
            self.analysis.source.snapshot()
            / "analyses"
            / self.analysis.manifest.analysis
            / self.directory
        ).resolve()
        candidate = (base / relative).resolve()
        if not candidate.is_relative_to(base):
            raise PermissionError(f"{relative} is outside the viewer of {self.name}")
        if not candidate.is_file():
            raise FileNotFoundError(f"{self.name} has no viewer file {relative}")
        return candidate

    def outputs(self) -> list[dict[str, object]]:
        """List the published outputs that carry a provenance sidecar.

        Each entry names the project, the participant (``None`` for project
        scope), the slot, the file name, its size, and the provenance record.
        """
        root = self.analysis.results_root
        found = []
        for sidecar in sorted(root.glob("project-*/**/*" + PROVENANCE_SUFFIX)):
            path = sidecar.with_name(sidecar.name.removesuffix(PROVENANCE_SUFFIX))
            if not path.is_file():
                continue
            relative = path.relative_to(root).parts
            participant = relative[1].removeprefix("sub-") if len(relative) == 3 else None
            try:
                provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found.append(
                {
                    "project": relative[0].removeprefix("project-"),
                    "participant": participant,
                    "slot": provenance.get("Slot"),
                    "file": path.name,
                    "size": path.stat().st_size,
                    "commit": provenance.get("Commit"),
                    "published": provenance.get("Published"),
                    "parameters": provenance.get("Parameters"),
                }
            )
        return found

    def output_file(self, project: str, file: str, participant: str | None = None) -> Path:
        """Return one published output named by project, participant, and file name.

        Raises FileNotFoundError when no published output has that name.
        """
        for entry in self.outputs():
            if (
                entry["project"] == project
                and entry["participant"] == (participant or None)
                and entry["file"] == file
            ):
                directory = self.analysis.results_root / f"project-{project}"
                if participant:
                    directory = directory / f"sub-{participant}"
                return directory / file
        raise FileNotFoundError(f"{self.name} has no published output {file} for {project}")


def viewers() -> list[AnalysisViewer]:
    """Return every analysis on a study's ``main`` that has a viewer and outputs.

    Studies or analyses that cannot be read are skipped.
    """
    found = []
    for study in studies():
        try:
            source = StudySource.at(study)
            names = source.analyses()
        except ValueError:
            continue
        for name in names:
            try:
                loaded = load_analysis(source, name)
            except ValueError:
                continue
            if loaded.manifest.viz is None:
                continue
            viewer = AnalysisViewer(loaded)
            if viewer.outputs():
                found.append(viewer)
    return found


def viewer(name: str) -> AnalysisViewer:
    """Return the viewer of ``STUDY/ANALYSIS``, or raise KeyError."""
    for candidate in viewers():
        if candidate.name == name:
            return candidate
    raise KeyError(f"No analysis viewer named {name}")


def content_type(path: Path) -> str:
    """Return the media type for a viewer or output file."""
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
