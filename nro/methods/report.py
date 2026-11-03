"""Assemble a methods report for the module lineages selected by one workflow."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from nro.engine.artifact_metadata import validate_metadata_fields
from nro.engine.bids import matches_selectors
from nro.methods.bibliography import (
    citation_labels,
    load_bibliography,
    reference_order,
    reference_runs,
    subset_bibtex,
)
from nro.methods.context import RUN_ENTITY_COLUMNS, ManifestRecord, flatten_keys
from nro.methods.document import MethodsDocument, Section, Table
from nro.methods.software import (
    SOFTWARE_COLUMNS,
    ImageRecord,
    observed_software,
    software_rows,
)
from nro.methods.templates import MethodsTemplate, module_templates, render_template
from nro.methods.writers import EXTENSIONS, WRITERS, tsv

SUPPORTED_MODULES = ("anat", "func", "clean")
MANIFEST_SUFFIXES = {
    "anat": "_desc-preprocessAnat_manifest.json",
    "func": "_desc-preprocessFunc_manifest.json",
    "clean": "_desc-clean_manifest.json",
}
RUN_COLUMNS = (*RUN_ENTITY_COLUMNS, "module", "field", "value")
_ENTITY_NAMES = {
    "ses": "session",
    "task": "task",
    "acq": "acquisition",
    "dir": "direction",
    "echo": "echo",
    "run": "run",
    "space": "space",
    "smoothing": "smoothing",
}


class MethodsReportError(ValueError):
    """Raised when a methods report cannot be produced for the requested lineages."""


@dataclass(frozen=True)
class Lineage:
    """One module lineage selected by the workflow."""

    module: str
    config_id: str
    directory: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class LineageManifests:
    """Fresh manifests for one lineage and the counts of excluded work items."""

    records: tuple[ManifestRecord, ...]
    excluded: Mapping[str, int] = field(default_factory=dict)


# Called with one lineage and every lineage in the report.
ManifestSource = Callable[[Lineage, Sequence[Lineage]], LineageManifests]


def provider(module: str) -> ModuleType:
    """Import a module's methods-context provider."""
    if module not in SUPPORTED_MODULES:
        raise MethodsReportError(f"Methods templates are not yet available for module {module!r}")
    return importlib.import_module(f"nro.modules.{module}.methods")


def workflow_lineages(workflow: Any, modules: Sequence[str] | None = None) -> tuple[Lineage, ...]:
    """Return the requested modules and their upstream closure in dependency order.

    ``workflow`` is a ``ResolvedWorkflow``. Directory labels are computed the same
    way the registry assigns them, so a report can be previewed before any work
    item exists. Without ``modules``, every supported module is reported.
    """
    from nro.definitions.store import CONFIGURATION_CLASSES, fingerprint
    from nro.orchestration.dependencies import direct_dependencies, primary_dependency
    from nro.orchestration.workflow_registry import lineage_directory_label

    lineage_fingerprints: dict[str, str] = {}
    directories: dict[str, str] = {}
    for name in CONFIGURATION_CLASSES:
        resolved = workflow.configurations[name]
        upstream = primary_dependency(name, resolved.values)
        lineage_fingerprints[name] = fingerprint(
            {
                "module": name,
                "config_id": resolved.config_id,
                "upstream": lineage_fingerprints[upstream] if upstream else None,
            }
        )
        directories[name] = lineage_directory_label(resolved.config_id, lineage_fingerprints[name])
    targets = list(modules or SUPPORTED_MODULES)
    unknown = sorted(set(targets) - set(CONFIGURATION_CLASSES))
    if unknown:
        raise MethodsReportError("Unknown module(s): " + ", ".join(unknown))
    closure: set[str] = set()
    pending = list(targets)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(direct_dependencies(name, workflow.configurations[name].values))
    unsupported = sorted(closure - set(SUPPORTED_MODULES))
    if unsupported:
        raise MethodsReportError(
            "Methods templates are not yet available for module(s): " + ", ".join(unsupported)
        )
    return tuple(
        Lineage(
            module=name,
            config_id=workflow.configurations[name].config_id,
            directory=directories[name],
            values=workflow.configurations[name].values,
        )
        for name in CONFIGURATION_CLASSES
        if name in closure
    )


def _entities(project: str, row: Mapping[str, Any]) -> dict[str, str]:
    entities = {"project": project, "participant": str(row["participant"])}
    for key, value in dict(row["entities"]).items():
        if key in _ENTITY_NAMES and value not in (None, ""):
            entities[_ENTITY_NAMES[key]] = str(value)
    return entities


@dataclass(frozen=True)
class ManifestSelection:
    """Participant and BIDS-entity selectors applied to registry work items.

    ``runs`` maps registry entity names, such as ``task`` or ``acq``, to the
    accepted values, or to None to require that the entity is absent. A
    selector applies to a lineage only when at least one of its work items
    records that entity, so a task selector narrows functional runs without
    removing session-level anatomical work items.
    """

    participants: tuple[str, ...] = ()
    runs: Mapping[str, tuple[str, ...] | None] = field(default_factory=dict)

    def applicable(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return the run selectors that apply to these work items."""
        recorded = {key for row in rows for key in dict(row["entities"])}
        return {key: value for key, value in self.runs.items() if key in recorded}

    def matches(self, row: Mapping[str, Any], selectors: Mapping[str, Any]) -> bool:
        """Return whether one work item satisfies the participant and run selectors."""
        if self.participants and str(row["participant"]) not in self.participants:
            return False
        return matches_selectors(
            {key: str(value) for key, value in dict(row["entities"]).items()}, selectors
        )


@dataclass(frozen=True)
class _Sample:
    """Participants and sessions that contribute selected downstream work items."""

    participants: frozenset[tuple[str, str]]
    sessions: frozenset[tuple[str, str, str]]
    without_session: frozenset[tuple[str, str]]

    def admits(self, project: str, row: Mapping[str, Any]) -> bool:
        participant = (project, str(row["participant"]))
        session = dict(row["entities"]).get("ses")
        if session in (None, ""):
            return participant in self.participants
        return (*participant, str(session)) in self.sessions or participant in self.without_session


def registry_manifests(
    registries: Mapping[str, Any], selection: ManifestSelection = ManifestSelection()
) -> ManifestSource:
    """Return a manifest source that pools fresh artifacts from project registries.

    ``registries`` maps each project name to its registry. Work items that the
    selection excludes are left out silently. Selected work items that are not
    fresh or lack a readable manifest are counted as excluded.

    When a run selector does not apply to a lineage, such as ``--task`` for
    session-level anatomy, that lineage keeps only the participants, and the
    sessions when both lineages record them, that have a fresh selected work
    item in another reported lineage to which every run selector applies.
    """
    cache: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = {}

    def rows_for(lineage: Lineage) -> list[tuple[str, Mapping[str, Any]]]:
        key = (lineage.module, lineage.directory)
        if key not in cache:
            cache[key] = [
                (project, row)
                for project, registry in registries.items()
                for row in registry.lineage_artifacts(project, lineage.module, lineage.directory)
            ]
        return cache[key]

    def sample(lineage: Lineage, lineages: Sequence[Lineage]) -> _Sample:
        anchors = []
        for other in lineages:
            if other == lineage:
                continue
            rows = rows_for(other)
            selectors = selection.applicable([row for _, row in rows])
            if set(selectors) == set(selection.runs):
                anchors.append((rows, selectors))
        if not anchors:
            raise MethodsReportError(
                f"Run selectors {', '.join(sorted(selection.runs))} do not apply to "
                f"{lineage.module}, and no reported module they apply to defines its sample"
            )
        selected = [
            (project, row)
            for rows, selectors in anchors
            for project, row in rows
            if row["artifact_state"] == "fresh" and selection.matches(row, selectors)
        ]
        sessions = {
            (project, str(row["participant"]), str(dict(row["entities"]).get("ses") or ""))
            for project, row in selected
        }
        return _Sample(
            participants=frozenset(key[:2] for key in sessions),
            sessions=frozenset(key for key in sessions if key[2]),
            without_session=frozenset(key[:2] for key in sessions if not key[2]),
        )

    def source(lineage: Lineage, lineages: Sequence[Lineage]) -> LineageManifests:
        rows = rows_for(lineage)
        selectors = selection.applicable([row for _, row in rows])
        allowed = None
        if set(selectors) != set(selection.runs):
            allowed = sample(lineage, lineages)
        suffix = MANIFEST_SUFFIXES[lineage.module]
        records: list[ManifestRecord] = []
        excluded: Counter[str] = Counter()
        for project, row in rows:
            if not selection.matches(row, selectors):
                continue
            if allowed is not None and not allowed.admits(project, row):
                continue
            if row["artifact_state"] != "fresh":
                excluded[row["artifact_state"]] += 1
                continue
            paths = [Path(path) for path in row["expected_outputs"] if str(path).endswith(suffix)]
            if len(paths) != 1:
                excluded["no_manifest"] += 1
                continue
            try:
                data = json.loads(paths[0].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                excluded["unreadable_manifest"] += 1
                continue
            records.append(ManifestRecord(entities=_entities(project, row), data=data))
        return LineageManifests(tuple(records), dict(sorted(excluded.items())))

    return source


def no_manifests(lineage: Lineage, lineages: Sequence[Lineage]) -> LineageManifests:
    """Return no manifests, for a configuration-only preview."""
    del lineage, lineages
    return LineageManifests(())


def validate_context(module: ModuleType, context: Mapping[str, Any]) -> None:
    """Check a context against its declared fields and reject undeclared fields."""
    validate_metadata_fields(
        context, module.CONTEXT_FIELDS, label=f"{module.TITLE} methods context"
    )
    declared = set(module.CONTEXT_FIELDS)
    undeclared = [
        name
        for name in flatten_keys(context)
        if name not in declared and not any(name.startswith(field + ".") for field in declared)
    ]
    if undeclared:
        raise MethodsReportError(
            f"{module.TITLE} methods context has undeclared field(s): " + ", ".join(undeclared)
        )


def configuration_coverage(module: str, values: Mapping[str, Any]) -> list[str]:
    """Return substantive configuration parameters that methods neither report nor excuse."""
    from nro.definitions.schema import scientific_values

    methods = provider(module)
    reported = set(methods.CONFIGURATION_FIELDS) | set(methods.METHODS_SILENT)
    return sorted(set(flatten_keys(scientific_values(module, dict(values)))) - reported)


def template_set_digest(templates: Sequence[MethodsTemplate]) -> str:
    """Hash template paths and contents so the footer identifies the exact text source."""
    digest = hashlib.sha256()
    for template in templates:
        digest.update(template.relative_path.encode())
        digest.update(b"\0")
        digest.update(template.path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class MethodsReport:
    """Rendered output files and the trace that produced them."""

    files: Mapping[str, str]
    trace: Mapping[str, Any]


def build_report(
    workflow: Any,
    *,
    output_format: str,
    modules: Sequence[str] | None = None,
    manifests: ManifestSource = no_manifests,
    overrides: Sequence[Path] = (),
    nro_version: str,
    images: Callable[[Sequence[str]], Mapping[str, ImageRecord]],
    pins: Mapping[str, str],
    package_versions: Mapping[str, str],
    bibliography: Path | None = None,
    scope: Mapping[str, Any] | None = None,
) -> MethodsReport:
    """Render the methods document, bibliography, tables, and trace for a workflow."""
    if output_format not in WRITERS:
        raise MethodsReportError(f"Unknown methods format {output_format!r}")
    entries = load_bibliography() if bibliography is None else load_bibliography(bibliography)
    lineages = workflow_lineages(workflow, modules)
    workflow_values = {name: config.values for name, config in workflow.configurations.items()}
    sections: list[Section] = []
    used_templates: list[MethodsTemplate] = []
    run_rows: list[tuple[str, ...]] = []
    image_modules: dict[str, set[str]] = {}
    observed: list[tuple[str, Any]] = []
    trace_lineages: list[dict[str, Any]] = []
    trace_sections: list[dict[str, Any]] = []
    titles: dict[str, str] = {}
    for lineage in lineages:
        methods = provider(lineage.module)
        titles[lineage.module] = methods.TITLE
        found = manifests(lineage, lineages)
        result = methods.methods(lineage.values, found.records, workflow=workflow_values)
        validate_context(methods, result.context)
        templates = module_templates(
            lineage.module, context_version=methods.CONTEXT_VERSION, overrides=overrides
        )
        used_templates.extend(templates)
        for template in templates:
            section = render_template(template, result.context)
            if section is None:
                continue
            sections.append(section)
            trace_sections.append(
                {
                    "template": template.relative_path,
                    "source": str(template.path),
                    "section": section.title,
                    "context": {
                        name: _trace_value(result.context, name) for name in template.requires
                    },
                }
            )
        run_rows.extend(fact.row(lineage.module) for fact in result.run_facts)
        for image in result.images:
            image_modules.setdefault(image, set()).add(lineage.module)
        observed.extend((lineage.module, use) for use in result.software)
        trace_lineages.append(
            {
                "module": lineage.module,
                "config_id": lineage.config_id,
                "directory": lineage.directory,
                "fresh_manifests": len(found.records),
                "excluded_work_items": dict(found.excluded),
            }
        )

    software = software_rows(
        image_modules=image_modules,
        observed=observed_software(observed),
        images=images(sorted(image_modules)),
        pins=pins,
        nro_version=nro_version,
        package_versions=package_versions,
    )
    cited = sorted(
        {
            key
            for section in sections
            for paragraph in section.paragraphs
            for run in paragraph
            if run.kind == "citation"
            for key in run.keys
        }
    )
    labels = citation_labels(entries, cited)
    references = tuple(
        reference_runs(entries[key], labels.years[key]) for key in reference_order(entries, cited)
    )
    run_table = Table("Run-level processing", RUN_COLUMNS, tuple(sorted(run_rows)))
    software_table = Table("Software", SOFTWARE_COLUMNS, software)
    tables = [
        Table(f"Supplementary Table S{number}. {table.title}", table.columns, table.rows)
        for number, table in enumerate(
            [table for table in (run_table, software_table) if table.rows], start=1
        )
    ]
    digest = template_set_digest(used_templates)
    lineage_text = ", ".join(f"{item.module}/{item.directory}" for item in lineages)
    footer = (
        f"Generated by nro {nro_version} from workflow {workflow.workflow_id} "
        f"({lineage_text}); template set {digest[:12]}."
    )
    document = MethodsDocument(
        sections=tuple(sections),
        module_titles=titles,
        references=references,
        tables=tuple(tables),
        footer=footer,
        cited_keys=tuple(cited),
    )
    trace = {
        "nro_version": nro_version,
        "workflow": workflow.workflow_id,
        "format": output_format,
        "scope": dict(scope or {"preview": True}),
        "template_set_sha256": digest,
        "lineages": trace_lineages,
        "sections": trace_sections,
        "citations": cited,
    }
    files = {
        f"methods.{EXTENSIONS[output_format]}": WRITERS[output_format](document, labels),
        "references.bib": subset_bibtex(entries, cited) if cited else "",
        "supplementary_runs.tsv": tsv(run_table),
        "supplementary_software.tsv": tsv(software_table),
        "methods.json": json.dumps(trace, indent=2, sort_keys=True) + "\n",
    }
    return MethodsReport(files=files, trace=trace)


def _trace_value(context: Mapping[str, Any], dotted: str) -> Any:
    value: Any = context
    for part in dotted.split("."):
        value = value[part]
    return value


def write_report(report: MethodsReport, destination: Path) -> list[Path]:
    """Write report files into ``destination``, replacing files with the same names."""
    from nro.engine.io import atomic_write_text

    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(report.files):
        path = destination / name
        atomic_write_text(path, report.files[name])
        written.append(path)
    return written
