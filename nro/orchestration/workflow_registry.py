"""Shared scientific workflow registration, independent of worker-pool storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nro.configuration.store import (
    CONFIGURATION_CLASSES,
    DERIVATIVE_CLASSES,
    UPSTREAM_CLASS,
    ResolvedConfiguration,
    ResolvedWorkflow,
    configuration_fingerprint,
    fingerprint,
)

WORKFLOW_SCHEMA = """
CREATE TABLE workflow_revisions (
    id INTEGER PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    definition_fingerprint TEXT NOT NULL,
    source_path TEXT NOT NULL,
    resolved_yaml TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(workflow_id, revision),
    UNIQUE(workflow_id, definition_fingerprint)
);

CREATE TABLE configuration_lineages (
    id INTEGER PRIMARY KEY,
    derivative_class TEXT NOT NULL,
    config_id TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    lineage_fingerprint TEXT NOT NULL,
    resolved_yaml TEXT NOT NULL,
    directory_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(derivative_class, lineage_fingerprint)
);

CREATE TABLE configuration_lineage_dependencies (
    configuration_lineage_id INTEGER NOT NULL REFERENCES configuration_lineages(id),
    upstream_configuration_lineage_id INTEGER NOT NULL REFERENCES configuration_lineages(id),
    role TEXT NOT NULL,
    PRIMARY KEY(configuration_lineage_id, upstream_configuration_lineage_id, role)
);

CREATE TABLE workflow_bindings (
    workflow_revision_id INTEGER NOT NULL REFERENCES workflow_revisions(id),
    derivative_class TEXT NOT NULL,
    configuration_lineage_id INTEGER NOT NULL REFERENCES configuration_lineages(id),
    PRIMARY KEY(workflow_revision_id, derivative_class)
);

"""


@dataclass(frozen=True)
class RegisteredWorkflow:
    """Registered revision, configuration lineages, and assigned output directories."""

    workflow_id: str
    revision: int
    revision_id: int
    fingerprint: str
    lineages: dict[str, int]
    lineage_fingerprints: dict[str, str]
    directories: dict[str, str]
    anatomy_lineage: int
    anatomy_lineage_fingerprint: str
    anatomy_directory: str
    created: bool

    @property
    def selector(self) -> str:
        """Return the workflow ID qualified by its registered revision."""
        return f"{self.workflow_id}@{self.revision}"

    def directory_for(self, configuration_class: str) -> str:
        """Return the public directory selected for one configuration class."""
        if configuration_class == "anat":
            return self.anatomy_directory
        if configuration_class == "func":
            return self.directories["preprocessing"]
        return self.directories[configuration_class]


CONFIGURATION_FILE_SUFFIX = {
    "anat": "anat",
    "func": "func",
    "clean": "clean",
    "dynconn": "dynconn",
    "microparcellation": "microparcellation",
    "networks": "networks",
    "firstlevels": "firstlevels",
}


def _lineage_configuration(
    workflow: ResolvedWorkflow, derivative_class: str
) -> ResolvedConfiguration:
    """Return the configuration payload that identifies one artifact lineage."""
    if derivative_class != "preprocessing":
        return workflow.configuration(derivative_class)
    anat = workflow.configuration("anat")
    func = workflow.configuration("func")
    config_id = (
        "main"
        if anat.config_id == func.config_id == "main"
        else f"{anat.config_id}+{func.config_id}"
    )
    values = {"anat": anat.values, "func": func.values}
    return ResolvedConfiguration(
        configuration_class="preprocessing",
        config_id=config_id,
        path=func.path,
        values=values,
        fingerprint=configuration_fingerprint("preprocessing", config_id, values),
    )


class WorkflowRegistry:
    """Register scientific configuration lineages through a supplied database connection.

    Hosts supply connection(write=...), and paths.workflows for execution snapshots.
    This component creates no worker pool and does not submit demand.
    """

    def _snapshot_workflow(
        self,
        workflow: ResolvedWorkflow,
        revision: int,
        directories: dict[str, str],
    ) -> None:
        from nro.orchestration.registry import _atomic_text, ensure_shared_directory

        destination = self.paths.workflows / workflow.workflow_id / f"{revision}_workflow.yml"
        data = {
            "workflow_id": workflow.workflow_id,
            "revision": revision,
            "definition_fingerprint": workflow.fingerprint,
            "selections": workflow.selections,
            "configurations": {
                configuration_class: {
                    "config_id": workflow.configurations[configuration_class].config_id,
                    "config_fingerprint": workflow.configurations[configuration_class].fingerprint,
                    "source": str(workflow.configurations[configuration_class].path),
                    "resolved": workflow.configurations[configuration_class].values,
                    "directory": (
                        directories["anatomy"]
                        if configuration_class == "anat"
                        else directories["preprocessing"]
                        if configuration_class == "func"
                        else directories[configuration_class]
                    ),
                }
                for configuration_class in CONFIGURATION_CLASSES
            },
        }
        _atomic_text(destination, yaml.safe_dump(data, sort_keys=False))

        runtime_directory = destination.parent / f"{revision}_runtime"
        ensure_shared_directory(runtime_directory)
        for configuration_class in CONFIGURATION_CLASSES:
            values = dict(workflow.configurations[configuration_class].values)
            if configuration_class == "func":
                values["anatomical_directory"] = directories["anatomy"]
            elif configuration_class in {"clean", "firstlevels"}:
                values["functional_directory"] = directories["preprocessing"]
                values["anatomical_directory"] = directories["anatomy"]
                if configuration_class == "firstlevels":
                    values["preprocessing_aroma"] = workflow.configuration("func").values[
                        "clean_ica_aroma"
                    ]
            elif configuration_class in {"dynconn", "microparcellation"}:
                values["anatomical_directory"] = directories["anatomy"]
                values["clean_directory"] = directories["clean"]
            elif configuration_class == "networks":
                values["anatomical_directory"] = directories["anatomy"]
                values["microparcellation_directory"] = directories["microparcellation"]
            suffix = CONFIGURATION_FILE_SUFFIX[configuration_class]
            directory = (
                directories["anatomy"]
                if configuration_class == "anat"
                else directories["preprocessing"]
                if configuration_class == "func"
                else directories[configuration_class]
            )
            runtime_path = runtime_directory / f"{directory}_{suffix}.yml"
            _atomic_text(runtime_path, yaml.safe_dump(values, sort_keys=False))

    def runtime_config_path(self, registered: RegisteredWorkflow, derivative_class: str) -> Path:
        """Return the stored runtime configuration path for a registered lineage."""
        if derivative_class not in CONFIGURATION_CLASSES:
            raise ValueError(f"Unknown configuration class: {derivative_class}")
        suffix = CONFIGURATION_FILE_SUFFIX[derivative_class]
        directory = (
            registered.anatomy_directory
            if derivative_class == "anat"
            else registered.directories["preprocessing"]
            if derivative_class == "func"
            else registered.directories[derivative_class]
        )
        return (
            self.paths.workflows
            / registered.workflow_id
            / f"{registered.revision}_runtime"
            / f"{directory}_{suffix}.yml"
        )

    def register_workflow(self, workflow: ResolvedWorkflow) -> RegisteredWorkflow:
        """Register resolved workflow configurations and return their lineage assignments."""
        from nro.orchestration.registry import utcnow

        created = False
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT id, revision FROM workflow_revisions WHERE workflow_id=? AND definition_fingerprint=?",
                (workflow.workflow_id, workflow.fingerprint),
            ).fetchone()
            if row:
                revision_id = int(row["id"])
                revision = int(row["revision"])
            else:
                latest = db.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM workflow_revisions WHERE workflow_id=?",
                    (workflow.workflow_id,),
                ).fetchone()[0]
                revision = int(latest) + 1
                resolved_yaml = yaml.safe_dump(
                    {
                        "selections": workflow.selections,
                        "configurations": {
                            derivative_class: _lineage_configuration(
                                workflow, derivative_class
                            ).values
                            for derivative_class in DERIVATIVE_CLASSES
                        },
                    },
                    sort_keys=False,
                )
                cursor = db.execute(
                    """
                    INSERT INTO workflow_revisions(
                        workflow_id, revision, definition_fingerprint, source_path,
                        resolved_yaml, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workflow.workflow_id,
                        revision,
                        workflow.fingerprint,
                        str(workflow.path),
                        resolved_yaml,
                        utcnow(),
                    ),
                )
                revision_id = int(cursor.lastrowid)
                created = True

            existing_bindings = db.execute(
                """
                SELECT wb.derivative_class, wb.configuration_lineage_id,
                       ci.lineage_fingerprint, ci.directory_label
                FROM workflow_bindings wb
                JOIN configuration_lineages ci ON ci.id=wb.configuration_lineage_id
                WHERE wb.workflow_revision_id=?
                """,
                (revision_id,),
            ).fetchall()
            if existing_bindings:
                lineages = {
                    str(item["derivative_class"]): int(item["configuration_lineage_id"])
                    for item in existing_bindings
                }
                directories = {
                    str(item["derivative_class"]): str(item["directory_label"])
                    for item in existing_bindings
                }
                lineage_fingerprints = {
                    str(item["derivative_class"]): str(item["lineage_fingerprint"])
                    for item in existing_bindings
                }
                # A workflow revision can become current again after a config
                # file is changed back. Refresh the mutable configuration-lineage
                # slot
                # so filesystem freshness checks always use the config that
                # was actually resolved for this invocation.
                for derivative_class, lineage_id in lineages.items():
                    resolved = _lineage_configuration(workflow, derivative_class)
                    db.execute(
                        """UPDATE configuration_lineages
                           SET config_fingerprint=?, resolved_yaml=? WHERE id=?""",
                        (
                            resolved.fingerprint,
                            yaml.safe_dump(resolved.values, sort_keys=False),
                            lineage_id,
                        ),
                    )
            else:
                lineages: dict[str, int] = {}
                directories: dict[str, str] = {}
                lineage_fingerprints: dict[str, str] = {}
                all_default_so_far = True
                allocated_new_label: str | None = None

                def allocate_new_label() -> str:
                    """Allocate one stable label for this new workflow suffix."""
                    nonlocal allocated_new_label
                    if allocated_new_label is None:
                        base = workflow.workflow_id
                        number = revision if revision > 1 else None
                        while True:
                            candidate = base if number is None else f"{base}-{number}"
                            occupied = db.execute(
                                "SELECT 1 FROM configuration_lineages WHERE directory_label=? LIMIT 1",
                                (candidate,),
                            ).fetchone()
                            if not occupied:
                                allocated_new_label = candidate
                                break
                            number = 2 if number is None else number + 1
                    return allocated_new_label

                for derivative_class in DERIVATIVE_CLASSES:
                    resolved = _lineage_configuration(workflow, derivative_class)
                    upstream_class = UPSTREAM_CLASS[derivative_class]
                    upstream_id = lineages.get(upstream_class) if upstream_class else None
                    upstream_lineage = (
                        lineage_fingerprints[upstream_class] if upstream_class else None
                    )
                    all_default_so_far = all_default_so_far and resolved.config_id == "main"
                    # A module-level configuration ID is a stable, mutable
                    # namespace.  Its contents are deliberately excluded from
                    # this identity: changing ``main_clean.yml`` must rebuild
                    # clean outputs in place, rather than create a second
                    # namespace.  Workflow IDs supply the first directory
                    # label assigned to a newly selected configuration path.
                    lineage = fingerprint(
                        {
                            "derivative_class": derivative_class,
                            "config_id": resolved.config_id,
                            "upstream": upstream_lineage,
                        }
                    )
                    existing_lineage = db.execute(
                        "SELECT id, directory_label FROM configuration_lineages WHERE derivative_class=? AND lineage_fingerprint=?",
                        (derivative_class, lineage),
                    ).fetchone()
                    if existing_lineage:
                        lineage_id = int(existing_lineage["id"])
                        directory = str(existing_lineage["directory_label"])
                        db.execute(
                            """UPDATE configuration_lineages
                               SET config_fingerprint=?, resolved_yaml=? WHERE id=?""",
                            (
                                resolved.fingerprint,
                                yaml.safe_dump(resolved.values, sort_keys=False),
                                lineage_id,
                            ),
                        )
                    else:
                        if all_default_so_far:
                            directory = "main"
                        else:
                            directory = allocate_new_label()
                        collision = db.execute(
                            "SELECT lineage_fingerprint FROM configuration_lineages WHERE derivative_class=? AND directory_label=?",
                            (derivative_class, directory),
                        ).fetchone()
                        if collision:
                            if all_default_so_far:
                                # A changed repository default is a new path,
                                # just like a changed named workflow.  The old
                                # default derivatives remain addressable at
                                # ``main``; the new content receives a numeric
                                # revision label.
                                directory = allocate_new_label()
                            else:
                                raise RuntimeError(
                                    f"Derivative directory {derivative_class}/{directory} is already assigned "
                                    "to an incompatible configuration lineage"
                                )
                        cursor = db.execute(
                            """
                            INSERT INTO configuration_lineages(
                                derivative_class, config_id, config_fingerprint, lineage_fingerprint,
                                resolved_yaml, directory_label, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                derivative_class,
                                resolved.config_id,
                                resolved.fingerprint,
                                lineage,
                                yaml.safe_dump(resolved.values, sort_keys=False),
                                directory,
                                utcnow(),
                            ),
                        )
                        lineage_id = int(cursor.lastrowid)
                        if upstream_id is not None:
                            db.execute(
                                "INSERT INTO configuration_lineage_dependencies("
                                "configuration_lineage_id, upstream_configuration_lineage_id, role"
                                ") VALUES (?, ?, ?)",
                                (lineage_id, upstream_id, upstream_class),
                            )
                    lineages[derivative_class] = lineage_id
                    directories[derivative_class] = directory
                    lineage_fingerprints[derivative_class] = lineage
                    db.execute(
                        "INSERT INTO workflow_bindings(workflow_revision_id, derivative_class, configuration_lineage_id) VALUES (?, ?, ?)",
                        (revision_id, derivative_class, lineage_id),
                    )

            # ``anat`` and ``func`` share the preprocessing derivative class,
            # but func-only changes must not create duplicate anatomical work.
            # Reuse the oldest preprocessing lineage with the same
            # anatomy-specific scientific settings and place its anatomy in
            # that lineage's directory.
            anatomy_fingerprint = workflow.configuration("anat").scientific_fingerprint
            anatomy_lineage = None
            for candidate in db.execute(
                """SELECT id, directory_label, resolved_yaml
                   FROM configuration_lineages
                   WHERE derivative_class='preprocessing' ORDER BY id"""
            ):
                try:
                    candidate_values = yaml.safe_load(candidate["resolved_yaml"])
                    if "fsaverage_template" in candidate_values:
                        candidate_values = {
                            **candidate_values["anat"],
                            "fsaverage_template": candidate_values["fsaverage_template"],
                            "container": candidate_values["container"],
                        }
                    else:
                        candidate_values = candidate_values["anat"]
                    candidate_fingerprint = configuration_fingerprint(
                        "anat",
                        workflow.configuration("anat").config_id,
                        candidate_values,
                        scientific=True,
                    )
                except (TypeError, ValueError, KeyError, yaml.YAMLError):
                    continue
                if candidate_fingerprint == anatomy_fingerprint:
                    anatomy_lineage = candidate
                    break
            if anatomy_lineage is None:
                raise RuntimeError("Registered preprocessing lineage could not be recovered")
            anatomy_lineage_id = int(anatomy_lineage["id"])
            anatomy_directory = str(anatomy_lineage["directory_label"])

        self._snapshot_workflow(workflow, revision, {**directories, "anatomy": anatomy_directory})
        return RegisteredWorkflow(
            workflow_id=workflow.workflow_id,
            revision=revision,
            revision_id=revision_id,
            fingerprint=workflow.fingerprint,
            lineages=lineages,
            lineage_fingerprints=lineage_fingerprints,
            directories=directories,
            anatomy_lineage=anatomy_lineage_id,
            anatomy_lineage_fingerprint=anatomy_fingerprint,
            anatomy_directory=anatomy_directory,
            created=created,
        )
