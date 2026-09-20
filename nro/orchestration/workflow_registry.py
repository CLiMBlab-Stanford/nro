"""Shared scientific workflow registration, independent of worker-pool storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nro.configuration.store import (
    CONFIGURATION_CLASSES,
    ResolvedWorkflow,
    fingerprint,
)
from nro.orchestration.dependencies import primary_dependency


@dataclass(frozen=True)
class RegisteredWorkflow:
    """Registered revision, module lineages, and assigned output directories."""

    workflow_id: str
    revision: int
    revision_id: int
    fingerprint: str
    lineages: dict[str, int]
    lineage_fingerprints: dict[str, str]
    directories: dict[str, str]
    created: bool

    @property
    def selector(self) -> str:
        """Return the workflow ID qualified by its registered revision."""
        return f"{self.workflow_id}@{self.revision}"

    def directory_for(self, configuration_class: str) -> str:
        """Return the public directory selected for one module."""
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


def lineage_directory_label(config_id: str, lineage_fingerprint: str) -> str:
    """Return the deterministic public directory for one module lineage."""
    if not config_id or not lineage_fingerprint:
        raise ValueError("Lineage directory identity cannot be empty")
    return f"{config_id}-{lineage_fingerprint[:12]}"


class WorkflowRegistry:
    """Register scientific module lineages through a supplied database connection.

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
                    "directory": directories[configuration_class],
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
                values["anat_directory"] = directories["anat"]
                values["fsaverage_template"] = workflow.configuration("anat").values[
                    "fsaverage_template"
                ]
            elif configuration_class in {"clean", "firstlevels"}:
                values["func_directory"] = directories["func"]
                values["anat_directory"] = directories["anat"]
            elif configuration_class in {"dynconn", "microparcellation"}:
                values["anat_directory"] = directories["anat"]
                values["clean_directory"] = directories["clean"]
            elif configuration_class == "networks":
                values["anat_directory"] = directories["anat"]
                source = str(values["connectivity_source"])
                values["source_directory"] = directories[source]
            suffix = CONFIGURATION_FILE_SUFFIX[configuration_class]
            directory = directories[configuration_class]
            runtime_path = runtime_directory / f"{directory}_{suffix}.yml"
            _atomic_text(runtime_path, yaml.safe_dump(values, sort_keys=False))

    def runtime_config_path(self, registered: RegisteredWorkflow, configuration_class: str) -> Path:
        """Return the stored runtime configuration path for a registered lineage."""
        if configuration_class not in CONFIGURATION_CLASSES:
            raise ValueError(f"Unknown configuration class: {configuration_class}")
        suffix = CONFIGURATION_FILE_SUFFIX[configuration_class]
        directory = registered.directories[configuration_class]
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
                            configuration_class: workflow.configurations[configuration_class].values
                            for configuration_class in CONFIGURATION_CLASSES
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
                SELECT wb.configuration_class, wb.module_lineage_id,
                       ci.lineage_fingerprint, ci.directory_label
                FROM workflow_bindings wb
                JOIN module_lineages ci ON ci.id=wb.module_lineage_id
                WHERE wb.workflow_revision_id=?
                """,
                (revision_id,),
            ).fetchall()
            if existing_bindings:
                lineages = {
                    str(item["configuration_class"]): int(item["module_lineage_id"])
                    for item in existing_bindings
                }
                directories = {
                    str(item["configuration_class"]): str(item["directory_label"])
                    for item in existing_bindings
                }
                lineage_fingerprints = {
                    str(item["configuration_class"]): str(item["lineage_fingerprint"])
                    for item in existing_bindings
                }
                # A workflow revision can become current again after a config
                # file is changed back. Refresh the mutable configuration-lineage
                # slot
                # so filesystem freshness checks always use the config that
                # was actually resolved for this invocation.
                for configuration_class, lineage_id in lineages.items():
                    resolved = workflow.configurations[configuration_class]
                    db.execute(
                        """UPDATE module_lineages
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

                for configuration_class in CONFIGURATION_CLASSES:
                    resolved = workflow.configurations[configuration_class]
                    upstream_class = primary_dependency(configuration_class, resolved.values)
                    upstream_id = lineages.get(upstream_class) if upstream_class else None
                    upstream_lineage = (
                        lineage_fingerprints[upstream_class] if upstream_class else None
                    )
                    # A module-level configuration ID is a stable, mutable
                    # namespace.  Its contents are deliberately excluded from
                    # this identity: changing ``main_clean.yml`` must rebuild
                    # clean outputs in place, rather than create a second
                    # namespace.  An upstream lineage remains part of the
                    # identity because it can change this module's outputs.
                    lineage = fingerprint(
                        {
                            "module": configuration_class,
                            "config_id": resolved.config_id,
                            "upstream": upstream_lineage,
                        }
                    )
                    existing_lineage = db.execute(
                        "SELECT id, directory_label FROM module_lineages WHERE configuration_class=? AND lineage_fingerprint=?",
                        (configuration_class, lineage),
                    ).fetchone()
                    if existing_lineage:
                        lineage_id = int(existing_lineage["id"])
                        directory = str(existing_lineage["directory_label"])
                        db.execute(
                            """UPDATE module_lineages
                               SET config_fingerprint=?, resolved_yaml=? WHERE id=?""",
                            (
                                resolved.fingerprint,
                                yaml.safe_dump(resolved.values, sort_keys=False),
                                lineage_id,
                            ),
                        )
                    else:
                        directory = lineage_directory_label(resolved.config_id, lineage)
                        collision = db.execute(
                            """SELECT lineage_fingerprint FROM module_lineages
                               WHERE configuration_class=? AND directory_label=? LIMIT 1""",
                            (configuration_class, directory),
                        ).fetchone()
                        if collision is not None:
                            raise RuntimeError(
                                "Lineage directory digest collision for "
                                f"{configuration_class}/{directory}"
                            )
                        cursor = db.execute(
                            """
                            INSERT INTO module_lineages(
                                configuration_class, config_id, config_fingerprint, lineage_fingerprint,
                                resolved_yaml, directory_label, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                configuration_class,
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
                                "INSERT INTO module_lineage_dependencies("
                                "module_lineage_id, upstream_module_lineage_id, role"
                                ") VALUES (?, ?, ?)",
                                (lineage_id, upstream_id, upstream_class),
                            )
                    lineages[configuration_class] = lineage_id
                    directories[configuration_class] = directory
                    lineage_fingerprints[configuration_class] = lineage
                    db.execute(
                        "INSERT INTO workflow_bindings(workflow_revision_id, configuration_class, module_lineage_id) VALUES (?, ?, ?)",
                        (revision_id, configuration_class, lineage_id),
                    )

        self._snapshot_workflow(workflow, revision, directories)
        return RegisteredWorkflow(
            workflow_id=workflow.workflow_id,
            revision=revision,
            revision_id=revision_id,
            fingerprint=workflow.fingerprint,
            lineages=lineages,
            lineage_fingerprints=lineage_fingerprints,
            directories=directories,
            created=created,
        )
