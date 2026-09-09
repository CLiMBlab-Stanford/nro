"""Serialize compiled scientific requests without importing a module catalog."""

from pathlib import Path

from nro.orchestration.contracts import InstanceSpec


def encode_spec(spec: InstanceSpec) -> dict:
    """Retain a complete recipe so unavailable inherited work can be computed locally."""
    return dict(
        key=spec.key,
        module=spec.module,
        project=spec.project,
        participant=spec.participant,
        entities=dict(spec.entities),
        scope=spec.scope,
        configuration_lineage_id=spec.configuration_lineage_id,
        directory_label=spec.directory_label,
        config_fingerprint=spec.config_fingerprint,
        runtime_config=str(spec.runtime_config),
        command=list(spec.command),
        dependencies=list(spec.dependencies),
        input_paths=[str(path) for path in spec.input_paths],
        expected_outputs=[str(path) for path in spec.expected_outputs],
        output_root=str(spec.output_root),
        output_prefix=spec.output_prefix,
        resource_class=spec.resource_class,
        memory_gb=spec.resources.memory_gb,
        max_memory_gb=spec.resources.max_memory_gb,
        output_format=spec.contract.output.format,
        processing=dict(spec.contract.processing),
    )


def decode_spec(value: dict) -> InstanceSpec:
    """Decode explicit output and processing contracts without scientific defaults."""
    if not isinstance(value, dict) or not value.get("output_format") or "processing" not in value:
        raise ValueError(
            "A compiled request must supply its complete output and processing contract"
        )
    fields = dict(value)
    for key in ("runtime_config", "output_root"):
        fields[key] = Path(fields[key])
    for key in ("input_paths", "expected_outputs"):
        fields[key] = tuple(Path(path) for path in fields[key])
    return InstanceSpec.create(**fields)


def export_workflow(scientific, registered) -> dict:
    """Detach the workflow and configuration lineage data needed for admission."""
    with scientific.connection() as db:
        return dict(
            revision=dict(
                db.execute(
                    "SELECT * FROM workflow_revisions WHERE id=?", (registered.revision_id,)
                ).fetchone()
            ),
            lineages=[dict(row) for row in db.execute("SELECT * FROM configuration_lineages")],
            bindings=[
                dict(row)
                for row in db.execute(
                    "SELECT * FROM workflow_bindings WHERE workflow_revision_id=?",
                    (registered.revision_id,),
                )
            ],
            dependencies=[
                dict(row) for row in db.execute("SELECT * FROM configuration_lineage_dependencies")
            ],
        )
