"""Immutable shared SQL captured by the first supported registry baselines."""

WORKFLOW_SQL = """
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

CREATE TABLE module_lineages (
    id INTEGER PRIMARY KEY,
    configuration_class TEXT NOT NULL,
    config_id TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    lineage_fingerprint TEXT NOT NULL,
    resolved_yaml TEXT NOT NULL,
    directory_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(configuration_class, lineage_fingerprint)
);

CREATE TABLE module_lineage_dependencies (
    module_lineage_id INTEGER NOT NULL REFERENCES module_lineages(id),
    upstream_module_lineage_id INTEGER NOT NULL REFERENCES module_lineages(id),
    role TEXT NOT NULL,
    PRIMARY KEY(module_lineage_id, upstream_module_lineage_id, role)
);

CREATE TABLE workflow_bindings (
    workflow_revision_id INTEGER NOT NULL REFERENCES workflow_revisions(id),
    configuration_class TEXT NOT NULL,
    module_lineage_id INTEGER NOT NULL REFERENCES module_lineages(id),
    PRIMARY KEY(workflow_revision_id, configuration_class)
);
"""
