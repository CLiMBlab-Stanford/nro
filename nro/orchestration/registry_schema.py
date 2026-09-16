"""Schema constants for the central scheduler registry."""

from nro.orchestration.workflow_registry import WORKFLOW_SCHEMA

APPLICATION_ID = 0x4E524F31  # ASCII "NRO1"
SCHEMA_VERSION = 20

SCHEMA_SQL = (
    """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


CREATE TABLE bids_projects (
    project TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    discovered_at TEXT NOT NULL
);

CREATE TABLE bids_participants (
    project TEXT NOT NULL REFERENCES bids_projects(project) ON DELETE CASCADE,
    participant TEXT NOT NULL,
    path TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY(project, participant)
);

"""
    + WORKFLOW_SCHEMA
    + """

CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    project TEXT NOT NULL,
    workflow_revision_id INTEGER NOT NULL REFERENCES workflow_revisions(id),
    target_module TEXT NOT NULL,
    selectors_json TEXT NOT NULL,
    concurrency INTEGER NOT NULL,
    partition_name TEXT,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE work_items (
    id INTEGER PRIMARY KEY,
    work_item_key TEXT NOT NULL UNIQUE,
    module TEXT NOT NULL,
    module_lineage_id INTEGER NOT NULL REFERENCES module_lineages(id),
    project TEXT NOT NULL,
    participant TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    artifact_state TEXT NOT NULL,
    artifact_reason TEXT,
    current_generation INTEGER NOT NULL DEFAULT 0,
    manifest_path TEXT NOT NULL,
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    max_memory_gb INTEGER NOT NULL DEFAULT 256,
    revision_fingerprint TEXT NOT NULL,
    artifact_contract_json TEXT NOT NULL,
    artifact_fingerprint TEXT NOT NULL,
    command_json TEXT NOT NULL,
    runtime_config_path TEXT NOT NULL,
    input_paths_json TEXT NOT NULL,
    output_root TEXT NOT NULL,
    output_prefix TEXT,
    expected_outputs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE work_item_dependencies (
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    upstream_work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    role TEXT NOT NULL,
    required_generation INTEGER,
    PRIMARY KEY(work_item_id, upstream_work_item_id, role)
);

CREATE TABLE request_work_items (
    request_id TEXT NOT NULL REFERENCES requests(id),
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    role TEXT NOT NULL,
    demand_state TEXT NOT NULL,
    PRIMARY KEY(request_id, work_item_id)
);

CREATE TABLE workers (
    id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    hostname TEXT NOT NULL,
    pid INTEGER NOT NULL,
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    slurm_job_id TEXT,
    state TEXT NOT NULL,
    lease_expires_at REAL,
    successor_submission_id INTEGER,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE attempts (
    id INTEGER PRIMARY KEY,
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    worker_id TEXT REFERENCES workers(id),
    state TEXT NOT NULL,
    revision_fingerprint TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    oom_detected INTEGER NOT NULL DEFAULT 0,
    process_group_id INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    error_type TEXT,
    error_message TEXT,
    log_path TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX one_active_attempt_per_work_item
ON attempts(work_item_id)
WHERE state IN ('queued', 'running', 'cancel_requested');

CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY,
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    attempt_id INTEGER REFERENCES attempts(id),
    direction TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER,
    mtime_ns INTEGER,
    digest_algorithm TEXT,
    digest TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE scheduler_submissions (
    id INTEGER PRIMARY KEY,
    intent_token TEXT NOT NULL UNIQUE,
    request_id TEXT REFERENCES requests(id),
    predecessor_worker_id TEXT REFERENCES workers(id),
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    state TEXT NOT NULL,
    slurm_job_id TEXT,
    submitted_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX work_item_module_participant ON work_items(module, participant);
CREATE INDEX attempt_state ON attempts(state);
CREATE INDEX request_state ON requests(state);

CREATE TABLE attempt_dependencies (
    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    upstream_work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    generation INTEGER NOT NULL,
    PRIMARY KEY(attempt_id, upstream_work_item_id)
);
CREATE INDEX dependency_readers ON attempt_dependencies(upstream_work_item_id);
CREATE TABLE artifact_mutations (
    work_item_id INTEGER PRIMARY KEY REFERENCES work_items(id),
    token TEXT NOT NULL
);

CREATE TABLE work_item_execution (
    work_item_id INTEGER PRIMARY KEY REFERENCES work_items(id),
    branch TEXT NOT NULL,
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    context_json TEXT NOT NULL,
    binding_sources_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    scientific_contract_json TEXT NOT NULL,
    UNIQUE(registry_id, logical_key)
);
CREATE TABLE request_owners (
    request_id TEXT PRIMARY KEY REFERENCES requests(id),
    branch TEXT NOT NULL,
    registry_id TEXT NOT NULL
);
CREATE TABLE request_plans (
    request_id TEXT PRIMARY KEY REFERENCES requests(id),
    payload_json TEXT NOT NULL
);
CREATE TABLE compiled_revisions (
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    PRIMARY KEY(registry_id,logical_key)
);
CREATE TABLE branch_work_items (
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    scientific_contract_json TEXT NOT NULL,
    PRIMARY KEY(registry_id, logical_key)
);
CREATE TABLE request_artifacts (
    request_id TEXT NOT NULL REFERENCES requests(id),
    work_item_id INTEGER NOT NULL REFERENCES work_items(id),
    PRIMARY KEY(request_id,work_item_id)
);
CREATE TABLE attempt_execution (
    attempt_id INTEGER PRIMARY KEY REFERENCES attempts(id),
    context_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    command_json TEXT NOT NULL
);
"""
)
