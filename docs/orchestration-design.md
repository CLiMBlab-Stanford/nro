# Orchestration design

Status: implemented and audited on 2026-09-03.

## Scope and vocabulary

The normative vocabulary and its relationships are defined in
[Core concepts](concepts.md). The planner-facing record hierarchy and execution
lifecycle are defined in
[Instance planning and execution](instance-lifecycle.md).

The scientific surface consists of five modules: `anat`, `func`, `clean`,
`microparcellation`, and `networks`. Their planner-facing metadata is assembled
in one closed built-in catalog. Module-specific instance construction lives in
each scientific package; the central planner owns cross-module traversal.

Reusable implementation primitives live in `nro/engine`, grouped by their
responsibility: BIDS and path handling, serialization, image IO, template and
FreeSurfer discovery, registration measurements, and runner-integrated image
operations. A scientific `module.py` composes these primitives into its own
workflow; it should contain only decisions and operations specific to that
module. Lab-wide planning, state, and workers remain in `nro/orchestration`.

## Configuration authority

`nro/configuration/store.py` is the sole resolver. Its fixed store is
`nro/configuration/files/`, containing one directory per derivative class and
one `workflows/` directory. Public interfaces accept identifiers only.

A derivative configuration contains only local parameters. A workflow selects
configuration IDs and therefore supplies upstream lineage. The complete base
for each class lives in its `main_<CLASS>.yml` file; resolution overlays a
selected named YAML file once. No Python configuration-default map exists. The
registry snapshots both the workflow and fully resolved class values; module
processes read the immutable runtime values without a second merge or another
configuration source.

Configuration lineages form a DAG independent of participant instances.
Equivalent lineages reuse a derivative directory. A changed definition creates
a workflow revision, while the all-`main` lineage reserves directory `main`.

## Instance specifications and contracts

Instance identity contains project, module, configuration-lineage ID,
participant, and applicable BIDS entities. Starting at `clean`, those entities
include one space and one smoothing value. Revision fingerprints contain the
module, configuration fingerprint, entities, and the explicit contract
version. Source-code hashes are not identity or freshness inputs.

An `InstanceSpec` contains an `InstanceIdentity`, `InstanceContract`,
`ExecutionRecipe`, and `ResourceRequest`. The instance contract records output
topology, dependency topology, entities, configuration, substantive direct
inputs, and explicit processing policy. Its normalized storage representation
is currently recorded in the registry and completion manifest under the field
name `artifact_contract`.

The current command is part of the execution recipe and may be reformatted
without making a derivative stale. Replanning an unchanged contract does not
replace a recipe attached to active demand.

## Registry

The lab has one private control store shared by every project:

```text
/juice6/u/nlp/climblab/.nro/
    registry.sqlite3
    registry.lock/
    manifests/
    requests/
    events/
    workers/
    snapshots/
    workflows/
```

Schema 14 is the only supported registry schema. There is no migration ladder:
an incompatible development registry must be reinitialized.

Source project and participant directories are discovered independently of
derivative planning. Project identity is also stored on requests and instances;
workers and scheduler submissions are lab-wide. Consequently, concurrency
limits and reusable worker capacity are enforced across simultaneous requests
from different projects.

All SQLite access is serialized by the same atomic directory lock. SQLite uses
rollback journaling rather than WAL for cross-node safety. Transactions are
short and never span scientific computation. Lock ownership records a unique
token, host, process, user, and Slurm identity; recovery requires a grace period
and positive evidence that the owner is terminal.

Registry state separates:

- artifact state: `missing`, `stale`, or `fresh`;
- request demand: `active`, `cancelled`, or `satisfied`;
- attempt state: `queued`, `running`, `cancel_requested`, `cancelled`,
  `success`, or `error`;
- derived scheduling state: `dormant`, `waiting`, `ready`, `running`, or
  `blocked`.

Historical attempt failure does not override a later fresh artifact
assessment. Cancellation applies to demand and attempts, not permanently to an
instance.

## Freshness

The filesystem is authoritative for current artifacts; the database is
authoritative for orchestration history. A successful attempt remains
historically successful even if its output later becomes stale.

Every completed instance receives a private manifest containing:

- exact resolved configuration and lineage;
- direct source inputs;
- required upstream generations and manifests;
- public output inventory and integrity records;
- runtime configuration, command, interpreter, attempt, and completion data.

Manifest version 3 has one public-output field, `public_outputs`, and no
source-implementation sentinel.

Each module creates one `Runner`, which creates and owns that module's
`RunnerGraph`. Step factories remain outside `Runner`: they receive the inputs
needed for one operation and return an immutable `Step`. The module's
`build_module()` function is the construction site that passes those steps to
`Runner.add_step()`, making the full sequence and its conditional branches
visible in one place. A factory never adds its own step, and a helper never
hides a sequence of steps behind a side effect. A factory may return named
references to its step's outputs when later steps need those paths, but it
still produces exactly one `Step`.

`Runner.execute()` freezes its owned graph before performing any freshness
check. Artifact state can make a declared step run or skip, but can never add,
remove, or redirect a step. Each resumable step declares nonempty exact inputs
and outputs. Several child commands may live in one step only when they form a
single atomic artifact operation with one freshness and recovery boundary.
Variable directory producers use a shared directory-artifact lifecycle. A
module may clear a directory it owns exclusively; when several space/smoothing
instances share a subject directory, it clears and validates only files owned
by the target prefix before writing a target-specific breadcrumb.

The persisted topology contract is loaded before execution. Nodes and edges
cannot vary with artifact freshness. A changed source-BIDS state may establish
a new contract; a code edit alone cannot. Module completion is rejected if a
planned node was not executed or skipped, if a step did not receive an exact
freshness decision, or if declared outputs are absent.

Missing private `WORK` intermediates do not stale an intact public boundary;
the module recreates them if it later needs to run. Changed existing private
artifacts, changed direct inputs, changed upstream generations, and missing or
changed public artifacts are freshness evidence.

## Multirun instances

`input_filter` belongs to the microparcellation configuration. The planner
discovers the raw BOLD universe, applies the filter, builds all required
run-level preprocessing and cleaning instances, and makes the participant-level
instance depend on the complete selected set. Adding or removing a matching run
changes that dependency set and requires replanning.

Space and smoothing are demand-driven. The planner does not enumerate every
possible pair. A request creates only its requested cross-product (default:
`fsnative` and `2mm`); a later request can add another pair without changing or
rerunning unrelated pairs. `func` remains shared because it publishes all
configured spaces, while each `clean`, `microparcellation`, and `networks`
instance represents exactly one pair.

## Workers and Slurm

Slurm jobs are reusable foreground workers, not one-job-per-module wrappers.
Workers claim compatible ready instances transactionally and supervise one
module subprocess at a time. A claim returns a typed `ExecutionEnvelope`; an
`ExecutionLauncher` owns subprocess creation and supervision. Workers renew
leases, poll durable cancellation, and terminate the whole subprocess group
when required.

Workers drain before wall-time and pre-submit `afterany` successors. The
registry, not queued Slurm count, enforces shared concurrency. Confirmed OOMs
increase the instance memory tier geometrically up to the request ceiling;
higher-tier workers may accept lower-tier instances.

## Observation and mutation

`status` and `log` are observational. They use read-only registry connections
and do not assess artifacts, reconcile requests, or cancel work. `run` and
workers own assessment and worker-pool growth. `set` changes live planner
settings without creating demand. `stop`, `purge`, and `publish` are explicit
mutation commands with distinct responsibilities.

## Publication

Publication freezes a satisfied request into a new standalone derivative
dataset. It stages copies or reflinks, embeds workflow-independent recursive
provenance, verifies copied content and live generations, optionally validates
BIDS structure, and atomically renames the staging directory. Existing
destinations are never overwritten.

## Architectural guardrails

Tests enforce these boundaries:

- exactly one `Runner` class and one `RunnerGraph` engine;
- graph creation occurs only inside `Runner`, and DAG mutation occurs only in
  each module's `build_module()` function;
- step factories return one `Step` without mutating a runner;
- no subprocess execution bypassing the runner in scientific modules;
- no outputless resumable steps;
- no derivative-member discovery by globs in downstream scientific modules;
- no public configuration paths or alternate stores;
- a closed explicit catalog with module-local instance planning;
- typed instance specifications and worker execution envelopes;
- no replacement of execution recipes attached to active demand;
- one schema with no implicit migration;
- request sharing, cancellation, lease recovery, OOM escalation, freshness,
  multirun expansion, and atomic publication behavior.
