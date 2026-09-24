# Orchestration design

## Scope and vocabulary

The normative vocabulary and its relationships are defined in
[Core concepts](concepts.md). The planner-facing record hierarchy and execution
lifecycle are defined in
[Work-item planning and execution](work-item-lifecycle.md).

The scientific surface includes `anat`, `func`, `clean`, `dynconn`,
`microparcellation`, `networks`, and `firstlevels`. Their planner-facing metadata is assembled
in one closed built-in catalog. Module-specific work-item construction lives in
each scientific package; the central planner owns cross-module traversal.

Reusable implementation primitives live in `nro/engine`, grouped by their
responsibility: BIDS and path handling, serialization, image IO, template and
FreeSurfer discovery, registration measurements, and runner-integrated image
operations. A scientific `module.py` composes these primitives into its own
workflow; it should contain only decisions and operations specific to that
module. Lab-wide planning, state, and workers remain in `nro/orchestration`.

## Configuration authority

`nro/configuration/store.py` resolves configurations and workflows from the
external [definitions store](definitions.md). Its `configs/` directory contains
one directory per configuration class. Workflows live in `workflows/`. Processing
requests accept identifiers; `nro paths` selects the store root.

A module configuration contains only local parameters. A workflow selects
configuration IDs and therefore supplies upstream lineage. The complete base
for each class lives in nro's packaged `main_<CLASS>.yml`; an optional external
`main` override and then a selected named YAML file are overlaid in that order.
No Python configuration-default map exists. The registry snapshots both the
workflow and fully resolved class values; module processes read the immutable
runtime values without a second merge or another configuration source.

Module lineages form a DAG independent of participant work items. Equivalent
lineages reuse a derivative directory. A changed definition creates a workflow
revision. Each module writes below
`derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`. The digest is derived from
the configuration ID and complete upstream lineage. It does not depend on
registration or discovery order. Separate `anat` and `func` lineages let
functional variants reuse identical anatomy without mixing their files.

## Work-item specifications and contracts

Work-item identity contains project, module, module-lineage ID,
participant, and applicable BIDS entities. Starting at `clean`, those entities
include one space and one smoothing value. Space and smoothing are request
entities, not configuration or directory identities. Revision fingerprints contain the
module, configuration fingerprint, entities, and the explicit contract
version. Source-code hashes are not identity or freshness inputs.

The implementation represents this identity with `WorkItemIdentity`. A
`WorkItemSpec` contains a `WorkItemIdentity`, `WorkItemContract`,
`ExecutionRecipe`, and `ResourceRequest`. The work-item contract records output
topology, dependency topology, entities, configuration, substantive direct
inputs, and explicit processing policy. Its normalized storage representation is
recorded in the registry's work-item and completion tables under the field name
`artifact_contract`.

The current command is part of the execution recipe and may be reformatted
without making a derivative stale. Replanning an unchanged contract updates
the recipe used by future attempts. An attempt that has already been claimed
keeps the immutable recipe captured when it started.

## Registries and scheduler

A shared deployment has one site-wide private control store. It contains one
scheduler registry and one scientific registry for each registered branch:

```text
CONTROL/
    shared/
        branches.json
        scheduler/registry.sqlite3
        cache/
        ingestion/
        promotions/
    branches/
        main/registry.sqlite3
        dev/registry.sqlite3
        <branch-id>/registry.sqlite3
```

Independent schema markers validate the private scheduler and scientific
registry layouts. They do not contribute to scientific freshness. Shared
maintenance migrates schemas at or after the supported baseline and reconstructs
older private state from durable public ownership records. It retains rollback state
until replacement registries pass integrity and identity checks. The
[migration contract](registry-migrations.md) derives each supported schema from an
immutable baseline and restricted, ordered changes.

Source project and participant directories are discovered independently of
derivative planning. Branch registries hold compiled scientific contracts,
workflow lineages, and observations. The scheduler registry alone holds demand,
attempts, workers, and submissions. Consequently, concurrency limits and
reusable worker capacity are enforced across simultaneous requests from
different projects and branches.

An automatically managed controller is the only long-lived client of the
shared scheduler SQLite database. Checkout processes may compile scientific
state in their branch-owned registries, which contain no attempts or worker-pool
state. On a Slurm site, work-producing commands submit the controller through
Slurm; local deployments run it directly. User commands and workers send requests
over TCP. Retryable requests and responses are recorded in SQLite before execution. A
replacement controller can replay an unresolved request.
Atomic launch election and fencing permit only the current controller to act.
The controller uses rollback journaling and short transactions; no transaction
spans scientific computation, Slurm waiting, network waiting, or bulk filesystem
work.

`Registry` is the sole connection and transaction facade. Focused internal
operations handle work-item registration, demand reconciliation, and status
projection using a connection supplied by that facade. They cannot open or commit
connections themselves, so decomposing registry logic does not create competing
lock or rollback authorities.

The controller publishes an atomic JSON read model after relevant changes.
Cached observation reads that snapshot without starting the controller or
opening SQLite. Bounded mutations use the live controller or a fenced local
one-shot coordinator. Only work supply and workers may submit a controller to
Slurm. Commands display a lightweight progress indicator while awaiting a
response. Controller queueing has no timeout; communication with a live
controller does. The controller exits after attempts, workers, submissions, and
pending requests remain idle for the configured grace period, which defaults to
12 hours. Installation and repair
use a deliberate shutdown barrier before entering their exceptional offline
maintenance phase.

### Private files outside SQLite

SQLite owns private scientific and scheduling facts. A small set of files remains
because those records cross a process, bootstrap, or filesystem-transaction boundary:

- the branch catalog and installed-release binding locate the databases before a
  controller can open them;
- immutable workflow, site, and source snapshots are execution inputs for pinned
  subprocesses;
- scheduler request rows let a replacement controller replay a request whose client or
  controller died before acknowledging it;
- the atomic status snapshot supports fast observation while no controller is live;
- logs and runner ledgers explain execution and support step-level resumption; and
- ingestion and promotion journals recover multi-file publication operations that a
  SQLite transaction cannot roll back.

These files are bootstrap records, immutable inputs, diagnostics, caches, or
write-ahead records for effects outside SQLite. They do not duplicate completed
work-item authority. In particular, nro has no private completion-manifest tree.

Registry state separates:

- artifact state: `missing`, `stale`, `corrupt`, or `fresh`;
- request demand: `active`, `cancelled`, or `satisfied`;
- attempt state: `queued`, `running`, `cancel_requested`, `cancelled`,
  `success`, or `error`;
- derived scheduling state: `dormant`, `waiting`, `ready`, `running`, or
  `blocked`.

Historical attempt failure does not override a later fresh artifact
assessment. Cancellation applies to demand and attempts, not permanently to a
work item.

## Freshness

The filesystem is authoritative for current artifacts; the database is
authoritative for orchestration history. A successful attempt remains
historically successful even if its output later becomes stale.

Every completed work item receives one normalized database record containing:

- exact resolved configuration and lineage;
- direct source inputs;
- required upstream generations;
- public output inventory and integrity records;
- command, implementation, attempt, and completion provenance.

The completion row and its input, output, and private-artifact rows commit in the
same transaction as the new generation. There is no private completion file to
race with the database or to outlive the registry identity it describes. Public
module manifests remain scientific derivative outputs: they index variable output
sets and provide portable metadata to downstream tools.

Requests likewise live only in the coordinator database. The scheduler does not
write a second per-request JSON archive. Files in private control storage are
limited to boundaries that SQLite cannot replace: service election, immutable execution
inputs, logs, filesystem-transaction journals, and
public ownership recovery.

Registration rejects two distinct work items that claim the same public output.
Fixed-output contracts may share a directory when their exact filenames are
disjoint. A prefixed contract owns that filename namespace, including files
listed by a variable-output completion index. Runner-graph freezing separately
rejects two internal steps that produce the same path.

Each module creates one `Runner`, which creates and owns that module's
`RunnerGraph`. Step factories remain outside `Runner`: they receive the inputs
needed for one operation and return an immutable `Step`. A larger named stage may
return an immutable `StagePlan`, which contains an ordered tuple of steps and typed
references to their downstream products. The module's `build_module()` function is
the sole construction site that passes these declarations to `Runner.add_step()` or
`Runner.add_steps()`. Factories and stage planners never mutate a runner as a side
effect.

`Runner.execute()` freezes its owned graph before performing any freshness
check. Artifact state can make a declared step run or skip, but can never add,
remove, or redirect a step. Each resumable step declares nonempty exact inputs
and outputs. Several child commands may live in one step only when they form a
single atomic artifact operation with one freshness and recovery boundary.
Variable directory producers use a shared atomic-directory-product lifecycle. A
module may clear a directory it owns exclusively; when several space/smoothing
work items share a subject directory, it clears and validates only files owned
by the target prefix before writing a target-specific breadcrumb.

The persisted runner contract separates declared topology from successful step
records. The runner saves topology before execution, then commits each step's
record after that step completes or passes its freshness checks. An interrupted
module therefore retains proof for its completed steps without treating failed
or unvisited steps as complete. Nodes and edges cannot vary with artifact
freshness. Each successful node records its explicit scientific parameters and
a normalized signature for an external command. When one node's scientific
declaration changes, the runner reruns that node and follows the graph to
invalidate its descendants. Independent nodes remain fresh. Presentation
changes, such as a new step label or equivalent long-option spelling, do not
invalidate outputs.

A changed source-BIDS state may establish a new topology contract; a code edit
alone cannot. Module completion is rejected if a planned node was not executed
or skipped, if a step did not receive an exact freshness decision, or if
declared outputs are absent. Modules must declare scientific parameters on the
steps that use them. The runner has no facility for silently adding one
configuration file as an input to every node.

Missing private `WORK` intermediates do not stale an intact artifact;
the module recreates them if it later needs to run. Changed existing private
intermediates, changed direct inputs, changed upstream generations, and missing
or changed products are freshness evidence.

## Multirun work items

`input_filter` belongs to the microparcellation configuration. The planner
discovers the raw BOLD universe, applies the filter, builds all required
run-level preprocessing and cleaning work items, and makes the participant-level
work item depend on the complete selected set. Adding or removing a matching run
changes that dependency set and requires replanning.

Space and smoothing are demand-driven. The planner does not enumerate every
possible pair. A request creates only its requested cross-product (default:
`fsnative` and `2mm`); a later request can add another pair without changing or
rerunning unrelated pairs. `func` remains shared because one run work item
publishes every supported space. Each `clean`, `dynconn`, `microparcellation`,
and `networks` work item represents exactly one pair.

## Workers and Slurm

Slurm jobs are reusable foreground workers, not one-job-per-module wrappers.
Workers request compatible ready work items through ordered scheduler events and
supervise one module subprocess at a time. A claim returns a typed
`ExecutionEnvelope`; an `ExecutionLauncher` owns subprocess creation and
supervision. Workers renew leases and check cancellation through direct scheduler
requests, then terminate the whole subprocess group when required. Their request
records remain available for replay until the scheduler commits them. Neither the
worker nor its scientific subprocess can open the scheduler database.

Workers drain before wall-time and request successor capacity from the
controller. The registry, not queued Slurm count, enforces the independent
general and GPU concurrency limits.
Confirmed OOMs increase the work-item memory tier geometrically up to the
request ceiling; higher-tier workers may accept lower-tier work items.

## Observation and mutation

`log` is observational. By default, `status` reads the last atomic scheduler
snapshot. `status --update` uses the live controller or a fenced one-shot
coordinator, performs authoritative assessment, updates the registry, and
publishes a new snapshot;
it can cancel an attempt whose registered contract has become obsolete. `run`
and worker events also assess artifacts and grow the worker pool. `set`, `stop`,
`purge`, and `publish` are explicit mutation commands with distinct
responsibilities. `scene` discovers files directly and does not require a live
controller. `render` uses that same discovery path before running Workbench's
headless image renderer.

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
- step factories return one `Step`, and named stages return immutable `StagePlan`
  values, without mutating a runner;
- focused registry operations use caller-owned connections and cannot connect,
  commit, or roll back;
- no subprocess execution bypassing the runner in scientific modules;
- no outputless resumable steps;
- no derivative-member discovery by globs in downstream scientific modules;
- no public configuration paths or alternate stores;
- a closed explicit catalog with module-local work-item planning;
- typed work-item specifications and worker execution envelopes;
- immutable execution recipes for claimed attempts, with current recipes used
  by later attempts;
- one generated schema per registry family, with explicit ordered migrations;
- request sharing, cancellation, lease recovery, OOM escalation, freshness,
  multirun expansion, and atomic publication behavior.
