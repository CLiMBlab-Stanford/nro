# Orchestration design

This page is for developers who change the planner, scheduler, registries,
workers, or runner. It describes how those parts divide responsibility and which
rules they must keep. [Running derivative workflows](orchestration.md) covers the
same system from a user's point of view, and
[Work-item planning and execution](work-item-lifecycle.md) defines the records
that pass between the parts. Shared vocabulary is defined in
[Core concepts](concepts.md).

## Code layout

The scientific modules are `anat`, `func`, `clean`, `dynconn`,
`microparcellation`, `networks`, and `contrasts`. One closed built-in catalog
collects the metadata the planner needs about each of them. Each scientific
package builds its own work items, and the central planner walks dependencies
across modules.

Reusable implementation primitives live in `nro/engine`. They are grouped by
responsibility: BIDS and path handling, serialization, image IO, template and
FreeSurfer discovery, registration measurements, and runner-integrated image
operations. A scientific `module.py` composes these primitives into its own
workflow and should contain only decisions and operations specific to that
module. Lab-wide planning, state, and workers stay in `nro/orchestration`.

## Configuration resolution

`nro/definitions/store.py` resolves configurations and workflows from the
external [definitions store](definitions.md). The store's `configs/` directory
has one directory per configuration class, and workflows live in `workflows/`.
Processing requests name configurations and workflows by identifier. `nro site`
selects the store root.

A module configuration contains only that module's local parameters. A workflow
selects configuration IDs, and that selection supplies the upstream lineage.

Each class has a complete base in nro's packaged `main_<CLASS>.yml`. An
optional external `main` override is overlaid on that base, and then the
selected named YAML file is overlaid on the result. There is no Python map of
configuration defaults. The registry snapshots the workflow and the fully
resolved values for each class. Module processes read those immutable runtime
values directly, with no second merge and no other configuration source.

## Lineages and derivative directories

Module lineages form a DAG that does not depend on participant work items.
Equivalent lineages reuse the same derivative directory. Changing a definition
creates a workflow revision.

Each module writes below
`derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`. The digest is computed from
the configuration ID and the complete upstream lineage, so it does not depend on
registration or discovery order. Because `anat` and `func` have separate
lineages, functional variants can reuse identical anatomy without mixing their
files.

## Work-item identity

A work item is identified by its project, module, module-lineage ID,
participant, and applicable BIDS entities. From `clean` onward, the entities
include one space and one smoothing value. Space and smoothing are request
entities. They do not belong to the configuration or to the directory name.

A revision fingerprint covers the module, the configuration fingerprint, the
entities, and the explicit contract version. Source-code hashes are not inputs
to identity or freshness.

The implementation stores the identity in `WorkItemIdentity`. A `WorkItemSpec`
bundles it with a `WorkItemContract`, an `ExecutionRecipe`, and a
`ResourceRequest`. The contract records output topology, dependency topology,
entities, configuration, substantive direct inputs, and explicit processing
policy. The registry's work-item and completion tables store its normalized
form in the `artifact_contract` field.
[Work-item planning and execution](work-item-lifecycle.md#the-work-item-specification)
describes each record and which changes make an output stale.

The command belongs to the execution recipe. Reformatting it does not make a
derivative stale. Replanning an unchanged contract updates the recipe for future
attempts, while an attempt that has already been claimed keeps the recipe it
captured when it started.

## Requests with several runs or targets

Some work items depend on a set of runs. `input_filter` belongs to the
microparcellation configuration. The planner discovers the raw BOLD runs,
applies the filter, and builds every run-level preprocessing and cleaning work
item the selected runs need. The participant-level work item then depends on
the complete selected set. Adding or removing a matching run changes that set
and requires replanning.

Space and smoothing pairs are created on demand. The planner does not enumerate
every possible pair. A request creates only the cross-product it asks for, which
defaults to `fsnative` and `2mm`. A later request can add another pair without
changing or rerunning the pairs that already exist. `func` stays shared because
one run work item publishes every supported space. Each `clean`, `dynconn`,
`microparcellation`, and `networks` work item represents exactly one pair.

## Registries and the control store

A shared deployment has one site-wide private control store. It holds one
scheduler registry and one scientific registry per registered branch:

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

Source projects and participant directories are discovered separately from
derivative planning. Branch registries hold compiled scientific contracts,
workflow lineages, and observations. Only the scheduler registry holds demand,
attempts, workers, and submissions. Because all of that lives in one place,
concurrency limits and reusable worker capacity apply across simultaneous
requests from different projects and branches.

Each registry layout has its own schema marker. The markers validate structure
and play no part in scientific freshness. Shared maintenance migrates schemas
at or after the supported baseline. Older private state is rebuilt from durable
public ownership records. Reconstruction starts from receipts that are backed by
outputs, and it keeps a missing ancestor only when it is part of a surviving
artifact's dependency chain. Maintenance keeps rollback state until the
replacement registries pass integrity and identity checks. The
[migration contract](registry-migrations.md) derives each supported schema from
an immutable baseline and a restricted, ordered list of changes.

`Registry` is the only connection and transaction facade. Focused internal
operations handle work-item registration, demand reconciliation, and status
projection. They receive a connection from the facade and cannot open, commit,
or roll back connections themselves. Splitting registry logic into smaller
operations therefore cannot create competing lock or rollback authorities.

## The controller

An automatically managed controller is the only long-lived client of the shared
scheduler database. Checkout processes may compile scientific state into their
own branch registries, which hold no attempts or worker-pool state.

On a Slurm site, commands that produce work submit the controller as a Slurm
job. Local deployments run it directly. User commands and workers talk to it
over TCP. Retryable requests and their responses are written to SQLite before
they run, so a replacement controller can replay a request that was never
resolved. Atomic launch election and fencing let only the current controller
act.

The controller uses rollback journaling and short transactions. No transaction
spans scientific computation, Slurm waits, network waits, or bulk filesystem
work. Queueing at the controller has no timeout, but communication with a live
controller does. Commands show a small progress indicator while they wait.

After relevant changes, the controller publishes a JSON read model in parts. A
small atomic manifest points to a reusable static work-item graph and to the
current dynamic execution state. Cached status reads join those parts without
starting the controller or opening SQLite. Bounded mutations go through the
live controller or a fenced local one-shot coordinator. `run` may submit the
scheduler, which runs pinned planning in an isolated child process. Workers may
restart a scheduler that their active work needs.

The registry keeps current state, not an unbounded execution history. Scheduler
maintenance keeps active demand, owned derivatives, unresolved diagnostics, and
the dependency closures of each. It removes superseded attempts and finished
allocation records. Request handles that can still be published keep their
terminal identities but not their full execution graphs.

The controller exits after attempts, workers, submissions, and pending requests
have been idle for the configured grace period, which defaults to 12 hours.
Installation and repair stop it through a deliberate shutdown barrier before
they start offline maintenance.

## Private files outside SQLite

SQLite holds the private scientific and scheduling facts. A few files remain
outside it because they cross a process, bootstrap, or filesystem-transaction
boundary:

- the branch catalog and installed-release binding, which locate the databases
  before a controller can open them;
- immutable workflow, site, and source snapshots, which pinned subprocesses read
  as execution inputs;
- the atomic status snapshot, which supports fast observation while no
  controller is running;
- logs and runner ledgers, which explain execution and support step-level
  resumption; and
- ingestion and promotion journals, which recover multi-file publication
  operations that a SQLite transaction cannot roll back.

None of these files duplicates the record of a completed work item. nro has no
private completion-manifest tree, and requests live only in the scheduler
database with no per-request JSON archive.

## Freshness and completion records

The filesystem is the authority for current artifacts. The database is the
authority for orchestration history. A successful attempt stays successful in
the history even if its output later becomes stale, and a past failure does not
override a later fresh assessment. Cancellation applies to demand and attempts,
never permanently to a work item. The
[state axes](work-item-lifecycle.md#independent-state-axes) list each kind of
state the registry tracks.

Every completed work item gets one normalized database record. It holds the
exact resolved configuration and lineage, the direct source inputs, the required
upstream generations, the public output inventory with integrity records, and
provenance for the command, implementation, attempt, and completion. The
completion row and its input, output, and private-artifact rows commit in the
same transaction as the new generation. With no private completion file, nothing
can race the database or outlive the registry identity it describes.

Public module manifests are scientific outputs. They index variable output sets
and give downstream tools portable metadata. Public paths use BIDS URIs or typed
site-resource identities, and absolute paths appear only in execution contexts
and private scheduler state. Converting ownership records changes only their
representation. It refreshes integrity evidence for the changed files and leaves
contracts, generations, and dependency edges alone. See
[portable derivative provenance](portable-provenance.md).

Registration rejects two distinct work items that claim the same public output.
Fixed-output contracts may share a directory when their exact filenames do not
overlap. A prefixed contract owns its whole filename namespace, including files
listed by a variable-output completion index.

Missing private `WORK` intermediates do not make an intact artifact stale. The
module recreates them if it needs to run again. Changed private intermediates
that still exist, changed direct inputs, changed upstream generations, and
missing or changed products all count as freshness evidence.

## The runner

Each module creates one `Runner`, and that runner creates and owns the module's
`RunnerGraph`. Step factories sit outside `Runner`. A factory receives the
inputs for one operation and returns an immutable `Step`. A larger named stage
may return an immutable `StagePlan`, which holds an ordered tuple of steps and
typed references to their downstream products. The module's `build_module()`
function is the only place that passes these declarations to
`Runner.add_step()` or `Runner.add_steps()`. Factories and stage planners never
change a runner as a side effect. Freezing the graph also rejects two internal
steps that produce the same path.

`Runner.execute()` freezes the graph before it checks freshness. Artifact state
can make a declared step run or skip, but it can never add, remove, or redirect
a step. Every resumable step declares a nonempty set of exact inputs and
outputs. One step may contain several child commands only when they form a
single atomic artifact operation with one freshness and recovery boundary.

Producers of variable directories share one atomic directory lifecycle. A module
may clear a directory that it owns exclusively. When several space and smoothing
work items share a subject directory, the module clears and validates only the
files owned by the target prefix, then writes a target-specific breadcrumb.

The persisted runner contract keeps the declared topology separate from the
records of successful steps. The runner saves the topology before execution and
commits each step's record once that step completes or passes its freshness
check. An interrupted module therefore keeps proof of the steps it finished, and
failed or unvisited steps are never treated as complete. Nodes and edges cannot
depend on artifact freshness.

Each successful node records its explicit scientific parameters and a
normalized signature of any external command. When one node's scientific
declaration changes, the runner reruns that node and invalidates its
descendants. Independent nodes stay fresh. Presentation changes, such as a new
step label or an equivalent long-option spelling, do not invalidate outputs.

A change in source BIDS state may establish a new topology contract. A code edit
alone cannot. Module completion fails if a planned node was neither executed nor
skipped, if a step did not receive an exact freshness decision, or if declared
outputs are missing. Modules must declare scientific parameters on the steps
that use them. The runner cannot silently add a configuration file as an input
to every node.

## Workers and Slurm

Slurm jobs are reusable foreground workers. Each one asks for compatible ready
work items through ordered scheduler events and supervises one module
subprocess at a time. A claim returns a typed `ExecutionEnvelope`, and an
`ExecutionLauncher` creates and supervises the subprocess. Workers renew leases
and check for cancellation through direct scheduler requests, and they stop the
whole subprocess group when told to. Their request records stay available for
replay until the scheduler commits them. Neither a worker nor its scientific
subprocess can open the scheduler database.

Workers drain before their wall time runs out and ask the controller for
successor capacity. The registry enforces the separate general and GPU
concurrency limits. The number of queued Slurm jobs does not. If an allocation
still expires while it runs an item or resource step, the scheduler records the
interrupted attempt and returns the work to the right ready queue. It does not
mark the work item as timed out or cancel its descendants. The same happens when
a signal reaches the worker while a command runs, even if the command traps
SIGTERM and exits with some other status.

The scheduler writes a new worker script for each successor, OOM retry, and pool
expansion. Wall time, CPUs, idle timeout, and drain time come from the current
site settings, and the source comes from the active scheduler binding. Only the
Slurm partition and account carry over from the requesting worker's profile. A
scheduler restarted by a worker also reads the current binding, so changes to
`slurm.scheduler.*` take effect at the next scheduler allocation.

A confirmed OOM raises the work item's memory tier geometrically, up to the
request ceiling. Workers at a higher tier may take lower-tier work items.

## Commands that observe or change state

`log` only observes. By default, `status` reads the last atomic scheduler
snapshot. `status --update` goes through the live controller or a fenced one-shot
coordinator. It runs an authoritative assessment, updates the registry, and
publishes a new snapshot, and it can cancel an attempt whose registered contract
is obsolete. `run` and worker events also assess artifacts and grow the worker
pool.

`set`, `stop`, `purge`, and `publish` are explicit mutation commands, each with
its own job. `scene` discovers files directly and does not need a live
controller. `render` uses the same discovery and then runs Workbench's headless
image renderer.

`publish` freezes a satisfied request into a new standalone derivative dataset.
It stages copies or reflinks, embeds recursive provenance that does not depend
on the workflow, verifies the copied content and live generations, optionally
validates the BIDS structure, and renames the staging directory atomically. It
never overwrites an existing destination.

## Rules enforced by tests

The test suite checks that these design rules hold:

- there is exactly one `Runner` class and one `RunnerGraph` engine;
- graphs are created only inside `Runner`, and the DAG changes only in each
  module's `build_module()` function;
- step factories return one `Step`, and named stages return immutable
  `StagePlan` values, without changing a runner;
- focused registry operations use connections owned by the caller and cannot
  connect, commit, or roll back;
- scientific modules run no subprocess outside the runner;
- no resumable step lacks outputs;
- downstream scientific modules do not find derivative members with globs;
- there are no public configuration paths or alternate stores;
- the module catalog is closed and explicit, and work-item planning stays in
  each module;
- work-item specifications and worker execution envelopes are typed;
- claimed attempts keep immutable execution recipes, and later attempts use the
  current recipe;
- each registry family has one generated schema with explicit ordered
  migrations; and
- request sharing, cancellation, lease recovery, OOM escalation, freshness,
  multirun expansion, and atomic publication behave as described here.
