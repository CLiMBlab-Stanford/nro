# Work-item planning and execution

This page is for developers who work on planning, scheduling, or freshness. It
defines the records a work item passes through from request to completion and
the rules for when an output becomes stale. The Python records, registry schema,
and command-line interface all use the vocabulary in
[Core concepts](concepts.md). [Orchestration design](orchestration-design.md)
describes the components that create and consume these records.

## From request to completion

A request moves through the following stages. The sections after the diagram
describe the records involved.

```text
selection + workflow
        │
        ▼
Planner constructs complete WorkItemSpecs
        │
        ▼
Registry merges logical work items and dependency edges
        │
        ▼
Request creates active demand for terminal and upstream work items
        │
        ▼
Filesystem assessment classifies artifacts
        │
        ▼
Worker claims one demanded, nonfresh work item whose upstreams are fresh
        │
        ▼
Registry creates an attempt and returns an ExecutionEnvelope
        │
        ▼
ExecutionLauncher supervises the module subprocess
        │
        ▼
Runner freezes its graph, executes or skips every step, and records the result
        │
        ├── A resource-specific step may yield to its matching worker pool
        │   and return the parent work item to the general queue
        │
        ▼
Worker validates public outputs and commits the completion record
```

## The work-item specification

The planner describes one complete unit of possible work with a `WorkItemSpec`.
It groups several records together and is not a separate kind of scientific
object:

```text
WorkItemSpec
├── WorkItemIdentity
├── WorkItemContract
├── ExecutionRecipe
├── ResourceRequest
├── scope
└── derivative directory label
```

The split exists so that operational changes do not cause false invalidation.
The contract says what substantive result is promised. The full specification
also says which work this is, how it runs, and what resources it may use. Only a
contract change makes an existing output stale:

| Change | Specification changed? | Contract changed? | Existing output stale? |
|---|---:|---:|---:|
| Increase memory | Yes | No | No |
| Reformat equivalent argv | Yes | No | No |
| Change a direct input | Yes | Yes | Yes |
| Change dependency topology | Yes | Yes | Yes |
| Change output format | Yes | Yes | Yes |
| Change required public metadata schema | Yes | Yes | Yes |
| Change substantive processing policy | Yes | Yes | Yes |

### WorkItemIdentity

`WorkItemIdentity` names one logical, schedulable work item. It contains the
project, module, participant, module lineage, applicable BIDS entities, and a
derived stable work-item key. Several requests can refer to the same work
through its identity. Identity says nothing about whether the work is needed or
fresh.

The stable key hashes the module-lineage fingerprint, not the registry's integer
lineage row ID. The same logical work item therefore keeps its key after a
registry repair.

### WorkItemContract

`WorkItemContract` is the promise that freshness checks compare against. It
contains:

- the resolved scientific configuration fingerprint;
- direct source inputs;
- upstream work-item dependencies;
- the public output root and target-specific prefix;
- required fixed outputs and the output format; and
- explicit processing details that affect what the derivative means or how it
  is built.

The normalized contract is fingerprinted, and a substantive change makes the
existing artifact stale. A source-code version does not stand in for this. When
code changes scientific behavior, the change must also update a configuration,
dependency, output, or explicit processing field.

The contract covers the artifact that the whole work item promises. Each
concrete file or atomic directory in that artifact is a product.

Modules may normalize their processing specification before comparison. For
example, `contrasts` uses a
[canonical compiled model definition](task-models.md#model-edits-and-freshness),
so equivalent ways of writing a model share one contract. The same
normalization applies to recorded contracts and to database completion records.
It never hides changes to dependencies, output requirements, or resolved
scientific configuration. The
[configuration compiler](configuration.md#scientific-settings-and-execution-snapshots)
keeps explicitly declared execution settings apart from scientific settings,
and full execution snapshots remain available as provenance.

Every built-in scientific module puts its public metadata schema in the explicit
processing part of the contract. The schema names each field and its
JSON-compatible type, with conditional schemas for distinct output domains. A
plain type name marks a required field. A mapping with `kind` and `default`
marks a field whose default also applies to older artifacts written before the
field existed. Module writers and freshness validators read the same
declaration.

Some schema changes are compatible with existing artifacts. Removing a field is
compatible because current readers no longer use it. Adding a field with a
default is compatible because its meaning for an older artifact is clear.
Unknown fields in recorded metadata are ignored. Adding a required field, making
an optional field required, or changing a field's type or default is
substantive, as is any other change to the scientific promise. Formatting, key
order, descriptive metadata outside the schema, and values that legitimately
depend on the input do not affect freshness.

### ExecutionRecipe

`ExecutionRecipe` tells a worker how to start the module process. It holds the
argument vector and the path to the immutable runtime configuration.

Spelling a command differently does not change freshness. Replanning an
unchanged contract may update the current recipe for future attempts, but an
attempt that a worker has already claimed keeps the recipe it captured at claim
time. A substantive contract change may cancel active work and set a new recipe
for the changed work.

### ResourceRequest

`ResourceRequest` describes what the scheduler must provide, such as the
resource class, the current memory tier, and the maximum memory. Resources
affect where and when work runs but not what the derivative means, so they do
not affect freshness.

### ExecutionEnvelope

When a worker claims a work item, the registry returns an `ExecutionEnvelope`.
It contains the claimed work-item and attempt IDs, the identity, the normalized
contract, the execution recipe, the exact inputs and outputs, and the log
destination. The registry decodes its SQL and JSON storage before returning the
envelope, so worker code never depends on database column names or handles raw
registry rows.

The scheduler keeps shared execution state current before workers claim
envelopes. On one scheduler-wide cadence, it reconciles Slurm submissions,
recovers orphaned attempts, and reassesses demanded artifacts. A claim does only
the smaller dependency update needed to expose newly ready work, so starting a
batch of workers does not repeat a full registry refresh for each one.

An attempt is orphaned when its worker's lease has expired and its run has
ended. Slurm keeps a job's ID when it requeues a preempted job, so a live job ID
is not enough to show that the worker is still running. The run has ended if the
job is pending again, is running on other nodes, or started after the worker
registered. The orphaned attempt returns to the ready queue with its demand
kept.

### ExecutionLauncher

`ExecutionLauncher` separates worker scheduling from process execution. The
current launcher runs a local subprocess, supervises its process group, passes
on cancellation, and sends heartbeats. Claim, retry, and completion logic in the
worker therefore does not depend on how the module process starts.

## Independent state axes

The registry tracks several kinds of state separately instead of folding them
into one status:

- **Artifact state:** `missing`, `stale`, `corrupt`, or `fresh`. An artifact is
  corrupt when existing files contradict their own contract or completion
  evidence.
- **Request state:** `active`, `cancelled`, or `satisfied`.
- **Demand state:** whether a particular request still needs a particular work
  item.
- **Attempt state:** `queued`, `running`, `cancel_requested`, `cancelled`,
  `success`, or `error`.
- **Worker state:** idle, running, draining, shutdown-requested, or terminal.
- **Displayed scheduling state:** derived states such as waiting, ready,
  running, blocked, and dormant.

Each axis answers a different question. An attempt that succeeded can later
have stale outputs. An earlier attempt error can coexist with a fresh artifact
from a later attempt. Cancelling one request does not erase a work item or its
history.

## Freshness

The planner always builds the complete module and work-item dependency graph
before it assesses freshness. Assessment compares the work-item contract and the
database completion record with current filesystem evidence:

- direct source inputs;
- required upstream generations;
- the public output inventory and integrity records; and
- existing private intermediates recorded for resumption.

Missing private intermediates alone do not invalidate an intact public artifact,
because they can be regenerated. Missing or changed products, changed private
intermediates that still exist, changed direct inputs, and changed upstream
generations are freshness evidence.

The command, interpreter path, memory tier, and source revision are execution or
provenance details. None of them is freshness evidence on its own.

### Concurrent assessment

A full assessment reads the selected graph and its ancestors in one registry
transaction and then checks the filesystem outside the lock. Before applying
its decisions, it checks that the graph, configurations, generations, and active
attempts still match what it read. A result computed before a concurrent
completion therefore cannot overwrite that newer completion.

Assessment retries up to three times on conflict. If contention persists,
workers defer the assessment until a later check-in, and `status --update` asks
the user to retry. Unrelated work, heartbeats, and memory-limit changes do not
force a retry. The snapshot's comparison token coordinates this check and is not
part of any artifact's scientific contract.

During branch verification, the invoking checkout recompiles the registered
selections. The central scheduler evaluates its detached snapshot against those
contracts and the filesystem evidence, then records the accepted observations
in the branch scientific registry. The evaluator never opens the branch registry
or imports branch modules.

### Input changes during execution

When a worker claims an attempt, the registry records the upstream work-item IDs
and generations it resolved, including transitive ancestors. These references
stay fixed even if replanning replaces the current dependency graph. They
describe the inputs actually chosen for that attempt, not the branch names or
locations used to find them.

Invalidation follows both the current dependencies and the captured inputs of
active attempts. When a producer changes, its affected consumers become invalid
and their running attempts are asked to cancel. Unrelated work continues.
Demand stays active, so cancelled work can retry with compatible inputs without
a new request. A retry never silently changes the requested scientific
configuration.

An upstream rebuild waits for those consumers to stop, so they cannot finish
with obsolete data. Cancelling a supervised command stops its whole process
group. If the shutdown state is unknown, replacement stays blocked, because an
expired worker lease alone does not prove the processes stopped. Completed
consumers also become stale when the upstream generation they require changes.

At completion, the coordinator transaction checks the attempt, contract,
captured inputs, and mutation barriers again before it records evidence and
advances the generation. A late completion cannot override a cancellation. nro
does not keep old derivative generations around so that invalidated attempts
can finish.

These rules are the same whether a dependency was resolved within one branch or
across branches. The producer's location changes where inputs are read from,
not how invalidation works.

## Module planning

The central planner owns dependency traversal. The closed catalog in
`nro/orchestration/catalog.py` declares the current modules with their
derivative classes, scopes, resource defaults, output formats, dependencies, and
planning functions.

Each `nro/modules/<module>/planning.py` function builds work items for its own
module only. It receives a resolved `SubjectPlanningContext` and the upstream
specifications that already exist, and it returns new `WorkItemSpec` objects
without changing the registry or the planner. Module-specific BIDS and output
decisions therefore stay with the scientific module, while one planner handles
cross-module traversal and scheduling.

When required source data do not exist, module planning may declare the
participant unavailable for the selected terminal module. This is a normal
selection outcome and counts as neither a partial work item nor an execution
failure. The planner records the reason, leaves that participant out of the
request group, and keeps planning other participants and projects. Unexpected
path or metadata failures are still errors and stop planning.
