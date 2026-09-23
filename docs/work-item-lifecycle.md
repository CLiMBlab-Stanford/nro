# Work-item planning and execution

This document defines the records and state transitions that connect planning,
scheduling, execution, and freshness. General scientific vocabulary is defined
in [Core concepts](concepts.md).

The Python records, registry schema, and command-line interface all follow the
vocabulary in [Core concepts](concepts.md).

## The work-item specification

The planner represents one complete unit of possible work with a
`WorkItemSpec`. It is an aggregate, not another kind of scientific object:

```text
WorkItemSpec
├── WorkItemIdentity
├── WorkItemContract
├── ExecutionRecipe
├── ResourceRequest
├── scope
└── derivative directory label
```

### WorkItemIdentity

`WorkItemIdentity` says which logical schedulable work item this is:

- project;
- module;
- participant;
- module lineage;
- applicable BIDS entities; and
- the derived stable work-item key.

Identity allows several requests to refer to the same logical work. It does not
say whether that work is currently needed or fresh.

The stable key hashes the module-lineage fingerprint, not the registry's
integer lineage row ID. The same logical work item therefore receives the same
key after registry repair.

### WorkItemContract

`WorkItemContract` is the freshness-relevant promise made by the work item. It
contains:

- the resolved scientific configuration fingerprint;
- direct source inputs;
- upstream work-item dependencies;
- the public output root and target-specific prefix;
- required fixed outputs and the output format; and
- explicit processing details that affect the meaning or construction of the
  derivative.

The normalized contract is fingerprinted. A substantive contract change makes
the existing artifact stale. Code that changes scientific behavior is
responsible for changing an appropriate configuration, dependency, output, or
explicit processing field; a source-code version is not a substitute for a
semantic contract.

Modules can normalize their processing specification before comparison.
Firstlevels uses a [canonical compiled task definition](task-models.md#model-edits-and-freshness)
so equivalent authoring forms share a contract. The same normalization applies
to recorded contracts and database completion records. It does not suppress changes
to dependencies, output requirements, or resolved scientific configuration.
The [configuration compiler](configuration.md#scientific-settings-and-execution-snapshots)
separates explicitly declared execution settings from scientific settings.
Full execution snapshots remain available as provenance.

Every built-in scientific module includes its public metadata schema in the
explicit processing portion of this contract. The schema names fields and their
JSON-compatible types, including conditional schemas for distinct output
domains. A plain type name marks a required field. A `kind` and `default`
mapping marks a field whose declared default also applies to older artifacts
that predate the field. Module writers and freshness validators consume the
same declaration.

Removing a schema field is compatible because current readers no longer use it.
Adding a field with a default is compatible because its meaning for an older
artifact is unambiguous. Unknown fields in recorded metadata are ignored.
Adding a required field, making an optional field required, changing a field's
type or default, or changing any other scientific promise remains substantive.
Formatting, key order, descriptive metadata outside the declared schema, and
values that legitimately depend on the input do not affect freshness.

`WorkItemContract` describes the artifact promised by the whole work item. Each
concrete file or atomic directory in that artifact is a product.

### ExecutionRecipe

`ExecutionRecipe` says how to invoke the module process:

- argument vector;
- immutable runtime configuration path.

Equivalent command spelling does not change derivative freshness. Replanning
an unchanged contract may update the current recipe for future attempts. An
attempt that a worker has already claimed keeps the immutable recipe captured
at claim time. A substantive contract change may cancel active work and
establish a new recipe for the changed work.

### ResourceRequest

`ResourceRequest` describes scheduler needs such as resource class, current
memory tier, and maximum memory. Resource changes affect where and when work can
run, not what the derivative means, so they do not affect freshness.

### WorkItemSpec

`WorkItemSpec` contains the four records above plus planner metadata such as
scope and derivative directory label. It is therefore a strict conceptual
superset of `WorkItemContract`:

- the contract answers **what substantive result is promised?**
- the complete specification answers **which work is this, how can it run, and
  what resources may it use?**

The distinction permits operational changes without spurious invalidation.

| Change | Specification changed? | Contract changed? | Existing output stale? |
|---|---:|---:|---:|
| Increase memory | Yes | No | No |
| Reformat equivalent argv | Yes | No | No |
| Change a direct input | Yes | Yes | Yes |
| Change dependency topology | Yes | Yes | Yes |
| Change output format | Yes | Yes | Yes |
| Change required public metadata schema | Yes | Yes | Yes |
| Change substantive processing policy | Yes | Yes | Yes |

## From request to completion

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

### ExecutionEnvelope

An `ExecutionEnvelope` is the typed boundary between the registry and worker.
It contains the claimed work-item and attempt IDs, identity, normalized
work-item contract, execution recipe, exact inputs and outputs, and log destination.

The registry decodes its SQL and JSON storage representation before returning
the envelope. Worker execution therefore does not depend on database column
names or manipulate a raw registry row.

The scheduler maintains shared execution state before workers claim envelopes.
It reconciles Slurm submissions, recovers orphaned attempts, and reassesses only
demanded artifacts on one scheduler-wide cadence. A worker claim performs the
smaller dependency-state update needed to expose newly ready work. Starting a
batch of workers therefore does not repeat a full registry refresh per worker.

### ExecutionLauncher

`ExecutionLauncher` is the boundary between worker scheduling and process
execution. The current launcher runs a local subprocess, supervises its process
group, propagates cancellation, and emits heartbeats. This leaves worker claim,
retry, and completion logic independent of how a module process is launched.

## Independent state axes

State is deliberately not compressed into one ambiguous status:

- **Artifact state:** `missing`, `stale`, `corrupt`, or `fresh`. Corruption
  means existing files contradict their own contract or completion evidence.
- **Request state:** `active`, `cancelled`, or `satisfied`.
- **Demand state:** whether a particular request still needs a particular
  work item.
- **Attempt state:** `queued`, `running`, `cancel_requested`, `cancelled`,
  `success`, or `error`.
- **Worker state:** idle, running, draining, shutdown-requested, or terminal.
- **Displayed scheduling state:** derived states such as waiting, ready,
  running, blocked, and dormant.

These axes answer different questions. A successful historical attempt can
have stale outputs later. A previous attempt error can coexist with a fresh
artifact produced by a later attempt. Cancelling one request does not erase a
work item or its history.

### Input changes during execution

Claiming an attempt records its resolved upstream work-item IDs and generations,
including transitive ancestors. These references remain unchanged if replanning
replaces the current dependency graph. They describe the inputs actually selected
for that attempt, not the branch names or locations used to discover them.

Invalidation follows both current dependencies and active attempts' captured
inputs. A changed producer invalidates affected consumers and requests cancellation
of their running attempts. Unrelated work continues. Demand remains active, so
cancelled work can retry with compatible inputs without a new request. A retry
does not silently change the requested scientific configuration.

An upstream rebuild waits for those consumers to stop, rather than letting them
finish using obsolete data. Cancellation of a supervised command stops its whole
process group. Unknown shutdown state continues to block replacement; an expired
worker lease alone is not proof that its processes stopped. Completed consumers
also become stale when their required upstream generation changes.

Completion validates the attempt, contract, captured inputs, and mutation barriers
again in the coordinator transaction before recording its evidence and advancing
its generation. A late completion cannot override cancellation. nro does not retain
old derivative generations to let invalidated attempts run to completion.

These rules apply to dependencies resolved across branch boundaries as well as
within one branch. The selected producer location affects input routing, not
invalidation semantics.

## Freshness boundary

The planner always constructs the complete module and work-item dependency graph
before assessing freshness. Assessment compares the work-item contract and
database completion record with current filesystem evidence, including:

- direct source inputs;
- required upstream generations;
- public output inventory and integrity records; and
- existing private intermediates recorded for resumption.

Missing private intermediaries alone do not invalidate an intact public
boundary because they can be regenerated. Missing or changed products, changed
existing private intermediates, changed direct inputs, or changed upstream
generations are freshness evidence.

The command, interpreter path, memory tier, and source revision are execution
or provenance details. They do not independently constitute freshness evidence.

### Concurrent assessment

A full assessment captures the selected graph and its ancestors in one registry
transaction. Filesystem checks run outside the lock. Publication then checks
that the graph, configurations, generations, and active attempts still match
the captured records before applying any decisions. A result computed before a
concurrent completion cannot overwrite that newer completion.

Assessment retries up to three times on conflict. Workers defer a persistently
contended assessment until a later check-in; `status --update` asks the user to
retry. Unrelated work, heartbeats, and memory-limit changes do not force retries.
The snapshot's comparison token coordinates publication and is not part of an
artifact's scientific contract.

During branch verification, the invoking checkout recompiles registered
selections. The central scheduler evaluates its detached snapshot against
those contracts and filesystem evidence, then records accepted observations
in the branch scientific registry. The evaluator never opens the branch
registry or imports branch modules.

## Module planning boundary

The central planner owns dependency traversal. The closed catalog in
`nro/orchestration/catalog.py` declares the current modules, their derivative
classes, scopes, resource defaults, output formats, dependencies, and planning
functions.

Each `nro/modules/<module>/planning.py` function constructs only work items for
its own module from a resolved `SubjectPlanningContext` and already constructed
upstream specifications. It returns new `WorkItemSpec` objects and does not
mutate the registry or planner as a side effect.

If required source data do not exist, module planning may declare that
participant unavailable for the selected terminal module. This is a normal
selection outcome, not a partially constructed work item and not an execution
failure. The central planner records the reason, omits that participant from
the request group, and continues planning other participants and projects.
Unexpected path or metadata failures remain errors and stop planning.

This separation keeps module-specific BIDS and output decisions beside the
scientific module while leaving cross-module traversal and scheduling under one
planner authority.
