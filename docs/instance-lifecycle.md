# Instance planning and execution

This document defines the records and state transitions that connect planning,
scheduling, execution, and freshness. General scientific vocabulary is defined
in [Core concepts](concepts.md).

## The instance specification

The planner represents one complete unit of possible work with an
`InstanceSpec`. It is an aggregate, not another kind of scientific object:

```text
InstanceSpec
├── InstanceIdentity
├── InstanceContract
├── ExecutionRecipe
├── ResourceRequest
├── scope
└── derivative directory label
```

### InstanceIdentity

`InstanceIdentity` says which logical schedulable instance this is:

- project;
- module;
- participant;
- configuration lineage;
- applicable BIDS entities; and
- the derived stable instance key.

Identity allows several requests to refer to the same logical work. It does not
say whether that work is currently needed or fresh.

The stable key hashes the configuration lineage fingerprint, not the registry's
integer lineage row ID. The same logical instance therefore receives the same
key after registry repair.

### InstanceContract

`InstanceContract` is the freshness-relevant promise made by the instance. It
contains:

- the resolved scientific configuration fingerprint;
- direct source inputs;
- upstream instance dependencies;
- the public output root and target-specific prefix;
- required fixed outputs and the output format; and
- explicit processing details that affect the meaning or construction of the
  derivative.

The normalized contract is fingerprinted. A substantive contract change makes
the existing instance output stale. Code that changes scientific behavior is
responsible for changing an appropriate configuration, dependency, output, or
explicit processing field; a source-code version is not a substitute for a
semantic contract.

Modules can normalize their processing specification before comparison.
Firstlevels uses a [canonical compiled task definition](task-models.md#model-edits-and-freshness)
so equivalent authoring forms share a contract. The same normalization applies
to recorded contracts and completion certificates. It does not suppress changes
to dependencies, output requirements, or resolved scientific configuration.
The [configuration compiler](configuration.md#scientific-settings-and-execution-snapshots)
separates explicitly declared execution settings from scientific settings.
Full execution snapshots remain available as provenance.

Every built-in scientific module includes its public metadata schema in the
explicit processing portion of this contract. The schema names required fields
and their JSON-compatible types, including conditional schemas for distinct
output domains. Module writers and freshness validators consume the same
declaration. Adding, removing, or changing a promised metadata field therefore
changes the artifact contract; formatting, key order, optional descriptive
metadata, and values that legitimately depend on the input do not.

`InstanceContract` describes the output boundary of the whole instance. An
artifact is merely one concrete file or directory participating in that
boundary.

### ExecutionRecipe

`ExecutionRecipe` says how to invoke the module process:

- argument vector;
- immutable runtime configuration path.

Equivalent command spelling does not change derivative freshness. Once a
recipe is attached to active demand, replanning the same unchanged contract
must not silently replace it. A substantive contract change may cancel active
work and establish a new recipe for the changed work.

### ResourceRequest

`ResourceRequest` describes scheduler needs such as resource class, current
memory tier, and maximum memory. Resource changes affect where and when work can
run, not what the derivative means, so they do not affect freshness.

### InstanceSpec

`InstanceSpec` contains the four records above plus planner metadata such as
scope and derivative directory label. It is therefore a strict conceptual
superset of `InstanceContract`:

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
Planner constructs complete InstanceSpecs
        │
        ▼
Registry merges logical instances and dependency edges
        │
        ▼
Request creates active demand for terminal and upstream instances
        │
        ▼
Filesystem assessment classifies instance outputs
        │
        ▼
Worker claims one demanded, nonfresh instance whose upstreams are fresh
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
        ▼
Worker validates public outputs and writes the completion manifest
```

### ExecutionEnvelope

An `ExecutionEnvelope` is the typed boundary between the registry and worker.
It contains the claimed instance and attempt IDs, identity, normalized instance
contract, execution recipe, exact inputs and outputs, manifest destination, and
log destination.

The registry decodes its SQL and JSON storage representation before returning
the envelope. Worker execution therefore does not depend on database column
names or manipulate a raw registry row.

### ExecutionLauncher

`ExecutionLauncher` is the boundary between worker scheduling and process
execution. The current launcher runs a local subprocess, supervises its process
group, propagates cancellation, and emits heartbeats. This leaves worker claim,
retry, and completion logic independent of how a module process is launched.

## Independent state axes

State is deliberately not compressed into one ambiguous status:

- **Instance output state:** `missing`, `stale`, or `fresh`.
- **Request state:** `active`, `cancelled`, or `satisfied`.
- **Demand state:** whether a particular request still needs a particular
  instance.
- **Attempt state:** `queued`, `running`, `cancel_requested`, `cancelled`,
  `success`, or `error`.
- **Worker state:** idle, running, draining, shutdown-requested, or terminal.
- **Displayed scheduling state:** derived states such as waiting, ready,
  running, blocked, and dormant.

These axes answer different questions. A successful historical attempt can
have stale outputs later. A previous attempt error can coexist with a fresh
instance produced by a later attempt. Cancelling one request does not erase an
instance or its history.

## Freshness boundary

The planner always constructs the complete module and instance dependency graph
before assessing freshness. Assessment compares the instance contract and
completion manifest with current filesystem evidence, including:

- direct source inputs;
- required upstream generations;
- public output inventory and integrity records; and
- existing private artifacts recorded for resumption.

Missing private intermediaries alone do not invalidate an intact public
boundary because they can be regenerated. Missing or changed public artifacts,
changed existing private artifacts, changed direct inputs, or changed upstream
generations are freshness evidence.

The command, interpreter path, memory tier, and source revision are execution
or provenance details. They do not independently constitute freshness evidence.

## Module planning boundary

The central planner owns dependency traversal. The closed catalog in
`nro/orchestration/catalog.py` declares the current modules, their derivative
classes, scopes, resource defaults, output formats, dependencies, and planning
functions.

Each `nro/<module>/planning.py` function constructs only instances of its own
module from a resolved `SubjectPlanningContext` and already constructed
upstream specifications. It returns new `InstanceSpec` objects and does not
mutate the registry or planner as a side effect.

If required source data do not exist, module planning may declare that
participant unavailable for the selected terminal module. This is a normal
selection outcome, not a partially constructed instance and not an execution
failure. The central planner records the reason, omits that participant from
the request group, and continues planning other participants and projects.
Unexpected path or metadata failures remain errors and stop planning.

This separation keeps module-specific BIDS and output decisions beside the
scientific module while leaving cross-module traversal and scheduling under one
planner authority.
