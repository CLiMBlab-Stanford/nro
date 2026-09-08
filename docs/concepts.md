# Core concepts

This document defines the language used to describe `nro`. These terms are
architectural contracts: code, tests, logs, and other documentation should use
them consistently.

## The scientific hierarchy

```text
workflow
  selects one configuration lineage for each derivative class

derivative class
  organizes a broad family of outputs
  is realized by one or more scientific modules

module
  is instantiated as schedulable work
  contains a complete graph of steps

instance
  owns one Runner and one RunnerGraph
  produces public and private artifacts
```

### Derivative class

A **derivative class** is a broad category of related results and configuration.
The current classes are `preprocessing`, `clean`, `microparcellation`, and
`networks`.

Classes organize configuration lineages and derivative directories. They are
not themselves schedulable. Most classes correspond to one module. The
`preprocessing` class is the deliberate exception: its configuration and
derivative tree are shared by the `anat` and `func` modules.

Use **class** only when discussing this organization. Do not use it as a synonym
for a Python class, module, or individual scheduled computation.

### Configuration

A **configuration** is the complete set of parameters for one derivative class.
It has a stable human-readable ID. Configuration files contain the authority;
Python does not maintain a second default parameter map.

A **configuration lineage** identifies a configuration together with its
upstream configuration choices. Equivalent lineages share a derivative
directory. A substantive configuration change changes the applicable instance
contract and therefore invalidates affected results when they are reassessed.

### Workflow

A **workflow** selects one configuration ID for each derivative class. It
defines a coherent path from source data through the available derivative
classes. A workflow does not contain scientific code and is not a runtime
sequence of steps.

### Module

A **module** is the complete scientific directed acyclic graph needed to
realize one kind of schedulable work. The current modules are `anat`, `func`,
`clean`, `microparcellation`, and `networks`.

A module's graph is determined entirely by resolved BIDS data and its workflow.
It is fully constructed before freshness is examined. Existing, missing,
fresh, stale, or invalid files may affect whether declared steps run, but may
never change which steps exist or how they depend on one another.

In the filesystem, a module's scientific construction belongs in
`nro/<module>/module.py`; planner-facing construction of its instances belongs
in `nro/<module>/planning.py`.

### Instance

An **instance** is one concrete, schedulable instantiation of a module. Its
identity includes its project, module, participant, configuration lineage, and
applicable BIDS entities. For example:

> `clean` for participant `t20`, one BOLD run, `fsnative` space, and 2 mm
> smoothing under a particular clean configuration lineage.

Instances can run concurrently when their dependencies permit it. Steps within
one instance are not independently scheduled to cluster workers.

### Step

A **step** is one immutable declaration of a scientific transform. It declares
exact inputs, exact outputs, and one action. A step may depend on other steps in
the same module.

A step is always added to the module graph before freshness is evaluated. At
runtime it receives exactly one decision: execute or skip. Closely related
child commands may share a step only when they form one atomic artifact
operation with one freshness and recovery boundary.

### Runner and runner graph

The **runner** is the one shared instance-internal execution engine,
`nro.orchestration.runner.Runner`. Every scientific module uses it; modules do
not define their own runner implementations.

Each runner creates and owns one **runner graph**, represented by
`RunnerGraph`. Module code adds steps through the runner. The runner freezes the
graph before checking freshness, then traverses it in dependency order and
records every execute-or-skip decision.

The module defines the scientific DAG. The runner implements its common
execution semantics. The planner schedules complete instances, not runner
steps.

### Artifact

An **artifact** is a concrete file or directory consumed or produced by an
instance or step. It is not schedulable and it has no independent worker.

Artifacts are classified by ownership and lifetime:

- A **public artifact** is a required result in the instance's derivative
  directory. Public artifacts form its durable output boundary.
- A **private artifact** is a resumable intermediary under `WORK`. It may be
  deleted after the public boundary is complete and can be regenerated when
  needed.

The registry sometimes reports the collective filesystem condition of an
instance as its **artifact state** (`missing`, `stale`, or `fresh`). This means
the state of the instance's required artifact set; it does not turn that set
into another kind of schedulable object.

## Orchestration hierarchy

```text
user request
  creates demand for terminal instances
  expands to demand for their upstream instances

planner
  constructs and registers complete InstanceSpecs
  assesses whether demanded instances need execution

worker
  claims one ready instance
  creates one attempt
  launches its immutable ExecutionEnvelope

runner
  executes or skips the instance's declared steps
```

### Planner

The **planner** turns selections and workflows into complete instance graphs,
registers them, evaluates demand, and supplies the shared worker pool. Its
scientific knowledge comes from the explicit built-in module catalog and the
module-local planning functions.

### Registry

The **registry** is the planner's durable lab-wide SQL state. It stores the
discovered source project/participant tree, workflows, instances, dependency
edges, requests, attempts, workers, and scheduler submissions. Ordinary source
discovery does not create derivative instances or demand. Registry bootstrap
and repair additionally discover existing files in nro-controlled derivative
locations and register the instances that own them, without creating demand.
The registry is the single locking and transaction authority; it is not a
second planner implementation.

The filesystem remains authoritative for whether derivative files currently
exist and match their completion records. The registry is authoritative for
orchestration identity, demand, and history.

Each owned derivative configuration root contains `.nro/lineage.json` and one
receipt under `.nro/instances/` for every instance it owns. These records keep
the configuration lineage, instance identity, dependencies, and artifact
contract with the derivative. Registry repair reads them before resolving the
workflows in the current configuration store. Removing or renaming a workflow
therefore does not make its existing derivatives invisible.

Ownership does not imply that an instance can be recomputed. A current workflow
must select its configuration lineage before the planner can request it. Status
reports a nonfresh historical instance as `Unavailable` when no current
workflow selects that lineage. A fresh historical instance remains `Success`
because its artifact is still usable. Purge can select either case without the
originating workflow.

### Request and demand

A **request** records a user's desire to obtain one or more terminal instances.
The planner expands it over upstream dependencies.

**Demand** is the active association between a request and an instance. Several
requests may share demand for the same instance. Cancelling one request does
not cancel work still demanded by another.

### Worker and attempt

A **worker** is a reusable Slurm or local process that claims and supervises one
ready instance at a time. It is cluster infrastructure, not a scientific
module.

An **attempt** is one historical execution of an instance by a worker. An
instance may have several attempts because it was interrupted, failed, became
stale, or was explicitly requested again. Attempt failure is history; it does
not permanently redefine the instance.

## Terms that are not interchangeable

| Terms | Distinction |
|---|---|
| Class / module | A class organizes derivative lineages; a module is a scientific DAG. |
| Module / instance | A module is the reusable DAG definition; an instance is one schedulable application of it. |
| Instance / artifact | An instance is work; an artifact is a file or directory used by that work. |
| Workflow / module | A workflow selects configurations; a module performs scientific computation. |
| Planner / runner | The planner schedules instances; the runner executes steps inside one instance. |
| Request / attempt | A request expresses demand; an attempt records one execution. |
| Space / smoothing | Both are instance selectors from `clean` onward. Nro filenames use the nonstandard `smoothing` entity because BIDS `scale` describes atlas granularity. |

The planner-facing record hierarchy is defined separately in
[Instance planning and execution](instance-lifecycle.md).
