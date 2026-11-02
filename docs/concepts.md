# Core concepts

This page defines the terms nro uses for its scientific and scheduling
structures. Code, tests, logs, and documentation use these terms with the
meanings given here. For an overview of how the pieces fit together, read
[How nro works](design.md) first.

## The scientific hierarchy

Each level below contains or applies the level after it.

```text
workflow
  selects one module configuration for each configuration class

configuration class
  defines the accepted parameter namespace for one module

module configuration
  contains the local scientific settings for one module

module
  defines one reusable kind of scientific work
  contains a complete graph of steps

module lineage
  combines one module configuration with its upstream configuration closure

work item
  applies one module lineage to selected data and applicable entities
  owns one Runner, which owns one RunnerGraph

artifact
  is the exact public output set promised by one work item

product
  is one file or atomic directory within an artifact
```

### Configuration class and module configuration

A **configuration class** is the set of parameters that one scientific module
accepts. The current classes are `anat`, `func`, `clean`, `dynconn`,
`microparcellation`, `networks`, and `contrasts`.

A **module configuration** gives a value for every local parameter of one
configuration class. It has a stable, readable ID. Configuration files are the
only source of these values. Python code does not keep a second map of default
parameters.

### Module lineage

A **module lineage** is a module configuration together with every upstream
module configuration it depends on. A lineage does not depend on participant,
run, space, smoothing, or any other data entity.

Several workflows can select the same lineage, and equivalent lineages share
one derivative namespace. A substantive change to a configuration changes the
contract of the work items that use it. nro marks the affected results stale
the next time it assesses them.

The anatomy lineage depends only on the `anat` configuration, so functional
lineages that select the same anatomy reuse it. Each downstream lineage
records its local configuration ID and its upstream lineage. Lineages with
incompatible inputs therefore cannot write into the same namespace.

### Workflow

A **workflow** selects one configuration ID for each configuration class. It
describes one coherent path from source data through the available modules. A
workflow contains no scientific code, and it is not a runtime sequence of
steps.

### Module

A **module** defines one reusable kind of scientific work and the graph of
steps inside it. The current modules have the same names as the configuration
classes. Each module lineage owns one namespace under
`derivatives/nro/MODULE/LINEAGE_ID/`.

The resolved BIDS data and the workflow fully determine a module's graph. nro
builds the whole graph before it checks freshness. Whether files exist, or are
fresh, stale, or invalid, can decide whether a declared step runs. It never
changes which steps exist or how they depend on each other.

All scientific module packages live under `nro/modules/`. Scientific
construction belongs in `nro/modules/<module>/module.py`, and the code that
builds work items for the planner belongs in
`nro/modules/<module>/planning.py`.

### Work item

A **work item** is one concrete, schedulable application of a module lineage.
Its identity combines the project, the module lineage, the selected data, and
the applicable BIDS entities. For example:

> `clean` for participant `t20`, one BOLD run, `fsnative` space, and 2 mm
> smoothing under a particular `clean` module lineage.

Work items run concurrently when their dependencies allow. A general worker
normally runs every step of a work item. A step that needs a different resource
class, such as a GPU, can be handed to a matching worker pool while the parent
work item waits. That step still belongs to the same fixed runner graph and
artifact contract.

### Step

A **step** is one immutable declaration of a scientific transform. It names its
exact inputs, its exact outputs, and one action. A step may depend on other
steps in the same module.

Every step is added to the module graph before freshness is evaluated. At
runtime the runner makes one decision for it: execute or skip. Several related
child commands may share one step only when they form one atomic operation on
an artifact, with one boundary for freshness and recovery.

Inputs create the data-dependency edges between steps. A step may also declare
scientific parameters that affect its result without naming a file. When an
input or a declared parameter changes, the runner reruns that step and its
descendants and leaves independent parts of the graph alone.

### Runner and runner graph

The **runner** is `nro.orchestration.runner.Runner`, the single engine that
executes steps inside a work item. Every scientific module uses it, and no
module defines its own runner.

Each runner creates and owns one **runner graph**, a `RunnerGraph`. Module code
adds steps through the runner. The runner freezes the graph, checks freshness,
then walks the graph in dependency order and records each execute-or-skip
decision.

The division of labor is as follows. The module defines the scientific graph,
the runner applies common execution rules to it, and the planner schedules
complete work items. The scheduler may send a step that declares a specific
resource to a matching worker. That step does not become a separate work item.

### Artifact and product

An **artifact** is the exact set of public outputs that one work item promises.
nro assesses completion and freshness against this set. An artifact cannot be
scheduled and has no worker of its own.

A **product** is one concrete file or atomic directory within an artifact.
Private intermediates under `WORK` are neither artifacts nor products. They
help interrupted work resume, and nro can regenerate them when needed.

One lineage namespace, `derivatives/nro/MODULE/LINEAGE_ID/`, can hold artifacts
from many work items. Participant and session directories organize the
products. Such a directory is not an artifact unless a work-item contract
declares it as an atomic product. An artifact is therefore neither the whole
lineage namespace nor, necessarily, one subject or session directory.

The registry reports the combined filesystem condition of an artifact as its
**artifact state**: `missing`, `stale`, or `fresh`.

Python interfaces, command-line output, ownership receipts, and registry tables
all use these terms directly.

## The orchestration hierarchy

These roles carry a user's request through to executed steps.

```text
user request
  creates demand for terminal work items
  expands to demand for their upstream work items

planner
  constructs and registers complete WorkItemSpecs
  assesses whether demanded work items need execution

worker
  claims one ready work item
  creates one attempt
  launches its immutable ExecutionEnvelope

runner
  executes or skips the work item's declared steps
```

### Request and demand

A **request** records that a user wants one or more terminal work items. The
planner expands it to include their upstream dependencies.

**Demand** is the active link between a request and a work item. Several
requests can demand the same work item. Cancelling one request does not cancel
work that another request still demands.

### Planner

The **planner** turns selections and workflows into complete graphs of work
items. It registers those graphs, evaluates demand, and feeds the shared worker
pool. Its knowledge of the science comes from the built-in module catalog and
the planning functions in each module.

### Registry

The registry has two coordinated parts.

The shared **scheduler registry** stores requests, attempts, workers, Slurm
submissions, and the general and GPU concurrency limits.

Each development branch has a **scientific registry**. It stores the branch's
discovered source tree, compiled workflows, work items, dependencies, and
artifact observations. Keeping these apart lets the scheduler coordinate every
branch without importing branch code.

Each registry controls locking and transactions for its own state. Neither one
is a second planner. The registries are the authority on orchestration
identity, demand, and history. The filesystem is the authority on whether
derivative files exist and match their completion records.

Ordinary source discovery does not create derivative work items or demand.
Registry bootstrap and repair go further: they also find existing files in
nro-controlled derivative locations and register the work items that own them.
They still create no demand.

### Ownership receipts

Each owned derivative configuration root contains `.nro/lineage.json`. It also
contains one receipt for every work item it owns, stored in the storage-only
`.nro/work_items/` directory. Together these records keep the module lineage,
work-item identity, dependencies, and artifact contract alongside the
derivative.

Registry repair reads the receipts before it resolves workflows from the
current definitions store. Removing or renaming a workflow therefore does not
hide its existing derivatives.

Owning a work item does not mean nro can recompute it. The planner can request
a lineage only when a current workflow selects it. If no current workflow
selects a historical work item's lineage, status reports it as `Unavailable`
when it is not fresh. A fresh historical work item stays `Success`, because
its artifact is still usable. Purge can select either case without the
original workflow.

### Worker and attempt

A **worker** is a reusable Slurm or local process. It claims one ready work
item at a time and supervises it. Workers are cluster infrastructure, not
scientific modules.

An **attempt** is one historical execution of a work item by a worker. A work
item can have several attempts, for example because it was interrupted, failed,
became stale, or was requested again. A failed attempt is recorded in history.
It does not permanently change the work item.

## Terms that are easy to confuse

| Terms | Distinction |
|---|---|
| Configuration class / module | A configuration class defines the accepted settings. The module defines the work graph. Their names currently correspond one to one. |
| Module configuration / module lineage | A module configuration holds local settings. A module lineage adds every upstream configuration. |
| Module / work item | A module is the reusable graph definition. A work item applies one module lineage to selected data and entities. |
| Work item / artifact | A work item is schedulable work. Its artifact is the exact public output set it promises. |
| Artifact / product | An artifact is a work item's public output set. A product is one file or atomic directory within it. |
| Workflow / module | A workflow selects configurations. A module performs scientific computation. |
| Planner / runner | The planner schedules work items. The runner executes steps inside one work item. |
| Request / attempt | A request expresses demand. An attempt records one execution. |
| Space / smoothing | Both select work items from `clean` onward. nro filenames use the nonstandard `smoothing` entity because BIDS `scale` describes atlas granularity. |

[Work-item planning and execution](work-item-lifecycle.md) defines the records
the planner uses for each work item.
