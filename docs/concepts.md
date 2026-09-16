# Core concepts

This document defines the language used to describe `nro`. These terms are
architectural contracts: code, tests, logs, and other documentation should use
them consistently.

## The scientific hierarchy

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

### Configuration class and configuration

A **configuration class** is the parameter namespace for one scientific module.
The current classes are `anat`, `func`, `clean`, `dynconn`,
`microparcellation`, `networks`, and `firstlevels`.

A **module configuration** is the complete set of local parameters for one
configuration class. It has a stable human-readable ID. Configuration files
contain the authority; Python does not maintain a second default parameter map.

A **module lineage** identifies one module configuration together with the
complete upstream closure of module configurations on which it depends. A
lineage is independent of participant, run, space, smoothing, and other data
entities. Several workflows can select the same lineage, and equivalent
lineages share a derivative namespace. A substantive configuration change
changes the applicable work-item contract and invalidates affected results when
they are reassessed.

Anatomy lineage depends only on the `anat` configuration. Functional lineages
that select the same anatomy reuse it. Each downstream lineage includes its
local configuration ID and upstream lineage, so incompatible inputs cannot
write into the same namespace.

### Workflow

A **workflow** selects one configuration ID for each configuration class. It
defines a coherent path from source data through the available modules. A
workflow does not contain scientific code and is not a runtime sequence of
steps.

### Module

A **module** defines one reusable kind of scientific work and its internal DAG.
The current modules are `anat`, `func`, `clean`, `dynconn`,
`microparcellation`, `networks`, and `firstlevels`. A module lineage owns one
namespace under `derivatives/nro/MODULE/LINEAGE_ID/`.

A module's graph is determined entirely by resolved BIDS data and its workflow.
It is fully constructed before freshness is examined. Existing, missing,
fresh, stale, or invalid files may affect whether declared steps run, but may
never change which steps exist or how they depend on one another.

All scientific module packages live under `nro/modules/`. Their scientific
construction belongs in `nro/modules/<module>/module.py`; planner-facing
work-item construction belongs in `nro/modules/<module>/planning.py`.

### Work item

A **work item** is one concrete, schedulable application of a module lineage.
Its identity includes its project, module lineage, selected data, and applicable
BIDS entities. For example:

> `clean` for participant `t20`, one BOLD run, `fsnative` space, and 2 mm
> smoothing under a particular `clean` module lineage.

Work items can run concurrently when their dependencies permit it. Steps within
one work item are not independently scheduled to cluster workers.

### Step

A **step** is one immutable declaration of a scientific transform. It declares
exact inputs, exact outputs, and one action. A step may depend on other steps in
the same module.

A step is always added to the module graph before freshness is evaluated. At
runtime it receives exactly one decision: execute or skip. Closely related
child commands may share a step only when they form one atomic artifact
operation with one freshness and recovery boundary.

Inputs create data-dependency edges. A step may also declare scientific
parameters that affect its transform without naming a file. If an input or
scientific declaration changes, the runner reruns that step and its descendants
while leaving independent subgraphs untouched.

### Runner and runner graph

The **runner** is the one shared work-item-internal execution engine,
`nro.orchestration.runner.Runner`. Every scientific module uses it; modules do
not define their own runner implementations.

Each runner creates and owns one **runner graph**, represented by
`RunnerGraph`. Module code adds steps through the runner. The runner freezes the
graph before checking freshness, then traverses it in dependency order and
records every execute-or-skip decision.

The module defines the scientific DAG. The runner implements its common
execution semantics. The planner schedules complete work items, not runner
steps.

### Artifact and product

An **artifact** is the exact set of public outputs promised by one work item.
It is the durable output boundary against which completion and freshness are
assessed. An artifact is not schedulable and has no independent worker.

A **product** is one concrete file or atomic directory within an artifact. A
private intermediary under `WORK` is neither an artifact nor a product. It may
support resumption and can be regenerated when needed.

The namespace `derivatives/nro/MODULE/LINEAGE_ID/` can contain artifacts from
many work items. Participant and session directories organize their products;
those directories are not themselves artifacts unless a work-item contract
declares one as an atomic product. An artifact is therefore neither the entire
lineage namespace nor necessarily one subject or session directory.

The registry reports an artifact's collective filesystem condition as its
**artifact state** (`missing`, `stale`, or `fresh`).

Python interfaces, command-line output, ownership receipts, and registry tables
use this vocabulary directly.

## Orchestration hierarchy

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

### Planner

The **planner** turns selections and workflows into complete work-item graphs,
registers them, evaluates demand, and supplies the shared worker pool. Its
scientific knowledge comes from the explicit built-in module catalog and the
module-local planning functions.

### Registry

The registry has two coordinated parts. The shared scheduler registry stores
requests, attempts, workers, Slurm submissions, and the site-wide concurrency
limit. Each development branch has a scientific registry that stores its
discovered source tree, compiled workflows, work items, dependencies, and
artifact observations. This separation lets the scheduler coordinate all
branches without importing their code.

Ordinary source discovery does not create derivative work items or demand.
Registry bootstrap and repair additionally discover existing files in
nro-controlled derivative locations and register the work items that own them,
without creating demand. The registries are the locking and transaction
authorities for their respective state; neither is a second planner
implementation.

The filesystem remains authoritative for whether derivative files currently
exist and match their completion records. The registries are authoritative for
orchestration identity, demand, and history.

Each owned derivative configuration root contains `.nro/lineage.json` and one
receipt under the storage-only `.nro/work_items/` directory for every work item
it owns. These records keep
the module lineage, work-item identity, dependencies, and artifact
contract with the derivative. Registry repair reads them before resolving the
workflows in the current definitions store. Removing or renaming a workflow
therefore does not make its existing derivatives invisible.

Ownership does not imply that a work item can be recomputed. A current workflow
must select its module lineage before the planner can request it. Status reports
a nonfresh historical work item as `Unavailable` when no current workflow
selects that lineage. A fresh historical work item remains `Success`
because its artifact is still usable. Purge can select either case without the
originating workflow.

### Request and demand

A **request** records a user's desire to obtain one or more terminal work items.
The planner expands it over upstream dependencies.

**Demand** is the active association between a request and a work item. Several
requests may share demand for the same work item. Cancelling one request does
not cancel work still demanded by another.

### Worker and attempt

A **worker** is a reusable Slurm or local process that claims and supervises one
ready work item at a time. It is cluster infrastructure, not a scientific
module.

An **attempt** is one historical execution of a work item by a worker. A work
item may have several attempts because it was interrupted, failed, became stale,
or was explicitly requested again. Attempt failure is history; it does not
permanently redefine the work item.

## Terms that are not interchangeable

| Terms | Distinction |
|---|---|
| Configuration class / module | A configuration class defines the accepted settings; the module defines the work graph. Their names currently correspond one-to-one. |
| Module configuration / module lineage | A module configuration contains local settings; a module lineage adds the complete upstream configuration closure. |
| Module / work item | A module is the reusable DAG definition; a work item applies one module lineage to selected data and entities. |
| Work item / artifact | A work item is schedulable work; its artifact is the exact public output set it promises. |
| Artifact / product | An artifact is a work item's public output set; a product is one file or atomic directory within it. |
| Workflow / module | A workflow selects configurations; a module performs scientific computation. |
| Planner / runner | The planner schedules work items; the runner executes steps inside one work item. |
| Request / attempt | A request expresses demand; an attempt records one execution. |
| Space / smoothing | Both select work items from `clean` onward. nro filenames use the nonstandard `smoothing` entity because BIDS `scale` describes atlas granularity. |

The planner-facing record hierarchy is defined separately in
[Work-item planning and execution](work-item-lifecycle.md).
