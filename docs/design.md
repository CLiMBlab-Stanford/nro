# Design and scope

nro supports repeated processing of many runs and participants on a shared
Linux cluster. It separates scientific computation from scheduling so modules
can describe their work without implementing worker management, retry logic,
or a second system of dependency tracking.

## Two dependency graphs

The planner builds a graph of **work items**: one participant's anatomy, one
functional run, or one subject-level connectivity result. Workers claim ready
work items from the site-wide registry. Within each work item, a shared `Runner`
owns a graph of `Step` declarations and executes them in dependency order.

```text
anat ──► func ──┬─► clean ──┬─► dynconn ────────────┐
  │             │           └─► microparcellation ──┴─► networks
  │             └─► firstlevels
  └── direct anatomical inputs ──► {clean, microparcellation, networks, firstlevels}
```

The sequence is not a chain of monolithic cluster jobs. There are many run-wise
`func` and `clean` work items, followed by subject-level aggregation. From
`clean` onward, space and smoothing are independent scheduling entities.
The module pages specify how `clean`, `microparcellation`, `networks`, and
`firstlevels` use their direct anatomy dependencies. Microparcellation needs
that dependency only for targets whose native geometry or mask comes from the
participant anatomy.

A networks configuration selects either dynconn or microparcellation as its
source. Workflow compilation fixes that choice before planning; it does not
create a runtime-dependent graph or demand the unselected source.

`firstlevels` follows a separate path from `func`, with a direct `anat`
dependency for geometry. One participant/model/space/smoothing work item fits its
runs independently and follows its Stats Models graph to publish within-subject
summaries. It does not consume `clean` or run population-level analyses.

Graphs are defined by BIDS inputs and resolved configuration before execution.
Freshness determines which declared steps execute, not which steps exist.
Factories return `Step` objects. Larger construction stages return immutable
`StagePlan` values containing steps and typed downstream products. Only the module
entry point passes these declarations to `Runner.add_step()` or
`Runner.add_steps()`. Module constructors must not execute image processing while
building the graph.

## Reuse and ownership

Workflows select one configuration for each module. Every module publishes below
`derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`. Functional variants that select equivalent
anatomy reuse the same anatomical work item and `anat` directory. Configuration
lineages include upstream choices, allowing equivalent workflows to share
results. A short lineage digest distinguishes configurations with the same name
but incompatible upstream inputs. Because the label is content-addressed, a
registry rebuild cannot rename a lineage by discovering it in a different
order. Directory labels are readable projections of lineage identity, not
substitutes for the full fingerprint.

Public artifacts are durable results under a project's `derivatives/nro` tree.
Private intermediates live under WORK and support resumption. Ownership receipts
form a hidden, lineage-local recovery index; SQLite remains authoritative during
normal operation. Registry reconstruction starts from receipts backed by declared
public outputs and retains the dependency receipts needed to describe those
artifacts. Purge removes unreferenced receipts and empty directories, but preserves
a missing upstream receipt while a surviving artifact still depends on it.
Completion requires the current output contract, not merely a directory or a
successful historical attempt.

Changes in configuration, inputs, dependencies, or output contracts can make
artifacts stale. Scheduler resources and command formatting are not scientific
changes. nro does not automatically infer that arbitrary code edits are
scientifically equivalent; maintainers must update substantive contracts when
they change methods or required metadata.

## Shared operation

The central scheduler registry stores demand, attempts, workers, and Slurm
submissions. Each registered Git branch also has a scientific registry for its
compiled contracts and artifact observations. The filesystem remains
authoritative for output existence and validity. Several requests can share a
work item; cancelling one request need not stop work still demanded by another.
Concurrency limits apply across projects and development branches.

An installed `main` release runs the scheduler. Registered development
checkouts submit compiled work through that scheduler, inherit compatible
ancestor artifacts, and write new outputs to branch-owned directories. Each
attempt runs from captured source selected at submission, so workers can serve
different branches without importing their modules into the scheduler. See
[branch registration](commands/branches.md) and
[development](development.md#branch-isolation-work).

## Naming and portability

Names preserve BIDS entities when practical, but working derivatives are not
claimed to be BIDS-validator compliant. In particular, `smoothing` is an nro
filename entity, and generated Workbench scenes use an nro-specific layout.
Do not infer metadata from a directory name when a manifest supplies it.

The default deployment uses Slurm, Singularity or Apptainer, and shared storage.
`run --local` executes a worker locally but still needs processing dependencies
and the configured filesystem. [Core concepts](concepts.md) defines the terms;
[work-item lifecycle](work-item-lifecycle.md) explains the typed records.
