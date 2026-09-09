# Design and scope

nro supports repeated processing of many runs and participants on a shared
Linux cluster. It separates scientific computation from scheduling so modules
can describe their work without implementing worker management, retry logic,
or a second system of dependency tracking.

## Two dependency graphs

The planner builds a graph of **instances**: one participant's anatomy, one
functional run, or one subject-level connectivity result. Workers claim ready
instances from the site-wide registry. Within each instance, a shared `Runner`
owns a graph of `Step` declarations and executes them in dependency order.

```text
anat ──► func ──► clean ──┬─► dynconn
  │                       └─► microparcellation ──► networks
  └──────────── anatomical geometry and transforms ─────────►
```

The sequence is not a chain of monolithic cluster jobs. There are many run-wise
`func` and `clean` instances, followed by subject-level aggregation. From
`clean` onward, space and smoothing are independent scheduling entities.
The module pages specify additional direct anatomy dependencies.

`firstlevels` forms a separate branch from `func`, with a direct `anat`
dependency for geometry. One participant/model/space/smoothing instance fits its
runs independently and follows its Stats Models graph to publish within-subject
summaries. It does not consume `clean` or run population-level analyses.

Graphs are defined by BIDS inputs and resolved configuration before execution.
Freshness determines which declared steps execute, not which steps exist.
Factories return `Step` objects; only `Runner.add_step()` mutates its graph.
Module constructors must not execute image processing while building the graph.

## Reuse and ownership

Workflows select configurations for derivative classes. `anat` and `func`
share the `preprocessing` class. Configuration lineages include upstream
choices, allowing equivalent workflows to share results. Directory labels are
not reliable substitutes for lineage identity.

Public artifacts are durable results under a project's `derivatives` tree.
Private intermediates live under WORK and support resumption. Ownership receipts
keep historical results discoverable after a workflow is renamed or removed.
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
authoritative for output existence and validity. Several requests can share an
instance; cancelling one request need not stop work still demanded by another.
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
filename entity, and subject-level CIFTI scenes use an nro-specific layout.
Do not infer metadata from a directory name when a manifest supplies it.

The default deployment uses Slurm, Singularity or Apptainer, and shared storage.
`run --local` executes a worker locally but still needs processing dependencies
and the configured filesystem. [Core concepts](concepts.md) defines the terms;
[instance lifecycle](instance-lifecycle.md) explains the typed records.
