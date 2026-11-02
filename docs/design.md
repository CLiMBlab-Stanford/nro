# How nro works

This page is for users who want a working picture of nro before they run it or
change its configuration. It covers how nro divides processing into work, when
it reuses earlier results, how several users share one scheduler, and how
outputs are named. [Core concepts](concepts.md) defines each term precisely.

nro is built for repeated processing of many runs and participants on a shared
Linux cluster. Scientific modules describe what to compute. A separate
scheduler decides when and where that computation runs, retries it after
failures, and tracks dependencies between results. Module code therefore does
not manage workers or keep its own dependency records.

## Two dependency graphs

nro plans work at two levels.

The outer graph connects **work items**. A work item is one schedulable piece
of processing, such as one participant's anatomy, one functional run, or one
participant's connectivity result. The planner builds this graph and registers
it in a site-wide registry. Workers then claim work items as their dependencies
finish.

The inner graph lives inside a single work item. Each module declares its
processing as `Step` objects, and a shared `Runner` executes those steps in
dependency order.

The outer graph for one participant looks like this:

```text
anat ──► func ──┬─► clean ──┬─► dynconn ────────────┐
  │             │           └─► microparcellation ──┴─► networks
  │             └─► contrasts
  └── direct anatomical inputs ──► {clean, microparcellation, networks, contrasts}
```

Each arrow can stand for many work items. nro creates one `func` and one
`clean` work item per run, and later modules aggregate runs at the subject
level. From `clean` onward, each space and smoothing level is a separate work
item, so a request for two spaces schedules two independent branches.

Several modules also read anatomy directly. The module pages describe how
`clean`, `microparcellation`, `networks`, and `contrasts` use that input.
`microparcellation` needs it only for targets whose native geometry or mask
comes from the participant's anatomy.

`networks` reads from either `dynconn` or `microparcellation`. The networks
configuration chooses one, and that choice is fixed when the workflow is
compiled, before planning. nro does not demand the source it did not choose,
and the graph never changes shape at runtime.

`contrasts` runs on `func` outputs and reads anatomy for geometry. It does not
use `clean`. Each contrasts work item covers one participant, model, space, and
smoothing level. It fits each run independently, then follows its
[Stats Models](task-models.md) graph to publish within-subject summaries. It
does not run population-level analyses.

Both graphs are fixed before anything runs. The BIDS inputs and the resolved
configuration decide which steps exist. Existing outputs only decide whether
each step runs or is skipped.

For module authors, this has a concrete consequence. Factories return `Step`
objects, and larger construction stages return immutable `StagePlan` values
that hold steps and typed downstream products. Only the module entry point
passes these declarations to `Runner.add_step()` or `Runner.add_steps()`.
Module constructors must not process images while they build the graph.

## Reuse of earlier results

A workflow selects one configuration for each module. Each module writes its
results below `derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`.

The lineage digest records every upstream configuration choice as well as the
module's own. Two workflows that make the same choices for a module and
everything upstream of it use the same directory and share its results. For
example, functional variants that select the same anatomy configuration reuse
one `anat` work item and one `anat` directory. When two configurations share
a name but depend on incompatible upstream inputs, the digest gives them
different directories.

The digest is computed from content, so rebuilding the registry cannot rename a
lineage because it found lineages in a different order. The directory label is
a readable summary of the lineage identity. It does not replace the full
fingerprint that nro stores.

nro writes two kinds of output. Public artifacts are the durable results under
a project's `derivatives/nro` tree. Private intermediates live under WORK and
let interrupted work resume.

A work item is complete only when its outputs satisfy the current output
contract. An existing directory or a past successful attempt is not enough.
Results become stale when their configuration, inputs, dependencies, or output
contract change. Scheduler resources and command formatting do not count as
scientific changes.

nro cannot tell whether an arbitrary code edit leaves results scientifically
equivalent. When maintainers change a method or its required metadata, they
must update the affected contracts themselves.

### Ownership receipts

Each lineage directory holds hidden ownership receipts. These receipts let nro
rebuild its registry from the files on disk. The SQLite registry remains the
authority during normal operation.

Reconstruction starts from receipts whose declared public outputs exist. It
keeps the dependency receipts needed to describe those artifacts. Purge removes
receipts that nothing references and deletes empty directories. It keeps a
receipt for a missing upstream result while a surviving artifact still depends
on it.

## Shared operation

One scheduler serves every project and every development branch at a site. Its
central registry records demand for work, attempts, workers, and Slurm
submissions. Each registered Git branch also has its own scientific registry,
which stores that branch's compiled contracts and observations of its
artifacts. The filesystem remains the authority on whether outputs exist and
are valid.

Several requests can share a work item. Cancelling one request does not stop
work that another request still needs. Concurrency limits apply across all
projects and branches together.

The scheduler runs from an installed `main` release. A registered development
checkout submits its compiled work to that scheduler. It reuses compatible
artifacts from its ancestor branches and writes new outputs to directories that
the branch owns. Each attempt runs from source captured at submission, so one
worker can serve different branches in turn without loading their code into the
scheduler. See [branch registration](commands/branches.md) and
[development](development.md#branch-isolation-work).

## Naming and portability

Output names keep BIDS entities where practical, but nro does not claim that
its working derivatives pass the BIDS validator. For example, `smoothing` is an
nro-specific filename entity, and generated Workbench scenes use an nro layout.
When a manifest supplies metadata, read it from the manifest instead of parsing
a directory name.

The default deployment uses Slurm, Singularity or Apptainer, and shared
storage. `run --local` runs a worker on the current machine. It still needs the
processing dependencies and the configured filesystem.

[Running derivative workflows](orchestration.md) explains what happens when you
request work. [Work-item planning and execution](work-item-lifecycle.md)
describes the typed records behind each work item.
