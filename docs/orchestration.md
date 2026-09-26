# Running derivative workflows

The orchestration layer turns user selections into scientific work, reuses
compatible artifacts, and runs ready work items through one shared worker pool.
It supports `anat`, `func`, `clean`, `dynconn`, `microparcellation`, `networks`,
and `firstlevels`.

The configured private control root keeps scheduler state and branch scientific
records in this layout:

```text
CONTROL/
  shared/scheduler/registry.sqlite3
  shared/scheduler/service/{progress,status.json}
  branches/<branch-id>/registry.sqlite3
```

The scheduler database is the sole authority for requests, attempts, workers,
Slurm submissions, and the concurrency limit. An ephemeral Slurm controller is
its only long-lived client. Commands and workers use direct TCP requests while
the controller is active. Retryable requests remain in SQLite until their result
commits. Cached status reads an atomic JSON snapshot.
A branch
database records that branch's compiled scientific contracts, workflow
lineages, and artifact observations. All projects and branches therefore share
worker capacity without requiring the scheduler to import development code.

The worker pool has separate general and GPU resource classes. General workers
claim complete work items. A runner may yield at a step marked for GPU
execution; the scheduler then records a durable step task, releases the general
worker, and starts a GPU worker only when that task is ready. The GPU worker
reconstructs the same fixed module graph, verifies the same inputs and contract,
and executes only the named step. The parent work item then returns to the front
of the general queue and resumes from its fresh step outputs.

GPU workers never claim complete work items and exit as soon as no GPU step is
ready. General and GPU pools have independent concurrency limits. `concurrency`
limits general derivative and ingestion work; `gpu_concurrency` defaults to one
and limits GPU step tasks. Resource classes, device identifiers, and handoffs
are execution policy. They do not change scientific contracts or freshness.

## Requests

The common selector interface narrows projects, participants, modules,
workflows, runs, spaces, smoothing levels, tasks, and models. Omitted selectors
mean all applicable values. A new `nro run` request is the workflow exception:
an omitted workflow selects `main` so that a bare request does not run every
defined workflow. For example:

```bash
nro run -p t20 -P nptl -m networks -w main -s fsnative -S 2
```

The planner compiles the requested terminal work items and their dependency
closure. Selecting an endpoint together with an upstream dependency does not
duplicate demand. A bare invocation derives the endpoints from each selected
workflow. In `main`, these are `dynconn`, `networks`, and `firstlevels`;
firstlevels uses model set `main` unless the request selects another model or
set. A networks configuration can select dynconn or microparcellation as its
source without making graph construction runtime-dependent.

From `clean` onward, space and smoothing are independent work item entities.
Multiple values request their cross-product. The planner does not register
unused combinations in advance. Exact run selectors use
`--run ENTITY=VALUE[,VALUE...]`; alternatives for one entity are OR choices,
while different entities are combined with AND.

Commands that operate on registered state do not repeat this expansion.
`run --resume` treats every omitted selector, including workflow, as all
matching registered work. It and `status --update` preserve each work item's
recorded workflow and entity values. Registry repair works backward from
derivatives that exist. These operations never combine selector values
collected from different rows.

See [work commands](commands/work.md) for every selector, execution resource,
status mode, cancellation rule, and repair boundary.

## Planning and execution

The planner registers complete work item specifications before submitting work.
Workers request ready work items from the controller and run each one in a
separate subprocess. A module constructs its complete runner graph before
freshness is checked. The shared runner then executes or skips each declared
step and writes the completion record.

Every step uses the same freshness and publication rules. A GPU handoff does
not strengthen or weaken upstream validation: claims capture the current
upstream generations, and completion is rejected if the work-item contract,
generation, or captured dependencies changed during either CPU or GPU
execution.

Development requests send their compiled graph and execution recipe to the
installed central scheduler. The scheduler resolves compatible ancestor
artifacts and branch-owned output paths. The attempt runs from captured source
and site settings chosen at submission; a worker can therefore serve different
branches in succession without retaining a branch's imports. See
[development](development.md#requests-and-execution) for inheritance and
cross-branch invalidation.

## Configuration and freshness

Workflows select one configuration ID per configuration class from the external
[definitions store](definitions.md). Resolution produces a complete scientific
configuration and a separate execution snapshot. Equivalent compiled
definitions share work even when their source YAML differs in formatting.

Work item contracts record substantive configuration, direct inputs, dependency
topology and generations, processing policy, and required outputs. Command
formatting, scheduler resources, Git identity, and source-capture paths are not
freshness inputs. The filesystem remains authoritative for whether outputs
exist and satisfy their contracts.

By default, `nro status` reports saved state without checking files.
`nro status --update` performs the authoritative assessment and updates the
registry; it may cancel an active attempt whose contract is obsolete. See
[work item planning and
execution](work-item-lifecycle.md) for the complete state model.

## Direct module execution

The installed `nro` interface is the normal entry point. Direct scientific
entry points remain available for diagnosis:

```bash
python -m nro.modules.anat -p t20 -P nptl -w main
python -m nro.modules.func -p t20 -P nptl -w main \
  --run task=Rest run=01
python -m nro.modules.clean -p t20 -P nptl -w main \
  --space fsnative --smoothing 2 --run task=Rest run=01
python -m nro.modules.dynconn -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.microparcellation -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.networks -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.firstlevels -p t20 -P nptl -w main \
  --task langlocSN --model main --space fsnative --smoothing 2
```

These commands use the same configuration compiler, execution context, runner,
and artifact contracts as scheduled work. They do not define alternate
scientific or freshness behavior.
