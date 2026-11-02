# Running derivative workflows

This page is for users who request processing with `nro run` and want to know
what nro does with that request. It covers how selectors become work, how nro
decides what to rerun, and how work reaches CPU and GPU workers. It applies to
the `anat`, `func`, `clean`, `dynconn`, `microparcellation`, `networks`, and
`contrasts` modules. [Orchestration design](orchestration-design.md) documents
the internals for developers.

## Requesting work

A request names the work you want with selectors. Selectors narrow projects,
participants, modules, workflows, runs, spaces, smoothing levels, tasks, and
models. An omitted selector means all applicable values, with one exception:
when a new `nro run` request omits the workflow, nro selects `main`. A bare
request therefore does not run every defined workflow.

```bash
nro run -p t20 -P nptl -m networks -w main -s fsnative -S 2
```

nro compiles the requested terminal work items and everything they depend on.
Selecting an endpoint together with one of its upstream modules does not
create duplicate demand. A request that names no module takes its endpoints
from each selected workflow. In `main`, the endpoints are `dynconn`,
`networks`, and `contrasts`. Contrasts uses model set `main` unless the request
selects another model or model set. The networks configuration chooses
`dynconn` or `microparcellation` as its source, and that choice is fixed before
planning.

From `clean` onward, space and smoothing are separate work-item entities.
Giving several values requests every combination. nro does not register
combinations that no request uses.

Exact run selectors take the form `--run ENTITY=VALUE[,VALUE...]`. Values for
one entity are alternatives, and different entities must all match. For
example, `--run task=Rest,Movie run=01` selects run 01 of either task.

Commands that act on work already in the registry do not expand selectors this
way. `run --resume` treats every omitted selector as all matching registered
work, including the workflow. Both `run --resume` and `status --update` keep
each work item's recorded workflow and entity values. Registry repair starts
from derivatives that already exist. None of these operations combine selector
values taken from different registry rows.

[Work commands](commands/work.md) lists every selector, execution resource,
status mode, cancellation rule, and repair boundary.

## What happens after you submit

`nro run` starts the scheduler if it is not already running, or joins the one
that is. The scheduler queues planning requests and handles them one at a time
in the order they arrive. Each request is planned from the requesting branch's
pinned source in a separate subprocess, which does not open the scheduler
registry. Separate scheduler threads keep coordinating workers while planning
runs. After the last activity, the scheduler stays available for 12 hours
before it exits.

The planner returns complete work-item specifications before any work is
submitted. Workers then ask the scheduler for ready work items and run each one
in its own subprocess. Inside a work item, the module builds its complete
runner graph before checking freshness. The shared runner executes or skips
each declared step and writes the completion record.

### Reuse of earlier plans

Planning a participant can take time, so nro caches each participant's plan.
The cache key covers the compiled workflow and module lineages, the relevant
first-level models and markup, protected site definitions, the execution
source, and the participant's source-BIDS inventory and metadata. Repeating an
equivalent request recovers the compiled dependency graph instead of
rebuilding every module plan. Later requests also see cache entries and
scientific revisions written by earlier ones, so they compile only scopes that
changed or are new.

Adding, removing, or changing relevant source metadata invalidates the entry.
For imaging files, planning fingerprints the header and the inventory, while
workers and authoritative assessment still check artifact inputs by content.
The cache only speeds up planning. It never decides whether an artifact is
fresh.

### Reuse of earlier results

When the scheduler admits a request, it reassesses only fresh artifacts whose
scientific contracts could satisfy the submitted graph, along with their
dependency closure. Missing, stale, and unrelated historical records cannot be
reused, so nro does not examine them.

Periodic maintenance applies the same check, limited to active attempts,
pending resource-specific steps, and their upstream closure. Queued work never
waits on a scan of unrelated history. A full filesystem reconciliation happens
only when you run `nro status --update`.

## Workers and resources

The worker pool has three resource classes: general, long-CPU, and GPU.

General and long-CPU workers claim complete work items and share the global CPU
concurrency limit. The long-CPU class is reserved for calibrated MSMAll
anatomy. Its workers start with 48 hours, eight CPUs, and at least 64 GB of
memory.

GPU workers run single steps. When a runner reaches a step marked for GPU
execution, it yields. The scheduler records a durable step task, releases the
general worker, and starts a GPU worker once that task is ready. The GPU worker
rebuilds the same fixed module graph, verifies the same inputs and contract,
and runs only the named step. The parent work item then returns to the front of
the general queue and resumes from the fresh step outputs. GPU workers never
claim complete work items, and they exit as soon as no GPU step is ready.

CPU and GPU pools have separate limits. The site setting `concurrency` limits
general derivative and ingestion work. `gpu_concurrency` limits GPU step tasks
and defaults to one. Resource classes, device identifiers, and handoffs affect
only how work runs. They do not change scientific contracts or freshness.

A GPU handoff leaves upstream validation unchanged. Every step follows the same
freshness and publication rules. A claim records the current upstream
generations. nro rejects completion if the work-item contract, its generation,
or those captured dependencies changed while the work ran on either the CPU or
the GPU.

## Configuration and freshness

Workflows select one configuration ID per configuration class from the external
[definitions store](definitions.md). Resolution produces a complete scientific
configuration and a separate execution snapshot. Compiled definitions that are
equivalent share work even when their YAML is formatted differently.
[Workflows and configuration](configuration.md) explains which settings are
scientific.

A work-item contract records the substantive configuration, direct inputs,
dependency topology and generations, processing policy, and required outputs.
Command formatting, scheduler resources, Git identity, and source-capture paths
do not affect freshness. The filesystem remains the authority on whether
outputs exist and satisfy their contracts.

By default, `nro status` reports saved state without checking files.
`nro status --update` performs the authoritative assessment and updates the
registry. It may cancel an active attempt whose contract is obsolete.
[Work-item planning and execution](work-item-lifecycle.md) describes the full
state model.

## Requests from development branches

A development checkout sends its compiled graph and execution recipe to the
installed central scheduler. The scheduler finds compatible artifacts from
ancestor branches and assigns output paths that the branch owns. Each attempt
runs from the source and site settings captured at submission, so one worker
can serve different branches in turn without keeping any branch's imports. See
[development](development.md#requests-and-execution) for inheritance and
cross-branch invalidation.

## Direct module execution

The installed `nro` command is the normal entry point. For diagnosis, each
module can also be run directly:

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
python -m nro.modules.contrasts -p t20 -P nptl -w main \
  --task langlocSN --model main --space fsnative --smoothing 2
```

These commands use the same configuration compiler, execution context, runner,
and artifact contracts as scheduled work. Scientific results and freshness
rules are the same either way.

## Scheduler internals

Most users can skip this section. It describes where the scheduler keeps its
state, which matters when you inspect the control root or diagnose the
scheduler itself.

The configured private control root has this layout:

```text
CONTROL/
  shared/scheduler/registry.sqlite3
  shared/scheduler/service/{progress,status.json,status-static-*,status-dynamic-*}
  branches/<branch-id>/registry.sqlite3
```

The scheduler database is the only authority for requests, attempts, workers,
Slurm submissions, and the concurrency limit. Its only long-lived client is a
Slurm controller job that runs while the scheduler is active. Commands and
workers send it direct TCP requests. A request that can be retried stays in
SQLite until its result commits.

Each branch database records that branch's compiled scientific contracts,
workflow lineages, and artifact observations. All projects and branches share
worker capacity this way, and the scheduler never imports development code.

Cached status reads an atomic JSON snapshot. The `status.json` manifest
combines a reusable static graph with a small dynamic-state component, so
routine execution changes do not rewrite the whole cached report.

The scheduler keeps current state, not an unbounded history of requests and
allocations. It preserves active demand, the provenance of existing
derivatives, unresolved diagnostics, and the dependency closures of all three.
Records for finished publications store only terminal identities, and plans for
reconciling inherited inputs are stored compressed.

Periodic compaction runs in transactions. It removes superseded attempts and
finished worker and Slurm records. It keeps every started attempt on a work
item's current revision, so `ELAPSED` in `nro status` includes interrupted and
retried attempts. Incremental vacuuming then reclaims the freed pages
gradually.
