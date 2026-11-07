# Request, inspect, and stop work

This page covers the commands for day-to-day processing: requesting
derivatives with `nro run`, following them with `nro status` and `nro log`,
and cancelling them with `nro stop`. All of these commands take the
[shared selectors](index.md#shared-selectors). `nro start` and the site
execution settings at the end of the page do not.

## `nro start`

`nro start` starts the shared scheduler without requesting any scientific work.
If a scheduler is already running, or its Slurm allocation is pending, the
command reports that state and exits without changing the allocation.

When the command creates a scheduler, `--partition`, `--account`,
`--scheduler-time-hours`, `--scheduler-memory-gb`, and `--scheduler-cpus` set
its Slurm allocation. Omitted options keep the site settings, which default to
24 hours, 8 GB, and 4 CPUs. `nro run` and `nro bidsify` accept the same options
for the scheduler they start. See [Slurm options](index.md#slurm-options).

```bash
nro start --partition sphinx --account nlp --scheduler-time-hours 24 \
  --scheduler-memory-gb 4 --scheduler-cpus 4
```

## `nro run`

`nro run` requests derivatives. It plans the selected terminal work items,
registers demand for them and for their upstream dependencies, checks which
outputs already exist and are fresh, and supplies workers for the rest. A fresh
intermediate is skipped even when a later work item that depends on it must
run. You do not need to list every space and smoothing combination in advance.

```bash
nro run -P nptl -p t20 -m networks -w main -s fsnative -S 2
```

### What a request selects

When `--module` is omitted, the request covers every endpoint of the selected
workflow. For `main` the endpoints are `dynconn`, `networks`, and `contrasts`.
Their shared dependencies are registered once. Participants without a matching
first-level model still run the connectivity modules.

An explicit `--module` restricts the endpoints. It does not add the modules
downstream of the one you name. Naming an endpoint together with its
dependencies adds no extra upstream demand when the endpoint already uses those
same work items, so `-m microparcellation networks` is equivalent to
`-m networks`. When an endpoint uses only some upstream runs, naming the
upstream module explicitly keeps demand for the remaining runs. This reduction
applies within each project and workflow of one invocation. It does not cancel
earlier requests.

For contrasts, select first-level models independently of the workflow:

```bash
nro run -m contrasts --analysis language --model-set main
```

Contrasts requests use model set `main` unless you name a model or a set.
Model sets affect which work a request selects. They do not affect whether an
artifact is fresh, and removing a model from a set does not cancel its existing
demand. See [first-level models](../task-models.md).

Most selectors match everything when omitted. A new request is the exception
for `--workflow`, which defaults to `main` so that a bare request does not
expand across every defined workflow. The space and smoothing defaults also
apply only to new requests. `--resume`, described below, uses neither default.

`--project` and `--participant` accept glob and regular-expression patterns.
See [shared selectors](index.md#shared-selectors).

### Options

| Option | Behavior and default |
| --- | --- |
| `--concurrency N` | Shared worker limit. Defaults to the site setting `execution.concurrency`. |
| `--worker-cpus N` | CPUs per worker, which also sets the thread count of scientific processes. Default 2. |
| `--worker-time-hours N` | Worker allocation length. Default 24 hours. |
| `--worker-memory-gb N` | Initial worker memory. Default 32 GB. |
| `--worker-max-memory-gb N` | Highest memory tier used for retries. Default 256 GB. |
| `--long-worker-time-hours N`, `--long-worker-cpus N` | Allocation for long CPU workers, which run work such as MSMAll anatomy. Defaults 48 hours and 8 CPUs. |
| `--partition`, `--account` | Slurm routing for workers and for a scheduler this command starts. Defaults come from site settings. |
| `--scheduler-time-hours N`, `--scheduler-memory-gb N`, `--scheduler-cpus N` | Allocation for the scheduler, used only when this command starts one. |
| `--worker-idle-timeout SECONDS` | How long an idle worker waits before exiting. Default 30 seconds. |
| `--drain-minutes N` | How long before allocation expiry a worker stops claiming work. Default 15. |
| `--local` | Run one worker locally instead of submitting a Slurm allocation. |
| `--no-submit` | Register and assess demand without launching workers. |
| `--no-inherit` | In a development branch, compute matching work locally instead of reusing ancestor artifacts. |
| `--resume` | Restore demand for matching resumable work already in the registry. |
| `--repair` | Rebuild registry records without creating demand. See below. |
| `--json` | Print the planning result as JSON. |

The Slurm defaults come from the site's `slurm.` settings, and the execution
defaults from its `execution.` settings, so a site can change them. Each Slurm
option matches its site key, as described in [Slurm options](index.md#slurm-options).

### How a request is executed

The terminal that runs `nro run` waits for its planning request to finish, but
it does no module planning itself. The request goes to the scheduler, which
handles planning requests in FIFO order. It runs each branch's pinned planner
in an isolated child process while its other threads keep coordinating workers.
Concurrent invocations share one scheduler.

If ready work needs worker capacity, `nro run` starts the scheduler controller
when none is running and sends requests to it over TCP. You do not manage the
controller. It stays available for 12 hours after work and workers become idle
and then exits, so a later request the same day does not wait for another Slurm
allocation.

`--no-submit` registers and plans demand but launches no derivative workers.
If the scheduler controller is running, the request goes to it. Otherwise
`nro run` starts a one-shot scheduler process that admits and plans the request
and then exits, without starting the long-lived controller. Either way the
command requires an activated main scheduler installation.

Each new request captures the source code and resolved site settings it will
run with. Editing the checkout after submission does not change the captured
source. Editing it during planning rejects the request, and you can retry.
Source capture does not freeze Python environments or container images. Read
the [development limits](../development.md#branch-isolation-work) before you
change dependencies or use a separate checkout with the shared pool.

The first request for a participant and a scientific selection compiles the
full dependency closure and saves it as a content-addressed planning entry.
Later equivalent requests reuse that entry. A new entry is selected
automatically after changes to any of these inputs: the source BIDS inventory or
metadata, imaging headers, compiled definitions, first-level models, markup,
protected site settings, or execution source. File timestamps can cause
conservative replanning, but they never make a derivative stale. Freshness
depends on artifact contracts and content evidence.

A request validates only the reusable artifacts in the submitted graph. It does
not reassess every historical work item in the project. Run
`nro status --update` when you want a full reconciliation of the registry with
the filesystem.

### Inheritance in development branches

Inheritance is on by default. `--no-inherit` affects only the new request. It
does not delete ancestor outputs or change the registered branch tree. The
option is unavailable until branch execution is activated.

### Resuming work

Use `--resume` when you cannot easily reconstruct the original sequence of
requests. A bare `nro run --resume` selects registered work with status
`Queued`, `Waiting`, `Stopped`, `Timeout`, or `Error`. It also selects work in
`Missing`, `Stale`, or `Blocked` that still has demand. Those three states do
not create demand on their own.

The shared selectors narrow the selection. Omitted selectors match all
resumable work and do not apply the workflow and target defaults of a new
request. Add `--workflow main` to limit resumption to that workflow.

Resume replans the selected work item identities against the current BIDS data
and definitions, then captures the current execution source. It never requests
work that has never had demand. If an old identity no longer exists under the
current scientific definitions, the command reports that it cannot resume it
instead of substituting different work. Add `--no-submit` to restore demand
without supplying workers. `--resume` cannot be combined with `--repair`.

### Repairing the registry

`--repair` rebuilds registry records from durable state and creates no new
demand. It always covers all projects, and it rejects every selector option.
Do not use it for routine error recovery or in unattended scripts. What it
repairs depends on the checkout and on whether the central scheduler is active.

In a development branch, `--repair` rebuilds that branch's scientific registry
from the admitted graph history. It cancels the branch's demand and asks before
stopping its active attempts. Other branches and the shared worker pool keep
running. The old scientific database is copied to
`registry-before-repair.sqlite3`. The repair does not replace the shared
scheduler database or recompute derivatives.

In the main checkout after central activation, `--repair` repairs the lab-wide
scheduler and asks before stopping the worker pool. This is a
[main-maintainer operation](releases.md#shared-scheduler-repair).

Before central activation, `--repair` asks before stopping active workers. It
then rebuilds private orchestration state and discovers BIDS sources and
existing owned derivatives. It does not rebuild scientific outputs merely to
repair registry state. Branch registrations, captured execution source and site
settings, and ingestion records are kept. Attempt and request history from the
replaced registry is not kept as active state.

This pre-activation recovery starts from the derivatives on disk. Ownership
records restore their work items directly, including work items whose original
workflow no longer exists. For unrecorded files in the output directories of a
current workflow, repair plans only the matching module, participant, run,
model, space, and smoothing, plus their required upstream work items. It does
not plan unrelated participants or unused combinations of space and smoothing.
Source discovery still catalogs every project and participant, but it creates
no demand.

## `nro status`

`nro status` reports registered work items with their states, errors, and
summaries of blocked dependencies. An empty registry prints the column headers
and no rows. Discovering source data does not by itself create rows, and
existing owned artifacts can be adopted without demand.

By default the command reads the last atomic scheduler snapshot. It does not
start a controller, check files, or recompile anything. `--update` runs a full
assessment, updates the registry, and reports the result. It uses the live
controller if there is one and otherwise a fenced local process. It never
submits a controller to Slurm.

After central activation, `--update` first recompiles the registered selections
with the scientific code of the invoking checkout, without creating demand.
Central code then verifies the resulting contracts against filesystem evidence.
Historical lineages that no current workflow represents keep their registered
contracts.

Output goes through `less`, with colors and pinned headers when the terminal
supports them. `--no-pager` prints directly, and `--json` prints structured
output.

### Reading the report

Rows are ordered by project, then participant, then module dependency depth.
Independent modules at the same depth appear in lexical order. Within a module,
lineage and the displayed entity values break ties. Status derives the module
order from the complete visible dependency graph, so selectors do not change
it.

The `ELAPSED` column shows the execution time accumulated by the current
scientific revision of a work item. It includes CPU and GPU attempts, retries,
and resource handoffs, among them attempts interrupted when a worker allocation
expired. It excludes time spent queued or waiting on dependencies. JSON output
reports the same value as `elapsed_seconds`.

The `LINEAGE` column identifies the complete configured route to a module.
`--show-lineages` appends the route and its registered workflow associations to
each row. Select an exact route with `-i`/`--lineage`, and write `MODULE/ID`
when an ID repeats. A workflow can reach a work item, but it does not own it.

Both reporting modes use the same states:

| State | Meaning |
| --- | --- |
| `Success` | The artifacts are fresh. |
| `Queued` | Demanded work whose dependencies are complete, waiting for worker capacity. |
| `Waiting` | Demanded work held behind unfinished dependencies. |
| `Blocked` | Demanded work held behind an upstream error. |
| `Running` | An attempt is executing. |
| `Error` | The latest attempt failed and the failure is unresolved. |
| `Timeout` | The work item reached its own execution limit. |
| `Stopping` | Cancellation is waiting for the worker to confirm. |
| `Stopped` | A user cancelled the demand for the latest attempt. |
| `Missing` | Required outputs or completion evidence are absent. |
| `Stale` | The result needs updating, for example because its contract or an upstream dependency changed. |
| `Corrupt` | Files exist but contradict their own artifact contract or completion evidence. |
| `Unavailable` | A historical lineage that no current workflow can request. |

`Missing` and `Stale` apply to registered work that has no active demand and no
active or unresolved failed attempt. `Stale` does not mean all the files still
exist. The JSON `reason` field gives the specific cause. A matching request
rebuilds a `Corrupt` artifact, and the corrupt state clears once the new result
is published.

`Timeout` refers only to a limit on the work item. When a reusable worker
reaches its Slurm wall time, its current attempt is interrupted and the work
item returns to `Queued`, because the allocation's lifetime is not a property
of the work item. `Stopped` is not an execution failure, and a new matching
`nro run` request returns the work to the queue.

Read the `reason` field and the upstream and downstream error summaries before
drawing conclusions about the filesystem. The submission state alone does not
tell you which files exist.

## `nro log`

`nro log` opens the current attempt logs for matching work items in `less`.
`--worker` opens the Slurm worker logs instead. Worker logs do not map one to
one onto derivatives, because each worker runs several work items. `--running`
keeps only matching work that has a running attempt, and it combines with the
standard selectors.

Logs keep a command's stdout and stderr separate. A message on stderr does not
by itself mean the command failed.

`nro log -m bidsify` opens the logs of matching bidsification requests. Here
`bidsify` selects ingestion logs and is not a scientific module. Project and
participant selectors apply, and `-r ses=LABEL` selects a BIDS session.
`--worker` does not change which bidsification logs are shown. With
`--running`, only sessions with an active ingestion stage are shown.

## `nro stop`

`nro stop` cancels your demand for the selected work and, by default, for its
downstream dependents. Demand from other users can keep shared work running.
`--only` limits cancellation to the selected derivative. `-f`/`--force` cancels
matching demand from all users and stops the shared attempt. Neither form
deletes outputs or history.

Two global controls stop work regardless of demand. `-W`/`--worker` stops every
worker and every active work-item attempt, and `--workers` is an alias.
`--scheduler` does the same and then stops the scheduler. Combining `--worker`
and `--scheduler` is the same as `--scheduler`. These controls do not accept
selectors, `--only`, or `--force`, and they do not remove demand or completed
outputs. They affect every user, so coordinate before using them.

## Site execution policy

The site stores durable concurrency limits for the shared worker pool:

```bash
nro site ls execution
nro site get execution.concurrency execution.gpu_concurrency
nro site set execution.concurrency=60 execution.gpu_concurrency=2
```

`execution.concurrency` defaults to 50 and `execution.gpu_concurrency` to 1.
`nro site set` validates and saves all assignments atomically, and it rejects
unknown keys. A live scheduler receives changes to these two limits
immediately. Without a scheduler, the saved values apply when coordination
resumes.

Changing a limit does not create demand or stop running workers. Workers see
the new limit during normal scheduler coordination. `nro run --concurrency`
overrides the limit for one invocation without changing the saved value.
