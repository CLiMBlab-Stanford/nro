# Request, inspect, and stop work

[Common selectors](index.md#shared-selectors) apply to every command here except `set`.

## `nro run`

Plan requested terminal work items, register upstream demand, assess outputs,
and supply workers. Fresh intermediates can be skipped even when the work item
needs execution. No upfront catalog of all space/smoothing combinations is needed.
The command admits demand without starting a persistent service. If ready work
needs worker capacity, it starts the ephemeral scheduler controller and sends
requests directly over TCP. Concurrent invocations share the same controller.
The controller remains available for 12 hours after work and workers become
idle, then exits. This avoids another Slurm wait when work is requested later
the same day. Users do not manage it. `--no-submit` never starts the controller.

Omitting `--module` requests every endpoint derived from the selected workflow.
For `main`, these are `dynconn`, `networks`, and `firstlevels`, with shared
dependencies registered once. Firstlevels selects model set `main` unless a
model or set is specified. Participants without a
matching task model can still run the connectivity paths. Explicit `--module`
restricts the endpoints requested; it does not request their downstream modules.

Selecting an endpoint together with its dependencies creates no extra upstream
demand when the endpoint covers those same work items. For example,
`-m microparcellation networks` is equivalent to `-m networks`. If an endpoint
uses only some upstream runs, an explicit upstream selection keeps demand for
the remaining runs. This reduction applies within each project and workflow
in one invocation; it does not cancel earlier requests.

```bash
nro run -P nptl -p t20 -m networks -w main -s fsnative -S 2
```

For firstlevels, select task models independently of the workflow:

```bash
nro run -m firstlevels --task langlocSN --model-set main
```

Model sets affect request selection, not artifact freshness. Existing demand
is not cancelled when a model leaves a set. See [task models](../task-models.md).

New requests capture source code and resolved site settings for execution.
Editing the checkout after submission does not change that captured source;
editing it during planning rejects the request so it can be retried. Source
capture does not freeze Python environments or container images. See the
[development limits](../development.md#branch-isolation-work) before maintaining
dependencies or using a separate checkout with the shared pool.

| Additional option | Behavior/default |
| --- | --- |
| `--concurrency N` | Shared limit, default 50. |
| `--cpus N` | CPUs and scientific-process threads per worker, default 2. |
| `--time HOURS` | Worker allocation duration, default 24 hours. |
| `--memory GB` | Initial worker memory, default 32 GB. |
| `--max-memory GB` | Maximum retry memory tier, default 256 GB. |
| `--partition`, `--account` | Site-configured Slurm routing. |
| `--worker-idle-timeout SECONDS` | Idle exit delay, default 30 seconds. |
| `--drain-minutes N` | Stop claiming work before allocation expiry, default 15. |
| `--local` | Execute a worker locally, without submitting a Slurm allocation. |
| `--no-submit` | Register/assess demand without launching workers. |
| `--no-inherit` | In a development branch, compute matching work locally instead of reusing ancestor artifacts. |
| `--resume` | Recreate demand for matching resumable work already known to the registry. |
| `--json` | Structured planning result. |
| `--repair` | Rebuild this branch's scientific registry after central activation; see below. |

Inheritance is enabled by default. `--no-inherit` affects only the new request;
it does not delete ancestor outputs or change the registered branch tree. The
option is unavailable before branch execution is activated.

Use `--resume` when the original sequence of requests is inconvenient to
reconstruct. A bare invocation selects registered work with status `Queued`,
`Waiting`, `Stopped`, `Timeout`, or `Error`. It also selects `Missing`, `Stale`, or
`Blocked` work that still has demand. Those three states do not create demand
on their own.
The shared selectors narrow the selection; omitted selectors mean all existing
resumable work rather than the usual workflow endpoints and default target.

Resume replans only the selected work item identities against the current BIDS
data and definitions, then captures the current execution source. It does not
request work that has never carried demand. If an old identity no longer exists
under the current scientific definitions, the command reports that it cannot
be resumed instead of substituting different work. Add `--no-submit` to restore
demand without supplying workers.

With an activated central scheduler, `--repair` covers the current branch across
all projects. It restores scientific records from the admitted graph history,
cancels that branch's demand, and asks before stopping its active attempts. Other
branches and the shared worker pool are retained. The old scientific database is
copied to `registry-before-repair.sqlite3`. This does not replace the shared
scheduler database or recompute derivatives.
Shared scheduler replacement is a separate
[main-maintainer operation](releases.md#shared-scheduler-repair).

Before central activation, `--repair` is site-wide even when selectors name one project. It asks before
stopping active workers, rebuilds private orchestration state, discovers BIDS
sources and existing owned derivatives, and creates no new demand. It does
not rebuild scientific outputs merely to repair registry state. Historical
attempt/request information in the replaced registry is not retained as active
state. Branch registrations, captured execution source and site settings, and
ingestion records are retained. Do not use repair as routine error recovery or
in unattended scripts.

That pre-activation recovery starts from derivatives on disk. Ownership records restore their
work items directly, including work items whose original workflow no longer exists.
For unrecorded files in a current workflow's output directories, repair plans only
the matching module, participant, run, model, space, and smoothing, plus required
upstream work items. It does not plan unrelated source participants or unused
combinations of spaces and smoothing levels. Source discovery still catalogs all
projects and participants; it does not create demand.

## `nro status`

Report registered work items, including errors and blocked-dependency summaries.
Source discovery alone does not create status rows; existing owned artifacts
can be adopted without demand. An empty registry prints headers without rows.

By default, status uses the last atomic scheduler snapshot without starting a
controller or checking files.
`--update` performs the full assessment, updates the registry, and reports the
result. It uses the live controller or a fenced local update process, without
submitting a controller to Slurm. Use `--json` for structured output or
`--no-pager` to bypass `less`.
The pager uses colors and pinned headers when supported.

After central activation, `--update` first recompiles registered selections
using the invoking checkout's scientific code, without creating demand. Central
code then verifies the resulting contracts and filesystem evidence. Default
status performs no recompilation or filesystem assessment.
Historical lineages no longer represented by current workflows retain their
registered contracts.

`Success` describes fresh artifacts. For registered work without active demand
or an active or unresolved failed attempt, `Missing` means required outputs or
completion evidence are absent. `Stale` means the result needs updating, for
example because its contract or an upstream dependency changed. The JSON `reason`
field gives the specific cause; `Stale` does not guarantee all files still exist.
`Corrupt` means files exist but contradict their own artifact contract or
completion evidence. A matching request rebuilds the artifact; successful
publication replaces the corrupt state.
Both reporting modes use the same status labels. `Queued` indicates demanded
work whose dependencies are complete and which is waiting for worker capacity.
`Waiting` indicates demanded work held behind unfinished dependencies.
`Blocked` indicates demanded work held behind an upstream error.
`Running` and `Error` describe current execution or an unresolved failed attempt.
`Timeout` means Slurm ended the worker allocation at its wall-time limit. Resume
that work with a larger `--time` value when the same limit would be insufficient.
`Stopping` means cancellation is awaiting worker confirmation. `Stopped` means
the latest attempt ended because a user cancelled its demand; it is not an
execution failure. A new matching `nro run` request returns that work to the
queue.
`Unavailable` can describe historical lineages no current workflow can request.
Inspect the reason field and upstream/downstream error summaries rather than
inferring filesystem state from submission state alone.

The `LINEAGE` column identifies the complete configured route to a module.
Pass `--show-lineages` to append the route and registered workflow associations
for each displayed work item. Use `-i`/`--lineage` to select an exact route.
Qualify a repeated ID as `MODULE/ID`. A workflow can reach a work item, but it
does not own that work item.

## `nro log`

Browse the current attempt logs for matching work items in `less`. Pass
`--worker` to browse the Slurm worker logs instead. Logs separate command stdout
and stderr; a message on stderr is not by itself a failed command. Worker logs
are not one-to-one with derivatives because workers execute multiple work items.
Pass `--running` to keep only matching work that currently has a running attempt.
The option composes with the standard selectors.

Use `nro log -m bidsify` to browse the dedicated logs for matching bidsification
requests. In this command, `bidsify` is a logging selector, not a scientific
module. Project and participant selectors apply, and `-r ses=LABEL` selects a
BIDS session. `--worker` does not change bidsification log selection.
For bidsification, `--running` keeps only sessions with an active ingestion stage.

## `nro stop`

Cancel selected demand and, by default, its downstream dependents. Other users'
demand can keep shared work alive. `--only` limits cancellation to the selected
derivative. `-f`/`--force` cancels matching demand from all users and stops the
shared attempt; it does not delete outputs or history.

`-W`/`--workers` shuts down the current user's site-wide worker pool without
cancelling work item demand. Worker shutdown and demand cancellation are distinct
operations. Coordinate with other users before installation maintenance.

## `nro set`

```bash
nro set ls
nro get
nro get concurrency gpu_concurrency
nro set concurrency=60
nro set gpu_concurrency=2
```

`nro set ls` lists every accepted name, its value constraint, its scope, and its
effect. Add `--json` for structured output. Updates accept one or more
`NAME=VALUE` pairs. `concurrency` updates active requests and
limits the general worker pool. `gpu_concurrency` is a persistent scheduler
setting, defaults to one, and limits resource-specific GPU steps independently.
Unsupported keys warn and are skipped. Malformed pairs or invalid recognized
values fail. `--json` returns the update result. The global site configuration
selects the registry and BIDS context.

Changing a limit does not create demand or immediately kill or launch workers.
Workers observe it during normal check-in; a later `nro run --resume` or worker
completion supplies newly useful capacity. Updating `concurrency` fails when no
active request can be changed. `gpu_concurrency` may be set before a GPU step is
ready.

`nro get` returns every current setting as `NAME=VALUE`; positional names limit
the result to those settings. `--json` returns a `settings` object instead.
Because `concurrency` is defined by active derivative and ingestion requests, it
is `unset` (`null` in JSON) when no active request supplies a limit.
