# Request, inspect, and stop work

[Common selectors](index.md#shared-selectors) apply to every command here except `set`.

## `nro run`

Plan requested terminal instances, register upstream demand, assess outputs,
and supply workers. Fresh intermediates can be skipped even when the instance
needs execution. No upfront catalog of all space/smoothing combinations is needed.

Omitting `--module` requests every workflow endpoint, currently `dynconn`,
`networks`, and `firstlevels`, with shared dependencies registered once. Firstlevels selects
model set `main` unless a model or set is specified. Participants without a
matching task model can still run the connectivity branches. Explicit `--module`
restricts the endpoints requested; it does not request their downstream modules.

Selecting an endpoint together with its dependencies creates no extra upstream
demand when the endpoint covers those same instances. For example,
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
| `--cpus N` | CPUs per Slurm worker, default 2. |
| `--time HOURS` | Worker allocation duration, default 24 hours. |
| `--memory GB` | Initial worker memory, default 32 GB. |
| `--max-memory GB` | Maximum retry memory tier, default 256 GB. |
| `--partition`, `--account` | Site-configured Slurm routing. |
| `--worker-idle-timeout SECONDS` | Idle exit delay, default 30 seconds. |
| `--drain-minutes N` | Stop claiming work before allocation expiry, default 15. |
| `--local` | Execute a worker locally, without submitting a Slurm allocation. |
| `--no-submit` | Register/assess demand without launching workers. |
| `--no-inherit` | In a development branch, compute matching work locally instead of reusing ancestor artifacts. |
| `--json` | Structured planning result. |
| `--repair` | Rebuild this branch's scientific registry after central activation; see below. |

Inheritance is enabled by default. `--no-inherit` affects only the new request;
it does not delete ancestor outputs or change the registered branch tree. The
option is unavailable before branch execution is activated.

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
instances directly, including instances whose original workflow no longer exists.
For unrecorded files in a current workflow's output directories, repair plans only
the matching module, participant, run, model, space, and smoothing, plus required
upstream instances. It does not plan unrelated source participants or unused
combinations of spaces and smoothing levels. Source discovery still catalogs all
projects and participants; it does not create demand.

## `nro status`

Report registered instances, including errors and blocked-dependency summaries.
Source discovery alone does not create status rows; existing owned artifacts
can be adopted without demand. An empty registry prints headers without rows.

`--cached` uses saved state only. Default mode predicts reassessment with cheaper
checks and does not persist it. `--verify` uses the full assessment method,
updates the registry, and reports the result. `--cached` and `--verify` are
mutually exclusive. Use `--json` for structured output or `--no-pager` to bypass
`less`. The pager uses colors and pinned headers when supported.

After central activation, verification first recompiles registered selections
using the invoking checkout's scientific code, without creating demand. Central
code then verifies the resulting contracts and filesystem evidence. Default
status previews processing-policy changes locally without publishing them.
Historical lineages no longer represented by current workflows retain their
registered contracts.

`Success` describes fresh artifacts. For registered work without active demand
or an active or unresolved failed attempt, `Missing` means required outputs or
completion evidence are absent. `Stale` means the result needs updating, for
example because its contract or an upstream dependency changed. The JSON `reason`
field gives the specific cause; `Stale` does not guarantee all files still exist.
All three reporting modes use the same status labels. `Queued` indicates pending
demanded work, while `Blocked` indicates work waiting on upstream errors.
`Running` and `Error` describe current execution or an unresolved failed attempt.
`Stopping` means cancellation is awaiting worker confirmation. `Stopped` means
the latest attempt ended because a user cancelled its demand; it is not an
execution failure. A new matching `nro run` request returns that work to the
queue.
`Unavailable` can describe historical lineages no current workflow can request.
Inspect the reason field and upstream/downstream error summaries rather than
inferring filesystem state from submission state alone.

## `nro log`

Browse matching worker logs in `less`. `-i`/`--instance-level` instead selects
the current attempt's instance logs. Logs separate command stdout and stderr;
a message on stderr is not by itself a failed command. Worker logs are not
one-to-one with derivatives because workers execute multiple instances.

## `nro stop`

Cancel selected demand and, by default, its downstream dependents. Other users'
demand can keep shared work alive. `--only` limits cancellation to the selected
derivative. `-f`/`--force` cancels matching demand from all users and stops the
shared attempt; it does not delete outputs or history.

`-W`/`--workers` shuts down the current user's site-wide worker pool without
cancelling instance demand. Worker shutdown and demand cancellation are distinct
operations. Coordinate with other users before installation maintenance.

## `nro set`

```bash
nro set concurrency=100
```

Accept one or more `NAME=VALUE` pairs. Currently only positive integer
`concurrency` is recognized; unsupported keys warn and are skipped. Malformed
pairs or invalid recognized values fail. `--json` returns the update result;
`--bids-root` selects the project-root context.

The update applies to active requests, not a permanent installation default.
It fails when no active requests can be updated. Workers observe the new limit
during normal check-in and claiming; changing the setting is not an immediate
kill/relaunch reconciliation operation.
