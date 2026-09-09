# Branch registration

`nro branch` registers Git branches and their checkouts with the shared nro
installation. Each registered branch has one scientific registry. Every
attached checkout of that branch resolves to the same database, regardless of
where the checkout lives.

After an approved main scheduler is activated, `run`, `status`, `log`, and
`stop` use the same selectors in development checkouts. The branch compiles its
scientific graph; a separate process using the central installation admits and
schedules it. Registration alone does not activate a scheduler. See
[development](../development.md#branch-isolation-work) for deployment requirements.

## Register and attach

From a checkout on `dev`:

```bash
nro branch register
nro branch show
```

From a new feature checkout:

```bash
nro branch register --parent dev
```

The branch name comes from Git. The command does not create a Git branch,
commit code, or switch branches. Feature branches default to parent `dev`.
`main` and `dev` are reserved on first registration, even if their checkouts do
not exist yet. Their parent relationship is fixed: `main -> dev`.

To use another checkout of an already registered branch:

```bash
nro branch attach --checkout /path/to/another/checkout
```

`register` rejects an existing branch name and directs the user to `attach`.
Attaching checks the checkout's actual Git branch. Detached checkouts and
checkout paths already attached to another branch are rejected. Merely creating
or checking out a Git branch does not register it with nro.

## Inspect and maintain

```bash
nro branch list
nro branch show feature/example --json
nro branch reparent feature/example --parent dev
nro branch retire feature/example
```

`show` without a name checks the current checkout's registration. `--checkout`
selects another checkout. `--json` provides structured output for every action;
`--bids-root` selects the shared registry context as it does for other commands.
`list` does not initialize an absent store.

Reparenting changes the declared inheritance tree, not Git history or artifact
locations. Cycles and parents outside the `main -> dev` spine are rejected.
Retirement preserves the branch name and database but rejects further checkout
authorization. Run either mutation from an attached checkout of the branch being
changed. When Git repository identity is available, attaching a checkout from a
different repository is rejected. These checks prevent routing mistakes among
trusted contributors; they are not an access-control boundary.

Reparent active children before retiring their parent. Retirement immediately
cancels the branch's demand and queued or running attempts. It also invalidates
affected readers and reconciles retained requests against the new topology.
Neither operation moves or deletes derivatives. With an activated scheduler,
topology changes run through its serialized maintenance interface.

## Development definitions

A branch installation reads the shared definitions store by default. Its
`create`, `edit`, and `delete` commands cannot change that store. To experiment
with configurations, workflows, models, or events, clone the definitions
repository to a separate directory and select it:

```bash
nro branch definitions --definitions /path/to/development-definitions
```

The directory must already contain a valid nro definitions store. Alternatively,
`nro definitions create /path/to/development-definitions` creates a starter store
without selecting it or copying the lab's definitions. Selection is central:
every installed checkout of the same branch uses the selected directory. No
files are copied or Git operations performed by the selection command.

The selected store must not overlap shared definitions, private control state,
or another branch's selected store. nro also rejects symbolic-link redirection.
Definition editing is limited to files inside the selected store.

To return to shared, read-only definitions:

```bash
nro branch definitions --shared
```

Selection does not change scientific observations. Subsequent planning compares
compiled definitions by content, using the existing freshness rules. Changing a
store path alone does not mark artifacts stale. Source and site capture records
the selected path; it does not snapshot the definitions directory.

## Storage and safeguards

The shared control directory contains the catalog and all branch databases:

```text
CONTROL/
  shared/
    branches.json
    scheduler/
      registry.sqlite3
      workers/
    ingestion/
    promotions/
    cache/
      implementations/
      execution-sites/
  branches/
    main/
      registry.sqlite3
      events/
    dev/registry.sqlite3
    feature%2Fexample/registry.sqlite3
```

`CONTROL/shared/scheduler/registry.sqlite3` is the production registry and sole
worker-pool authority. It currently holds production scientific state as well.
The new branch databases hold scientific contracts and validator observations;
they have no workers, submissions, execution claims, or concurrency settings.
Registration alone neither initializes another scheduler nor submits work.

The site setting `development` controls the separate development output tree.
Its lab default is `/juice6/u/nlp/climblab/NRO_DEV`; outside the lab the proposed
default is `~/nro/development`. Development outputs use
`development/BRANCH_ID/{BIDS,WORK}/PROJECT/derivatives`. Scientific inputs always
start with the shared site's raw BIDS data.

Requests retain their complete compiled graph, including computations hidden
behind inherited artifacts. If an inherited input becomes unavailable, workers
reconsider that graph and compute the needed work locally. They do not request
an ancestor rebuild or load the branch's scientific modules. A newer admitted
scientific revision rejects an older delayed handoff from another checkout.

Branch status and logs show that branch's registered selections. `stop` cancels
only that branch's requests, including with `--force`; shared producer work
requested by another branch remains intact. Inherited artifacts remain read-only.

`ControlPaths` supplies this layout to installation, registry, ingestion,
branch registration, and execution-cache code. Branch directories also hold
manifests and scientific configuration snapshots when used by processing.
Shared worker logs remain with the scheduler; ingestion and caches are site-wide.
With central execution activated, `run --repair` rebuilds only the current
branch's scientific database. It preserves the shared scheduler, other branches'
demand, ingestion, and public outputs. The previous database is retained as
`registry-before-repair.sqlite3`; admitted records supply the replacement.

nro rejects the previous flat layout before creating another scheduler. No
automatic relocation or alternate-path fallback exists; use
[nro cutover](cutover.md) as a separate maintenance step when retaining old state.

Each database is bound to its branch name, registration identity, and shared
control path. nro rejects a misplaced database or symlink redirection. It does
not search checkouts for local databases, merge conflicting copies, or silently
recreate a missing registered database. Registration identities and database
schema versions are not scientific freshness inputs.

Catalog edits use a shared filesystem lock and reject changes based on an
outdated catalog revision. A new branch becomes visible only after its database
exists. Interrupted publication keeps a pending record; the next registration
initialization resumes it with the same identity. Scientific record updates
also check the revision that the caller read, so competing checkouts cannot
silently replace one another's edits. Verified observations carry that revision
through central assessment and cannot be attached to a replacement contract.

Production registry repair preserves the branch catalog and databases. A
branch's scientific schema is checked when reading its scientific records, not
when inspecting its identity or registering another branch. A schema change in
one branch therefore does not require repairing every branch.

These checks protect trusted developers against mistakes. They are not an
access-control boundary against someone who can directly modify shared files.
