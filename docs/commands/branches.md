# Branch registration

Use `nro branch` to register a Git branch and its checkouts with the shared nro
installation, so that you can run processing from a development checkout.
Each registered branch has its own scientific registry. Every checkout attached
to the branch uses that same database, wherever the checkout lives.

Once a main scheduler release is installed and activated, `run`, `status`,
`log`, and `stop` accept the same selectors in a development checkout as on
`main`. The branch compiles its own scientific graph, and a separate process
running the central installation admits and schedules the work. Registering a
branch does not activate a scheduler. See
[development](../development.md#branch-isolation-work) for deployment
requirements, and [Develop on a branch](../branching-tutorial.md) for a
complete Git and nro workflow.

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

nro takes the branch name from Git. It does not create a Git branch, commit
code, or switch branches, and creating or checking out a Git branch does not
register it with nro. Feature branches default to the parent `dev`. The names
`main` and `dev` are reserved on first registration, even if their checkouts do
not exist yet, and their relationship is fixed as `main -> dev`.

`register` rejects a name that is already registered and tells you to use
`attach`. To use another checkout of a registered branch:

```bash
nro branch attach --checkout /path/to/another/checkout
```

Attaching checks which Git branch the checkout is actually on. nro rejects
detached checkouts and checkouts already attached to another branch. When the
Git repository identity is available, it also rejects a checkout from a
different repository.

## Inspect and maintain

```bash
nro branch list
nro branch show feature/example --json
nro branch reparent feature/example --parent dev
nro branch retire feature/example
```

`show` without a name reports the current checkout's registration, and
`--checkout` selects another checkout. Every action accepts `--json` for
structured output. `list` does not create the store if it is absent. The global
site configuration selects the shared registry and BIDS context.

Run `reparent` and `retire` from an attached checkout of the branch you are
changing. Reparenting changes the declared inheritance tree. It does not change
Git history or where artifacts are stored. Cycles, and parents outside the
`main -> dev` spine, are rejected.

Retiring keeps the branch name and database but stops authorizing its
checkouts. It immediately cancels the branch's demand and its queued or running
attempts, invalidates affected readers, and reconciles retained requests
against the new tree. Reparent any active children before retiring their
parent. Neither operation moves or deletes derivatives. With an activated
scheduler, both run through the scheduler's serialized maintenance interface.

These checks prevent routing mistakes among trusted contributors. They are not
an access-control boundary.

## Development definitions

By default a branch reads the shared definitions store, and `nro def edit` and
`nro def rm` cannot change it from a branch. To experiment with configurations,
workflows, models, or events, clone the definitions repository to a separate
directory and select it:

```bash
nro branch definitions --definitions /path/to/development-definitions
```

The directory must already hold a valid nro definitions store and must not
contain `site/site.yml`. The branch inherits that protected document from the
shared repository. Other definitions resolve through the registered branch
tree: the current branch first, then each parent from nearest to farthest, then
the shared repository. A private store therefore needs only its overrides. To
start one from the packaged starter instead of the lab's definitions, run
`nro def init /path/to/development-definitions`, then select it.

The selection is central, so every installed checkout of the branch uses the
selected directory. The selection command copies no files and runs no Git
operations. Create and publish branch definitions with nro's
[authoring commands](authoring.md). The store manifest rejects direct edits.
Installing the branch may migrate its private store in place, but never
modifies the shared store or a parent's private store.

The selected store must not overlap the shared definitions, private control
state, or another branch's store. nro also rejects symbolic-link redirection
and any private copy of the protected site document. Definition edits are
limited to files inside the selected store.

To return to the shared, read-only definitions:

```bash
nro branch definitions --shared
```

`--definitions` and `--shared` cannot be combined.

Changing the selection does not change any scientific observation. Later
planning compares compiled definitions by content under the usual freshness
rules, so changing the store path alone does not make artifacts stale. Source
capture records the nearest branch definitions path, and parent lookup follows
the central branch tree. The execution-site snapshot always comes from the
shared protected document. Neither copies a definitions directory.

## Running work on a branch

Status and logs on a branch show that branch's registered selections. `stop`
cancels only that branch's requests, including with `--force`. Shared producer
work requested by another branch keeps running. Inherited artifacts are
read-only.

Development outputs go to a separate tree set by the site setting
`development`. The lab default is `/juice6/u/nlp/climblab/NRO_DEV`, and the
proposed default outside the lab is `~/nro/development`. Outputs are written to
`development/BRANCH_ID/{BIDS,WORK}/PROJECT/derivatives`. Scientific inputs
always start from the shared site's raw BIDS data.

A request keeps its complete compiled graph, including computations hidden
behind inherited artifacts. If an inherited input becomes unavailable, workers
reconsider that graph and compute the missing work for the branch. They do not
ask an ancestor branch to rebuild, and they do not load the branch's scientific
modules. If a newer scientific revision has been admitted, a delayed handoff of
an older revision from another checkout is rejected.

With central execution activated, `run --repair` rebuilds only the current
branch's scientific database. The shared scheduler, other branches' demand,
ingestion, and public outputs are left alone. The previous database is kept as
`registry-before-repair.sqlite3`. Repair restores work items from current
public ownership records whose artifacts still have their declared files,
workflow history from on-disk workflow snapshots, and reproducibility bindings
in both the branch registry and the shared scheduler from current definitions.
Repair creates no demand.

## Storage and safeguards

The shared control directory holds the branch catalog and every branch
database:

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

`CONTROL/shared/scheduler/registry.sqlite3` is the production registry and the
only authority over the worker pool. It currently also holds production
scientific state. Branch databases hold scientific contracts and validator
observations. They have no workers, submissions, execution claims, or
concurrency settings. Shared worker logs stay with the scheduler, and ingestion
and caches are shared by the whole site. Branch directories also hold
scientific configuration snapshots when processing uses them. `ControlPaths`
provides this layout to the installation, registry, ingestion, branch
registration, and execution-cache code.

Each database is bound to its branch name, registration identity, and shared
control path. nro rejects a misplaced database and symlink redirection. It does
not search checkouts for local databases, merge conflicting copies, or silently
recreate a missing registered database. Registration identities and database
schema versions do not affect scientific freshness.

Catalog edits take a shared filesystem lock and are rejected if they are based
on an outdated catalog revision. A new branch becomes visible only after its
database exists. If registration is interrupted, a pending record remains, and
the next registration resumes it with the same identity. Updates to scientific
records also check the revision the caller read, so competing checkouts cannot
silently overwrite each other's edits. Verified observations carry that
revision through central assessment and cannot be attached to a replacement
contract.

nro rejects the previous flat layout before creating another scheduler. There
is no automatic relocation and no fallback to an alternate path. Archive or
remove an obsolete private store before initializing the current layout.

Production registry repair keeps the branch catalog and current databases. nro
checks a branch's scientific schema when it reads the branch's scientific
records, not when it inspects the branch's identity or registers another
branch. Shared installation quiesces the worker pool and then migrates schemas
at or after the supported baseline. For an older schema, it first restores each
development branch's scheduler mappings from that branch's public derivative
tree, then rebuilds the branch database from the recovered contracts.

These safeguards protect trusted developers from mistakes. They are not an
access-control boundary against anyone who can modify shared files directly.
