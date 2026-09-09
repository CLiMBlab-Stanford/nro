# Development and documentation

This is an early-development codebase. Shared execution requires an installed main
release and registered development checkouts. Installing or maintaining the shared
checkout records and activates its tagged release.

## Release and compatibility policy

The project follows [Semantic Versioning](https://semver.org/) beginning with
version 0.0.1. `main` is the released production branch; `dev` is the integration
branch. Changes reach `main` through pull requests, and every merge advances the
version by at least one patch. The repository check on pull requests verifies that
the proposed version is later than the version at the target commit.

Patch releases preserve supported behavior. Minor releases add functionality and may
change interfaces while the major version remains zero. Prefer migration notes and
small compatibility paths when they help current users. Do not retain compatibility
code that materially harms readability, maintenance, runtime, or correctness.

Release tags use `vMAJOR.MINOR.PATCH` and do not determine derivative freshness.
Artifact contracts and scientific inputs remain the source of freshness decisions.
See [installed releases](commands/releases.md) for publication and scheduler activation.

## Branch isolation work

The development spine is `main -> dev -> feature branches`. Main owns production
outputs. Development branches inherit compatible outputs through their registered
parents and write beneath the site's `development` path. Scientific inputs always
come from shared raw BIDS. Branch identity governs ownership, not freshness.
The first release is `0.0.1`.

[Branch registration](commands/branches.md) creates one central scientific database
per branch. All authorized checkouts of a branch share that database. Catalog and
scientific updates require the revision the caller read. Interrupted registration
can resume without allocating another identity. Git merges do not change the
explicit parent tree; use `branch reparent` after a merge when appropriate.

Branch databases contain compiled scientific contracts, workflow lineages, and
revision-matched validator observations. They contain no worker pool. The shared
scheduler is the sole authority for demand, attempts, concurrency, and physical
input dependencies.
The scientific and scheduler databases have independent schema markers. These
are storage-format checks, not scientific freshness inputs.

### Requests and execution

Once [a main scheduler is installed and active](commands/releases.md), branch
`run`, `status`, `log`, `stop`, and `set` use the central implementation.
The branch's planner compiles its complete graph and records scientific revisions.
It sends that graph, workflow records, and execution recipes as JSON to a separate
process running central Python and source. The central process does not open the
branch's scientific database or import its modules. Delayed requests cannot replace
a newer admitted scientific revision.

Admission resolves fresh compatible ancestors, records read dependencies without
demanding ancestor computation, and routes owned work to branch directories.
Every request retains its complete graph, including computations hidden by reused
outputs. If an inherited input becomes unavailable, workers resolve the stored graph
again and request local computation. Cancelled targets are not restored by that
resolution. A newer scientific request supersedes an older incompatible graph.
Use `run --no-inherit` to request branch-local computation when an otherwise
compatible ancestor artifact should not be reused.

One long-lived central worker can run jobs from different branches. Each attempt
executes in a separate subprocess with its selected source, interpreter, configuration,
and input/output context. All seven scientific modules accept that context. Runner
checks declared output destinations at construction and execution. Shared raw data
cannot be redirected to debug BIDS. These checks protect trusted developers against
mistakes; they are not an operating-system sandbox for arbitrary code.

Attempts retain their selected dependency generations until shutdown. Changes to an
input invalidate and cancel readers across branch boundaries; replacing inputs waits
for shutdown confirmation. Completion checks reject obsolete attempts. Generation
zero remains valid for adopted derivatives whose producer predates scheduler
registration. It does not invent a historical code version.

### Freshness and execution snapshots

Branch-side compilation supplies scientific expectations. Scheduler assessment
checks those registered contracts, filesystem evidence, and dependency generations.
`status --verify` first recompiles registered selections in the invoking checkout
without creating demand, then asks central code to assess the compiled results.
Default status also previews code-defined processing-policy changes locally;
`--cached` does neither. Historical lineages no longer represented by current
workflows retain their recorded contracts.
`AssessmentSnapshot` captures a consistent graph; `AssessmentReport` returns its
decisions. Publication rejects the entire report if relevant state changed.
Worker heartbeats and resource changes alone do not reject an assessment.

Source identity, Git status, release numbers, and execution snapshot paths do not
make scientific artifacts stale. No freshness check invokes historical source.
Output members remain relative to artifact roots in inheritance comparisons;
processing settings, direct raw inputs, and dependency semantics remain substantive.

Requests capture source and resolved site settings under
`CONTROL/shared/cache/{implementations,execution-sites}`. Source includes package
code and resources, including uncommitted development edits. It excludes environments,
Git metadata, and caches. Launchers verify captures before importing code.
Configuration snapshots are separate. Environments, images, and definitions stores
are not copied; coordinate maintenance of those resources.

Execution snapshots are temporary. Cleanup waits until there is no outstanding
demand, active attempt, worker, submission, or queued/running ingestion. This preserves
retries and replacement workers. Collection retains the collecting process's own
source and site until a later invocation. Manual `purge --cache` is site-wide and
does not change scientific freshness.

### Installation and maintenance

The [installer](installation.md#development-checkouts) can prepare a branch environment
and the working-directory-aware launcher without changing the default central
installation. Development definitions default to shared read-only use; select a
[private definitions store](commands/branches.md#development-definitions) to edit them.

Request admission and branch-scoped status, logs, and demand cancellation are connected.
Global concurrency and explicit worker shutdown also route to the central implementation.
Branch purge removes only branch-owned outputs. Branch repair restores the
scientific database from admitted records without replacing the shared scheduler.
It cancels that branch's demand and asks before stopping its active attempts.
Reparenting and retirement are also serialized by the central scheduler.
Retirement cancels the branch's active work and reconciles affected consumers;
it does not remove its files.
Replacing the shared scheduler itself is a separate
[main-maintainer operation](commands/releases.md#shared-scheduler-repair), with
global shutdown and a retained backup.
The installer rejects environment maintenance while queued or running work uses
that environment. Admission also holds the installation's maintenance lock.

Debug bidsification has separate record, staging, and receipt paths and can publish
only into branch-local BIDS. Those files never supply scientific inputs. Its
reviewer and stages use captured branch code; stages share the central pool's
capacity, shutdown recovery, and cache retention. Production publication requires
an installed main release. Debug ingestion tests patches before release; it does
not create an alternate source dataset for scientific development. A captured
development stage receives only its branch-local path context; it does not open
the shared scheduler database.

[Promotion](commands/promotion.md) compiles the accepting checkout's current
contracts, checks the dependency closure, and copies eligible outputs after an
explicit accepted-PR attestation. It retains producer provenance and source files.
Promotion does not merge code, change parents, or retire branches. Its durable
per-file journal restores pre-transaction files after interruption or finishes
ownership metadata when the scheduler transaction already committed.

[Private-state cutover](commands/cutover.md) is available for a quiescent store and
retains a rollback copy. Existing canonical derivatives remain the main-owned baseline
with their original provenance. Do not attribute them to release 0.0.1 after the fact.

The test suite exercises different source catalogs through a persistent worker and
through the actual central JSON service. It checks output isolation, global capacity,
inherited reads, local fallback, cancellation, obsolete-completion rejection, and
scientific assessment after source-cache removal. Release publication, shared
installation, and live registry repair remain explicit operator actions.

## Tests

Install with `--dev` to include the locked test dependencies:

```bash
./install --maintain --dev
.nro-env/bin/python -m pytest -q
```

This default suite omits integration tests that exercise complete scheduler and
private-state transactions. Run every test before a release or a major merge:

```bash
.nro-env/bin/python -m pytest -q -m "integration or not integration"
```

Run only the integration tier with `-m integration`.

Installation tests isolate site settings and mock scheduler mutations. Scientific
unit tests use small synthetic inputs; they do not replace visual inspection
and deployment-specific container tests on real acquisitions.

## Python style

Ruff defines the Python formatting and baseline lint rules. Check both before
handing off a change:

```bash
.nro-env/bin/ruff format --check nro tests docs/conf.py
.nro-env/bin/ruff check nro tests docs/conf.py
```

Use `.nro-env/bin/ruff format nro tests docs/conf.py` to apply formatting. Tests may
import nro after fixture setup, so only `E402` is ignored under `tests/`. New ignores
need a specific reason and should be narrower than a repository-wide rule.

Agent-facing plans under `plans/` are maintained design guidance. Update them when
the corresponding behavior changes and remove superseded instructions. Migration
reports belong under `plans/evidence/` and do not define current behavior.

## Documentation build

```bash
python3.12 -m venv /tmp/nro-docs
/tmp/nro-docs/bin/pip install -r docs/requirements.txt
/tmp/nro-docs/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Read the Docs uses `.readthedocs.yaml` with
the same requirements and treats build warnings as failures. Connecting the
repository to a Read the Docs project is a separate hosting action; this
configuration does not create a hosted site.

Markdown pages use MyST. AutoAPI reads Python source without importing the
scientific package. Add docstrings for public classes and methods where the
code is defined; API pages are generated and should not be edited directly.
Configuration examples are included from their source YAML. When changing a
step, update its method page, configuration meaning, artifact contract, and tests.

Follow the repository's
[writing policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/WRITING_POLICY.md)
for prose. Follow its
[contribution policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/CONTRIBUTING.md)
for attribution and publication. The
[agent instructions](https://github.com/CLiMBlab-Stanford/nro/blob/dev/AGENTS.md)
govern agent behavior. No documentation build should modify a live registry or
download scientific images.
