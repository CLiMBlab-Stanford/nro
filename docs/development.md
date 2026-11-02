# Development and documentation

This guide is for developers working on nro itself. It covers the release
policy, how development branches are isolated from production, and how to run
the tests, style checks, and documentation build. For the step-by-step Git,
registration, promotion, and release sequence, follow
[Develop on a branch](branching-tutorial.md).

nro is in early development. Shared execution requires an installed `main`
release and registered development checkouts. Installing or maintaining the
shared checkout records its tagged release and activates it.

## Release and compatibility policy

The project follows [Semantic Versioning](https://semver.org/), starting at
version 0.0.1. `main` is the released production branch, and `dev` is the
integration branch. Changes reach `main` through pull requests. Every merge
advances the version by at least one patch, and a pull-request check verifies
that the proposed version is later than the version at the target commit.

Patch releases preserve supported behavior. Minor releases add functionality
and, while the major version is zero, may change interfaces. Prefer migration
notes and small compatibility paths when they help current users. Do not keep
compatibility code that materially harms readability, maintenance, runtime, or
correctness.

Release tags use the form `vMAJOR.MINOR.PATCH`. Tags do not determine
derivative freshness. Freshness decisions come from artifact contracts and
scientific inputs. See [installed releases](commands/releases.md) for
publication and scheduler activation.

## Branch isolation work

Development follows the spine `main -> dev -> feature branches`. `main` owns
production outputs. A development branch inherits compatible outputs through
its registered parents and writes its own outputs beneath the site's
`development` path. Scientific inputs always come from shared raw BIDS. Branch
identity decides who owns an output. It does not affect freshness.

[Branch registration](commands/branches.md) creates one central scientific
database per branch, shared by all authorized checkouts of that branch. Catalog
and scientific updates must name the database revision the caller read. An
interrupted registration can resume without allocating another identity. A Git
merge does not change the registered parent tree. Run `branch reparent` after a
merge when the parent should change.

A branch database holds compiled scientific contracts, workflow lineages, and
validator observations matched to their revisions. It holds no worker pool. The
shared scheduler alone controls demand, attempts, concurrency, and physical
input dependencies.

The scientific and scheduler databases each carry their own schema marker.
These markers check storage format and are not scientific freshness inputs.
Their schemas change through the [registry migration contract](registry-migrations.md).
Scientific artifact contracts change through a separate
[contract-migration chain](artifact-contract-migrations.md). Any added setting
must declare both its current default and its historical meaning. Do not use
the current default to guess how an older derivative was produced.

### Requests and execution

Once [a main scheduler is installed and active](commands/releases.md), the
branch commands `run`, `status`, `log`, `stop`, and `set` use the central
implementation.

To plan a request, the scheduler launches the branch's pinned planner in an
isolated child process. The child compiles the complete graph, records
scientific revisions, and returns workflow records and execution recipes as
JSON. The scheduler never opens the branch's scientific database or imports its
modules. A delayed request cannot replace a newer admitted scientific revision.

Admission then resolves fresh compatible ancestors. It records reads of their
outputs without demanding that the ancestors compute anything, and it routes
the branch's own work to branch directories. Each request keeps its complete
graph, including computations skipped because outputs were reused. If an
inherited input later becomes unavailable, workers resolve the stored graph
again and request local computation. That resolution does not restore cancelled
targets. A newer scientific request supersedes an older incompatible graph. Use
`run --no-inherit` to compute locally when a compatible ancestor artifact
should not be reused.

One long-lived central worker can run jobs from different branches. Each
attempt runs in its own subprocess with its selected source, interpreter,
configuration, and input/output context. All seven scientific modules accept
that context. Runner checks declared output destinations when it is constructed
and again during execution, and shared raw data cannot be redirected to debug
BIDS. These checks protect trusted developers from mistakes. They are not an
operating-system sandbox for arbitrary code.

An attempt keeps its selected dependency generations until shutdown. When an
input changes, nro invalidates and cancels its readers, including readers on
other branches. Replacing the input waits for those readers to confirm
shutdown, and completion checks reject obsolete attempts. Generation zero stays
valid for adopted derivatives whose producer predates scheduler registration.
It does not stand for an invented historical code version.

### Freshness and execution snapshots

The branch compiles the scientific expectations. The scheduler assesses them by
checking the registered contracts, filesystem evidence, and dependency
generations.

Plain `status` reads the current registry. It does not recompile definitions or
check the filesystem. `status --update` first recompiles the registered
selections in the invoking checkout without creating demand, then asks the
central code to assess the compiled results. Historical lineages that current
workflows no longer represent keep their recorded contracts.

An `AssessmentSnapshot` captures a consistent graph, and an `AssessmentReport`
returns the decisions made from it. Publication rejects the whole report if
relevant state changed in the meantime. Worker heartbeats and resource changes
alone do not reject an assessment.

Source identity, Git status, release numbers, and execution snapshot paths never
make a scientific artifact stale, and no freshness check runs historical source.
Inheritance comparisons treat output members as paths relative to their artifact
roots. Processing settings, direct raw inputs, and dependency semantics remain
substantive.

Each request captures its source and resolved site settings under
`CONTROL/shared/cache/{implementations,execution-sites}`. The captured source
includes package code and resources, including uncommitted development edits.
It excludes environments, Git metadata, and caches. Publication and scientific
worker launchers verify every captured file before importing code. Short-lived
scheduler commands validate the capture manifest and launcher without rereading
package resources. Configuration snapshots are captured separately.
Environments, images, and definitions stores are not copied, so coordinate any
maintenance of those resources with running work.

Execution snapshots are temporary. Cleanup waits until no demand, active
attempt, worker, submission, or queued or running ingestion remains, which
keeps retries and replacement workers working. The collecting process keeps its
own source and site until a later invocation. Manual `purge --cache` is
site-wide and does not change scientific freshness.

### Installation and maintenance

The [installer](installation.md#development-checkouts) can prepare a branch
environment and the working-directory-aware launcher without changing the
default central installation. Development definitions default to shared
read-only use. Select a
[private definitions store](commands/branches.md#development-definitions) to
edit them.

Request admission, branch-scoped status, logs, and demand cancellation all go
through the central implementation, as do global concurrency and explicit worker
shutdown. Admission also holds the installation's maintenance lock, and the
installer rejects environment maintenance while queued or running work uses
that environment.

Branch maintenance commands affect only the branch:

- Branch purge removes only branch-owned outputs.
- Branch repair restores the scientific database from admitted records without
  replacing the shared scheduler. It cancels the branch's demand and asks before
  stopping its active attempts.
- Reparenting and retirement are serialized by the central scheduler.
  Retirement cancels the branch's active work and reconciles affected consumers,
  but it does not remove files.

Replacing the shared scheduler itself is a separate
[main-maintainer operation](commands/releases.md#shared-scheduler-repair) that
requires a global shutdown and keeps a backup.

Debug bidsification lets you test ingestion patches before a release. It uses
its own record, staging, and receipt paths and can publish only into
branch-local BIDS. Those files never supply scientific inputs, so debug
ingestion does not create an alternate source dataset for scientific
development. Its reviewer and stages run captured branch code. The stages share
the central pool's capacity, shutdown recovery, and cache retention. A captured
development stage receives only its branch-local path context and does not open
the shared scheduler database. Production publication requires an installed
`main` release.

[Promotion](commands/promotion.md) moves eligible outputs into an accepting
branch after an explicit attestation that the pull request was accepted. It
compiles the accepting checkout's current contracts, checks the dependency
closure, and copies the outputs, keeping producer provenance and source files.
Promotion does not merge code, change parents, or retire branches. A durable
per-file journal covers interruption. Recovery restores the pre-transaction
files, or finishes the ownership metadata if the scheduler transaction already
committed.

The test suite runs different source catalogs through a persistent worker and
through the real central JSON service. It checks output isolation, global
capacity, inherited reads, local fallback, cancellation, rejection of obsolete
completions, and scientific assessment after the source cache is removed.
Release publication, shared installation, and live registry repair remain
explicit operator actions.

## Tests

Install with `--dev` to include the locked test dependencies, then run the
default suite:

```bash
./install --maintain --dev
.nro-env/bin/python -m pytest -q
```

The default suite omits integration tests that exercise complete scheduler and
private-state transactions. Locally, run the focused tests for the files and
behavior you changed. `nro dev test` selects them for you, as described in
[scoped validation](architecture-map.md#scoped-validation). It and its
sphere-specific variants run the repository's formatting and lint checks before
pytest, so style failures appear before you open a pull request.

Pull-request checks run the complete suite before a release can merge:

```bash
.nro-env/bin/python -m pytest -q -m "integration or not integration"
```

Use `-m integration` to run only the integration tier.

When a change touches installation, orchestration schemas, migrations, or
release management, also run the maintenance rehearsal locally:

```bash
./install --rehearse-upgrade
```

The main release check runs formatting, lint, the complete test suite, the
isolated upgrade rehearsal, registry migration validation, and a warning-free
documentation build. This clean-environment check is the authoritative release
gate. Configure `Version advances` as a required check in the repository
settings so that a version cannot merge after one of these steps fails. The
rehearsal covers orchestration and source replacement. It does not probe site
containers or Slurm.

Installation tests isolate site settings and mock scheduler mutations.
Scientific unit tests use small synthetic inputs. The upgrade test crosses real
process and source-snapshot boundaries in temporary storage. None of these
replace visual inspection or deployment-specific container tests on real
acquisitions.

## Python style

Ruff defines the Python formatting and baseline lint rules. Check both before
handing off a change:

```bash
.nro-env/bin/ruff format --check nro tests docs/conf.py
.nro-env/bin/ruff check nro tests docs/conf.py
```

Run `.nro-env/bin/ruff format nro tests docs/conf.py` to apply formatting. Tests
may import nro after fixture setup, so `E402` is the only rule ignored under
`tests/`. A new ignore needs a specific reason and should be narrower than a
repository-wide rule.

Plans under `plans/` are maintained design guidance for agents. Update a plan
when the behavior it describes changes, and remove superseded instructions.
Migration reports belong under `plans/evidence/` and do not define current
behavior.

## Documentation build

Build the documentation in a separate environment:

```bash
python3.12 -m venv /tmp/nro-docs
/tmp/nro-docs/bin/pip install -r docs/requirements.txt
/tmp/nro-docs/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`. Read the Docs uses `.readthedocs.yaml`
with the same requirements and also treats build warnings as failures. This
configuration does not create a hosted site. Connecting the repository to a
Read the Docs project is a separate hosting action.

Markdown pages use MyST. AutoAPI reads Python source without importing the
scientific package. Add docstrings for public classes and methods where the code
defines them. The API pages are generated, so do not edit them directly.
Configuration examples are included from their source YAML. When you change a
step, update its method page, configuration meaning, artifact contract, and
tests together. No documentation build should modify a live registry or
download scientific images.

Prose follows the repository's
[writing policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/WRITING_POLICY.md).
Attribution and publication follow the
[contribution policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/CONTRIBUTING.md),
and the
[agent instructions](https://github.com/CLiMBlab-Stanford/nro/blob/dev/AGENTS.md)
govern agent behavior.
