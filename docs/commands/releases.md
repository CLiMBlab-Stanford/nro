# Installed releases

A shared nro installation runs a tagged release of `main`. Maintainers of the
shared installation use `./install` to record and activate releases,
`nro release` to inspect them, and `nro release --repair-scheduler` to rebuild
shared scheduling state.

## Install and record a release

A shared installation runs the exact release checked out on `main`. When
`./install` succeeds, it records that release and activates it for workers.
The installer requires a clean tree, an annotated tag that matches the package
version, and a tagged commit on `origin/main`. It records the tag, commit,
tree, tagger, installing user, and main registry identity. A new site can start
with the current release and needs no records for earlier versions.

To see the recorded history, run `nro release` without a version:

```bash
nro release
```

Records are stored in `CONTROL/shared/releases.json` under the same edit lock
as branch registrations. Use `./install` for routine installation and
upgrades. It infers the release from Git and does not ask you to repeat
pull-request metadata. The older explicit attestation options of
`nro release` remain for installations created before releases were bound
automatically.

## Scheduler activation

The shared installer marks its verified installation as the one future workers
use, so no separate activation step is needed. After updating a shared
checkout to a new release, run:

```bash
./install --maintain
```

If work is active, the installer offers three choices: let current attempts
finish, stop them immediately while keeping their demand, or cancel the
maintenance. Activation writes
`CONTROL/shared/scheduler/implementation.json`, which records the central
checkout, the dependency interpreter, the site settings path, a source
fingerprint, and the installed release. The checkout's installation record
binds that interpreter to the verified application snapshot that interactive
commands use.

To activate an approved installation without running the installer, run:

```bash
nro release --activate
```

This makes the approved checkout the scheduler installation for future
workers. `--checkout PATH` selects a checkout other than the current directory.
Activation does not change Git refs or users' launchers. Approval and activation are separate operations, so `--activate`
cannot be combined with `--pr`, `--attest-merged`, or `--bootstrap`. If you
pass a version, it must match the approved release. Activation fails when:

- the checkout is not a ready personal or shared installation;
- the installation's registry or BIDS root belongs to another site;
- its interpreter is missing; or
- the pool is not quiescent: the scheduler service is running, or there is
  outstanding demand, an active attempt, worker, or Slurm submission, or
  registry maintenance in progress. Stop or drain that work first.

After activation, Slurm submissions and `run --local` use this orchestration
installation. Job subprocesses use their own registered scientific code and
environments. If the central source is missing or has changed, the work fails
instead of falling back to the requesting branch's code. Before activation, a
standalone installation uses its own worker code. A development installation
has no such fallback.

Before starting workers, nro checks the recorded commit, tree, version,
environment, site, and source fingerprint. Registered development checkouts
then use the central execution service with their own scientific code and
output paths.

## Git release policy

`main` holds released code and is the repository's default branch. Changes
reach it only through pull requests. Every merge must raise the version in
`pyproject.toml` to a later `MAJOR.MINOR.PATCH`. The smallest allowed change is
the next patch version.

The `Version advances` check enforces this rule on pull requests. It also
requires formatting, lint, the full test suite, an isolated upgrade rehearsal,
and a documentation build with no warnings. Make it a required check in the
repository settings. Run `./install --rehearse-upgrade` before merging. Pass an
explicit Git ref when the baseline should not be the newest reachable release
tag.

nro follows [Semantic Versioning](https://semver.org/). Major version zero
means initial development, so interfaces may change between minor releases.
During the 0.x series, use patch releases for compatible fixes and minor
releases for new features or intentional interface changes. Compatibility code
may stay when it has a clear use and does not add much code complexity,
runtime cost, or maintenance burden.

### Publish a release

Tag an approved release as `vMAJOR.MINOR.PATCH`. The package version is the
same without the `v`. Release tags must be annotated. Never move or replace a
published release tag.

Pushing a version tag starts the GitHub Release workflow. The workflow rejects
a tag whose version differs from `pyproject.toml`, whose target is not on
`main`, or whose name is not a plain semantic version prefixed with `v`. For a
valid tag, it creates a GitHub Release with generated notes and marks it as the
latest release.

After the workflow succeeds, edit the release so that it opens with a short
summary of the important changes, written by a person. Generated notes may
follow the summary. Check both the workflow result and the final release
description. A tag without a GitHub Release, or a release with only generated
notes, is not a complete publication.

Versions, tags, and GitHub Releases have no effect on scientific freshness,
which depends only on artifact contracts and inputs.

## Shared scheduler repair

On a development branch, `nro run --repair` repairs only that branch's
scientific database. On `main`, it replaces the shared scheduling state. The
release command runs the same shared repair:

```bash
nro release --repair-scheduler
```

`--repair-scheduler` cannot be combined with other `nro release` options.
Shared repair asks for confirmation when work is active. It stops the entire
worker pool, discards active scheduling history, and keeps a backup with an
index of the original paths. Branch scientific databases, runtime
configurations, ingestion records, and public derivatives stay in place.
Completion evidence belonged to the replaced scheduler database, so it survives
only inside the backup. Repair needs the old worker-control state to stay
readable long enough to confirm that workers have shut down.

A branch scientific database whose schema is at or after the supported baseline
is migrated, and the previous database is kept. An older schema is rebuilt from
durable ownership records and public artifacts, and later requests register any
remaining current scientific contracts as needed.

Repair rediscovers main artifacts from disk without creating demand. Other
branches register their outputs against current compiled contracts the next
time work is requested.

After a release update, `./install --maintain` performs any required migration
or reconstruction before it installs and activates the release.
