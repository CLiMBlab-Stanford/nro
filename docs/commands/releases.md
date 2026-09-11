# Installed releases

A successful shared installation registers and activates the exact release checked
out on `main`. The installer requires a clean tree, an annotated tag matching the
package version, and a tagged commit on `origin/main`. It records the tag, commit,
tree, tagger, installation user, and main registry identity. A new site can begin
with the current release; it does not need records for earlier versions.

Omit the version from `nro release` to inspect the recorded history:

```bash
nro release
```

Records are stored in `CONTROL/shared/releases.json` under the same edit lock as
branch registrations. The older explicit attestation options remain available for
installations created before automated release binding. Routine installation and
upgrades should use `./install`, which infers the release from Git and does not ask
the maintainer to repeat pull-request metadata.

## Scheduler activation

The shared installer designates its verified installation for future workers. No
separate activation command is needed. After updating a shared checkout to a new
release, run:

```bash
./install --maintain
```

If work is active, the installer asks to drain it. The drain preserves demand and
queued ingestion while running work finishes. Activation records the central
checkout, interpreter, site settings path, source fingerprint, and installed
release in `CONTROL/shared/scheduler/implementation.json`.

Once activated, Slurm submissions and `run --local` use this orchestration
installation. Job subprocesses use their separately registered scientific code
and environments. Missing or changed central source fails rather than selecting
the requesting branch's implementation. Before activation, a standalone
installation uses its own worker code; a development installation cannot use
that fallback.

Runtime validation checks the recorded commit, tree, version, environment, site,
and source fingerprint before starting workers. Registered development checkouts
then use the central execution service with their own scientific code and output
paths.

## Git release policy

`main` contains released code and is the repository's default branch. Changes enter
it through pull requests. Every merge must update `pyproject.toml` to a later
`MAJOR.MINOR.PATCH` version. The smallest permitted change is the next patch version.
The main-version check runs on pull requests to enforce this rule.

Use patch releases for compatible fixes and minor releases for new features or
intentional interface changes during the 0.x series. Tag an approved release as
`vMAJOR.MINOR.PATCH`; the package version omits the `v`. Do not move or replace a
published release tag.

Pushing a version tag starts the GitHub Release workflow. It rejects a tag whose
version differs from `pyproject.toml`, whose target is not on `main`, or whose name is
not a plain semantic version prefixed by `v`. Release tags must be annotated. A valid
tag creates a GitHub Release with generated notes and marks it as the latest release.
After the workflow succeeds, edit the release so that it begins with a brief
human-written summary of the important changes. Generated notes can follow the
summary. Verify the workflow and the final release description. A tag without its
GitHub Release, or a release containing only generated notes, is not a complete
publication.

Version changes, tags, and GitHub Releases remain separate from scientific freshness,
which is based on artifact contracts and inputs.

nro follows [Semantic Versioning](https://semver.org/). Major version zero denotes
initial development, so interfaces may change between minor releases. Compatibility
code may be retained when it has a clear use and does not materially increase code
complexity, runtime cost, or maintenance burden.

## Shared scheduler repair

`nro run --repair` repairs the current branch's scientific database after central
activation. To replace the shared scheduler itself, use the designated, approved
main checkout:

```bash
nro release --repair-scheduler
```

The command always asks for confirmation. It stops the entire pool, discards
active scheduling history, and keeps a backup with an index of original paths.
Branch scientific databases, runtime configurations, ingestion records, and
public derivatives remain in place. Private completion certificates are archived
because their registry IDs belong to the old database.

Main artifacts are rediscovered from disk without creating demand. Other branches
register their outputs against current compiled contracts when work is next
requested. The operation does not migrate an obsolete schema. It requires the
old worker-control tables to remain readable so shutdown can be confirmed.
After a release update, run `./install --maintain` to install and activate it.
