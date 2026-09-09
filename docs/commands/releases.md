# Release attestation

`nro release` records a maintainer's approval of a `main` release. Normal approval
attests that a PR was approved and merged. The initial bootstrap is recorded
separately because 0.0.1 created the `main` branch directly. The command does not
query a hosting service or supply a cryptographic signature. This policy assumes
trusted maintainers.

Prepare the version change in the reviewed PR. After merging it, use a clean,
registered `main` checkout:

```bash
nro release 0.0.2 --pr https://HOST/OWNER/REPO/pull/NUMBER --attest-merged
```

The directly created first `main` branch has no preceding PR. Record that one-time
bootstrap explicitly:

```bash
nro release 0.0.1 --bootstrap
```

Bootstrap approval is accepted only for 0.0.1 when the release history is empty.
It reads the `v0.0.1` tag and verifies that commit is an ancestor of the current
clean `main` checkout. The checkout may therefore already contain a later release.
Bootstrap cannot be combined with `--pr` or `--attest-merged`. All later approvals
require the merged PR reference and explicit merge attestation.

The version must match the committed `pyproject.toml`. The first release must be
`0.0.1`. Each later release must advance the version by at least one patch and
descend from the previous approved commit. Duplicate approvals and dirty checkouts
are rejected. Configure the human maintainer's Git `user.name` and `user.email`
first; nro records those values and the executing Unix UID.

Omit the version to list approvals:

```bash
nro release
```

`--checkout PATH` selects the source checkout. `--bids-root PATH` selects the
shared control context. Records are stored in `CONTROL/shared/releases.json`,
under the same edit lock as branch registrations. Approval does not tag,
push, deploy code, change the package version, or modify derivatives. Version
metadata is separate from scientific freshness.

## Scheduler activation

After approving a release and installing its checkout in shared mode, designate
that installation for future workers:

```bash
nro release 0.0.1 --activate
```

Activation requires no outstanding demand, workers, allocations, or running or
queued ingestion. It records the central checkout, interpreter, site settings
path, and approved release in `CONTROL/shared/scheduler/implementation.json`.
It does not change the user's command launcher, Git refs, or derivatives.

Once activated, Slurm submissions and `run --local` use this orchestration
installation. Job subprocesses use their separately registered scientific code
and environments. Missing or changed central source fails rather than selecting
the requesting branch's implementation. Before activation, a standalone
installation uses its own worker code; a development installation cannot use
that fallback.

Approval and activation are separate operations. An approval alone does not
change the worker implementation. Activation enables registered checkouts to use
the central execution service with their own scientific code and output paths.

## Git release policy

`main` contains released code and is the repository's default branch. Changes enter
it through pull requests. Every merge must update `pyproject.toml` to a later
`MAJOR.MINOR.PATCH` version. The smallest permitted change is the next patch version.
The main-version check runs on pull requests to enforce this rule.

Use patch releases for compatible fixes and minor releases for new features or
intentional interface changes during the 0.x series. Tag an approved release as
`vMAJOR.MINOR.PATCH`; the package version omits the `v`. Do not move or replace a
published release tag. Version changes and Git tags remain separate from scientific
freshness, which is based on artifact contracts and inputs.

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
After a release update, activate the approved implementation separately.
