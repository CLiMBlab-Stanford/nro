# Installation, cleanup, and publication

## `./install` and `nro setup`

Both use the same installer. First setup accepts `--mode personal`,
`--mode shared`, or `--mode branch`, or asks interactively. A new checkout uses
branch mode automatically when the user's dispatcher has a default installation.
An existing shared checkout connects
the caller's launcher unless `--maintain` is given. See the
[installation guide](../installation.md) for effects and permissions.

| Option | Meaning |
| --- | --- |
| `--mode personal/shared/branch` | Declare the first installation's role; branch mode reuses an existing site read-only. |
| `--maintain` | Permit shared environment/resource maintenance. |
| `--drain` | Permit noninteractive shared maintenance to drain the worker pool while preserving demand. |
| `--site PATH` | Choose the generated definitions-locator file during first setup. |
| `--bin-dir PATH` | Place the user's launcher in this directory. |
| `--default` | Make this checkout the user's default outside registered checkouts. |
| `--replace-launcher` | Back up and replace a recognized older fixed-path launcher. |
| `--convert-to-branch` | Convert a quiescent shared checkout to branch mode. |
| `--non-interactive` | Use supplied settings without prompts. |
| `--offline` | Forbid new resource downloads; require cached/installed dependencies. |
| `--without-oslom` | Omit OSLOM and its Python dependencies for this invocation. |
| `--with-bidsify` | Include Flywheel, dcm2bids, DICOM, and Google Drive dependencies. |
| `--without-marss` | Omit the official MARSS package and its Python dependencies. |
| `--dev` | Include locked test dependencies. |
| `--accept-qunex-license` | Acknowledge terms for unattended image acquisition. |
| `--local` | Do not require Slurm. |

Ctrl-C exits with status 130 and one cancellation message. Incomplete shared
setup resumes with `./install --maintain`; accepting defaults does not eliminate
the need for license files and suitable host permissions.

## `nro paths`

With no subcommand, show proposed defaults and ask whether to accept all. If
declined, prompt independently for every setting. `show` prints the locator,
protected definition, resolved values, and their sources. `set key=value ...`
validates and atomically saves updates in the protected definitions document.
`--maintain` is required to edit a shared site's settings. This command never
moves datasets, artifacts, or registries. See the installation guide for all keys.

## `nro doctor`

Report the checkout, branch, environment, executable, and site settings selected
for this invocation. The default checks Python package and configured-resource
availability, filesystem access, and Slurm executables. `--deep` additionally
imports scientific libraries, validates all definitions, starts containers, and
checks tool availability, template checksums, and image receipts. `--local`
makes Slurm optional; `--without-oslom` makes OSLOM optional. `--json` emits
structured results. Failed required checks exit nonzero. Doctor does not install
resources or process subject data.

## `nro purge`

Delete owned public derivatives, private intermediates, and associated logs.
Selectors narrow scope; absent selectors mean all. Ownership receipts and
module contracts distinguish nro outputs from unrelated derivatives.

```bash
nro purge -P nptl -p t20 -m clean --dry-run
nro purge -P nptl -p t20 -m clean
```

The normal invocation displays a pageable list of public/private removal paths
and asks for confirmation. `-f`/`--force` skips confirmation. `--dry-run` reports
without deleting. `-l`/`--logs` removes matching attempt logs and inactive-worker
logs only. `--json` supplies a structured result; `--work-root` overrides the
private-file root. Selected work items must have no active attempts.

Use `--excluded` after adding source paths to markup:

```bash
nro purge --excluded --dry-run
nro purge --excluded
```

This mode finds owned work items whose recorded raw inputs are excluded by their
current markup and follows the registered DAG through every downstream consumer.
It purges that complete affected closure and withdraws its demand, so a clean run
or subject-level aggregate cannot preserve or protect a derivative contaminated
by an excluded run. Other selectors narrow the affected roots; downstream
consumers are still included even when they fall outside a module or entity
selector. Work configured with `markup: null` is unaffected. `--excluded` cannot
be combined with `--logs` or `--cache`, and it still refuses active attempts.

Deleting an upstream artifact also invalidates its consumers and requests
cancellation of their active attempts, even when those consumers were not selected
for deletion. Their files are not purged. Purge withdraws existing demand for
the selected work and for targets that depend on it; unrelated targets from the
same request remain active. The command reserves the selected
outputs and waits up to 30 seconds for confirmed consumer shutdown without holding
the registry lock while waiting. If shutdown is not confirmed, it deletes no
outputs; retry after the attempts stop. Invalidation and cancellation remain in
effect. `--dry-run` does not invalidate or cancel work.

After successful deletion, the scheduler removes purged work-item records that
no surviving registered artifact references. A missing upstream record remains
when it is needed to describe a surviving downstream artifact's DAG. Purging that
downstream artifact later makes the retained ancestor eligible for removal.

After an interrupted purge, repeat the operation to recover its mutation lock and
reservation. The scheduler retains an approved purge and finishes it before later
requests, even if the invoking terminal disconnects. Large purges report their
validation and deletion progress on stderr. Registry repair refuses unresolved
mutation reservations. This prevents repair from discarding the barrier while
files might still be changing.

Deletion is destructive and does not move files to trash. A bare invocation
selects every controlled derivative owned by the current branch across all
projects, including historical lineages. Inherited outputs are excluded. Force
does not mean it is safe to delete a shared user's needed work. After removing
selected files, purge also removes empty parent directories within the controlled
derivative, WORK, control, and log trees. It preserves the roots of those trees.

Use `--cache` to remove unused executable-source snapshots and captured site
settings across the shared installation:

```bash
nro purge --cache --dry-run
nro purge --cache
nro purge --cache --force
```

This mode ignores all artifact selectors, including project, participant,
module, and workflow. It uses the BIDS and registry roots from the global site
configuration. It does not remove derivatives, logs, definitions, container
images, or dependency downloads, and cannot be combined with `--logs`.

Cache cleanup is conservative: outstanding demand, attempts, workers, scheduler
submissions, or queued/running ingestion defer collection across the whole
pool. Unknown registry state also preserves the cache. `--force` skips the
prompt; it never overrides these checks. State is checked again after
confirmation. Old snapshots are not needed to assess derivative freshness.

Normal requests and worker shutdown also attempt cleanup automatically. A
process running from a snapshot retains its own source and site file; a later
command can reclaim those after it exits. Unknown entries and partial capture
directories are not automatically removed.

## `nro gc`

Remove unclaimed files from nro's public derivative and private WORK namespaces
while preserving registered artifacts and work directories:

```bash
nro gc -P nptl -p t20 -m clean --dry-run
nro gc -P nptl -p t20 -m clean
```

`gc` accepts the same artifact selectors as `purge`. Selectors limit the
directories inspected; they do not make other registered artifacts eligible for
deletion. A bare invocation checks every project and module owned by the current
branch. Inherited branch data lies outside the branch's writable roots and is not
collected.

The command reports public and WORK files separately and asks for confirmation.
Use `-f`/`--force` to skip the prompt, `--dry-run` to report without deleting, or
`--json` for structured output. Collection refuses a scope with queued, running,
or cancel-requested attempts. It verifies the candidate set again after
confirmation and removes directories made empty by the selected files.

Ownership is conservative. Fixed artifacts are protected by exact paths or
filename prefixes. Directory-producing artifacts and registered work-item
directories protect their complete directory trees because third-party tools can
create files that are not known in advance. As a result, `gc` removes orphaned
lineages and unmatched files in shared directories but does not remove an
unrecognized file placed inside an owned directory tree. Control records, logs,
and executable caches are outside this command's scope.

## `nro project rename`

Rename a BIDS project after stopping its demand and active work:

```bash
nro project rename climblab_multisession climb
nro project rename climblab_multisession climb --execute
```

The first command is a read-only preview. It reports every managed directory
move and the affected work items, ownership receipts, Workbench scenes,
BIDSification records, structured metadata, symbolic links, and definition
files. `--execute` applies that exact kind of migration. The command is
available only from the registered main checkout.

The rename covers the shared BIDS and WORK projects, every registered branch's
development BIDS and WORK projects, project log directories, scheduler and
branch registries, ownership receipts, BIDSification records, Workbench scene
links, absolute symbolic links into renamed roots, markup, and BIDSification
project routing. Directory moves must be atomic on their filesystem. The command
refuses symbolic-link roots, an existing destination, active demand, active
attempts or resource steps, and active
BIDSification for the source project.

Raw BIDS symbolic links are materialized during execution. Regular files become
hard links when their target is on the same filesystem; otherwise they become
metadata-preserving copies. Directory links become metadata-preserving directory
copies. Symbolic links inside derivatives remain links. Absolute derivative
links into the renamed project are updated without following their targets.

Numerical scientific outputs are not rewritten. Structured derivative and WORK
metadata containing absolute project paths are translated, and the registry
updates observations for those exact files. Project-sensitive work-item
identities and current control contracts are translated so that existing
artifacts retain their state under the new project name. Definition changes use
the managed definitions transaction and update its integrity manifest.

Execution creates a recovery journal under the shared private control store and
backs up the scheduler registry, branch registries, and edited metadata. A
failure before the registry commit reverses directory moves and restores edited
state. Keep external readers and writers stopped until the command completes.

## `nro migrate dataset`

Bring source BIDS metadata and nro-owned derivative metadata into the current
representation:

```bash
nro migrate dataset -P PROJECT
nro migrate dataset -P PROJECT --dry-run
nro migrate dataset -P PROJECT -f
```

The default pages through a read-only plan and asks for confirmation before applying
it. `--dry-run` stops after the preview. `-f`/`--force` applies the plan without a
prompt. Execution covers the selected source project plus main and registered branch
derivatives. It requires a quiet worker pool, removes obsolete source `EventsFile`
fields, converts durable contracts to field-level source metadata snapshots, and
updates integrity records without advancing artifact generations. A durable journal
rolls back failed or interrupted file conversion. See [portable derivative
provenance](../portable-provenance.md) for the reference model and migration boundary.

## `nro publish`

```bash
nro publish REQUEST_ID /destination -P PROJECT
```

Assess a completed request, copy terminal public files to a staging directory,
verify checksums, embed recursive provenance, and recheck source generations
before publishing. `destination` must not already exist. The source root comes
from the global site configuration. `--no-validate` skips the optional external
BIDS validator.
Otherwise the validator runs only if available on PATH.

Publication writes `dataset_description.json` and `.nro-publication.json`.
It is a separate snapshot operation, not an assertion that every working
derivative format meets the BIDS validator. Inspect the published files and
validation output before distributing the dataset; the command does not upload
anything or create a release.
