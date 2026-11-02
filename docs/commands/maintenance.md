# Installation, cleanup, and publication

This page covers the commands that maintain an installation and its data. It
starts with setup, site settings, and `nro doctor`. It then describes the three
cleanup commands, `purge`, `gc`, and `freshen`, followed by the project-level
operations `project rename` and `migrate`, and finally `publish`.

## `./install` and `nro setup`

`./install` and `nro setup` run the same installer. The first setup of a
checkout declares its role with `--mode personal`, `--mode shared`, or
`--mode branch`, or the one-letter forms `p`, `s`, and `b`. The interactive
prompt accepts the same values. A new checkout uses branch mode automatically
when your dispatcher already has a default installation. Running setup in an
existing shared checkout connects your launcher to it, unless you pass
`--maintain`. The [installation guide](../installation.md) describes the
effects of each mode and the permissions it needs.

| Option | Meaning |
| --- | --- |
| `--mode personal/shared/branch` | Declare the role of the first installation. Branch mode reuses an existing site read-only. |
| `--maintain` | Allow maintenance of the shared environment and resources. |
| `--drain` | Allow noninteractive shared maintenance to drain the worker pool. Demand is preserved. |
| `--site PATH` | Choose the location of the generated definitions-locator file during first setup. |
| `--default-parent-dir PATH` | Choose the parent directory for the storage and resource paths proposed during first setup. |
| `--bin-dir PATH` | Put your launcher in this directory. |
| `--default` | Make this checkout your default outside registered checkouts. |
| `--replace-launcher` | Back up and replace a recognized older fixed-path launcher. |
| `--convert-to-branch` | Convert a quiescent shared checkout to branch mode, keeping its environment. |
| `--non-interactive` | Use the supplied settings without prompting. |
| `--offline` | Forbid new resource downloads. Dependencies must already be cached or installed. |
| `--without-oslom` | Omit OSLOM and its Python dependencies for this invocation. |
| `--without-marss` | Omit the official MARSS package and its Python dependencies. |
| `--without-viewer` | Omit Connectome Workbench support for scene viewing and rendering. |
| `--with-bidsify` | Include the Flywheel, dcm2bids, DICOM, and Google Drive dependencies. |
| `--with-cicada` | Include the pinned pycicada classification backend. |
| `--with-lesion` | Include the lesion-masking dependencies. |
| `--dev` | Include the locked test dependencies. |
| `--accept-qunex-license` | Accept the QuNex terms so images can be acquired unattended. |
| `--local` | Do not require Slurm. |

Ctrl-C exits with status 130 and prints one cancellation message. To resume an
incomplete shared setup, run `./install --maintain`. Accepting the defaults
does not remove the need for license files and suitable host permissions.

## `nro site`

`nro site` reads and changes site settings:

* `ls [PREFIX]` lists settings with their effective values and lifecycles.
* `get [KEY ...]` prints selected values.
* `set KEY=VALUE ...` validates the updates and saves them atomically in the
  protected definitions document.
* `edit` opens the interactive editor.
* `validate` checks the current document.

On a shared site, changes to storage and installation resources require
`--maintain`. Changes to live execution limits are passed to a running
scheduler automatically, as described under
[site execution policy](work.md#site-execution-policy). No site operation
starts a scheduler only to apply a setting.

`nro paths`, `nro get`, and `nro set` remain as compatibility aliases.

## `nro doctor`

`nro doctor` reports the checkout, branch, environment, executable, and site
settings that this invocation selects, and checks that they work. By default it
checks Python packages, configured resources, filesystem access, and the Slurm
executables. `--deep` also imports the scientific libraries, validates all
definitions, starts containers, and checks tool availability, template
checksums, and image receipts.

`--local` makes Slurm optional and `--without-oslom` makes OSLOM optional.
`--json` prints structured results. The command exits nonzero when a required
check fails. It does not install resources or process subject data.

## `nro purge`

`nro purge` deletes owned public derivatives, private intermediates, and their
logs. Selectors narrow the scope, and an omitted selector matches everything.
nro tells its own outputs apart from unrelated derivatives by ownership
receipts and module contracts.

```bash
nro purge -P nptl -p t20 -m clean --dry-run
nro purge -P nptl -p t20 -m clean
```

The command shows a pageable list of the public and private paths it will
remove and asks for confirmation. `-f`/`--force` skips the confirmation, and
`--dry-run` reports without deleting. `-l`/`--logs` removes only matching
attempt logs and logs of inactive workers. `--json` prints a structured result,
and `--work-root` overrides the root for private files. The selected work items
must have no active attempts.

Deletion is permanent. Files are not moved to a trash directory. A bare
`nro purge` selects every controlled derivative owned by the current branch in
every project, including historical lineages. Inherited outputs are excluded.
`--force` only skips the prompt. It does not make it safe to delete work that
another user needs. After removing files, purge removes empty parent
directories within the controlled derivative, WORK, control, and log trees, and
keeps the roots of those trees.

### Effects on other work

Purge withdraws existing demand for the selected work and for the targets that
depend on it. Unrelated targets from the same request stay active.

Deleting an upstream artifact also invalidates its consumers and requests
cancellation of their active attempts, even when those consumers were not
selected. Their files are not deleted. Purge reserves the selected outputs and
waits up to 30 seconds for the consumers to confirm shutdown, without holding
the registry lock while it waits. If shutdown is not confirmed, purge deletes
nothing, and you can retry after the attempts stop. The invalidation and
cancellation stay in effect. `--dry-run` neither invalidates nor cancels work.

After a successful deletion, the scheduler removes purged work-item records
that no surviving registered artifact refers to. A purged upstream record stays
when a surviving downstream artifact needs it to describe its DAG. Purging that
downstream artifact later makes the ancestor record eligible for removal.

### Interrupted purges

Repeat an interrupted purge to recover its mutation lock and reservation. The
scheduler keeps an approved purge and finishes it before later requests, even
if the invoking terminal disconnects. Large purges report validation and
deletion progress on stderr. Registry repair refuses to run while a mutation
reservation is unresolved, so that it cannot discard the barrier while files
might still be changing.

### Removing work that uses excluded inputs

After you exclude source paths in markup, run purge with `--excluded`:

```bash
nro purge --excluded --dry-run
nro purge --excluded
```

This mode finds owned work items whose recorded raw inputs are now excluded by
their current markup. It follows the registered DAG to every downstream
consumer, purges that whole closure, and withdraws its demand. A clean run or a
subject-level aggregate therefore cannot keep a derivative that was built from
an excluded run. Other selectors narrow the affected roots, but downstream
consumers are included even when they fall outside a module or entity
selector. Work configured with `markup: null` is unaffected. `--excluded`
cannot be combined with `--logs` or `--cache`, and it still refuses work with
active attempts.

### Removing unused source snapshots

`--cache` removes unused executable-source snapshots and captured site settings
across the whole shared installation:

```bash
nro purge --cache --dry-run
nro purge --cache
nro purge --cache --force
```

This mode ignores all artifact selectors, including project, participant,
module, and workflow. It uses the BIDS and registry roots from the global site
configuration. It does not remove derivatives, logs, definitions, container
images, or downloaded dependencies, and it cannot be combined with `--logs`.

Cache cleanup is conservative. Any outstanding demand, attempt, worker,
scheduler submission, or queued or running ingestion defers collection for the
whole pool. Unknown registry state also preserves the cache. `--force` skips
the prompt but never overrides these checks, and the state is checked again
after confirmation. Old snapshots are not needed to assess derivative
freshness.

Normal requests and worker shutdown also try to clean the cache automatically.
A running process keeps its own source snapshot and site file, and a later
command can reclaim them after the process exits. Unknown entries and partial
capture directories are never removed automatically.

## `nro gc`

`nro gc` removes unclaimed files from nro's public derivative namespace and its
private WORK namespace. Registered artifacts and work directories are kept.

```bash
nro gc -P nptl -p t20 -m clean --dry-run
nro gc -P nptl -p t20 -m clean
```

`gc` accepts the same artifact selectors as `purge`. Selectors limit the
directories inspected. They never make a registered artifact eligible for
deletion. A bare invocation checks every project and module owned by the
current branch. Inherited branch data lies outside the branch's writable roots
and is never collected.

The command reports public and WORK files separately and asks for
confirmation. `-f`/`--force` skips the prompt, `--dry-run` reports without
deleting, and `--json` prints structured output. Collection refuses a scope
that has queued, running, or cancel-requested attempts. It checks the candidate
set again after confirmation and removes directories that the deletion leaves
empty.

Ownership is conservative. Artifacts with fixed outputs protect their exact
paths or filename prefixes. Artifacts that produce directories, and registered
work-item directories, protect their whole directory trees, because
third-party tools can create files that nro cannot predict. As a result, `gc`
removes orphaned lineages and unmatched files in shared directories, but it
leaves an unrecognized file inside an owned directory tree. Control records,
logs, and executable caches are outside the scope of `gc`.

## `nro freshen`

`nro freshen` recovers completed derivatives after an upstream artifact was
rebuilt by mistake, without rerunning work whose science did not change:

```bash
nro freshen -P climb -p c001 -m func clean --dry-run
nro freshen -P climb -p c001 -m func clean
```

The command validates matching work items in dependency order. It accepts a
stale item only when its current scientific contract, direct-input
fingerprints, dependency set, and recorded artifacts are all unchanged. It
refuses missing or corrupt artifacts, changed contracts or inputs, and active
attempts. When an upstream item is rejected, its selected descendants are
rejected too.

For each accepted item, `freshen` moves output timestamps past those of the
validated inputs and rebinds the completion records to the current upstream
generations. It does not advance artifact generations or change file contents.
Use it only to correct a wrong invalidation. After a scientific change, rerun
the work instead.

By default the command pages through the validation report and asks for
confirmation. `--dry-run` stops after the preview, and `-f`/`--force` skips the
confirmation. `--json` prints structured output in either of those
noninteractive modes.

## `nro project rename`

`nro project rename OLD NEW` renames a BIDS project. Stop the project's demand
and active work first.

```bash
nro project rename climblab_multisession climb
nro project rename climblab_multisession climb --dry-run
```

The command opens a read-only preview in the pager and then asks for
confirmation. The preview lists every managed directory move and the affected
work items, ownership receipts, Workbench scenes, BIDSification records,
structured metadata, and definition files. `--dry-run` stops after the
preview, `-f`/`--force` skips the confirmation, and `--json` prints
machine-readable output. The command runs only from the registered main
checkout.

The command starts the central scheduler if it is not already running. Both the
preview and the rename run in the scheduler's Slurm allocation, not in your
login shell.

### What a rename changes

The rename covers the shared BIDS and WORK projects, the development BIDS and
WORK projects of every registered branch, project log directories, the
scheduler and branch registries, ownership receipts, BIDSification records,
Workbench scene metadata, markup, and BIDSification project routing.

Numerical scientific outputs are not rewritten. Indexed structured metadata in
derivatives and WORK that contains absolute project paths is translated, and
the registry updates its observations of those exact files. Work item
identities and current control contracts that depend on the project name are
translated, so existing artifacts keep their state under the new name.
Definition changes go through the managed definitions transaction and update
its integrity manifest.

### Requirements

Run `nro migrate -P PROJECT` before renaming. Migration materializes raw-data
links and repairs nonportable links in nro derivatives, and rename relies on
that. Rename inventories SQL artifact records and the fixed ownership and scene
namespaces, but it does not repeat migration's traversal of raw data or
scientific derivatives. Relative links stay valid because each project tree
moves as a unit.

Directory moves must be atomic on their filesystem. The command refuses
symbolic-link roots, an existing destination, active demand, active attempts or
resource steps, and active BIDSification for the source project.

### Preview, execution, and recovery

Until you confirm, the validated preview is kept in uncompressed private
control state. It survives a scheduler restart and does not expire. On
execution, the command checks that the branch topology, definitions, managed
roots, work-item count, guarded files and links, and active-work barriers still
match the preview. It then applies the prepared changes without repeating
discovery. Any relevant change makes the command fail, and you must preview
again. Successful execution or interactive cancellation deletes the preview.

Execution creates a recovery journal in the shared private control store and
backs up the scheduler registry, the branch registries, and the edited
metadata. A failure before the registry commit reverses the directory moves and
restores the edited state. Keep external readers and writers stopped until the
command completes.

## `nro migrate`

`nro migrate` brings source BIDS metadata and nro-owned derivative metadata up
to the current representation:

```bash
nro migrate -P PROJECT
nro migrate -P PROJECT --dry-run
nro migrate -P PROJECT -f
```

Pass several project IDs after `-P` to migrate them together. Omitting `-P`
selects every BIDS project. The command rejects the other scientific
selectors.

By default the command pages through a read-only plan and asks for
confirmation. `--dry-run` stops after the preview and `-f`/`--force` applies the
plan without a prompt. Before preparing the preview, the command starts or
joins the central Slurm scheduler. Starting it creates no scientific demand.

### What migration changes

Execution covers the selected source projects and their derivatives in main and
in every registered branch. It requires an idle worker pool. It makes these
changes:

* Removes obsolete `EventsFile` fields from source metadata.
* Converts durable contracts to field-level snapshots of source metadata.
* Updates integrity records without advancing artifact generations.
* When an older anatomical derivative links to a template inside a container,
  copies a versioned FreeSurfer `fsaverage` template into the project's hidden
  nro template namespace and replaces the link with a relative link inside the
  project.
* Replaces raw BIDS symbolic links with hard links when the target is on the
  same filesystem, and with metadata-preserving copies otherwise. Directory
  links become metadata-preserving directory copies.

A durable journal rolls back an incomplete file conversion and resumes an
interrupted registry update. See
[portable derivative provenance](../portable-provenance.md) for the reference
model and the limits of migration.

### Preview and execution

The scheduler writes the prepared file rewrites to uncompressed private control
state before it shows the preview. The preparation survives a scheduler restart
and does not expire. On execution, nro verifies each file or link it will
modify, without rescanning or reparsing the project. A changed target requires
a new preview. Unrelated files added after the preview are left for a later
migration. Successful execution or interactive cancellation deletes the
preparation, and a newer preview for the same projects and nro version replaces
an older one.

## `nro publish`

`nro publish` copies the public outputs of a completed request into a new
dataset directory:

```bash
nro publish REQUEST_ID /destination -P PROJECT
```

The command assesses the request and copies its terminal public files to a
staging directory. It verifies checksums, embeds recursive provenance, and
checks the source generations again before publishing. The destination must not
exist yet. The source root comes from the global site configuration.

The published dataset includes `dataset_description.json` and
`.nro-publication.json`. If the external BIDS validator is on `PATH`, the
command runs it, and `--no-validate` skips it. Publishing creates a snapshot. It
does not claim that every working derivative format passes the BIDS validator.
Inspect the published files and the validation output before you distribute the
dataset. The command does not upload anything or create a release.
