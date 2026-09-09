# Definitions stores

nro reads scientific definitions from a directory outside its installation.
The site owns this directory and can track it in a separate Git repository.
Code upgrades do not replace its contents.

`nro paths show` displays the selected `definitions` path. Its default is
`/juice6/u/nlp/climblab/nro-definitions` when lab storage is accessible,
otherwise `~/nro/definitions`. An explicit site setting takes precedence.

```text
DEFINITIONS/
├── configs/CLASS/ID_CLASS.yml
├── workflows/ID_workflow.yml
├── models/TASK/VARIANT.yml
├── events/TASK/index.yml
├── events/TASK/*.tsv
└── bidsify/PROFILE.yml
```

Configurations and workflows control processing; [task models](task-models.md)
define predictors and contrasts. [Event tables](event-files.md) supply stimulus
timing during bidsification. [Ingestion profiles](commands/bidsify.md) describe
Flywheel servers, acquisition rules, and conversion resources. Credentials,
registry databases, imaging data, and generated outputs belong elsewhere.

## Create and select a store

```bash
nro definitions create /data/lab/nro-definitions
nro paths set definitions=/data/lab/nro-definitions
```

Creation copies packaged starter configurations and workflows, creates empty
model and event catalogs, and adds an ingestion profile with no configured
servers. It validates the staged files before publication and refuses an
existing destination, even an empty directory. It does not change site settings
or initialize Git. Both commands accept the installation's usual Python-module
invocation through `python -m nro.bin.COMMAND`.

Omit the path to create the currently selected store. `./install` creates that
store if missing and validates it if present. Installation never merges new
starters into an existing store. Adopt changes to defaults explicitly through
normal definition editing and review.

On shared installations, changing the selected path requires
`nro paths set definitions=PATH --maintain` and an inactive worker pool. Selecting
a path does not move files. Copy and validate the destination before switching.
Shared registry users should select the same definitions store.

Publication uses a non-replacing rename when supported. On shared filesystems
without that operation, nro reserves the destination and marks it incomplete
until publication finishes. Interrupted publication leaves the marker in place;
readers reject that store. Inspect and move the incomplete directory aside before
recreating it. Creation never overwrites it on retry.

## Validate and edit

```bash
nro definitions validate
nro definitions validate /data/lab/nro-definitions --json
```

Omitting the path checks the selected store. Validation reads all definitions,
including unused variants, and reports malformed filenames, missing defaults,
invalid configuration keys, broken workflow references, invalid task models,
bad event tables, unindexed TSVs, and invalid ingestion profiles. It rejects
symlinks in definition directories and cross-task event references. Empty model
and event catalogs and empty Flywheel server mappings are valid starting points.

Validation does not contact Flywheel, load imaging data, check credentials,
submit jobs, or change registry state. It cannot establish scientific suitability
or availability of referenced software and external event files. Use `nro doctor`
for dependency checks. Both store commands support `--json` and exit nonzero on
failure.

Use [create, edit, and delete](commands/authoring.md) for configurations,
workflows, and task models. These commands target the external store. Their
existing staged validation, writer locks, and concurrent-edit checks still
apply. Edit event catalogs and ingestion profiles directly, then validate the
store. Keep unrelated notes outside the structured definition directories.

There is no fallback to packaged starters or registry-owned profiles when an
active definition is absent. Missing required files are errors. Packaged
starters exist only to initialize new stores and document default parameters.

## Version control and reproducibility

Track the store's definitions and README in its own repository. The generated
`.gitignore` excludes editor files and nro's authoring locks. Review files for
credentials and sensitive information before committing. nro never commits,
pulls, pushes, or changes branches automatically.

Freshness compares compiled scientific content. Moving the store, editing
comments, committing changes, or changing Git branches without changing that
content does not itself invalidate derivatives. Changing a referenced scientific
resource path can still change a contract; moving the definitions directory
does not rewrite those resource paths.

Resolved execution snapshots remain the authority for submitted attempts.
Configuration edits affect subsequent resolution, not saved snapshots. Git
history supplements these records but is not required to run nro, and Git
revision identifiers do not participate in scientific fingerprints.
