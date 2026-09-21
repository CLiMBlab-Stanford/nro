# Definitions stores

nro reads scientific definitions from a directory outside its installation.
The site owns this directory and can track it in a separate Git repository.
Installation migrations may rewrite definitions when their schema changes, but
they preserve the repository and leave ordinary text changes for review.

`nro paths show` displays the selected `definitions` path. Its default is
`/juice6/u/nlp/climblab/nro-definitions` when lab storage is accessible,
otherwise `~/nro/definitions`. An explicit site setting takes precedence.

```text
DEFINITIONS/
├── site/site.yml
├── configs/CLASS/ID_CLASS.yml
├── workflows/ID_workflow.yml
├── models/TASK/VARIANT.yml
├── markup/ID_markup.yml
├── hardware/gradient_unwarping.yml
├── events/TASK/index.yml
├── events/TASK/*.tsv
├── bidsify/PROFILE.yml
└── scanplans/parser.py
```

The protected `site/site.yml` document defines shared storage, execution
resources, and ingestion sources. Configurations and workflows control
processing; [task models](task-models.md)
define predictors and contrasts. Source markup selects manual anatomical inputs
and excludes known-bad BIDS paths. [Event tables](event-files.md) supply stimulus
timing during bidsification. [Ingestion profiles](commands/bidsify.md) describe
conversion rules and worker resources. A
[site scan-plan parser](commands/scanplans.md) may connect those profiles to a
local or Google Drive source. Credentials, registry databases, imaging data,
and generated outputs belong elsewhere.

## Protected site settings

The shared definitions repository is the only authority for `site/site.yml`.
This file records facts that every checkout connected to the scheduler must
share:

- storage roots for BIDS, work, development outputs, and private control state;
- external software and resource locations;
- the container runtime, Slurm partitions, account, and bind paths; and
- Flywheel servers, destination-to-source project mappings, scan-plan sources,
  credential variable names, and existing-session rules.

The document stores credential variable names, never credential values. Keep it
under version control with the rest of the shared definitions repository.
`nro paths set` edits this document after installation. Review and commit that
change through the site's normal definitions process.

Each checkout retains a generated TOML file containing only the absolute path
to the shared definitions repository. It is disposable installation state;
`./install` can recreate it. Submitted attempts still receive a full, immutable
TOML snapshot, so later site edits cannot change running work.

Development branches may select private definitions repositories for
configurations, workflows, models, markup, events, hardware policy, conversion
profiles, and scan-plan parser code. A private repository must omit
`site/site.yml`. Definitions resolve from the current branch, then each
registered parent in order, and finally the shared repository. A file in a
nearer store replaces the file with the same category and ID in every later
store. Branch selection rejects a private site document, and central admission
rejects a request prepared against a different protected site.

The document has five top-level fields:

```yaml
version: 1
storage:
  bids: /data/BIDS
  work: /scratch/nro
  development: /data/NRO_DEV
  registry: /data/.nro
resources:
  images: /opt/nro/images
  gradient_coefficients: /opt/nro/gradient-coefficients
  templates: /opt/nro/templateflow
  workbench: /opt/workbench/bin_linux64/wb_command
  oslom: /opt/oslom/oslom_undir
  license: /opt/freesurfer/license.txt
execution:
  runtime: apptainer
  partition: compute
  viewing_partition: interactive
  account: ''
  binds: []
bidsify:
  default_server: null
  default_project: null
  servers: {}
  project_sources: {}
  scanplans: {location: null, credential_env: null}
  session_rules: []
  event_rules: []
```

`qunex`, `synthstrip`, `synbold`, `gradient_unwarp`, `fastsurfer`,
`fastsurfer_data`, and `mni_template` are
optional resource overrides. When omitted, nro derives them from `images` or
`templates`. The optional `lesion` Python extra supplies automatic
stroke-lesion masking. An empty account is valid for Slurm sites that do not use
one.

## Gradient-unwarping hardware

`hardware/gradient_unwarping.yml` maps inherited BIDS acquisition metadata to
site policy. It is optional. With no matching profile, `gradient_unwarping: auto`
leaves the acquisition unchanged. A private branch definitions store that omits
this file inherits it from the shared site store. If neither store defines it,
nro uses an empty packaged catalog.

```yaml
version: 1
profiles:
  impulse_7t_head_coil:
    match:
      Manufacturer: Siemens
      ManufacturersModelName: MAGNETOM Terra.X Impulse Edition
      ReceiveCoilName: Neurocam_7T
      MagneticFieldStrength:
        minimum: 6.5
        maximum: 7.5
    action: unwarp
    coefficients: coeff_IMPULSE.grad
    acquisition_metadata:
      gradient_coil_model: Impulse head gradient (Athena)
```

String and list matches are exact. Numeric ranges may define inclusive
`minimum` and `maximum` values. Each acquisition may match at most one profile.
`action: unwarp` requires a coefficient filename relative to the site's
`gradient_coefficients` directory. `action: already_corrected` declares that the
matched hardware needs no nro correction and cannot name coefficients.

The optional `acquisition_metadata` mapping records scanner facts that exported
DICOMs may omit. nro accepts these assertions only from the shared site's
hardware catalog; a development definitions override cannot change them:

```yaml
acquisition_metadata:
  nonlinear_gradient_correction: true
  gradient_correction_mode: 2D
  gradient_coil_model: SIGNA UHP gradient
```

`gradient_correction_mode` accepts `2D`, `3D`, or `none`. A 2D or 3D mode
requires `nonlinear_gradient_correction: true`; `none` requires `false`.
During bidsification, series-level DICOM evidence takes precedence. A site
assertion fills a missing value, while disagreement blocks publication for
review. Profiles should identify one scanner installation closely enough that
their assertions apply to every matching acquisition.

An `unwarp` profile normally respects `NonlinearGradientCorrection: true` and
does not repeat correction. Set `override_existing_correction: true` only when
the site knows that field is incorrect for the matched hardware. The artifact
contract records the profile, matched metadata, decision, method, and coefficient
SHA-256 digest. The absolute coefficient path is execution metadata and does not
define scientific identity.

## Source markup

A markup document records exceptions to ordinary BIDS discovery. Its top level
is the BIDS project because participant labels need not be unique across
projects. Project and participant entries are optional, as are all fields in a
participant entry.

```yaml
nptl:
  sub-t20:
    T1w: ses-anat/anat/sub-t20_ses-anat_T1w.nii.gz
    T2w:
      - ses-anat/anat/sub-t20_ses-anat_run-1_T2w.nii.gz
      - ses-anat/anat/sub-t20_ses-anat_run-2_T2w.nii.gz
    exclude:
      - ses-bad/func/sub-t20_ses-bad_task-rest_bold.nii.gz
    lesion: true
```

Paths are relative to the participant directory. `T1w` and `T2w` accept one
path or a list. A marked modality supplies the candidates for that modality;
otherwise, nro uses all discovered, non-excluded candidates. The anatomical
module applies its configured selection strategy independently to the T1w and
T2w candidate sets. Their selected sources may come from different sessions.
`exclude` must be a list; each entry hides that path and everything below it
from nro source discovery and metadata inheritance. A selected anatomical path
cannot also be excluded.
`lesion` is an optional boolean. `true` selects lesion-aware anatomical
reconstruction for that participant; omission and `false` select ordinary
reconstruction. This flag is a processing instruction, not a diagnosis.

Every module configuration has a `markup` field. Its default is `main`, which
selects `markup/main_markup.yml`; set it to `null` to ignore markup. The packaged
`main` document is empty, so an existing definitions store without a `markup`
directory retains ordinary discovery. All module configurations selected by
one workflow must select the same markup ID. This guarantees that the workflow
DAG represents one consistent view of source BIDS.

Use the ordinary authoring commands to manage markup:

```bash
nro create markup main
nro edit markup main
nro delete markup alternative
```

Planning captures the resolved participant entry in the artifact contract.
Workers therefore use the planned view even if the central document changes
during an attempt. Later assessment compares the source inputs selected by the
current document. An edit that changes no selected source content does not by
itself make an artifact stale.

## Create and select a store

```bash
nro definitions create /data/lab/nro-definitions
nro paths set definitions=/data/lab/nro-definitions
```

Creation copies packaged workflows, named configuration examples, and an empty
`main` markup document. It creates empty model and event catalogs and adds a
conversion profile. A shared or personal store also gets a complete protected
site document. A store created from a development installation omits that
document and inherits the shared site. Packaged `main` configurations
remain in the nro installation and are inherited rather than copied. Creation
validates the staged files before
publication and refuses an existing destination, even an empty directory. It
does not change site settings or initialize Git. Both commands accept the
installation's usual Python-module invocation through `python -m nro.bin.COMMAND`.

Omit the path to create the currently selected store. `./install` creates that
store if missing and validates it if present. Installation never merges new
named examples into an existing store. Package upgrades supply new defaults
automatically unless the store deliberately overrides them.

On shared installations, changing the authoritative path requires
`nro paths set definitions=PATH --maintain` and an inactive worker pool. Selecting
a path does not move files. Copy and validate the destination before switching.
Every checkout connected to one scheduler inherits the same site document.

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

Each store has a `.nro-definitions.yml` manifest containing its schema version
and the SHA-256 digest of every managed definition. YAML and Python definitions
also begin with a notice that names the supported authoring commands. Readers
reject direct additions, removals, and edits because those changes bypass
validation and branch ownership checks.

Omitting the path checks the selected store. Validation reads all definitions,
including unused variants, and reports malformed filenames, missing packaged
defaults, invalid configuration keys, broken workflow references, invalid task
models, invalid source markup, bad event tables, unindexed TSVs, invalid
ingestion profiles, an invalid protected site document when present, and an
invalid scan-plan parser interface. It rejects
symlinks in definition directories and cross-task event references. Empty model
and event catalogs and empty Flywheel server mappings are valid starting points.
Installation requires `site/site.yml` in the authoritative store. Branch
selection requires it to be absent from a private store.

Validation does not contact Flywheel, load imaging data, check credentials,
submit jobs, or change registry state. It cannot establish scientific suitability
or availability of referenced software and external event files. Use `nro doctor`
for dependency checks. Both store commands support `--json` and exit nonzero on
failure.

Use [create, edit, and delete](commands/authoring.md) for configurations,
workflows, task models, and source markup. Use `nro paths set` for protected
site values. For event catalogs, ingestion profiles, hardware policy, and
scan-plan parsers, publish one or more local files as a validated transaction:

```bash
nro definitions apply \
  --file events/mytask/index.yml=./index.yml \
  --file events/mytask/main.tsv=./main.tsv
nro definitions apply --file bidsify/main.yml=./main.yml
```

`nro definitions edit RELATIVE_PATH` is available for an existing definition
that has no specialized editor. Transactions stage the complete store, update
its manifest, validate every definition and reference, and then replace the
affected files. A validation failure leaves the store unchanged. The editor
retains unpublished work in private, ignored draft directories and offers it on
the next edit of the same definition. Keep unrelated notes outside the
structured definition directories.

If a direct edit has already occurred, `nro definitions apply` can adopt it
only when the transaction names every drifted path. This recovery behavior does
not make direct editing a supported workflow.

Every configuration class inherits its packaged `main` definition. A store may
omit `main_CLASS.yml`; if present, that file is a partial site override. Named
configuration IDs and workflows still require matching external files, and
there is no registry-owned fallback. Deleting an external `main` restores the
packaged values.

## Version control and reproducibility

Track the store's definitions and README in its own repository. The generated
`.gitignore` excludes editor files, private definition drafts, and nro's
authoring locks. Review files for credentials and sensitive information before
committing. nro never commits, pulls, pushes, or changes branches automatically.

Freshness compares compiled scientific content. Moving the store, editing
comments, committing changes, or changing Git branches without changing that
content does not itself invalidate derivatives. Changing a referenced scientific
resource path can still change a contract; moving the definitions directory
does not rewrite those resource paths.

Resolved execution snapshots remain the authority for submitted attempts.
Configuration edits affect subsequent resolution, not saved snapshots. Git
history supplements these records but is not required to run nro, and Git
revision identifiers do not participate in scientific fingerprints.

## Schema migrations

`./install --maintain` migrates the shared definitions store before activating
a new shared release. It applies each required schema migration to a staged
copy, validates the result, and then rewrites the affected files in place. A
rollback journal lets a later invocation recover if publication is interrupted.
The store remains human-readable and its Git history records the resulting
changes.
Commit those changes after review.

A development installation never migrates the shared store or a parent's
private store. It may migrate only the private store selected for its own
branch. Inherited stores must already match the installed schema; otherwise the
installation stops and asks the appropriate owner to run maintenance.
