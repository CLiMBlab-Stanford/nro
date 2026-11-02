# Definitions stores

This page is for site maintainers and users who author definitions. A
definitions store is the directory, outside the nro installation, that holds a
site's scientific definitions and protected site settings. The page explains
how to create, select, validate, and edit a store, and what each part of it
contains.

The site owns the store and can track it in a separate Git repository.
Installation migrations may rewrite definitions when their schema changes. They
preserve the repository and leave the resulting text changes for review.

`nro site get storage.definitions` shows which store is selected. An explicit
site setting takes precedence. Otherwise the default is
`/juice6/u/nlp/climblab/nro-definitions` when lab storage is accessible, and
`~/nro/definitions` when it is not.

## Layout

```text
DEFINITIONS/
├── site/site.yml
├── configs/CLASS/ID_CLASS.yml
├── workflows/ID_workflow.yml
├── models/ANALYSIS/VARIANT.yml
├── markup/ID_markup.yml
├── hardware/gradient_unwarping.yml
├── events/TASK/index.yml
├── events/TASK/*.tsv
├── bidsify/PROFILE.yml
└── scanplans/parser.py
```

| Path | Contents |
| --- | --- |
| `site/site.yml` | The protected site document: shared storage, execution resources, and ingestion sources. See [protected site settings](#protected-site-settings). |
| `configs/`, `workflows/` | [Configurations and workflows](configuration.md) that control processing. |
| `models/` | [First-level models](task-models.md) that define predictors and contrasts. |
| `markup/` | [Source markup](#source-markup) that selects manual anatomical inputs and excludes known-bad BIDS paths. |
| `hardware/` | [Gradient-unwarping hardware](#gradient-unwarping-hardware) policy. |
| `events/` | [Event tables](event-files.md) that supply stimulus timing during bidsification. |
| `bidsify/` | [Ingestion profiles](commands/bidsify.md) that describe conversion rules and worker resources. |
| `scanplans/` | An optional [site scan-plan parser](commands/scanplans.md) that connects those profiles to a local or Google Drive source. |

Credentials, registry databases, imaging data, and generated outputs do not
belong in the store.

Every configuration class inherits its packaged `main` definition from the nro
installation. A store may omit `main_CLASS.yml`. If the file is present, it is
a partial site override. Named configuration IDs and workflows still need
matching files in the store, and the registry supplies no fallback. Deleting an
external `main` restores the packaged values.

## Create and select a store

```bash
nro def init /data/lab/nro-definitions
nro site set storage.definitions=/data/lab/nro-definitions
```

`nro def init` copies the packaged workflows, named configuration examples, and
an empty `main` markup document. It creates empty model and event catalogs and
adds a conversion profile. A shared or personal store also gets a complete
protected site document. A store created from a development installation omits
that document and inherits the shared site. The packaged `main` configurations
stay in the nro installation and are inherited, not copied.

Creation validates the staged files before publishing them. It refuses an
existing destination, even an empty directory. It does not change site
settings or initialize Git. Omit the path to create the currently selected
store. Both commands also run as `python -m nro.bin.COMMAND`, like the other
installation commands.

`./install` creates the selected store if it is missing and validates it if it
exists. Installation never merges new named examples into an existing store.
Package upgrades supply new defaults automatically unless the store overrides
them on purpose.

On shared installations, changing the authoritative path requires
`nro site set storage.definitions=PATH --maintain` and an inactive worker pool.
Selecting a path does not move any files, so copy and validate the destination
before switching. Every checkout connected to one scheduler inherits the same
site document.

Publication uses a non-replacing rename when the filesystem supports one. On
shared filesystems without that operation, nro reserves the destination and
marks it incomplete until publication finishes. If publication is interrupted,
the marker stays and readers reject the store. Inspect the incomplete directory
and move it aside before you create the store again. Creation never overwrites
it on retry.

## Validate and edit

```bash
nro def validate
nro def validate /data/lab/nro-definitions --json
```

Each store has a `.nro-definitions.yml` manifest. It records the store's schema
version and the SHA-256 digest of every managed definition. YAML and Python
definitions also begin with a notice that names the supported authoring
commands. Readers reject files added, removed, or edited directly, because
those changes bypass validation and branch ownership checks.

`nro def validate` checks the selected store when you omit the path. It reads
every definition, including unused variants, and reports:

- malformed filenames and missing packaged defaults;
- invalid configuration keys and broken workflow references;
- invalid task models and invalid source markup;
- bad event tables and unindexed TSVs;
- invalid ingestion profiles and an invalid protected site document, when one
  is present; and
- an invalid scan-plan parser interface.

It also rejects symlinks in definition directories and event references that
cross tasks. Empty model and event catalogs and empty Flywheel server mappings
are valid starting points. Installation requires `site/site.yml` in the
authoritative store, and branch selection requires it to be absent from a
private store.

Validation does not contact Flywheel, load imaging data, check credentials,
submit jobs, or change registry state. It cannot show that the definitions suit
the data scientifically, or that referenced software and external event files
are available. Use `nro doctor` to check dependencies. Both store commands
support `--json` and exit nonzero on failure.

### Editing definitions

Use [definition authoring](commands/authoring.md) for configurations,
workflows, first-level models, and source markup. Use `nro site set` for
protected site values.

Event catalogs, ingestion profiles, hardware policy, and scan-plan parsers have
no typed editor. Publish them from local files as one validated transaction:

```bash
nro def apply \
  --file events/mytask/index.yml=./index.yml \
  --file events/mytask/main.tsv=./main.tsv
nro def apply --file bidsify/main.yml=./main.yml
```

`nro def edit file RELATIVE_PATH` edits or creates one of these untyped
definitions interactively.

A transaction stages the complete store, updates its manifest, validates every
definition and reference, and only then replaces the affected files. If
validation fails, the store is left unchanged. The editor keeps unpublished
work in private, ignored draft directories and offers it the next time you edit
the same definition. Keep unrelated notes outside the structured definition
directories.

If someone has already edited files directly, bare `nro def apply` validates
all the current drift and adopts it as one transaction. An explicit
transaction must name every drifted path. This recovery path exists for
mistakes. Direct editing is still not a supported way to work.

## Protected site settings

The shared definitions repository is the only authority for `site/site.yml`.
The file records facts that every checkout connected to the scheduler must
share:

- storage roots for BIDS, work, development outputs, and private control state;
- external software and resource locations;
- the container runtime, execution policy, Slurm allocation profiles, and bind
  paths; and
- Flywheel servers, mappings from destination to source projects, scan-plan
  sources, credential variable names, and rules for existing sessions.

The document stores the names of credential variables, never their values.
Keep it under version control with the rest of the shared definitions
repository. After installation, edit it with `nro site set`, then review and
commit the change through the site's normal definitions process.

Each checkout keeps a generated TOML file that contains only the absolute path
to the shared definitions repository. This file is disposable installation
state, and `./install` can recreate it. Each submitted attempt still receives a
full, immutable TOML snapshot, so later site edits cannot change running work.

The document has six top-level fields:

```yaml
version: 4
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
  binds: []
  concurrency: 50
  gpu_concurrency: 1
  worker_idle_timeout: 30
  worker_drain_minutes: 15
slurm:
  partition: compute
  viewing_partition: interactive
  account: ''
  scheduler: {time_hours: 24, memory_gb: 8, cpus: 4}
  worker: {time_hours: 24, memory_gb: 32, max_memory_gb: 256, cpus: 2}
  long_worker: {time_hours: 48, cpus: 8}
  viewer: {time_hours: 12, memory_gb: 32, cpus: 2}
bidsify:
  default_server: null
  default_project: null
  servers: {}
  project_sources: {}
  scanplans: {location: null, credential_env: null}
  session_rules: []
  event_rules: []
```

The resources `qunex`, `synthstrip`, `synbold`, `gradient_unwarp`,
`fastsurfer`, `synthstroke_data`, and `mni_template` are optional overrides.
When they are omitted, nro derives them from `images` or `templates`. The
optional `lesion` Python extra supplies automatic stroke-lesion masking. An
empty `account` is valid for Slurm sites that do not use accounts.

### Branch definitions stores

A development branch may select a private definitions repository for
configurations, workflows, models, markup, events, hardware policy, conversion
profiles, and scan-plan parser code. A private repository must omit
`site/site.yml`.

Definitions resolve from the current branch first, then from each registered
parent in order, and finally from the shared repository. A file in a nearer
store replaces the file with the same category and ID in every later store.
Branch selection rejects a private site document. Central admission rejects a
request prepared against a different protected site.

### Private Flywheel credentials

`site/site.yml` defines Flywheel server IDs, hosts, and permitted projects. It
does not name environment variables or contain API keys. Each user registers a
key with `nro fw addkey SERVER`. See
[bidsification](commands/bidsify.md#before-first-use).

Keys live in the ignored `.definition-secrets` directory. It belongs to the
shared store but is not part of the store's schema or integrity manifest. Key
files are grouped by Unix user, use mode `0620`, and inherit the store's group.
Their directories let group members list, replace, and delete keys without
reading them. Store migrations and branch inheritance do not copy this
directory.

nro keeps every store file group-writable. It keeps every store directory
group-writable, group-searchable, and setgid. Existing read access is left as
it is, so making a private file maintainable does not make it group-readable.

## Gradient-unwarping hardware

`hardware/gradient_unwarping.yml` maps inherited BIDS acquisition metadata to
site policy for gradient unwarping. The file is optional. When no profile
matches, `gradient_unwarping: auto` leaves the acquisition unchanged. A private
branch store that omits the file inherits it from the shared site store. If
neither store defines it, nro uses an empty packaged catalog.

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

String and list matches are exact. A numeric range may set inclusive `minimum`
and `maximum` values. Each acquisition may match at most one profile.

A profile's `action` is one of two values. `action: unwarp` requires a
coefficient filename relative to the site's `gradient_coefficients` directory.
`action: already_corrected` declares that the matched hardware needs no
correction from nro, and it cannot name coefficients.

An `unwarp` profile normally respects `NonlinearGradientCorrection: true` and
does not correct the image a second time. Set
`override_existing_correction: true` only when the site knows that field is
wrong for the matched hardware. The artifact contract records the profile, the
matched metadata, the decision, the method, and the coefficient SHA-256 digest.
The absolute coefficient path is execution metadata and does not define
scientific identity.

### Acquisition metadata

The optional `acquisition_metadata` mapping records scanner facts that exported
DICOMs may omit. nro accepts these assertions only from the shared site's
hardware catalog. A development definitions override cannot change them.

```yaml
acquisition_metadata:
  nonlinear_gradient_correction: true
  gradient_correction_mode: 2D
  gradient_coil_model: SIGNA UHP gradient
```

`gradient_correction_mode` accepts `2D`, `3D`, or `none`. A `2D` or `3D` mode
requires `nonlinear_gradient_correction: true`, and `none` requires `false`.

During bidsification, series-level DICOM evidence takes precedence. A site
assertion fills in a missing value, and a disagreement blocks publication until
someone reviews it. A profile should identify one scanner installation closely
enough that its assertions hold for every matching acquisition.

## Source markup

A markup document records exceptions to ordinary BIDS discovery. Use it to
choose anatomical inputs by hand, hide known-bad source files, request
lesion-aware reconstruction, or fix the runs used for MSMAll calibration.

The top level of the document is the BIDS project, because participant labels
need not be unique across projects. Project and participant entries are
optional, and so is every field in a participant entry.

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
  sub-t21:
    msmall:
      rest:
        - ses-rest1/func/sub-t21_ses-rest1_task-rest_run-01_bold.nii.gz
        - ses-rest2/func/sub-t21_ses-rest2_task-rest_run-01_bold.nii.gz
```

Paths are relative to the participant directory. The fields are:

- `T1w`, `T2w`: One path or a list. A marked modality supplies the candidates for that
  modality. For an unmarked modality, nro uses every discovered candidate that
  is not excluded. The anatomical module applies its configured selection
  strategy to the T1w and T2w candidates separately, so the selected sources
  may come from different sessions.

- `exclude`: A list. Each entry hides that path, and everything below it, from nro source
  discovery and metadata inheritance. A selected anatomical path cannot also be
  excluded.

- `lesion`: An optional boolean. `true` selects lesion-aware anatomical reconstruction
  for the participant. Omitting it or setting `false` selects ordinary
  reconstruction. The flag is a processing instruction, not a diagnosis.

- `msmall.rest`: An optional, fixed list of resting-state BOLD images used to estimate a fixed
  MSMAll surface registration. Every path must resolve below the participant
  directory. nro derives the matching metadata and opposite-PE fieldmap pair
  from BIDS. It does not infer or extend the calibration set. MSMAll
  calibration requires T1w and T2w anatomy, FreeSurfer reconstruction, and
  compatible fieldmaps for every selected run. In the initial implementation it
  cannot be combined with `lesion: true`.

### Selecting markup

Every module configuration has a `markup` field. Its default, `main`, selects
`markup/main_markup.yml`. Set it to `null` to ignore markup. The packaged
`main` document is empty, so a store without a `markup` directory keeps
ordinary discovery. All module configurations selected by one workflow must
select the same markup ID, so the workflow's graph reflects one consistent view
of source BIDS.

Manage markup with the typed definition editor:

```bash
nro def edit markup
nro def rm markup alternative
```

Planning captures the resolved participant entry in the artifact contract.
Workers therefore use the planned view even if the central document changes
during an attempt. Later assessment compares the source inputs that the
current document selects. An edit that changes no selected source content does
not by itself make an artifact stale.

## Version control and reproducibility

Track the store's definitions and README in their own repository. The
generated `.gitignore` excludes editor files, private definition drafts, and
nro's authoring locks. Check files for credentials and sensitive information
before committing. nro never commits, pulls, pushes, or changes branches on its
own.

Freshness compares compiled scientific content. Moving the store, editing
comments, committing, or switching Git branches does not invalidate
derivatives unless that content changes. Changing a referenced scientific
resource path can still change a contract, and moving the definitions
directory does not rewrite those resource paths.

Resolved execution snapshots remain the authority for submitted attempts.
Editing a configuration affects later resolution, not saved snapshots. Git
history adds to these records, but nro does not need it to run, and Git
revision identifiers are not part of scientific fingerprints.

## Schema migrations

`./install --maintain` migrates the shared definitions store before it
activates a new shared release. It applies each required schema migration to a
staged copy, validates the result, and then rewrites the affected files in
place. A rollback journal lets a later run recover if publication is
interrupted. The store stays human-readable, and its Git history shows the
resulting changes. Commit them after review.

A development installation never migrates the shared store or a parent's
private store. It may migrate only the private store selected for its own
branch. Inherited stores must already match the installed schema. If one does
not, installation stops and asks that store's owner to run maintenance.
