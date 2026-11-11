# Installation

This guide is for people who install, maintain, or develop against an `nro`
installation. It covers each installation mode, the settings setup asks for,
the resources it downloads, and how to check and maintain the result. For a
first personal setup, the [new-installation quickstart](quickstart.md) is
shorter.

## Installation modes

Run `./install` from a Linux checkout. The script finds its own checkout even
when you invoke it from another directory. First setup asks for one of three
modes and saves the choice in the untracked `.nro-installation.json`:

- `personal` installs a tagged `main` release for one user or account.
- `shared` installs a tagged `main` release that other users connect to. See
  [shared installations](#shared-installations).
- `branch` installs a development checkout that uses an existing site. See
  [development checkouts](#development-checkouts).

Personal and shared installations both publish the scheduler implementation.
Ordinary `run` commands start or join the scheduler, which launches pinned
planner children inside its allocation. Personal mode does not plan or submit
scientific workers directly from the calling process. Only installation
maintenance may access the registry before the scheduler binding exists.

Pass `--local` when setting up a host without Slurm.

## What setup installs

The host needs `curl` or `wget`, `sha256sum`, and `tar`. It does not need its
own Python. The `./install` bootstrap is a POSIX shell script. It selects a
checksum-pinned standalone uv archive for the host architecture and libc, then
installs the repository's exact uv-managed CPython version. On a platform
without a pinned uv archive, it fails before changing the installation.

The bootstrap environment is content-addressed under `.nro-bootstraps/`, and
the stable `.nro-bootstrap` symlink selects its active version. That
environment contains only the pinned bootstrap Python package. The standalone
uv binary lives under `.nro-tools/`. Application dependency layers use the same
managed Python version and the locked packages in `uv.lock`. The interpreter
supplies SQLite 3.35 or newer, independent of the host's Python and SQLite
libraries. Python distributions downloaded by uv stay in `.nro-python`, so a
shared environment does not depend on the maintainer's home directory. Setup
does not modify existing Conda environments.

Personal and shared installations use staged, immutable application and
dependency layers. Pulling new commits into the checkout therefore does not
change the active command until setup completes. A failed or interrupted setup
leaves the previous installation active, and a failed first setup publishes no
active installation. The one-time conversion of a legacy bootstrap directory is
journaled and recoverable. Branch installations instead use an editable
environment in the checkout's `.nro-env`.

## Optional components

Setup includes the components most sites need. These options change what it
installs:

- `--dev`: Include the locked test dependencies.
- `--with-bidsify`: Install Flywheel, DICOM, and Google Drive ingestion dependencies. The
  [bidsification guide](commands/bidsify.md) covers the required validator,
  server credentials, and staging configuration.
- `--with-lesion`: Install PyTorch, MONAI, safetensors, and SOCKS proxy support for
  lesion-aware anatomy, and acquire the pinned SynthStroke model. See
  [lesion-aware anatomy](#lesion-aware-anatomy).
- `--with-cicada`: Install the pinned pycicada classifier. CICADA is an optional
  functional-processing backend. Installations that do not request it neither
  install its package nor check that it is available.
- `--without-marss`: Omit the pinned official MARSS package, which setup includes by default.
  `diagnose` does not import MARSS, but its default `auto` mode needs it
  whenever the multiband-factor threshold calls for correction.
- `--without-oslom`: Skip OSLOM and its Python dependencies for this setup or maintenance run.
  ICA and clustering do not need OSLOM. Existing OSLOM binaries are not
  deleted.
- `--without-viewer`: Skip the Workbench image and launcher checks when scene viewing and
  rendering are not needed. Scientific processing is unaffected, and a later
  installation without the flag restores viewer support.

Ordinary setup always acquires pinned FreeSurfer 7.4.1 and FastSurfer 2.5.4
images for the selectable anatomical reconstruction engines.

## The launcher

Setup installs the launcher as `~/.local/bin/nro`. If setup reports that
`~/.local/bin` is not on your `PATH`, add it to your shell startup file. No
environment activation is needed.

Inside the directory tree of an installed checkout, the launcher selects that
checkout. Elsewhere, it selects your default installation. Connecting another
checkout does not change the default. Run `./install --default` to make the
connected checkout your default. This changes only your own launcher index. It
does not change other users' defaults or the checkout's write authority, and
working-directory selection still takes precedence inside connected checkouts.
Launcher checkout bindings live in `.nro-launchers.json` beside the launcher,
not in the shared registry.

The managed launcher must be the `nro` command that `PATH` selects. A shell
alias or an activated environment's own `nro` can bypass it. Use `type -a nro`
to see how the shell resolves the command. The launcher excludes ambient Python
import paths when it selects an interpreter.

If your installed command is the earlier fixed-path nro launcher, add
`--replace-launcher` to authorize its replacement. The installer checks the old
launcher's format and installation record, keeps a `nro.previous-...` backup
beside it, and installs the dispatcher. Without `--default`, the previous
default is preserved. Unrelated commands, modified shell scripts, and symlinks
are refused even with this flag. Choose another directory with `--bin-dir` or
move them aside yourself. The installer prints the resulting default and any
backup path.

`python -m nro.bin.run` remains available through the installation's
interpreter. It does not use working-directory selection.

## Site settings

Durable site settings live in `site/site.yml` in the
[definitions repository](definitions.md#protected-site-settings). Keep that
repository under version control. The document covers storage roots, external
resources, execution settings, and ingestion sources shared by every checkout
connected to the scheduler.

### Choosing paths during setup

The path editor shows all proposed settings before it asks for changes. Accept
them all to keep the displayed values. If you decline, it prompts for each
setting in turn. Enter keeps a value and Tab completes paths. A final
confirmation saves the choices. Use `-` for an empty Slurm account.

When `/juice6/u/nlp/climblab` is readable and traversable, the editor offers
the lab defaults. Otherwise it proposes paths under your resolved home
directory, such as `~/nro/bids`, `~/nro/work`, `~/nro/templateflow`, and related
resource directories. The generic private-control root is `~/nro/.nro`,
independent of the checkout location. Users joining a shared deployment should
select its existing root. Branch installations inherit it from the selected
site.

The editor does not create these directories. Previously configured paths and
environment overrides are kept even if they are unavailable. Interactive and
noninteractive setup use the same resolved defaults.

During a first personal or shared installation,
`./install --default-parent-dir /path/to/parent` proposes `/path/to/parent`
instead of `~/nro` for all site storage and resource paths. The option does not
move the checkout, launcher, or site locator. It cannot modify an existing site
or branch installation.

### The definitions repository and site locator

The `definitions` setting selects the site-owned repository. It defaults to
`/juice6/u/nlp/climblab/nro-definitions` when lab storage is accessible and to
`~/nro/definitions` otherwise. Setup creates a missing store from generic
starters and validates an existing one. New stores have no first-level models,
event tables, or configured Flywheel servers.

Each installation also has a generated TOML locator that contains only the
definitions repository path. Personal installations default to
`$XDG_CONFIG_HOME/nro/site.toml`, or `~/.config/nro/site.toml`. Setup
records the resolved path, so a symlinked home directory does not appear in
it. Shared
installations default to `.nro-site.toml` beside the checkout. Choose another
locator with `./install --site /absolute/site.toml` during first setup.
`NRO_SITE_CONFIG` selects one before an installation is registered. Personal,
shared, and branch installations enforce their recorded locator so that every
command reaches the scheduler and registry configured during installation.

Older sites kept settings in the TOML file. During the first maintenance run
after the current layout was introduced, setup moves those settings and the
protected ingestion fields into `site/site.yml` and reduces the TOML file to its
locator. Commit the migrated definitions repository after review.

Definitions stores have their own schema. Shared maintenance applies required
migrations to a staged copy, validates it, and then rewrites the affected files
in place. This keeps the files readable and preserves the repository's Git
history. A branch installation may migrate only its selected private overlay.
Shared and parent stores are read-only to a branch and must already use the
current schema.

### Viewing and changing settings

`nro site` reads and edits the site document after setup:

```bash
nro site ls
nro site set storage.bids=/data/BIDS storage.work=/scratch/nro
nro site set execution.runtime=/usr/bin/apptainer
nro site get slurm.partition slurm.scheduler.time_hours
```

`nro site ls` lists storage, resources, execution policy, Slurm profiles, and
bidsification defaults with their effective values and when changes take
effect. `nro site set` updates `site/site.yml`. It does not relocate files or
migrate a registry.

On a shared site, storage and installation-resource edits require `--maintain`,
write permission, and a stopped worker pool. Operational settings do not.
Changes to live scheduler limits are relayed automatically when a scheduler
exists, and otherwise apply when coordination resumes. Worker attempts keep
their immutable execution snapshot.

### Resource and Slurm settings

`workbench` names `wb_command`, and `wb_view` is expected beside it. `qunex`,
`synthstrip`, `synbold`, `gradient_unwarp`, `fastsurfer`, `synthstroke_data`,
and `mni_template` can override individual resources that are otherwise derived
from their parent directories. `binds` is a TOML list. Generic defaults omit
CLIMBLAB's `/juice6` bind.

`partition` routes scientific workers. `viewing_partition` separately routes
the persistent, user-specific X11 allocation that `nro scene --open` creates.
CLIMBLAB uses `john` for viewing, and both partitions use the `nlp` account.
The viewer profile defaults to 2 CPUs and 32 GB for up to 12 hours and is
configurable under `slurm.viewer`. Viewer allocations do not enter nro's worker
pool or shared concurrency accounting.

`flywheel_server` and `flywheel_project` are optional defaults for
`nro bidsify`. The project uses the `GROUP/PROJECT` form, and command-line
values override these defaults for one invocation:

```bash
nro site set bidsify.default_server=cni bidsify.default_project=cashain/climblab
```

Personal installations also honor `NRO_WORK_PATH`, `NRO_WB_COMMAND`,
`TEMPLATEFLOW_HOME`, and `FS_LICENSE`. Every installation reads the BIDS root
from the protected site document. The engine supplies the FreeSurfer license
and thread settings to processing commands.

Scientific YAML refers to resources with explicit `site:KEY` references.
Resolving them with the unmodified CLIMBLAB defaults preserves existing
workflow fingerprints. Changing a resolved resource path can still change a
configuration fingerprint, because this release does not infer scientific
equivalence after a resource moves.

### Private state

Private state uses the
[shared/branch hierarchy](commands/branches.md#storage-and-safeguards). A
control store in the older flat layout predates the supported registry baseline
and is rejected before a scheduler is created. Archive or remove that obsolete
private state before initializing the current layout. Installation and repair
never adopt it.

## Dependencies and downloads

Python requirements are declared in `pyproject.toml` and resolved in `uv.lock`.
Maintainers regenerate the lock when they change dependencies. Normal setup
uses the existing lock and never upgrades dependencies implicitly.

### Container images

Setup obtains QuNex 1.5.1, SynthStrip 1.7, SynBOLD-DISCO 1.4, and FastSurfer
2.5.4 from pinned OCI digests. A site with an active gradient-unwarping profile
also gets the pinned HCP base image that provides the correction tools. Sites
without such a profile neither download nor require that image.

Existing configured images are reused and tested. Downloads use temporary paths
and resource locks, then publish completed files atomically. Image receipts
record the source and the SHA-256 of the generated SIF. A receipt documents
acquisition, not the history of a resource that already existed.

After a large container's checksum is verified, nro associates it with the
file's device, inode, size, and timestamps, so unchanged images avoid repeated
multi-gigabyte reads. A changed identity triggers a new checksum before the
cached identity is updated. Container identity checks inspect pinned OCI
metadata without starting scientific software on the installation host.
Explicit execution probes belong to `nro doctor --deep`.

Review the [QuNex access terms](https://qunex.yale.edu/access/) before
approving downloads. For unattended setup, `--accept-qunex-license` accepts
them without a prompt.

### Container runtime

The default `singularity` or `apptainer` runtime setting selects nro's managed,
unprivileged Apptainer 1.4.5-1 bundle. Setup verifies the bundle checksum,
installs its complete relocatable tree beside the image directory, records a
file inventory, and stores the absolute executable in the site definition.
Compute nodes therefore do not need a system Singularity or Apptainer command.
Offline setup reuses the verified tree or its cached bundle, and reports the
missing artifact otherwise.

The bundle includes `squashfuse_ll` and `fuse-overlayfs`, which let Apptainer
mount a SIF image without unpacking it. When `/dev/fuse` is readable and
writable, setup checks and scientific jobs append the bundle's `bin` directory
to `PATH` so Apptainer can find these helpers. Helpers already on `PATH` take
precedence. Without FUSE, Apptainer unpacks each image into a temporary sandbox
at every launch. That is slow for large images and fails on hosts that refuse
unprivileged overlay mounts.

To use an administrator-provided runtime, set `execution.runtime` to its
absolute executable. Setup keeps that choice and does not fall back to the
managed bundle. Both choices require host support for unprivileged containers,
and nro cannot change kernel or cluster security policy. See the
[Apptainer installation guide](https://apptainer.org/docs/admin/1.4/installation.html).

### Workbench

Workbench 2.2.1 is installed from the checksum-pinned
[`nro-workbench`](https://github.com/CLiMBlab-Stanford/nro-workbench)
Apptainer image. The image contains the GUI, Qt, X11, and software-rendering
libraries that `nro scene`, `nro view`, and `nro render` use, so these commands
do not depend on compatible host graphics libraries. Small `wb_command` and
`wb_view` launchers bind the site's managed paths into the image. A working
executable configured outside an nro-managed Workbench directory remains an
explicit native override.

Scientific modules run their Workbench commands inside the pinned QuNex image,
so `--without-viewer` does not affect processing.

### Templates

The template catalog pins MNI T1w and GM probability maps and fsaverage surface
geometry by S3 object version and checksum. It includes the 41k geometry needed
by the default fsaverage6 target. Setup derives the matching midthickness
surface deterministically from the pinned white and pial surfaces. It downloads
only these selected files, and neither setup nor processing needs the
TemplateFlow Python client.

### OSLOM

Setup includes OSLOM, `igraph`, and `leidenalg` by default. If `oslom_undir` is
absent, setup downloads the official
[OSLOM source](http://www.oslom.org/software.htm), checks its pinned SHA-256,
and builds the undirected solver with `g++`. The host must provide the
compiler. Before publishing the executable, setup fits the bundled example
graph and saves a receipt with the source checksum, compiler, build arguments,
and binary checksum. Existing binaries are reused.

The official archive is served over HTTP. It is the downloader's only HTTP
exception, and it requires the pinned checksum. The checksum detects changes to
the accepted archive, but it does not authenticate the first HTTP download.

### Lesion-aware anatomy

The `lesion` extra installed by `--with-lesion` provides nro's built-in
SynthStroke adapter. nro owns this adapter so the anatomical graph does not
depend on an unversioned site wrapper. The adapter checks the pinned model
configuration and weight hashes, runs inference on a canonical 1 mm grid, and
restores the lesion probability map to the exact source grid before
thresholding.

Setup downloads the pinned SynthStroke model revision from Hugging Face once,
into the site-managed `synthstroke_data` directory. Every model file is checked
against a pinned SHA-256 before publication. Workers use these resources
read-only, so a scientific job never downloads or silently replaces model
weights. A site may instead set `lesion.masker_command` to an executable that
implements the same command interface.

The lesion scaffold is synthesized on a CPU with the `mri_synthsr` model
bundled in the pinned FreeSurfer image. It needs no separate checkpoint download
or lesion-only GPU allocation. Shared installations add the stack with
`./install --maintain --with-lesion`. Development branches use
`./install --with-lesion` from their checkout.

### FreeSurfer license

Supply your own
[FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html).
Setup reports a missing license as optional, and it never requests credentials
or registers on your behalf. FreeSurfer-dependent work fails before execution
until the configured license file exists.

## Unattended and offline setup

For unattended setup, prepare the site file first:

```bash
./install --mode personal --site /path/to/site.toml \
  --non-interactive --accept-qunex-license
```

`--offline` forbids new resource downloads and tells uv to use its cache. It
requires the cached standalone uv archive, managed interpreter, bootstrap
package, Python dependencies, and every required external resource.

Ctrl-C exits with a cancellation message and status 130. Interrupted path
editing leaves the saved settings unchanged. An interrupted maintenance run
leaves the preceding installation active. Rerunning it recovers any incomplete
bootstrap transition and resumes from the installed resources. To resume shared
setup, use `./install --maintain`.

## Checking an installation

`nro doctor` first reports the checkout, branch, environment, executable, and
site settings selected for the invocation. It then checks Python packages,
resource availability, directory access, and Slurm commands. It does not import
the scientific libraries or parse every definition. `--local` makes the Slurm
checks optional. OSLOM checks are mandatory unless you pass `--without-oslom`.
`--json` returns structured results.

`nro doctor --deep` goes further without processing subject data. It:

- imports the scientific libraries and validates every stored definition;
- starts each container and checks the required tools and license bind in
  QuNex;
- checks the HCP Pipelines, MSM, pyFIX, Octave, FIX-model, template, and
  dedrifting resources that optional MSMAll anatomy requires; and
- verifies pinned template checksums and existing image acquisition receipts.

Setup runs the deep checks before it marks an installation ready. Run
`nro doctor --deep` inside a compute allocation as well as on the login node to
check shared mounts and runtime policy there. A GUI display is only needed to
open Workbench scenes.

## Shared installations

A shared installation lets many users run the same release and worker pool.
The shared checkout must be on a clean `main` commit with an annotated release
tag that matches `pyproject.toml` and belongs to `origin/main`. The first
maintainer selects shared mode:

```bash
./install --mode shared
```

The installer registers or attaches the checkout as `main`, records the
checked-out release, and activates its environment for future workers. It uses
local Git state, so no separate `nro branch` or `nro release` command is
needed. A new site can start at the current release without recording every
earlier release.

Other users then run `./install` in the same checkout. It reads the
installation record and creates their launcher. It does not synchronize Python
dependencies or write into the checkout. Users cannot connect to an incomplete
installation.

### Permissions

Users must be able to read and traverse the shared checkout, environment, and
site file. Restrict write access to maintainers through filesystem ownership or
ACLs. Shared setup creates new files with a readable umask, but it does not
rewrite permissions on existing trees. BIDS derivatives, the work directory, and
the registry need the site's shared write permissions independently of the
software tree.

The shared checkout supplies the installed scheduler implementation. New work
captures its selected source before submission, so editing the checkout does
not change attempts that have already launched. Use registered development
checkouts for feature work instead of editing the shared installation.

### Maintenance

Maintenance is an explicit step:

```bash
./install --maintain
```

The installer works through the active scheduler or a fenced one-shot update
process. Maintenance does not start a scheduler, request a Slurm allocation, or
wait for capacity in the scientific partition. Shared setup verifies container
images by checksum without launching them on the login host. To test container
execution, run `nro doctor --deep` in the same kind of Slurm allocation that
scientific work uses.

When shared work is active, the installer reports it and offers three choices:

- A graceful drain prevents new claims, lets running derivative and ingestion
  stages finish, and then stops the workers.
- An immediate stop interrupts running work and keeps its demand so it can be
  queued again.
- Cancelling makes no registry change.

For either stop, the installer waits for Slurm to confirm that every allocation
has ended before changing the environment. Installation continues once the
pool is quiet. Pressing Ctrl-C after choosing an action leaves the maintenance
barrier and the chosen action in place. Repeat `./install --maintain` to
resume. Noninteractive maintenance refuses an active pool unless `--drain`
authorizes a graceful drain.

### Upgrading to a new release

After you update `main` to a newer tagged release, `./install --maintain`
records and activates that release automatically. Runtime checks compare the
installation's commit, tree, package version, environment, site, and source
fingerprint with the active release record.

Maintenance first audits every scheduler and branch database and every public
ownership record without changing them, and it reports all corrupt or obsolete
records together. Schema differences call for migration or reconstruction.
SQLite or branch-identity failures stop maintenance before the active
installation changes.

Once the pool is quiet, schemas at or after the supported baseline are
migrated on staged database copies. Each copy must match the generated target
schema and pass foreign-key and SQLite integrity checks before an atomic swap.
Older schemas are reconstructed from durable ownership records and public
artifacts. The previous scheduler state and every changed branch database are
kept as rollback copies. A durable phase record lets a repeated invocation
resume the same maintenance transaction. Corrupt or obsolete derivatives are
reported and never silently assigned to a new lineage. See
[registry migrations](registry-migrations.md).

Shared maintenance keeps third-party dependencies separate from nro source.
The dependency environment lives under `.nro-environments/dependencies-*`,
addressed by the locked dependency set, selected extras, Python version, and
platform. A later release reuses it when these inputs are unchanged. A changed
lock, extra selection, Python version, or platform creates a new environment
without modifying the active one. During the first maintenance run after this
layout was introduced, the installer asks uv to verify the active environment
against the selected locked dependencies and adopts it when they match, which
avoids rebuilding an equivalent environment.

The candidate nro source is stored separately as a verified, content-addressed
application snapshot. Setup and command launch combine that snapshot with the
selected dependency interpreter. After resource checks and registry maintenance
pass, the installer publishes the scheduler binding and installation record
while it holds the installation barrier. A failed cutover restores both
previous records. A successful cutover may remove dependency environments that
are no longer active.

### Rehearsing an upgrade

The pull-request release gate rehearses the upgrade from the target release.
Run the same check locally when you change installation, orchestration schemas,
migrations, or release management:

```bash
./install --rehearse-upgrade
```

By default the baseline is the newest release tag. Pass a Git ref after the
option to select another one. The rehearsal clones that revision into a
temporary directory, gives it an isolated registry and BIDS root, and then
replaces its executable source with the current working tree. It exercises any
required schema migration or reconstruction, candidate maintenance, and the
real one-shot scheduler. It does not alter the configured site, shared
installation, user launcher, or scientific data. It also does not install or
validate third-party scientific software. Use `nro doctor --deep` for those
deployment checks.

### Recovering direct definition edits

Changes made to the definitions store outside the authoring commands normally
block setup. To keep an intentional direct edit, run
`./install --adopt-definitions`, adding `--maintain` for a shared installation.
The installer validates and publishes all current drift in the definitions
store before continuing. Validation runs in the staged candidate environment,
so recovery does not depend on packages in the system Python. Development
branches cannot use this option.

### Replacing the original shared development checkout

An existing shared installation can stay in use while you prepare a separate
shared checkout. Select the existing site with `--site` when you install the
replacement. Do not reset the registry or copy a virtual environment or
`.nro-installation.json` between checkouts. If the generated definitions
locator lives inside the old development checkout, recreate it from the
protected definitions repository rather than copying it as configuration.

When the replacement shared installation is ready, connect it as your default:

```bash
./install --default --replace-launcher
```

`--replace-launcher` is needed only for the earlier fixed-path launcher. Each
user connects their own default, and one user's choice does not change another
user's shell resolution. The successfully installed tagged checkout is already
the active main release.

To convert the old shared checkout to development mode, schedule a maintenance
window with all clients. Finish or cancel demand, stop workers and pending
allocations, and resolve active ingestion and review leases. Then run this from
the old checkout:

```bash
./install --convert-to-branch
```

Conversion requires a named non-main Git branch and a different, ready shared
default. It uses that default's site file, which must lie outside the
development checkout and resolve to the same paths and settings as the original
site. Conversion registers or attaches the branch and keeps the original record
in `.nro-installation-transition.json`. It does not synchronize dependencies,
delete the environment, move artifacts, reset registries, create demand, or
change your default.

An interrupted conversion blocks scientific execution. Repeat the same command
to resume, and do not restore the old shared record by hand. If only the
launcher connection failed after conversion, reconnect with the same command.
Other users whose default still selects the converted checkout must connect to
the replacement shared installation. Their launchers will not silently fall
back to production.

After conversion, the checkout uses branch-owned outputs, compatible ancestor
inputs, and the shared scheduler.

## Development checkouts

A development checkout is a branch installation that shares a site's scheduler
and worker pool but writes its own outputs. If your launcher already has a
default installation, running `./install` in a new checkout selects branch mode
and reuses that site's settings. Otherwise, select an existing site explicitly:

```bash
./install --mode branch --site /path/to/site.toml
```

Branch setup creates an editable environment in the checkout's `.nro-env`,
checks shared dependencies without installing them, and registers or attaches
the Git branch with the central store. It does not update the production
environment, site settings, or definitions. Branch mode rejects `--maintain`,
and `main` cannot be installed as a development branch.

The branch stays selected in the checkout's subdirectories. Outside the
checkout, the launcher selects the previous default. If only a branch has been
installed, there is no default outside the checkout. An incomplete
installation, a switched or detached branch, or a revoked registration produces
an error instead of falling back to production.

Branch processing requires an
[installed, active main scheduler](commands/releases.md). The normal commands
then use branch-owned outputs, compatible ancestor inputs, and the shared
worker pool. Development installations cannot edit shared definitions or site
settings. Use `nro branch`, `nro doctor`, and `nro site ls` to inspect the
setup. See [development](development.md#branch-isolation-work) for execution
and maintenance boundaries.
