# Installation

Run `./install` from a Linux checkout. The script finds its own checkout even
when invoked from another directory. First setup asks for `personal`, `shared`,
or `branch` mode and saves that role in the untracked `.nro-installation.json`.

The bootstrap requires Python 3.11 or newer with `venv` and pip support.
It installs uv 0.8.22 in `.nro-bootstrap` and uses `uv.lock` to create an
editable Python 3.12 environment in `.nro-env`. Python distributions downloaded
by uv stay in `.nro-python`, so a shared environment does not depend on the
maintainer's home directory. Existing Conda environments are not modified.
Pass `--dev` to include test dependencies.
Pass `--with-bidsify` for Flywheel ingestion dependencies. The
[bidsification guide](commands/bidsify.md) covers the required validator,
server credentials, and staging configuration.
Pass `--local` when setting up a host without Slurm.

The launcher goes in `~/.local/bin/nro`. Add `~/.local/bin` to your shell's PATH
if setup reports that it is absent. No environment activation is needed.
The launcher selects an installed checkout when invoked inside its directory
tree, and the user's default installation elsewhere. Adding another checkout
does not change that default. Use `./install --default` to explicitly select
the successfully connected checkout as your default. This changes only your
launcher index, not other users' defaults or the checkout's write authority.
Working-directory selection still takes precedence inside connected checkouts.

If your installed command is the earlier fixed-path nro launcher, add
`--replace-launcher` to authorize replacement. The installer checks its format
and installation record, retains a `nro.previous-...` backup beside it, and
installs the dispatcher. Without `--default`, its previous default is preserved.
Unrelated commands, modified shell scripts, and symlinks are refused even with
this flag; choose `--bin-dir` or move them aside explicitly. The installer prints
the resulting default and any retained backup path.
`python -m nro.bin.run` remains available through the installation's interpreter;
it does not use working-directory selection.

## Development checkouts

With a default installation already connected to the user's launcher, running
`./install` in another, new checkout selects branch mode and reuses that site's
settings. Without that connection, select an existing site explicitly:

```bash
./install --mode branch --site /path/to/site.toml
```

Branch setup creates an editable environment in that checkout's `.nro-env`,
checks shared dependencies without installing them, and registers or attaches
the Git branch with the central store. It does not update the production
environment, site settings, or definitions. `--maintain` is rejected in branch
mode. Main cannot be installed as a development branch.

The branch remains selected in checkout subdirectories. After leaving the
checkout, the launcher selects the previous default. If only a branch has been
installed, there is no outside-checkout default. An incomplete installation,
switched or detached branch, or revoked registration produces an error instead
of falling back to production. The launcher excludes ambient Python import
paths when selecting an interpreter.

The managed launcher must be the `nro` command selected by PATH. A shell alias
or an activated environment's own `nro` command can bypass it; use `type -a nro`
to inspect shell resolution. Launcher checkout bindings live in
`.nro-launchers.json` beside the user launcher, not in the shared registry.

Branch processing requires an [installed, active main scheduler](commands/releases.md).
The normal commands then use branch-owned outputs, compatible ancestor inputs,
and the shared worker pool. Shared definition/site editing remains blocked from
development installations. Use `nro branch`, `nro doctor`, and `nro paths show`
to inspect setup. See [development](development.md#branch-isolation-work) for
execution and maintenance boundaries.

## Shared installations

The shared checkout must be on a clean `main` commit with an annotated release
tag that matches `pyproject.toml` and belongs to `origin/main`. The first
maintainer selects shared mode:

```bash
./install --mode shared
```

The installer registers or attaches the checkout as `main`, records the checked-out
release, and activates its environment for future workers. These actions use local
Git state and require no separate `nro branch` or `nro release` commands. A new site
can start at the current release without recording every earlier release.

Subsequent users run the same `./install`. It reads the installation record
and creates their launcher. It does not synchronize Python dependencies or
write into the checkout. An incomplete installation cannot onboard users.

Maintenance is explicit:

```bash
./install --maintain
```

When shared work is active, the installer reports it and asks permission to drain
the pool. Confirmation places the registry in installation maintenance, prevents
new claims, lets running derivative and ingestion stages finish, stops workers and
allocations, and preserves demand. Installation continues when the pool is quiet.
Ctrl-C leaves the maintenance barrier in place; repeat `./install --maintain` to
resume. `--drain` provides the same authorization for noninteractive maintenance.
Without that option, noninteractive maintenance refuses an active pool.

After updating `main` to a newer tagged release, `./install --maintain` records and
activates that release automatically. Runtime checks compare the installation's
commit, tree, package version, environment, site, and source fingerprint with the
active release record. The installer never repairs or resets the registry silently.

The shared checkout, environment, and site file must be readable and traversable
by users. Restrict write access to maintainers using filesystem ownership or
ACLs. Shared setup creates new files with a readable umask; it does not rewrite
permissions on existing trees. BIDS derivatives, WORK, and the registry need
the site's shared write permissions independently of the software tree.

The shared checkout supplies the installed scheduler implementation. New work
captures its selected source before submission; editing the checkout does not
change already launched attempts. Use registered development checkouts for
feature work instead of editing the shared installation.

### Replacing the original shared development checkout

An existing shared installation can remain in use while a separate shared
checkout is prepared. Select the existing site with `--site` when installing
the replacement; do not reset the registry or copy a virtual environment or
`.nro-installation.json` between checkouts. If the site TOML lives inside the
old development checkout, first copy those settings to a durable shared location
or into the replacement shared checkout, keeping their values unchanged.

After the replacement shared installation is ready, connect it as your default:

```bash
./install --default --replace-launcher
```

The replacement flag is needed only for the earlier fixed-path launcher.
Each user connects their own default; one user's installation does not change
another user's shell resolution. The successfully installed tagged checkout is
already the active main release.

To convert the old shared checkout to development mode, coordinate a maintenance
window with all clients, finish or cancel demand, and stop workers and pending
allocations. Resolve active ingestion and review leases too. Then, from that
checkout:

```bash
./install --convert-to-branch
```

Conversion requires a named non-main Git branch and a different, ready shared
default. It uses that default's site file, which must be outside the development
checkout and resolve to the same paths and settings as the original site.
It does not synchronize dependencies, delete the environment, move artifacts,
reset registries, or change your default. It registers or attaches the branch
and retains the original record in `.nro-installation-transition.json`.

Interrupted conversion blocks scientific execution. Repeat the same command to
resume; do not restore the old shared record manually. If only launcher connection
failed after conversion, reconnect with the same command. Other users whose
default still selects the converted checkout must connect to the replacement
shared installation. They will not silently fall back to production.

After conversion, the checkout uses branch-owned outputs, compatible ancestor
inputs, and the shared scheduler. Conversion itself creates no demand and moves
no derivatives.

## Site settings

Personal settings default to `$XDG_CONFIG_HOME/nro/site.toml`, or
`~/.config/nro/site.toml`. Shared settings default to `.nro-site.toml` beside
the checkout. Choose a different file with `./install --site /absolute/site.toml`
during first setup. `NRO_SITE_CONFIG` selects a site file for an unregistered
or personal installation; shared installations enforce their recorded file.

The path editor displays all proposed paths before asking for changes. It
offers lab defaults when `/juice6/u/nlp/climblab` is readable and traversable.
Otherwise, setup proposes paths under the user's resolved home directory:
`~/nro/bids`, `~/nro/work`, `~/nro/templateflow`, and related resource directories.
The generic private-control root is `~/nro/.nro`, independent of checkout
location. Users joining a shared deployment should select its existing root;
branch installation inherits it from the selected site.
These are suggestions, not directories created by the editor. Previously
configured paths and environment overrides are retained even if unavailable.
Interactive and noninteractive setup use the same resolved defaults.

The `definitions` path selects a separate, site-owned
[definitions store](definitions.md). It defaults to
`/juice6/u/nlp/climblab/nro-definitions` when lab storage is accessible and to
`~/nro/definitions` otherwise. Setup creates a missing store from generic
starters and validates an existing store without replacing its files. New stores
have no task models, event tables, or configured Flywheel servers.

Accept all defaults to keep the displayed settings. If declined, the editor
prompts for each setting independently. Enter keeps that value; Tab completes
paths. A final confirmation saves the choices.
Use `-` for an empty Slurm account. Scripted editing is also available:

```bash
nro paths show
nro paths set bids=/data/BIDS work=/scratch/nro
nro paths set runtime=/usr/bin/apptainer
```

Shared editing requires `--maintain`, write permission, and a stopped worker
pool. Changes update settings for new commands; they do not relocate files or
migrate a registry. Worker scripts carry the selected site filename explicitly.

Private state uses the [shared/branch hierarchy](commands/branches.md#storage-and-safeguards).
An existing flat-layout control store is rejected before any new scheduler is
created. Use the explicit [cutover command](commands/cutover.md) during a
coordinated maintenance window; neither installation nor registry repair silently
moves or adopts the old store.

The site file accepts `definitions`, `bids`, `work`, `registry`, `images`, `templates`,
`workbench`, `oslom`, `license`, `runtime`, `partition`, `account`, and `binds`.
`workbench` names `wb_command`; `wb_view` is expected beside it. `qunex`,
`synthstrip`, `synbold`, and `mni_template` can override individual resources
otherwise derived from their parent directories. `binds` is a TOML list.
Generic defaults omit CLIMBLAB's `/juice6` bind.

Personal installations also honor `NRO_BIDS_PATH`, `NRO_WORK_PATH`,
`NRO_WB_COMMAND`, `TEMPLATEFLOW_HOME`, and `FS_LICENSE`. Shared installations
use the site file for these values. `nro paths show` reports each value's
source. The engine supplies the FreeSurfer license and thread settings to
processing commands.

Scientific YAML uses explicit `site:KEY` resource references. Resolution with
the unmodified CLIMBLAB defaults preserves existing workflow fingerprints.
Changing a resolved resource path can still change a configuration
fingerprint; this release does not infer scientific equivalence after resource
relocation.

## Dependencies and downloads

Python requirements are declared in `pyproject.toml` and resolved in `uv.lock`.
Maintainers regenerate the lock when changing dependencies. Normal setup uses
the existing lock and does not upgrade dependencies implicitly.

Setup obtains QuNex 1.5.1, SynthStrip 1.7, and SynBOLD-DISCO 1.4 from pinned
OCI digests. Existing configured images are reused and tested. Downloads use
temporary paths and resource locks, then publish completed files atomically.
Image receipts record the source and the generated SIF's SHA-256. A receipt
documents acquisition, not historical provenance for pre-existing resources.

Workbench 2.2.1 is installed from its official Linux archive when absent;
the archive is checked against a pinned SHA-256 before extraction.
Automatic installation requires an x86_64 host and a target ending in
`workbench/bin_linux64/wb_command` or `workbench/bin_rh_linux64/wb_command`.
An existing incomplete Workbench directory is reported rather than overwritten.

The template catalog pins MNI T1w and GM probability maps and fsaverage surface
geometry by S3 object version and checksum. It downloads only these selected
files. The TemplateFlow Python client is not needed at runtime or setup.

The runtime installer reuses Singularity or Apptainer when available. Otherwise
it can install unprivileged Apptainer 1.4.5 using the upstream installer, whose
script checksum is pinned. This requires `curl`, `rpm2cpio`, `cpio`, and host
support for unprivileged containers. It cannot change kernel or cluster policy.
See the [Apptainer installation guide](https://apptainer.org/docs/admin/1.4/installation.html).

Review the [QuNex access terms](https://qunex.yale.edu/access/). Supply your
[FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html);
setup does not request credentials or register on your behalf.

Setup includes OSLOM, `igraph`, and `leidenalg` by default. If `oslom_undir` is absent,
setup downloads the official [OSLOM source](http://www.oslom.org/software.htm),
checks its pinned SHA-256, and builds the undirected solver using `g++`.
The host must provide the compiler. Setup fits the bundled example graph before
publishing the executable and saves a receipt with the source checksum, compiler,
build arguments, and binary checksum. Existing binaries are reused.

The official archive is served over HTTP. This is the sole HTTP exception in
the downloader and requires the pinned checksum. That checksum detects changes
to the accepted archive; it does not authenticate the initial HTTP acquisition.
ICA and clustering do not need OSLOM. Pass `--without-oslom` to skip it and
its Python dependencies for a setup or maintenance invocation. Existing OSLOM
binaries are not deleted when this option is used.

For unattended setup, prepare the site file first:

```bash
./install --mode personal --site /path/to/site.toml \
  --non-interactive --accept-qunex-license
```

`--offline` forbids new resource downloads and tells uv to use its cache.
It requires an existing uv bootstrap, cached Python dependencies, and all
required external resources. An interrupted setup leaves the installation
marked incomplete; rerunning resumes from installed resources.
Ctrl-C exits with a cancellation message and status 130. Interrupted path
editing leaves the saved settings unchanged. To resume shared setup, use
`./install --maintain`.

## Checking an installation

`nro doctor` first reports the checkout, branch, environment, executable, and site
settings selected for the invocation. It then checks Python package and resource
availability, directory access, and Slurm commands without importing the scientific
libraries or parsing every definition. `--local` makes Slurm optional;
OSLOM checks are mandatory by default; `--without-oslom` makes them optional.
`--json` returns structured results.

`nro doctor --deep` also imports the scientific libraries, validates every stored
definition, starts each container, checks the required tools and license bind in
QuNex, and verifies pinned template checksums and existing image acquisition
receipts. These probes do not process subject data. Run the
command within a compute allocation as well as on the login node to check shared mounts and
runtime policy there. Setup runs deep checks before marking an installation
ready. A GUI display is only needed when opening Workbench scenes.
