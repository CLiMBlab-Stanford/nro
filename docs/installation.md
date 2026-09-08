# Installation

Run `./install` from a Linux checkout. The script finds its own checkout even
when invoked from another directory. First setup asks for `personal` or
`shared` mode and saves that role in the untracked `.nro-installation.json`.

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
An existing, different `nro` launcher is not overwritten; select `--bin-dir`
or explicitly move the old launcher. `python -m nro.bin.run` remains available
through the installation's Python interpreter.

## Shared installations

The first maintainer selects shared mode:

```bash
./install --mode shared
```

Subsequent users run the same `./install`. It reads the installation record
and creates their launcher. It does not synchronize Python dependencies or
write into the checkout. An incomplete installation cannot onboard users.

Maintenance is explicit:

```bash
./install --maintain
```

Stop or drain the worker pool first, and coordinate with users so nobody
submits work during maintenance. The installer refuses maintenance when the
registry records active workers or pending allocations, unless a successful
Slurm query confirms those allocations are no longer in the queue. Scheduler
errors block maintenance. Records without a Slurm job ID must be resolved
through the normal worker controls. The installer never
silently repairs or resets the registry.

The shared checkout, environment, and site file must be readable and traversable
by users. Restrict write access to maintainers using filesystem ownership or
ACLs. Shared setup creates new files with a readable umask; it does not rewrite
permissions on existing trees. BIDS derivatives, WORK, and the registry need
the lab's usual shared write permissions independently of the software tree.

Editable code can change underneath running processes. This installer does
not isolate revisions or make separate experimental checkouts safe to mix
with the production worker pool.

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
The generic registry default is `../../.nro` relative to the repository root,
preserving the lab registry's relative location.
These are suggestions, not directories created by the editor. Previously
configured paths and environment overrides are retained even if unavailable.
Runtime defaults do not change until the proposed settings are saved.

The `definitions` path selects a separate, lab-owned
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

The site file accepts `definitions`, `bids`, `work`, `registry`, `images`, `templates`,
`workbench`, `oslom`, `license`, `runtime`, `partition`, `account`, and `binds`.
`workbench` names `wb_command`; `wb_view` is expected beside it. `qunex`,
`synthstrip`, `synbold`, and `mni_template` can override individual resources
otherwise derived from their parent directories. `binds` is a TOML list.
Generic defaults omit the lab's `/juice6` bind.

Personal installations also honor `NRO_BIDS_PATH`, `NRO_WORK_PATH`,
`NRO_WB_COMMAND`, `TEMPLATEFLOW_HOME`, and `FS_LICENSE`. Shared installations
use the site file for these values. `nro paths show` reports each value's
source. The engine supplies the FreeSurfer license and thread settings to
processing commands.

Scientific YAML uses explicit `site:KEY` resource references. Resolution with
the unmodified lab defaults preserves existing workflow fingerprints. Changing
a resolved resource path can still change a configuration fingerprint; this
release does not infer scientific equivalence after resource relocation.

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

`nro doctor` reports Python packages, resource files, executable availability,
directory access, and Slurm commands. `--local` makes Slurm optional;
OSLOM checks are mandatory by default; `--without-oslom` makes them optional.
`--json` returns structured results.

`nro doctor --deep` also starts each container, checks the required tools and
license bind in QuNex, and verifies pinned template checksums and existing
image acquisition receipts. These probes do not process subject data. Run the
command within a compute allocation as well as on the login node to check shared mounts and
runtime policy there. Setup runs deep checks before marking an installation
ready. A GUI display is only needed when opening Workbench scenes.
