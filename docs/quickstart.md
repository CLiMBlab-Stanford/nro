# Quickstart for a new installation

This guide is for people setting up `nro` at a new site or on a personal Linux
system. CLIMBLAB members should use the
[internal installation quickstart](climblab-quickstart.md).

## Prepare the host

`nro` needs Linux on x86-64, Git, and Python 3.11 or newer with `venv` and pip
support. Acquire a
[FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html)
before setup. The default OSLOM installation also needs a C++ compiler.

Cluster installations need Slurm commands on `PATH`. Their data, work,
registry, installation, and processing resources must be visible at the same
paths on login and compute nodes. A local installation can omit Slurm. The host
must provide Singularity or Apptainer or support the installer's unprivileged
Apptainer setup.

## Install

Clone the repository and enter the checkout:

```bash
git clone https://github.com/CLiMBlab-Stanford/nro.git
cd nro
```

Create a personal installation on a Slurm system:

```bash
./install --mode personal
```

Use `--local` on a machine without Slurm:

```bash
./install --mode personal --local
```

Outside CLIMBLAB, the path editor proposes directories under `~/nro`. Review
every setting. In particular:

- `bids` contains project directories with source BIDS data.
- `work` contains temporary and resumable processing files.
- `registry` contains private scheduler state, logs, and source snapshots.
- `definitions` contains workflows, configurations, models, and event files.
- `images`, `templates`, `workbench`, and `oslom` contain processing resources.
- `license` must name an existing FreeSurfer license.
- `runtime` selects Singularity or Apptainer.
- `partition` and `account` must match the Slurm site. Enter `-` for no account.

The Slurm fields are unused with `--local`. Review the linked QuNex terms before
approving downloads. Setup creates an editable `.nro-env`, obtains missing
resources, checks them, and installs `~/.local/bin/nro`. Repeat the same command
after correcting any reported failure.

Read [installation](installation.md) before creating a multiuser shared
installation, using offline setup, or changing resource acquisition.

## Check the installation

For Bash, add the launcher directory to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then run:

```bash
source ~/.bashrc
hash -r
type -a nro
nro doctor                 # Slurm installation
nro doctor --local         # installation without Slurm
nro paths show
```

No environment activation is needed. Run `nro doctor --deep` inside a compute
allocation when compute nodes have different mounts or container policies. Use
`nro doctor --deep --local` when Slurm is not part of the installation.

## Add data and request work

Place each BIDS dataset in its own project directory beneath the configured
`bids` root:

```text
BIDS_ROOT/
    example/
        dataset_description.json
        sub-01/
```

Validate the source dataset through your site's BIDS procedure. Then request a
small, explicitly selected target:

```bash
nro run -P example -p 01 -m anat
```

Use a local worker when the installation does not use Slurm:

```bash
nro run -P example -p 01 -m anat --local
```

Inspect progress and logs:

```bash
nro status -P example -p 01
nro log -P example -p 01 -m anat -i
```

Requesting a downstream endpoint automatically adds its dependencies. For
example:

```bash
nro run -P example -p 01 -m microparcellation
```

Add `--local` to each request when Slurm is unavailable. Omitted selectors
usually mean all matches, so avoid bare `nro run` and `nro purge` commands until
you understand their scope.

Continue with the [command reference](commands/index.md),
[module guides](modules/index.md), and [configuration guide](configuration.md).
