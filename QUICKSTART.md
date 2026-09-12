# Quickstart for a new installation

This guide is for people setting up `nro` at a new site or on a personal Linux
system. CLIMBLAB members using the existing lab installation should follow the
[CLIMBLAB quickstart](CLIMBLAB_QUICKSTART.md) instead.

`nro` expects MRI source data organized as BIDS. It installs its Python
environment and can download the supported processing containers, Workbench,
templates, and OSLOM. It does not install Git, a host compiler, or cluster
software.

## Prepare the host

Before setup, you need:

- Linux on an x86-64 system;
- Git and Python 3.11 or newer with `venv` and pip support;
- a C++ compiler if you want the default OSLOM installation;
- enough persistent storage for BIDS data, derivatives, work files, containers,
  and templates;
- a [FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html);
- Slurm commands on `PATH` for cluster execution, or permission to process on
  the local host; and
- Singularity or Apptainer, or a host that supports the installer's unprivileged
  Apptainer setup.

Storage used by cluster workers must be mounted at the same paths on the login
and compute nodes. Ask the cluster administrator about the container runtime,
Slurm partition and account, filesystem permissions, and local policy before
installing on a managed system.

## Get the source

Clone the repository and enter the checkout:

```bash
git clone https://github.com/CLiMBlab-Stanford/nro.git
cd nro
```

The installation is editable. Commands use this checkout, so keep it at a
stable absolute path after setup.

## Choose paths and install

For one user or one administrative account, start with a personal installation:

```bash
./install --mode personal
```

If this machine has no Slurm service, use:

```bash
./install --mode personal --local
```

The installer displays proposed settings. Outside CLIMBLAB, storage defaults to
directories under `~/nro`. Accept them only if that location has enough space
and is visible from every compute node that will run work.

The main settings are:

| Setting | Purpose |
| --- | --- |
| `bids` | Directory containing one subdirectory per BIDS project. |
| `work` | Intermediate processing files. |
| `registry` | Private scheduler state, logs, and source snapshots. |
| `definitions` | Workflows, module configurations, task models, and event files. |
| `development` | Isolated outputs for development branches. |
| `images`, `templates`, `workbench`, `oslom` | Installed processing resources. |
| `license` | Existing FreeSurfer `license.txt`. |
| `runtime` | Singularity or Apptainer command or absolute executable path. |
| `partition`, `account` | Slurm submission settings. Enter `-` for no account. |
| `viewing_partition` | Slurm partition for X11 scene-viewing jobs. |

Enter the path to your existing FreeSurfer license when prompted. For Slurm,
replace the proposed partition and account with values for your cluster. The
partition and account are unused with `--local`.

Review the linked QuNex terms before approving downloads. Setup creates a locked
Python environment in `.nro-env`, downloads missing resources, checks them, and
installs a launcher in `~/.local/bin`. If setup is interrupted or a check fails,
correct the reported problem and repeat the same `./install` command.

For a multiuser deployment, one maintainer should create a shared installation
from a clean, tagged `main` checkout. Installation records and activates that
release automatically; other users then connect to it. Read the
[installation guide](docs/installation.md#shared-installations) before choosing
`--mode shared`.

## Make the command available

Add the launcher's directory to your shell startup file. For Bash, add this line
to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Load the change and verify the installation:

```bash
source ~/.bashrc
hash -r
type -a nro
nro doctor                 # Slurm installation
nro doctor --local         # installation without Slurm
nro paths show
```

No environment activation is required for `nro` commands. If another executable
or an activated environment takes precedence, use `~/.local/bin/nro` directly
while correcting `PATH`.

Run `nro doctor --deep` inside a compute allocation if login and compute nodes
have different mounts or container policies. Use `nro doctor --deep --local`
when Slurm is not part of the installation.

## Add a BIDS project

Place an existing BIDS dataset beneath the configured `bids` directory. `nro`
treats each direct child as a project:

```text
BIDS_ROOT/
    example/
        dataset_description.json
        sub-01/
        sub-02/
```

Use your BIDS validation procedure before processing. `nro` can ingest selected
Flywheel data through its optional bidsification system, but that is a separate
workflow with additional dependencies and server configuration. See the
[bidsification guide](docs/commands/bidsify.md).

## Request the first result

Start with explicit project, participant, and module selectors:

```bash
nro run -P example -p 01 -m anat
```

For an installation created with `--local`, run one local worker explicitly:

```bash
nro run -P example -p 01 -m anat --local
```

The planner discovers the selected BIDS data, registers demand, reuses fresh
outputs, and adds required upstream work. Results go under the project's
`derivatives` directory; temporary and resumable files go under the configured
`work` directory.

Follow the request with:

```bash
nro status -P example -p 01
nro log -P example -p 01 -m anat -i
```

By default, `nro status` quickly reports the registry's saved state. Run
`nro status --update` when you need it to check files and update that state.

Once anatomical processing works, request a downstream endpoint such as
`microparcellation`. Required functional preparation and cleaning are added
automatically:

```bash
nro run -P example -p 01 -m microparcellation
```

Add `--local` again when the site does not use Slurm.

Omitted selectors usually mean all matching data. Avoid bare `nro run` and
`nro purge` commands until you understand their scope. Purge deletes controlled
derivatives and logs after confirmation.

## Next steps

- [General installation and maintenance](docs/installation.md)
- [Command-line reference](docs/commands/index.md)
- [Module inputs, processing, and outputs](docs/modules/index.md)
- [Configuration and workflows](docs/configuration.md)
- [Definitions stores](docs/definitions.md)
- [Troubleshooting work](docs/commands/work.md)
