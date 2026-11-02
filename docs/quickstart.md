# Quickstart for a new installation

This guide takes you from a fresh Linux host to a first processed result. It is
for people setting up `nro` at a new site or on a personal system. CLIMBLAB
members should use the [CLIMBLAB quickstart](climblab-quickstart.md) instead,
because the lab installation is already configured.

`nro` expects MRI source data organized as BIDS. Setup installs its own Python
environment and downloads the supported processing containers, Workbench,
templates, and OSLOM. It does not install Git, a host compiler, or cluster
software.

## Prepare the host

Before setup, you need:

- Linux on an architecture covered by the installer's pinned uv archives;
- Git, `curl` or `wget`, `sha256sum`, and `tar`;
- a C++ compiler if you want the default OSLOM installation;
- enough persistent storage for BIDS data, derivatives, work files, containers,
  and templates;
- Slurm commands on `PATH` for cluster execution, or permission to process on
  the local host; and
- a host that permits unprivileged containers.

The host does not need its own Python. By default, setup installs a managed
Apptainer runtime, so compute nodes do not need a system Singularity or
Apptainer command.

On a cluster, the data, work, registry, installation, and processing resources
must be mounted at the same paths on login and compute nodes. Before installing
on a managed system, ask the cluster administrator about container support, the
Slurm partition and account, filesystem permissions, and local policy.

You also need a
[FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html)
before FreeSurfer-dependent processing runs. Setup does not require it.

## Get the source

Clone the repository and enter the checkout:

```bash
git clone https://github.com/CLiMBlab-Stanford/nro.git
cd nro
```

Keep the checkout at a stable absolute path. The installation records it, and
the launcher selects installations by checkout.

## Install

For one user or one administrative account, create a personal installation:

```bash
./install --mode personal
```

On a machine without Slurm, add `--local`:

```bash
./install --mode personal --local
```

The installer first shows the proposed site settings. Outside CLIMBLAB, storage
defaults to directories under `~/nro`. Accept them only if that location has
enough space and is visible from every compute node that will run work. If you
decline, the editor asks for each setting in turn.

| Setting | Purpose |
| --- | --- |
| `bids` | Directory containing one subdirectory per BIDS project. |
| `work` | Temporary and resumable processing files. |
| `registry` | Private scheduler state, logs, and source snapshots. |
| `definitions` | Version-controlled repository of protected site settings, workflows, module configurations, task models, and event files. |
| `development` | Isolated outputs for development branches. |
| `images`, `templates`, `oslom` | Installed processing resources. |
| `workbench` | Optional native scene viewer and renderer. |
| `license` | FreeSurfer `license.txt`. The file may be added after setup. |
| `runtime` | Container runtime. The default bare name selects managed Apptainer; an absolute path selects an administrator-provided Singularity or Apptainer executable. |
| `partition`, `account` | Slurm submission settings. Enter `-` for no account. Unused with `--local`. |
| `viewing_partition` | Slurm partition for X11 scene-viewing jobs. |
| `flywheel_server`, `flywheel_project` | Optional default ingestion source. Enter `-` to leave either unset. |

The installer saves these values in `site/site.yml` inside the definitions
repository. Keep that repository under version control. The small local TOML
file is only a generated pointer to the repository, and setup can recreate it.

Review the linked QuNex terms before you approve downloads. Setup then builds a
locked Python environment, downloads missing resources, checks them, and
installs the launcher `~/.local/bin/nro`. If setup is interrupted or a check
fails, correct the reported problem and run the same `./install` command again.

Personal installations run from a staged copy of the checkout. Pulling new
commits does not change the active command until you run `./install` again.

A multiuser deployment needs a shared installation instead. One maintainer
creates it from a clean, tagged `main` checkout, and other users connect to it.
Read [shared installations](installation.md#shared-installations) before using
`--mode shared`. The [installation guide](installation.md) also covers offline
setup, optional components, and resource acquisition.

## Check the installation

Add the launcher's directory to your shell startup file. For Bash, add this line
to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Load the change and check the installation:

```bash
source ~/.bashrc
hash -r
type -a nro
nro doctor                 # Slurm installation
nro doctor --local         # installation without Slurm
nro site ls
```

No environment activation is needed. The first `nro` reported by `type -a`
should be `~/.local/bin/nro`. If an activated environment or shell alias takes
precedence, use `~/.local/bin/nro` directly while you correct `PATH`.

If compute nodes have different mounts or container policies from the login
node, run `nro doctor --deep` inside a compute allocation. Use
`nro doctor --deep --local` when the installation does not use Slurm.

## Add a BIDS project

Place each BIDS dataset in its own directory beneath the configured `bids`
root. `nro` treats each direct child as a project:

```text
BIDS_ROOT/
    example/
        dataset_description.json
        sub-01/
        sub-02/
```

Validate the dataset with your site's BIDS procedure before processing. `nro`
can also ingest selected Flywheel data, but that workflow needs extra
dependencies and server configuration. See the
[bidsification guide](commands/bidsify.md).

## Request the first result

Start with a small, explicitly selected target:

```bash
nro run -P example -p 01 -m anat
```

On an installation created with `--local`, add `--local` to run one local
worker:

```bash
nro run -P example -p 01 -m anat --local
```

The planner finds the selected BIDS data, records your request, reuses fresh
outputs, and adds any required upstream work. Results go under the project's
`derivatives` directory. Temporary and resumable files go under the configured
`work` directory.

Follow the request:

```bash
nro status -P example -p 01
nro log -P example -p 01 -m anat
```

By default, `nro status` quickly reports the registry's saved state. Run
`nro status --update` when you need it to check files and update that state.

Once anatomical processing works, request a downstream endpoint such as
`microparcellation`. The planner adds the required functional preparation and
cleaning:

```bash
nro run -P example -p 01 -m microparcellation
```

Add `--local` again when the site does not use Slurm.

## Selector scope

Omitted selectors mean all matching values. A new `nro run` request has one
exception: it uses workflow `main` when `--workflow` is omitted. A bare request
can still cover every participant in every project. With `nro run --resume`, an
omitted workflow matches every workflow with resumable work.

Avoid bare `nro run` and `nro purge` commands until you understand their scope.
After confirmation, purge deletes controlled derivatives and logs.

## Next steps

- [Installation and maintenance](installation.md)
- [Command reference](commands/index.md)
- [Module inputs, processing, and outputs](modules/index.md)
- [Configuration and workflows](configuration.md)
- [Definitions stores](definitions.md)
- [Troubleshooting work](commands/work.md)
