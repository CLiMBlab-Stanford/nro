# CLIMBLAB quickstart

This guide is for CLIMBLAB members joining the lab's internal `nro`
installation. Outside users who need to configure their own site should follow
the [new-installation quickstart](QUICKSTART.md). The lab's processing software,
data paths, registry, and Slurm worker pool are already configured. Each lab
member only needs to connect their account to the shared installation.

## Connect your account

Log in to a lab server and run the installer from the shared checkout:

```bash
cd /juice6/u/nlp/climblab/nro
./install --default
```

This creates a launcher at `~/.local/bin/nro` and selects the shared checkout as
your default installation. It does not create a separate environment, download
software, or change the shared configuration. Do not use `--maintain`; that
option is reserved for coordinated maintenance of the shared installation.

Add the launcher's directory to your `PATH` if it is not already present. For
Bash, add this line to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Load the change in your current shell:

```bash
source ~/.bashrc
hash -r
```

No Conda environment or Python virtual environment needs to be activated. The
launcher selects the Python environment maintained with the shared checkout.

## Check the setup

Confirm which command your shell will run, then check the installation and the
configured lab paths:

```bash
type -a nro
nro doctor
nro paths show
```

The first path reported by `type -a nro` should be
`~/.local/bin/nro`. `nro doctor` should report `OK` for every required check.
The paths command should show the shared CLIMBLAB data, work, definitions,
TemplateFlow, and registry locations.

If another `nro` command appears first, an activated environment, shell alias,
or shell function may be taking precedence. Deactivate the environment or
remove the alias, then run `hash -r`. You can use `~/.local/bin/nro` directly
while correcting your shell configuration.

## Inspect existing work

Most commands accept the same selectors. Project uses uppercase `-P`; participant
uses lowercase `-p`:

```bash
nro status -P nptl -p t20
```

Replace `nptl` and `t20` with the project and participant you need. Participant
labels may be given with or without the `sub-` prefix. By default, status reads
the saved registry state without checking files. Use `nro status --update` for
a full validation that updates the registry before reporting.

Omitted selectors usually mean **all matching data**. During onboarding, always
provide a project and participant. In particular, a bare `nro run` can request
work across the lab, and a bare `nro purge` can delete every derivative under
`nro` control after confirmation.

## Request processing

Request work through a named endpoint. The planner adds any required upstream
modules:

```bash
nro run -P nptl -p t20 -m microparcellation
```

An explicit upstream module does not need to be listed. For example,
`-m microparcellation` includes required `anat`, `func`, and `clean` work.
Existing fresh artifacts are reused.

Without `-m`, `run` requests every endpoint in the selected workflow. Current
defaults are workflow `main`, space `fsnative`, and 2 mm FWHM smoothing. Select
other spaces or smoothing values only when needed:

```bash
nro run -P nptl -p t20 -m networks -s fsnative T1w -S 0 2
```

Multiple spaces and smoothing values request their cross-product. Use
`nro run --help` to see task, model, run, workflow, and other selectors.

## Follow a request

Check status or inspect logs for the selected work:

```bash
nro status -P nptl -p t20
nro log -P nptl -p t20 -m func -i
```

The `-i` log view groups output by processing instance. Status reports blocked
work and shows which upstream error caused the block.

To withdraw your matching request without deleting completed derivatives:

```bash
nro stop -P nptl -p t20
```

Other users' demand for the same work remains active. Do not use `nro purge`
until you have read the [maintenance guide](docs/commands/maintenance.md); purge
removes shared derivatives and logs, not just your request.

## Where to go next

- [Quickstart for a new installation](QUICKSTART.md)
- [Project overview and common commands](README.md)
- [General quickstart](docs/quickstart.md)
- [Command-line reference](docs/commands/index.md)
- [Processing module guides](docs/modules/index.md)
- [Viewing completed outputs](docs/commands/viewing.md)
- [Task-model documentation](docs/task-models.md)

If installation checks fail or the shared installation reports that setup is
incomplete, contact its maintainer. Do not repair paths, update dependencies, or
modify the shared checkout as an onboarding step.
