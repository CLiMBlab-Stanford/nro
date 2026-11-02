# nro

[![Documentation](https://readthedocs.org/projects/nro/badge/?version=latest)](https://nro.readthedocs.io/en/latest/)
[![Tests](https://github.com/CLiMBlab-Stanford/nro/actions/workflows/main-release-version.yml/badge.svg)](https://github.com/CLiMBlab-Stanford/nro/actions/workflows/main-release-version.yml)
[![Release](https://github.com/CLiMBlab-Stanford/nro/actions/workflows/publish-release.yml/badge.svg)](https://github.com/CLiMBlab-Stanford/nro/actions/workflows/publish-release.yml)

`nro` processes anatomical and functional MRI from BIDS datasets. It produces
preprocessed images, cleaned time courses, dynamic-connectivity series, small
brain parcels, individualized functional networks, and task-effect maps. You
request the results you want. nro finds their dependencies, reuses existing
work, and distributes ready work across a site-wide Slurm worker pool or local
workers.

Each kind of result comes from a _module_ with configurable settings. A
_workflow_ selects a configuration for each module. The main processing graph
contains these modules:

- `anat` prepares anatomical images.
- `func` prepares each functional run.
- `clean` removes unwanted signal from functional data.
- `dynconn` packages cleaned vertex- or voxel-level signals for interactive
  dynamic-connectivity viewing.
- `microparcellation` divides the brain into small regions and measures their
  connectivity.
- `networks` estimates individualized functional networks from either
  microparcel connectivity or dynamic-connectivity time series.
- `contrasts` estimates task effects from `func` outputs using registered task
  models.

The arrows show dependencies:

```text
[anat] ──► [func] ──┬──► [clean] ──┬──► [dynconn] ────────────┐
                    │              └──► [microparcellation] ──┴──► [networks]
                    └──► [contrasts]

[anat] ── "direct anatomical inputs" ──► {clean, microparcellation, networks, contrasts}

Only one of the two configured inputs to `networks` is active in a workflow.
```

Module outputs live under
`BIDS/PROJECT/derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`. Their files
use BIDS-like names, but nro does not claim full BIDS compliance. Changed inputs,
configurations, or output contracts make results stale, while equivalent
requests share existing work. [How nro works](docs/design.md) explains this
model.

## Installation

Choose the guide for your situation:

- The [new-installation quickstart](docs/quickstart.md) is for outside users
  configuring `nro`, its storage, and its processing dependencies.
- The [CLIMBLAB quickstart](docs/climblab-quickstart.md) is for lab members
  connecting to the existing internal installation.

Both start by running `./install` from a Linux checkout. The installer needs
`curl` or `wget`, `sha256sum`, and `tar`, but not a host Python. It installs a
launcher in `~/.local/bin`, so no environment activation is needed. The
[installation guide](docs/installation.md) covers shared deployments,
unattended setup, permissions, and dependency acquisition.

## Basic usage

Run the complete `main` workflow for one participant of project `example`:

```bash
nro run -p 01 -P example
```

With no `--module`, the request reaches every workflow endpoint: currently
`dynconn`, `networks`, and `contrasts`. Shared upstream work runs once. To stop
at a particular module or select several participants:

```bash
nro run -p 01 02 -P example -m microparcellation
```

Check progress and inspect the logs of a failed task:

```bash
nro status -P example
nro log -p 01 -P example -m func
```

`nro help` gives task-oriented guidance from the installed CLI, and
`nro help --command COMMAND` shows a command's current syntax. The
[command reference](docs/commands/index.md) covers spaces and smoothing,
alternative workflows, task models, viewing, rendering, and cleanup.

## Documentation

The latest documentation is published at
<https://nro.readthedocs.io/en/latest/>. Its source is under `docs/`, starting
from [docs/index.md](docs/index.md).

| Topic | Guide |
| --- | --- |
| First use | [New-installation quickstart](docs/quickstart.md), [CLIMBLAB quickstart](docs/climblab-quickstart.md), [installation](docs/installation.md) |
| How work is organized | [How nro works](docs/design.md), [concepts](docs/concepts.md), [work-item lifecycle](docs/work-item-lifecycle.md) |
| Commands | [Command reference](docs/commands/index.md), [viewing and rendering](docs/commands/viewing.md), [maintenance and cleanup](docs/commands/maintenance.md) |
| Configuration | [Workflows and parameters](docs/configuration.md), [definitions stores](docs/definitions.md), [task models](docs/task-models.md) |
| Scientific processing | [Module guides](docs/modules/index.md), [denoising](docs/methods/denoising.md), [software and methods sources](docs/methods/software.md) |
| Cloud acquisition | [Flywheel-to-BIDS ingestion](docs/commands/bidsify.md) |
| Development | [Architecture and validation scopes](docs/architecture-map.md), [API guide](docs/api.md), [development guide](docs/development.md), [branching tutorial](docs/branching-tutorial.md) |

## Versions and development

nro follows [Semantic Versioning](https://semver.org/), and `main` contains
released code. Each published version has an annotated Git tag and a
corresponding GitHub Release. During the 0.x series, public interfaces may
still change. `nro --version` reports the release that the selected
installation uses. The
[development guide](docs/development.md#release-and-compatibility-policy)
describes the release and compatibility policy.

Development work happens in branch installations, which have editable
environments. Install with `--dev` to include test dependencies, then run the
default development suite from the checkout:

```bash
.nro-env/bin/python -m pytest -q
```

The [development guide](docs/development.md#tests) gives commands for the
integration tier and the complete suite. The documentation build needs only
`docs/requirements.txt`. It does not install neuroimaging containers or import
the processing package.

## AI-assisted development

This codebase was developed almost entirely with AI coding assistance. Human
maintainers selected the requirements, reviewed the generated code, ran the
applicable tests, and accept responsibility for the published result.

The tools and models used are recorded in
[AI_PROVENANCE.md](AI_PROVENANCE.md). Where available, individual commits also
contain `Assisted-by:` trailers.
