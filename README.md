# nro

`nro` processes anatomical and functional MRI from BIDS datasets. It produces
preprocessed images, cleaned time courses, dynamic-connectivity series, small
brain parcels, and individualized functional networks. Users request results;
the orchestration layer finds their dependencies and distributes ready work
across a site-wide Slurm worker pool.

The design separates scientific computation from scheduling. Modules declare
their complete steps before execution; a common runner handles freshness,
logging, and resumption. Public outputs have explicit contracts. Changed inputs,
configurations, or contracts can invalidate results, while equivalent requests
share existing work.

Derivatives are produced by _modules_ with configurable settings. Sequences of
configured modules are organized into _workflows_.

The main processing graph contains these modules:

- `anat` prepares anatomical images.
- `func` prepares each functional run.
- `clean` removes unwanted signal from functional data.
- `dynconn` packages cleaned vertex- or voxel-level signals for interactive
  dynamic-connectivity viewing.
- `microparcellation` divides the brain into small regions and measures their
  connectivity.
- `networks` groups those regions into individualized functional networks.
- `firstlevels` estimates task effects from `func` outputs. Registered task
  models specify run, session, and subject contrasts; volume and surface
  results include effect, variance, t, and degrees-of-freedom maps.

The arrows show dependencies:

```text
[anat] ──► [func] ──┬──► [clean] ──┬──► [dynconn]
                    │              └──► [microparcellation] ──► [networks]
                    └──► [firstlevels]

[anat] ── "direct anatomical inputs" ──► {clean, networks, firstlevels}
```

Version 0.0.1 is the first release. nro follows
[Semantic Versioning](https://semver.org/), and `main` contains released code.
Each published version has an annotated Git tag and a corresponding GitHub Release.
During the 0.x series, public interfaces may still change. The project retains
older behavior when the benefit is clear and the implementation remains small,
readable, and inexpensive to run. See the
[development guide](docs/development.md#release-and-compatibility-policy) for
the release policy and the [branching tutorial](docs/branching-tutorial.md) for
development and release workflows.

## Installation

Choose the guide for your site:

- [New installation quickstart](QUICKSTART.md) for outside users configuring
  `nro`, its storage, and its processing dependencies.
- [CLIMBLAB quickstart](CLIMBLAB_QUICKSTART.md) for lab members connecting to
  the existing internal installation.

From a Linux checkout, run `./install`. First setup creates an editable,
locked Python environment and asks where data, work files, private state, and
processing resources belong. It can use Slurm or run workers locally.

The user launcher is installed in `~/.local/bin`; no environment activation is
needed. The bootstrap requires Python 3.11 or newer with `venv` support and an
existing FreeSurfer license. Some managed hosts need an administrator to enable
the container runtime. See [installation details](docs/installation.md) for
shared deployments, unattended setup, permissions, and dependency acquisition.

```bash
export PATH="$HOME/.local/bin:$PATH"
nro doctor
```

## Basic usage

Run the complete workflow for one participant:

```bash
nro run -p 01 -P example
```

With no `--module`, requests reach every workflow endpoint: currently `dynconn`,
`networks`, and `firstlevels`. Firstlevels selects models in set `main` for
matching tasks. Shared upstream work runs once.

Run through a particular module, or select several participants:

```bash
nro run -p 01 02 -P example -m microparcellation
```

`clean` and downstream modules default to `fsnative` at 2 mm smoothing.
Request only the space/smoothing combinations you need; multiple values
produce their cross-product:

```bash
nro run -p 01 -P example -m networks \
  --space fsnative T1w --smoothing 0 2
```

The `main` workflow estimates networks with ICA. Select repeated connectivity
clustering with `-w clustering`, or the slower OSLOM backend with `-w oslom`.

Request task-effect maps separately:

```bash
nro models list
nro run -p 01 -P example -m firstlevels -s fsnative -S 2
```

New definitions stores start without task models. Add reviewed models to set
`main` when they should run by default. See
[first-level models](docs/modules/firstlevels.md) for model registration,
supported BIDS Stats Models features, and inference limitations.

Create a model draft from a task's BIDS events, or edit an existing definition:

```bash
nro create model newtask
nro edit model newtask/main
```

The same commands support `config CLASS/ID` and `workflow ID`. See
[definition authoring](docs/commands/authoring.md) for validation, local drafts,
and shared-store safeguards.

Check progress and inspect a failed task:

```bash
nro status -P example
nro log -p 01 -P example -m func -i
```

Create one Workbench scene from matching completed derivatives:

```bash
nro scene -p 01 -P example -s fsnative -S 2 --open
```

The default scene links to its source derivatives. Add `--publish` to copy its
inputs into a portable scene directory. `--open` runs `wb_view` through an X11
Slurm allocation on the site's viewing partition; it does not consume an nro
worker slot.

Common controls are:

```bash
nro stop -p 01 -P example
nro set concurrency=100
nro purge -p 01 -P example
```

`purge` previews and confirms destructive cleanup. A bare purge selects all
nro-controlled derivatives across all projects; `--logs` limits cleanup to
eligible logs.

The direct module form remains equivalent, for example:

```bash
.nro-env/bin/python -m nro.bin.run -p 01 -P example
```

Registration quality-control images can be generated separately:

```bash
nro qc registration 01 -p example
```

The equivalent engine entry point is
`.nro-env/bin/python -m nro.qc registration 01 -p example`.

Workflow settings are selected with `-w`; the default is `main`. Configuration
files live in a separate [definitions store](docs/definitions.md). Use
`nro paths show` to find it and `nro definitions validate` to check its contents.

## Documentation

The [documentation home](docs/index.md) links the full guide. Pages build as a
Sphinx site compatible with Read the Docs.

| Topic | Guide |
| --- | --- |
| First use | [New installation quickstart](QUICKSTART.md), [CLIMBLAB quickstart](CLIMBLAB_QUICKSTART.md), [installation](docs/installation.md) |
| Work structure | [Design](docs/design.md), [concepts](docs/concepts.md), [instance lifecycle](docs/instance-lifecycle.md) |
| Scientific processing | [Module guides](docs/modules/index.md), [denoising](docs/methods/denoising.md), [software and methods sources](docs/methods/software.md) |
| Configuration | [Workflows and parameters](docs/configuration.md) |
| Command-line interface | [Command reference](docs/commands/index.md) |
| Cloud acquisition | [Flywheel-to-BIDS ingestion](docs/commands/bidsify.md) |
| Python development | [API guide](docs/api.md), [development and documentation builds](docs/development.md) |

The module guides describe branch conditions and public artifact layouts.
Working derivatives use BIDS-like names but are not claimed to be fully BIDS
compliant. Publication and destructive cleanup are separate user commands.

## Development

Run the test suite with:

```bash
.nro-env/bin/python -m pytest -q
```

The default command runs the development suite. The
[development guide](docs/development.md#tests) gives commands for the integration
tier and the complete suite.

Install with `--dev` to include test dependencies. The documentation build needs
only `docs/requirements.txt`; it does not install neuroimaging containers or
import the processing package. See the development guide for the build command.

## AI-assisted development

This codebase was developed almost entirely with AI coding assistance. Human
maintainers selected the requirements, reviewed the generated code, ran the
applicable tests, and accept responsibility for the published result.

The tools and models used are recorded in
[AI_PROVENANCE.md](AI_PROVENANCE.md). Where available, individual commits also
contain `Assisted-by:` trailers.
