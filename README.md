# nro

`nro` processes anatomical and functional MRI from BIDS datasets. It produces
preprocessed images, cleaned time courses, small brain parcels, and individualized
functional networks. Users request results; a lab-wide orchestration layer finds
their dependencies and distributes ready work across a shared Slurm worker pool.

The design separates scientific computation from scheduling. Modules declare
their complete steps before execution; a common runner handles freshness,
logging, and resumption. Public outputs have explicit contracts. Changed inputs,
configurations, or contracts can invalidate results, while equivalent requests
share existing work.

Derivatives are produced by _modules_ with configurable settings, and sequences of configured
modules are organized into _workflows_.

The main processing sequence contains five modules:

1. `anat` prepares anatomical images.
2. `func` prepares each functional run.
3. `clean` removes unwanted signal from functional data.
4. `microparcellation` divides the brain into small regions and measures their
   connectivity.
5. `networks` groups those regions into individualized functional networks.

A separate `firstlevels` branch estimates task effects from `func` outputs.
Registered task models specify run, session, and subject contrasts; volume and
surface results include effect, variance, t, and degrees-of-freedom maps.

This is still an early project. It intentionally supports a small, current set
of workflows instead of preserving old commands and formats.

## Installation

From a Linux checkout, run:

```bash
./install
```

First setup asks whether this is a personal or shared installation and walks
through the default paths. It installs a locked Python environment in editable
mode, reuses or obtains processing resources, and creates a user command
launcher. The bootstrap needs Python 3.11 or newer with `venv` support.
Provide an existing FreeSurfer license when prompted. Some cluster hosts need
an administrator to install or enable the container runtime.

For an existing shared installation, `./install` only connects your account to
the shared environment. Maintainers use `./install --maintain` after stopping
the worker pool. Source edits remain immediately visible to new commands.

Inspect or update paths with `nro paths`; check dependencies with `nro doctor`.
The lab paths remain the defaults. See [installation details](docs/installation.md)
for unattended setup, permissions, and dependency acquisition. OSLOM is included
by default; use `--without-oslom` to omit it.

The launcher is installed in `~/.local/bin`. Add that **directory**, not the
launcher file, to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
nro doctor
```

Activation is unnecessary for `nro` commands. The environment lives in
`.nro-env`; explicit `python -m ...` commands must use that interpreter or an
activated environment containing nro.

## Basic usage

Run the complete workflow for one participant:

```bash
nro run -p t20 -P nptl
```

With no `--module`, requests reach every workflow endpoint: currently `networks`
and `firstlevels`. Firstlevels selects models in set `main` for matching tasks.
Shared upstream work runs once.

Run through a particular module, or select several participants:

```bash
nro run -p t12 t20 -P nptl -m microparcellation
```

`clean` and downstream modules default to `fsnative` at 2 mm smoothing.
Request only the space/smoothing combinations you need; multiple values
produce their cross-product:

```bash
nro run -p t20 -P nptl -m networks \
  --space fsnative T1w --smoothing 0 2
```

The `main` workflow estimates networks with ICA. Select repeated connectivity
clustering with `-w clustering`, or the slower OSLOM backend with `-w oslom`.

Request task-effect maps separately:

```bash
nro models list
nro run -p t20 -P nptl -m firstlevels -s fsnative -S 2
```

The lab's `langlocSN/main` model belongs to model set `main`. New definitions
stores start without task models; create models for your experiments. See
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
nro status -P nptl
nro log -p t20 -P nptl -m func -i
```

Open the Workbench scene stored with a completed subject-level derivative:

```bash
nro wb_view microparcellation -p t20 -P nptl
nro wb_view networks -p t20 -P nptl -s fsnative -S 2
```

Microparcellation and network artifacts are grouped first by space and
smoothing, then by participant. Each participant directory contains its
scientific outputs and a relocatable Workbench scene.

Common controls are:

```bash
nro stop -p t20 -P nptl
nro set concurrency=100
nro purge -p t20 -P nptl
```

`purge` previews and confirms destructive cleanup. A bare purge selects all
nro-controlled derivatives across all projects; `--logs` limits cleanup to
eligible logs.

The direct module form remains equivalent, for example:

```bash
python -m nro.bin.run -p t20 -P nptl
```

Registration quality-control images can be generated separately:

```bash
nro qc registration t20 -p nptl
```

The equivalent engine entry point is
`python -m nro.qc registration t20 -p nptl`.

Workflow settings are selected with `-w`; the default is `main`. Configuration
files live in a separate [definitions store](docs/definitions.md). Use
`nro paths show` to find it and `nro definitions validate` to check its contents.

## Documentation

The [documentation home](docs/index.md) links the full guide. Pages build as a
Sphinx site compatible with Read the Docs.

| Topic | Guide |
| --- | --- |
| First use | [Quickstart](docs/quickstart.md), [installation](docs/installation.md) |
| Work structure | [Design](docs/design.md), [definitions](docs/concepts.md), [instance lifecycle](docs/instance-lifecycle.md) |
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
