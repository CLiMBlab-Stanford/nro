# nro

`nro` runs the lab's fMRI processing workflows. It starts with a BIDS dataset
and builds the anatomical, functional, connectivity, and network results used
for analysis.

The project is designed for long-running work on a shared cluster. A run can
stop and resume without repeating valid work. Dependencies are explicit, and a
shared worker pool limits how much work the lab sends to the cluster at once.
Configuration lives in files rather than being hidden in Python code.

The current workflow has five modules:

1. `anat` prepares anatomical images.
2. `func` prepares each functional run.
3. `clean` removes unwanted signal from functional data.
4. `microparcellation` divides the brain into small regions and measures their
   connectivity.
5. `networks` groups those regions into individualized functional networks.

This is still an early project. It intentionally supports a small, current set
of workflows instead of preserving old commands and formats.

## Installation

Install `nro` into the prepared lab environment in editable mode:

```bash
cd /juice6/u/nlp/climblab/code/nro
conda activate climbprep
python -m pip install --editable .
```

This installs one `nro` command that imports directly from this development
tree. Python source edits are therefore visible to newly started commands and
workers without reinstalling. Reinstall only after changing package metadata
or the installed command definition.

The lab's standard paths and tool locations are set in
`nro/configuration/files/`.

## Basic usage

Run the complete workflow for one participant:

```bash
nro run -p t20 -P nptl
```

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

Check progress and inspect a failed task:

```bash
nro status -P nptl
nro log -p t20 -P nptl -m func -i
```

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
files are under `nro/configuration/files/`.

More detailed notes are available in
[docs/core concepts](docs/concepts.md),
[instance planning and execution](docs/instance-lifecycle.md),
[orchestration usage](docs/orchestration.md), and the
[orchestration design](docs/orchestration-design.md).

## Development

Run the test suite with:

```bash
.venv/bin/pytest -q
```
