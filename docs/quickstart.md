# Quickstart

## Install or join a shared installation

Use Linux, Python 3.11 or newer with `venv`, access to your BIDS projects,
and a FreeSurfer license. From the checkout:

```bash
./install
```

On first setup, choose personal or shared mode and review the paths. OSLOM,
the container images, Workbench, templates, and Python dependencies are obtained
or reused. A host administrator may need to provide a container runtime or C++
compiler. Subsequent users of an initialized shared checkout run the same
command to create their launcher without changing shared dependencies.

Add the **directory** containing that launcher to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
nro doctor
```

Put the export in your shell startup file to keep it. The launcher uses the
checkout's `.nro-env/bin/python`; activation is unnecessary. Installation is
editable: new commands read the checkout, not a frozen copy. Do not edit shared
code or update dependencies beneath running workers. See [installation](installation.md)
for maintenance, licenses, offline use, and path configuration.

## Check paths and request work

```bash
nro paths show
nro run -P nptl -p t20 -m microparcellation --no-submit
nro run -P nptl -p t20 -m microparcellation
```

Replace project and participant IDs with your data. `--no-submit` registers and
assesses demand but does not launch workers; it is not a read-only preview.
Omit `-m` to request every workflow endpoint, currently `networks` and
`firstlevels`. Firstlevels uses model set `main` for matching tasks. Missing
projects or participants mean all matches, so review selectors before running
a bare command.

Defaults are workflow `main`, space `fsnative`, and smoothing 2 mm FWHM.
Select additional independent combinations on demand:

```bash
nro run -P nptl -p t20 -s fsnative T1w -S 0 2
```

This requests four space/smoothing combinations from `clean` onward. The
`anat` and `func` prerequisites are shared. Run selection matches BIDS entities:

```bash
nro run -P nptl -p t20 -r task=rest,langlocSN run=01,02
```

Values within an entity are alternatives; different entities must all match.

## Inspect, view, and stop

```bash
nro status -P nptl
nro status -P nptl --verify
nro log -P nptl -p t20 -m func -i
nro wb_view microparcellation -P nptl -p t20
nro stop -P nptl -p t20
```

Default status predicts freshness using inexpensive checks. `--cached` reports
saved state; `--verify` performs a full reassessment and updates the registry.
Stopping demand does not delete completed results. `nro stop --workers` stops
the current user's worker pool without cancelling demand.

`nro purge` deletes results and logs after a preview and confirmation. A bare
purge selects every nro-controlled derivative across projects. Read its
[reference](commands/maintenance.md) before using it.

The [module guides](modules/index.md) describe where to find outputs. All
commands also support `python -m nro.bin.COMMAND` when using the installation's
Python interpreter. Plain `python` may select a different environment.
