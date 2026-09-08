# Running derivative workflows

The orchestration layer plans and runs instances of the five supported modules:
`anat`, `func`, `clean`, `microparcellation`, and `networks`. All projects share
one lab-wide registry and worker pool stored by default in:

```text
/juice6/u/nlp/climblab/.nro/registry.sqlite3
```

An editable installation exposes every executable in `nro/bin` as an
`nro COMMAND` subcommand: `run`, `status`, `log`, `set`, `stop`, `purge`,
`publish`, and `qc`. The equivalent `python -m nro.bin.COMMAND` forms remain
available. The `qc` executable delegates to the extensible engine in
`nro/qc/`; `python -m nro.qc` invokes that engine directly.
Their implementation uses the private machinery in `nro.orchestration`.

## Plan and run

```bash
nro run -p t12 t20 \
  -P nptl -m networks -w main --partition sphinx --concurrency 25
```

From `clean` onward, space and smoothing are independent instance
entities. The defaults are `fsnative` and 2 mm. Supplying multiple values
requests only their cross-product; possible pairs are not pre-registered:

```bash
nro run -p t20 -P nptl -m networks \
  --space fsnative T1w --smoothing 0 2
```

The corresponding filenames include both entities, for example
`_space-fsnative_smoothing-2mm_`. Cleaned runs retain the source BIDS
subject/session layout. Microparcellation and network artifacts use
`space-<space>_smoothing-<mm>mm/sub-<participant>/` so each target has an
independent, relocatable directory.

Each requested module defines terminal instances: the planner adds every
required upstream instance. Within a project and workflow, targets already
covered by another selected target's dependencies create no separate demand.
Coverage is checked by instance, so upstream runs outside a downstream module's
input selection remain requested. Existing requests are unchanged.
Run-level `func` and `clean` requests expand from source BIDS only;
derivative contents never expand the source run universe. Use exact BIDS entity
selectors when narrowing those modules:

```bash
nro run -p t20 -P nptl -m clean \
  --run ses=ex31792 task=Rest dir=LR run=01
```

An empty invocation targets every module with no downstream consumers, currently
`networks` and `firstlevels`, for every source-BIDS participant in every
discovered project. It uses workflow `main`, model set `main`, concurrency 50,
and the configured partition. Common upstream instances are shared between
branches. An explicit `--module` restricts the requested endpoints.
Participant filters are matched across projects, so `-p t12` does not
require a project when only one project contains `sub-t12`.

A source subject directory is discoverable even when it cannot support the
requested module. The planner treats missing required source data as participant
unavailability, reports the participant and reason, and continues with every
eligible match. For example, a participant with no T1w or T2w image is skipped
instead of aborting a lab-wide request. Such a selection is not registered as
an instance or request. An invocation that selects only unavailable work exits
with the reported reason.

Important options:

- `--repair` destroys the entire lab-wide private `.nro` registry state,
  creates the current schema, and discovers the source BIDS project and
  participant directories. It restores instances from ownership receipts
  stored with nro derivatives, including instances whose workflows are no
  longer in the configuration store. It also scans current workflow locations
  for artifacts created before ownership receipts were introduced. The
  registry registers each artifact with its dependency graph and then assesses
  it. This discovery does not create requests or demand, remove public
  derivatives, or submit workers.
  Selection options cannot accompany this lab-wide operation.
  Repair freezes the worker pool before replacing registry state. If workers or
  pending worker submissions exist, it asks for confirmation, requests every
  user's workers to stop, cancels their Slurm allocations, and waits until all
  worker processes are inactive. Declining the prompt leaves the registry and
  workers unchanged.
  If no controlled artifacts exist, `nro status` immediately after repair
  prints its table header with no derivative rows.
- `--no-submit` creates demand without submitting workers.
- `--local` runs a worker in the current allocation.
- `--concurrency` limits active module instances shared by overlapping requests.
- `--space` and `--smoothing` accept one or more targets for `clean`,
  `microparcellation`, and `networks`; `-s` is the short form for space and
  `-S` is the short form for smoothing.
- `--memory` and `--max-memory` set the initial and maximum adaptive memory
  tiers.
- `--time`, `--cpus`, `--partition`, and `--account` define Slurm workers.
- `--drain-minutes` stops workers from claiming work near allocation expiry.

Workers reuse logical instances across users and requests. Confirmed OOMs
double an instance's required memory until the configured ceiling. An OOM at
the ceiling is terminal.

## Configuration

The [definitions store](definitions.md) is external to the installation and
selected through `nro paths`. Processing requests select configuration and
workflow IDs within that store.

A workflow file selects one configuration ID per derivative class:

```yaml
# DEFINITIONS/workflows/expreg_workflow.yml
preprocessing: expreg
clean: main
microparcellation: main
networks: main
```

Omitted classes select `main`. Each class's `main` file contains its complete
defaults; another named class file contains only local overrides of that YAML
base. For example, a subject-level multirun selection belongs to the
microparcellation configuration:

```yaml
input_filter:
  task: Rest
  dir: [LR, RL]
```

Workflow lineage determines upstream derivative directories. The planner
writes fully resolved, immutable runtime snapshots inside `.nro/workflows/`;
workers consume those values directly and do not merge defaults again.

Source-image metadata follow the BIDS inheritance principle. Every applicable
JSON sidecar from the dataset root through the image directory is merged, and
every contributing file is retained in the instance's direct-input contract.
The main preprocessing configuration requests SynBOLD-DisCo for fieldmapless
distortion correction. For an individual run whose effective metadata lack a
valid `PhaseEncodingDirection` or usable readout time, functional preprocessing
records the reason and falls back to ordinary anatomical SyN for that run.

Changing a workflow or selected configuration creates a registry revision.
Equivalent configuration lineages are shared; distinct lineages receive
distinct derivative directories.

## Status and logs

By default, status performs a lightweight, read-only preview using the current
module processing contracts, declared inputs and outputs, and dependency graph:

```bash
nro status
nro status -p t20 -P nptl -m clean --json
nro status -p t20 -P nptl -w expreg \
  --run ses=ex31792 task=Rest --space fsnative --smoothing 2
nro status --cached
nro status --verify
```

`--cached` reports only the state already saved in the registry and does no
filesystem checking. The default preview detects cheap warning signs such as a
changed module processing contract, missing direct input, missing declared
output, or stale upstream derivative, but does not fingerprint large files or
write its conclusions back. `--verify` runs the authoritative thorough artifact
assessment, updates the registry, and then reports the newly saved state. A
verified processing-contract change also requests cancellation of an active
attempt so it cannot publish an obsolete derivative.

Logs use the same project, participant, workflow, module, run, space, and
smoothing filters. The default view opens Slurm worker logs;
`--instance-level` (`-i`) opens current instance logs. Options specific to one
command remain on that command rather than becoming part of the common
selection interface.

```bash
nro log -p t20 -P nptl -m func \
  --run ses=ex31792 task=Rest
nro log -p t20 -P nptl -m func -i
```

All matches open newest-first in one `less` session.

## Live settings

Change the shared concurrency limit for every active request with:

```bash
nro set concurrency=100
```

The update does not create demand or retry failed instances. Workers read the
current limit whenever they claim work. A lower limit therefore takes effect as
running instances finish, while a higher limit permits the existing workers to
expand the pool naturally as they complete instances and expose runnable work.
Any number of `NAME=VALUE` assignments may be supplied. Unsupported names are
ignored with a warning.

## Cancellation

```bash
nro stop -p t20 -P nptl -m clean
nro stop -p t20 -P nptl -m func \
  --run dir=RL run=01
```

Stopping removes selected demand and, by default, its downstream dependents.
It preserves demand shared by another request. `--force` cancels matching
demand across users and signals a shared running attempt. `--only` prevents
downstream cancellation. `--workers` shuts down the current user's worker jobs
without cancelling instance demand.

## Purge

```bash
nro purge
nro purge -p t20 -P nptl -m func \
  --run ses=ex31792 task=Rest run=01 --dry-run
nro purge -p t20 -P nptl -m func clean --force
nro purge --logs
```

Purge follows the common selection rules: every omitted selector means “all.”
A bare invocation therefore selects every registered nro instance in every
project. Only paths owned by those instances are removed; unrelated derivative
trees are left in place. A scoped purge does not traverse upstream or
downstream dependencies.

The default operation removes the selected public derivatives, private
completion state and `WORK` products, matching terminal attempt logs, and logs
from workers that are no longer running. `--logs` (`-l`) removes only the two
log categories. Active attempt and worker logs are always preserved, and
derivative deletion refuses instances with active attempts.

Before deletion, the command pages a report of the selected public and private
paths and asks for confirmation. `--force` (`-f`) skips confirmation;
`--dry-run` reports without deleting. JSON mutation calls must include
`--force`.

## Publication

```bash
nro publish REQUEST_ID OUTPUT_DIR -P nptl
```

Publication requires a satisfied, fresh request. It copies files into a new
standalone derivative dataset, verifies checksums and generations, optionally
runs an available BIDS validator, and publishes by atomic rename without
overwriting an existing destination.

## Direct module execution

Direct execution uses the same workflow store and private runtime snapshot
mechanism:

```bash
python -m nro.anat -p t20 -P nptl -w expreg
python -m nro.func -p t20 -P nptl -w expreg \
  --run ses=ex31792 task=Rest run=01
python -m nro.clean -p t20 -P nptl -w expreg \
  --space fsnative --smoothing 2 --run ses=ex31792 task=Rest run=01
python -m nro.microparcellation -p t20 -P nptl -w expreg \
  --space fsnative --smoothing 2
python -m nro.networks -p t20 -P nptl -w expreg \
  --space fsnative --smoothing 2
```

The run selector interface is `--run ENTITY=VALUE[,VALUE...] ...`. Values for
one entity are alternatives; different entities are combined. For example,
`--run task=langlocSN,spatialFIN dir=LR` selects either task in the LR
direction.

Each direct entry point invokes its package's `module.py`. Every module creates
one common `Runner` and explicitly adds externally constructed `Step` objects.
The runner owns the resulting graph, freezes it after construction, and then
makes runtime freshness decisions. The scientific modules define their steps;
they do not provide independent execution engines.
