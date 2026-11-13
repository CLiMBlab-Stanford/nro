# Command-line reference

These pages describe every `nro` command, its options, and its defaults. Use
them when you know what you want to do and need the exact syntax or the
behavior behind it.

## Getting help

`nro help` prints task guides and a list of the installed commands. `nro help
TOPIC` explains a common workflow. For the syntax of one command, run `nro help
--command COMMAND` or `nro COMMAND --help`. Every command accepts `-h` and
`--help`. `nro --version` prints the package version of the selected
installation.

Each public executable module in `nro/bin` becomes a command. With the correct
interpreter, `python -m nro.bin.COMMAND` is equivalent to `nro COMMAND`. Other
Python modules in the engine are not exposed as commands.

All commands exit nonzero on a parse failure or an operational error. Progress
and error messages can go to stderr even when a command offers `--json`. Every
command uses the BIDS root from the global site configuration.

## Where each command is documented

| Page | Commands |
| --- | --- |
| [Request, inspect, and stop work](work.md) | `start`, `run`, `status`, `log`, `stop`, and the execution settings in `site` |
| [Installation, cleanup, and publication](maintenance.md) | `./install`, `setup`, `site`, `doctor`, `purge`, `gc`, `freshen`, `project rename`, `migrate`, `publish` |
| [Viewing and quality control](viewing.md) | `scene`, `wb_view`, `viz`, `render`, `qc registration` |
| [First-level model registration](models.md) | `models` |
| [Manage definitions](authoring.md) | `def edit`, `def init`, and other `def` subcommands |
| [Bidsify](bidsify.md) | `bidsify` |
| [Configure scan plans](scanplans.md) | scan-plan parsers used by `bidsify` |
| [Branch registration](branches.md) | `branch` |
| [Installed releases](releases.md) | `release` and shared-scheduler repair |
| [Accepting development artifacts](promotion.md) | `promote` |
| [Methods reporting](../methods/reporting.md) | `methods` |
| [Study analyses](studies.md) | `study` |
| [Stimulus annotations](../stimulus-files.md) | `stimuli sync` |

```{toctree}
:maxdepth: 1

work
maintenance
viewing
models
authoring
bidsify
scanplans
branches
releases
promotion
studies
```

`nro def` also manages the site-owned definitions store itself. `nro def init [PATH]` creates
one. `nro def validate [PATH]` checks every definition and reference without
changing files. `nro def apply` publishes local files through a transaction
that covers the full store, and `nro def migrate` updates the store to the
current schema in place. See [definitions stores](../definitions.md) for the
layout, the validation scope, and the version-control policy.

`nro fw`, which has no page of its own, manages your private Flywheel keys. `nro fw addkey SERVER` stores a key
for a server configured for the site. `nro fw list` shows which servers have
usable keys, and `nro fw removekey SERVER` removes one. See
[bidsification](bidsify.md).

## Shared selectors

Most commands that act on derivatives choose their work items with the same
selector options. These commands are `run`, `status`, `stop`, `log`, `purge`,
`gc`, `freshen`, `promote`, `scene`, and `render`. Each command adds its own
options, documented on its page.

| Option | Meaning |
| --- | --- |
| `-p`, `--participant ID ...` | Participant labels, with or without `sub-`. |
| `-P`, `--project PROJECT ...` | Projects beneath the configured BIDS root. |
| `-m`, `--module MODULE ...` | One or more scientific modules. |
| `-w`, `--workflow WORKFLOW ...` | Workflow IDs from the definitions store. |
| `-i`, `--lineage ID ...` | Module-lineage IDs shown by `nro status`. Use `MODULE/ID` when an ID is repeated across modules. |
| `-r`, `--run KEY=VALUE[,VALUE...] ...` | BIDS run entities to match jointly. |
| `-s`, `--space SPACE ...` | Target spaces, for modules from `clean` onward. |
| `-S`, `--smoothing MM ...` | Smoothing FWHM in mm, as nonnegative integers. |
| `--task TASK ...` | Source BIDS tasks. Combined with `--run task=...`, the two lists intersect. |
| `--analysis ANALYSIS ...` | Contrasts analysis labels, the first part of `ANALYSIS/VARIANT`. |
| `--model MODEL ...` | Contrasts variants, or qualified `ANALYSIS/VARIANT` IDs. |
| `--model-set SET ...` | Model sets declared in first-level model YAML. |

In `-r`, different keys combine with AND and the values for one key combine
with OR. Run selectors describe acquisitions, so `run=01` alone may match
several runs that differ by task or direction. An empty value such as `task=`
requires the entity to be absent. Space and smoothing values expand as a
cross-product.

### Patterns

The data selectors `--project`, `--participant`, `--task`, and each `-r` value
accept patterns. A value containing `*`, `?`, or `[` is a case-sensitive glob.
A value starting with `re:` is a regular expression that must match the whole
value. Any other value matches exactly. Quote patterns so the shell does not
expand them:

```bash
nro run -P 'nptl*' -p 't2?' -m networks
nro status -p 're:t(20|21|3[0-9])' -r 'task=lang*' 'run=0[1-3]'
```

Commas separate `-r` alternatives, as in `task=rest,lang*`, but commas inside
`(...)`, `[...]`, or `{...}` belong to the pattern, so `run=re:0{1,2}` stays
whole. `--task` and `-r task=` intersect when both list exact tasks. Give only
one of them when either uses a pattern.

Project and participant patterns match directory names beneath the BIDS root,
and participant patterns match IDs without `sub-`. A participant pattern
searches the selected projects, or every project when none is selected. Each
pattern is replaced by the IDs it matches before the command runs, so a request
records exact IDs. A project or participant pattern that matches nothing is an
error, as is an invalid regular expression. Registry records for a participant
whose BIDS directory was removed can still be selected by exact ID only.

Task and `-r` patterns are matched against each run's entities, both when
`nro run` plans work and when other commands filter registered work.

The other selectors, `--module`, `--workflow`, `--lineage`, `--space`,
`--analysis`, `--model`, and `--model-set`, accept exact values only. `nro run`
uses several of them to decide which work to create, so a pattern there is
rejected.

Inspection and cleanup commands treat an omitted selector as unrestricted. A
new `nro run` request is the exception, because it applies planning defaults:

* An omitted `--module` selects every endpoint of the module graph, currently
  `dynconn`, `networks`, and `contrasts`. Explicit modules restrict the targets,
  and their upstream dependencies are included.
* Workflow, space, and smoothing default to `main`, `fsnative`, and `2`.
* Omitted participants and projects select every match.

Contrasts requests that name no model or model set use model set `main`. An
explicit `--model` bypasses that default, and explicit model and set filters
intersect. Inspection and cleanup apply no default model set. See
[first-level models](../task-models.md) for examples and freshness rules.

A bare `purge` covers every project owned by the current branch and excludes
inherited outputs. A bare `gc` checks every public and private derivative
namespace of the current branch.

Some commands have their own parsers: `publish`, `site`, the installation
commands, `viz`, and `qc registration`. Their pages document their options. Do
not assume the shared short flags mean the same thing there.

## Slurm options

Every `slurm.` site setting has a command-line option of the same name. The
option is the part of the key after `slurm.`, with periods and underscores
replaced by hyphens. An option you omit keeps the site value. The options
affect only allocations that the command itself creates.

| Site key | Option | Accepted by |
| --- | --- | --- |
| `slurm.partition` | `--partition NAME` | `start`, `run`, `bidsify` |
| `slurm.account` | `--account NAME` | `start`, `run`, `bidsify`, `viz`, `scene` |
| `slurm.scheduler.time_hours` | `--scheduler-time-hours N` | `start`, `run`, `bidsify` |
| `slurm.scheduler.memory_gb` | `--scheduler-memory-gb N` | `start`, `run`, `bidsify` |
| `slurm.scheduler.cpus` | `--scheduler-cpus N` | `start`, `run`, `bidsify` |
| `slurm.worker.time_hours` | `--worker-time-hours N` | `run`, `bidsify` |
| `slurm.worker.memory_gb` | `--worker-memory-gb N` | `run`, `bidsify` |
| `slurm.worker.max_memory_gb` | `--worker-max-memory-gb N` | `run`, `bidsify` |
| `slurm.worker.cpus` | `--worker-cpus N` | `run`, `bidsify` |
| `slurm.long_worker.time_hours` | `--long-worker-time-hours N` | `run` |
| `slurm.long_worker.cpus` | `--long-worker-cpus N` | `run` |
| `slurm.viewer_partition` | `--viewer-partition NAME` | `viz`, `scene` |
| `slurm.viewer.time_hours` | `--viewer-time-hours N` | `viz`, `scene` |
| `slurm.viewer.memory_gb` | `--viewer-memory-gb N` | `viz`, `scene` |
| `slurm.viewer.cpus` | `--viewer-cpus N` | `viz`, `scene` |

`start`, `run`, and `bidsify` can each start the shared scheduler, so they
share the scheduler options. These apply only when no scheduler is running or
pending. A running scheduler keeps its allocation until it exits. For example:

```bash
nro start --partition sphinx --scheduler-memory-gb 16
nro run -P nptl -m networks --partition sphinx --worker-cpus 8
```

`--partition` and `--account` apply to both the scheduler and the workers that
a command starts. `bidsify` takes worker resources from each request's
ingestion profile, and the worker options override the profile when given. The
viewer options apply to `nro viz` and to `nro scene --open`. To change a
default for everyone, set the site key with `nro set`.

## Finding source images

`nro find` prints absolute paths to source BIDS images. It does not inspect
derivatives or registry records. It accepts the project, participant, run, and
task selectors, with the [pattern rules](#patterns) above:

```bash
nro find -P nptl -p 're:t(12|20)' -r 'task=Rest*' 'run=0[1-3]'
nro find -P nptl -r datatype=anat suffix=T1w,T2w
```

Unlike the derivative commands, `-r` here also accepts image-level entities
such as `datatype` and `suffix`, and a pattern that matches nothing prints
nothing. `--json` prints
the selected paths as a JSON array. The command rejects derivative-only
selectors: module, workflow, lineage, model, space, and smoothing.
