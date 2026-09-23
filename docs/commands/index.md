# Command-line reference

Run `nro help` for task guides and an inventory of installed commands. Use
`nro help TOPIC` for common workflows, `nro help --command COMMAND` for the
current command syntax, or `nro COMMAND --help` directly. `nro --version`
reports the selected installation's package version.

Commands are discovered from public executable modules in `nro/bin`;
`python -m nro.bin.COMMAND` is equivalent when using the correct interpreter.
The engine's Python modules are not automatically exposed as commands.

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
```

`nro definitions create [PATH]` initializes a site-owned definitions store.
`nro definitions validate [PATH]` checks every definition and reference without
changing files. `nro definitions apply` publishes local files through a
full-store transaction, and `nro definitions migrate` applies the current store
schema in place. See [definitions stores](../definitions.md) for their layout,
validation scope, and version-control policy.

`nro fw addkey SERVER` stores the current user's private Flywheel key for a
configured site server. `nro fw list` reports which servers have usable keys,
and `nro fw removekey SERVER` removes one. See [bidsification](bidsify.md).

## Shared selectors

`run`, `status`, `stop`, `log`, `purge`, `gc`, `promote`, `scene`, and `render` use the
common selector engine. Their remaining options are command-specific. `find`
uses the project, participant, run, and task subset to inspect source BIDS
images.

| Option | Meaning |
| --- | --- |
| `-p`, `--participant ID ...` | Participant labels, with or without `sub-`. |
| `-P`, `--project PROJECT ...` | Projects beneath the configured BIDS root. |
| `-m`, `--module MODULE ...` | One or more scientific modules. |
| `-w`, `--workflow WORKFLOW ...` | Workflow IDs from the definitions store. |
| `-i`, `--lineage ID ...` | Select module-lineage IDs shown by `nro status`; accepts `MODULE/ID` to disambiguate. |
| `-r`, `--run KEY=VALUE[,VALUE...] ...` | Joint BIDS run-entity matching. |
| `-s`, `--space SPACE ...` | Target spaces from clean onward. |
| `-S`, `--smoothing MM ...` | Nonnegative integer FWHM in mm. |
| `--task TASK ...` | Tasks; intersects `--run task=...` when both are supplied. |
| `--model MODEL ...` | Firstlevels variants or qualified `TASK/VARIANT` IDs. |
| `--model-set SET ...` | Memberships declared in task YAML. |

Run selectors combine different keys with AND and values for a key with OR.
Space and smoothing values expand as a cross-product. Run selectors describe
acquisitions; `run=01` alone is not necessarily unique across tasks or directions.

For `run`, omitted `--module` selects every endpoint of the module graph,
currently `dynconn`, `networks`, and `firstlevels`. Explicit modules restrict the targets;
their upstream dependencies are included. Workflow, space, and smoothing default
to `main`, `fsnative`, and `2`. Missing participants/projects select all
matches. Inspection and cleanup commands ordinarily leave omitted selectors
unrestricted. Bare `purge` covers all projects owned by the current branch;
inherited outputs are excluded. Bare `gc` checks every current-branch public and
private derivative namespace. `publish`, `set`,
installation commands, and `qc registration` have different parsers, documented
on their pages; do not assume the common short flags apply to them.

Firstlevels requests without explicit model/set selection use model set `main`.
An explicit model bypasses that default; explicit model and set filters
intersect. Inspection and cleanup have no default model-set restriction.
See [task models](../task-models.md) for examples and freshness rules.

All commands support `-h`/`--help`. Parse failures and operational errors exit
nonzero. Progress/errors may go to stderr even when a command offers `--json`.
Commands always use the BIDS root from the global site configuration.

## Finding source images

`nro find` prints absolute paths to source BIDS images and does not inspect
derivatives or registry records. Project names are exact. Participant, task,
and `-r` entity values are full regular expressions:

```bash
nro find -P nptl -p 't(12|20)' -r 'task=Rest.*' 'run=0[1-3]'
nro find -P nptl -r datatype=anat suffix='T1w|T2w'
```

Different entity keys combine with AND. Comma-delimited expressions for one
key combine with OR, as in `task=Rest,langloc`. Parenthesized groups, character
classes, and repetition bounds can contain commas without being split. An empty
expression, such as `task=`, requires that entity to be absent. `--json` emits
the selected paths as a JSON array. Derivative-only selectors such as module,
workflow, lineage, model, space, and smoothing are rejected.
