# Command-line reference

Run `nro COMMAND --help` for the installed parser. Commands are discovered from
public executable modules in `nro/bin`; `python -m nro.bin.COMMAND` is equivalent
when using the correct interpreter. The engine's Python modules are not
automatically exposed as commands.

```{toctree}
:maxdepth: 1

work
maintenance
viewing
models
authoring
bidsify
branches
releases
promotion
cutover
```

`nro definitions create [PATH]` initializes a site-owned definitions store.
`nro definitions validate [PATH]` checks every definition and reference without
changing files. See [definitions stores](../definitions.md) for their layout,
validation scope, and version-control policy.

## Shared selectors

`run`, `status`, `stop`, `log`, `purge`, `promote`, and `scene` use the common selector
engine. Their remaining options are command-specific.

| Option | Meaning |
| --- | --- |
| `-p`, `--participant ID ...` | Participant labels, with or without `sub-`. |
| `-P`, `--project PROJECT ...` | Projects beneath the configured BIDS root. |
| `-m`, `--module MODULE ...` | One or more scientific modules. |
| `-w`, `--workflow WORKFLOW ...` | Workflow IDs from the definitions store. |
| `-r`, `--run KEY=VALUE[,VALUE...] ...` | Joint BIDS run-entity matching. |
| `-s`, `--space SPACE ...` | Target spaces from clean onward. |
| `-S`, `--smoothing MM ...` | Nonnegative integer FWHM in mm. |
| `--task TASK ...` | Tasks; intersects `--run task=...` when both are supplied. |
| `--model MODEL ...` | Firstlevels variants or qualified `TASK/VARIANT` IDs. |
| `--model-set SET ...` | Memberships declared in task YAML. |
| `--bids-root PATH` | Override the directory containing projects. |

Run selectors combine different keys with AND and values for a key with OR.
Space and smoothing values expand as a cross-product. Run selectors describe
acquisitions; `run=01` alone is not necessarily unique across tasks or directions.

For `run`, omitted `--module` selects every endpoint of the module graph,
currently `dynconn`, `networks`, and `firstlevels`. Explicit modules restrict the targets;
their upstream dependencies are included. Workflow, space, and smoothing default
to `main`, `fsnative`, and `2`. Missing participants/projects select all
matches. Inspection and cleanup commands ordinarily leave omitted selectors
unrestricted. Bare `purge` covers all projects owned by the current branch;
inherited outputs are excluded. `publish`, `set`,
installation commands, and `qc registration` have different parsers, documented
on their pages; do not assume the common short flags apply to them.

Firstlevels requests without explicit model/set selection use model set `main`.
An explicit model bypasses that default; explicit model and set filters
intersect. Inspection and cleanup have no default model-set restriction.
See [task models](../task-models.md) for examples and freshness rules.

All commands support `-h`/`--help`. Parse failures and operational errors exit
nonzero. Progress/errors may go to stderr even when a command offers `--json`.
