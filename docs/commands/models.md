# First-level model registration

`nro models` lists, inspects, validates, registers, and compiles the YAML
first-level models in the selected definitions store. It does not open the
registry, request fits, or change source BIDS data. To create a model from
events or to make a reviewed change to an existing one, use the commands in
[create and edit](authoring.md).

```bash
nro models list
nro models show langlocSN/main
nro models validate langlocSN /path/to/model.yml
nro models register langlocSN/alternative /path/to/model.yml
nro models compile langlocSN/main --config main
```

| Subcommand | Behavior |
| --- | --- |
| `list` | Print each `ANALYSIS/VARIANT` ID with its set memberships. `-` means the model belongs to no set. |
| `show ID` | Print the validated YAML of one model. |
| `validate ANALYSIS FILE` | Check that the file uses supported model syntax. Nothing is written and no dataset is inspected. |
| `register ANALYSIS/VARIANT FILE` | Validate the file and publish it as `models/ANALYSIS/VARIANT.yml`. Registering an existing ID is an error. |
| `compile ID [--config CONFIG]` | Print the Stats Models template combined with a contrasts configuration, `main` by default. |

Each part of a model ID may contain letters, digits, and hyphens. A
single-task model uses `ANALYSIS` as its source task. A multi-task model
declares a nonempty `tasks` list.

The template that `compile` prints includes session and subject summaries. It
shows the model before any data are attached. During execution, nro omits the
session nodes for data without sessions and saves run-specific documents with
the observed nuisance columns and the resolved event HRFs. Whether each
variable is available and each contrast is estimable is checked only when the
data are present.

Registering a model requires write permission on the definitions store. On a
shared installation, coordinate model changes with its maintainers.
Registration alone requests no work. Select models in `nro run` with
`--analysis`, `--task`, `--model`, or `--model-set`. Contrasts requests default
to set `main`. See [first-level models](../task-models.md) for syntax and
membership rules, and [contrasts](../modules/contrasts.md) for processing and
output layout.

`python -m nro.bin.models` exposes the same parser.
