# Task model registration

`nro models` manages YAML task models in the selected definitions store. It does not
open the registry, request fits, or modify source BIDS data.

Use [create and edit](authoring.md) for event-based initialization and reviewed
changes to existing definitions. The commands here provide inspection,
validation, registration of local files, and compilation.

```bash
nro models list
nro models show langlocSN/main
nro models validate langlocSN /path/to/model.yml
nro models register langlocSN/alternative /path/to/model.yml
nro models compile langlocSN/main --config main
```

`list` prints task/variant IDs and set memberships (`-` means no set).
`show` prints validated YAML. `validate TASK FILE` checks supported task syntax
without writing or inspecting a dataset. `register TASK/VARIANT FILE` validates
and publishes `models/TASK/VARIANT.yml`; an existing ID is an error. Identifiers
allow letters, digits, and hyphens in each part. Task comes from the registration,
not a field inside the model.

`compile` prints a Stats Models template combined with the selected firstlevels
configuration. This template includes session and subject summaries. Actual
execution omits session nodes for data without sessions and saves run-specific
documents with the observed nuisance columns and resolved event HRFs. Variable
availability and numerical estimability are checked when the data are available.

The user must have write permission to the definitions store. Coordinate
model changes with maintainers of a shared installation. Registration alone
does not request work. Select models with `--task`, `--model`, or `--model-set`;
firstlevels requests default to set `main`. See [task models](../task-models.md)
for syntax and membership rules, and [firstlevels](../modules/firstlevels.md)
for processing and output layout.

`python -m nro.bin.models` exposes the same parser.
