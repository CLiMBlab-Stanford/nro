# Manage definitions

`nro def` manages individual definitions and complete
[definitions stores](../definitions.md). These operations do not open the
registry or request scientific work.

## Edit or create a definition

`edit` opens an existing definition or creates it when it is missing:

```bash
nro def edit model newtask
nro def edit model newtask alternative
nro def edit config clean alternative
nro def edit workflow experiment
nro def edit markup main
```

Model arguments are `TASK [VARIANT]`; an omitted variant means `main`. Config
arguments are `CLASS ID`. Workflow and markup definitions take one ID. The
configuration classes are `anat`, `func`, `clean`, `microparcellation`,
`dynconn`, `networks`, and `firstlevels`.

The command opens a private draft with `$VISUAL`, then `$EDITOR`. If neither is
set, it tries `nano` and then `vi`. Editor commands may contain options without
shell evaluation; a GUI editor must wait until editing finishes, as in
`code --wait`.

When the editor writes the draft, nro validates it, prints the diff, and
publishes it. Exiting without writing leaves the stored definition unchanged.
There is no separate save prompt. An unchanged edit also leaves the definition
untouched.

Invalid drafts can be reopened. Interrupted, conflicting, failed, and invalid
edits retain a private draft in the definitions store. The next edit offers to
recover it or start over. Drafts are separated by operating-system user,
ignored by Git, and excluded from store validation.

Publication checks that the stored bytes have not changed since editing began.
It then validates the changed definition and definitions that depend on it in
one transaction. For example, changing a config rechecks workflows that select
it. Use `nro def validate` for a complete store audit.

## Initialize models, configs, and workflows

A missing model can be inferred from source BIDS events:

```bash
nro def edit model newtask -P myproject -p 01 02
nro def edit model newtask alternative --from newtask main
```

Without `--from` or `--file`, model initialization discovers source BOLD runs
with the task label and resolves their inherited event tables. Omitted project
and participant selectors mean all matches under the configured BIDS root.
Pass `--events FILE ...` to use explicit local tables instead. The initializer
prefers `trial_type`; `--conditions COLUMN` resolves another shared condition
column noninteractively.

The generated model contains the observed condition union, the SPM HRF, and one
contrast per condition against the implicit baseline. New inferred and copied
models have `model_set: []`. Add `model_set: main` when the model should enter
default requests. Review the design before fitting; event discovery cannot
establish scientific suitability or estimability.

A missing config starts as an empty override with its packaged `main` values in
comments. Add only values that should differ. `--from ID` copies another config
in the same class:

```bash
nro def edit config clean another --from alternative
nro def edit workflow another --from experiment
```

Workflow drafts list every configuration class with `main` selected. Creating
a config does not select it in a workflow.

## Local and noninteractive files

`--output` writes an initialized local draft without registering it:

```bash
nro def edit model newtask --events run1_events.tsv run2_events.tsv --output model.yml
nro def edit config clean alternative --output clean.yml
```

The output must be outside the definitions store and must not already exist.
Publish a local file without opening an editor with `--file`:

```bash
nro def edit model newtask --file model.yml
nro def edit config clean alternative --file clean.yml
```

Supplying `--file` is an explicit publication request, so no confirmation flag
is required. Validation and conflict checks still apply. `--from`, `--file`,
and event-discovery options cannot be combined as competing draft sources.

Definitions without a typed editor use their store-relative path:

```bash
nro def edit file hardware/gradient_unwarping.yml
nro def edit file bidsify/main.yml --file ./main.yml
```

This path form validates the complete staged store before publication. Use
`nro paths set` instead of `file` for protected `site/site.yml` settings.

## List definitions

`ls` reports active definitions and the inheritance layer supplying each one:

```bash
nro def ls config
nro def ls config anat
nro def ls model
nro def ls model langlocSN
nro def ls workflow
nro def ls markup
nro def ls file
```

Add `--json` for machine-readable output. Config listings include packaged
`main` definitions even when the external store does not override them.

## Remove and regenerate

`rm` removes one definition after confirmation. It does not remove derivatives,
logs, registry records, or demand. `--yes` is available for noninteractive use.

```bash
nro def rm model spatialFIN main
nro def edit model spatialFIN -P nptl

nro def rm config clean alternative
nro def rm workflow experiment
nro def rm markup alternative
```

Removing an external class `main` override restores the packaged defaults.
Removing a named config reports workflows that still select it, and validation
prevents removal while unresolved references remain. Removal reports a
temporary recovery copy. Copy it elsewhere if it must survive temporary-file
cleanup.

## Store operations

The same command initializes, migrates, and validates whole stores:

```bash
nro def init /data/lab/nro-definitions
nro def migrate
nro def validate
```

`nro def apply` publishes several local files in one validated transaction.
See [definitions stores](../definitions.md) for store selection, schema
migrations, and version-control policy.
