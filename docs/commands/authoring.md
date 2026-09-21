# Create, edit, and delete definitions

`nro create`, `nro edit`, and `nro delete` manage task models, scientific
configs, workflows, and source markup
in the selected [definitions store](../definitions.md). They validate and
save drafts or remove definitions without opening the registry or requesting
work. Deletion does not require a valid definition.

Publication verifies the store manifest, then validates the changed definition
and definitions that depend on it. For example, changing a config rechecks the
workflows that select it. Unrelated models and event catalogs are not reparsed.
Use `nro definitions validate` when a complete store audit is needed.

```bash
nro create model newtask
nro create config clean/alternative
nro create workflow experiment
nro create markup main

nro edit model newtask/main
nro edit config clean/alternative
nro edit workflow experiment
nro edit markup main
```

Model IDs are `TASK/VARIANT`; a task alone means `TASK/main`. Config IDs are
`CLASS/ID`, where class is `anat`, `func`, `clean`, `microparcellation`,
`dynconn`, `networks`, or `firstlevels`. Each module has its own configuration
class and derivative directory. Workflow and markup IDs have no class prefix.
Markup files select manual anatomicals and known-bad source paths by BIDS
project and participant; see [source markup](../definitions.md#source-markup).

If a create target exists, the interactive command announces that it is opening
the existing definition for editing. Creation-only options such as `--from`
are errors in this case. Noninteractive create never replaces an existing
definition. Edit requires an existing target.

## Model drafts from events

```bash
nro create model newtask -P myproject -p 01 02
nro create model newtask/alternative --from newtask/main
```

Without `--from` or `--file`, creation discovers source BOLD runs bearing the
task label and resolves their inherited event tables. Omitted project and
participant selectors mean all matches beneath the configured BIDS root;
the command does not accept a per-invocation root. Alternatively, pass one or
more local tables with `--events FILE ...`. Event files are read, but images
are not.

The initializer prefers `trial_type` as the condition column. If there is no
common `trial_type`, it asks for a common column; `--conditions COLUMN` supplies
the choice explicitly. A `trial_type` column present in only some tables needs
an explicit decision: narrow discovery or specify another common column.
Missing tables, malformed timing, and ambiguous BIDS inheritance are errors.

The draft contains the union of observed conditions, the SPM HRF, and one
contrast per condition against the implicit baseline. Run aggregation is set by
the selected firstlevels configuration.
Weights use condition shorthand, such as `E: {E: 1}`, under a shared
`conditions: trial_type`. Labels that themselves start with the column prefix
are explicitly qualified to preserve their meaning.
Contrast names are made filename-safe while retaining the original condition
labels in their weights. Conditions missing from individual tables remain in
the model. The console report lists event columns, condition counts and table
coverage, and unlabeled events. A shared inherited table is counted once, so
table counts need not equal run counts.

Other categorical or numeric event columns are reported but not added as
predictors. Between-condition comparisons, modulator transformations, and HRF
exceptions require scientific judgment. Review the generated baseline and
design before fitting. See [task models](../task-models.md) for the YAML syntax.

New inferred and copied models have `model_set: []`. Add `model_set: main`
when a model should enter default requests. A model outside all sets can still
be explicitly selected with `nro run -m firstlevels --model TASK/VARIANT`.
Importing a local file with `--file` preserves its membership as written.

## Config and workflow drafts

A new config starts with no overrides and includes commented packaged defaults
as reference. Add only the settings that should differ. Omitted keys continue
to follow the class's packaged `main` configuration and any external `main`
override. To copy an existing override file:

```bash
nro create config clean/another --from clean/alternative
nro create workflow another --from experiment
```

Copying a class's `main` config also produces an empty override draft with
commented defaults, so it does not pin every default. Config copies must belong
to the same class. Workflow drafts list every class and its selected config ID;
change the appropriate entry to use a new config. Creating a config alone does
not select it in any workflow.

## Editor and save behavior

The commands open a private temporary copy with `$VISUAL`, then `$EDITOR` if
VISUAL is unset. They fall back to `nano`, then `vi`, when available. Editor options
are supported without shell evaluation; GUI editors must use an option that
waits until editing is finished, such as `code --wait`.

On exit, nro validates the draft, prints a diff, and asks before saving.
Invalid drafts can be reopened. Cancelled or failed edits retain the draft and
print its path. An unchanged edit does not rewrite the stored file.

Validation checks model syntax, config keys and basic value types, and workflow
references. Duplicate YAML keys are rejected. It does not establish numerical
estimability, scientific suitability, or every parameter constraint enforced
during execution. Scientific changes may affect artifact freshness when the
registry is next assessed; model-set changes alone do not.

Saving uses an atomic replacement and checks that the stored content has not
changed since editing began. Concurrent authoring commands use file and store
locks. The saved definition and store manifest are published in one validated
transaction. Direct edits are rejected on later reads. Permission errors do not
trigger elevation or redirection to another store. Coordinate changes to
shared definitions with other users.

## Local files and noninteractive use

To prepare a local draft without registering it:

```bash
nro create model newtask --events run1_events.tsv run2_events.tsv --output model.yml
nro create config clean/alternative --output clean.yml
```

`--output` creates a new file outside the central store and never overwrites an
existing file. It cannot be combined with `--file` or `--yes`.

To publish a locally edited file without opening an editor:

```bash
nro create model newtask/main --file model.yml --yes
nro edit config clean/alternative --file clean.yml --yes
```

`--file` replaces the editor step; `-y`/`--yes` skips save confirmation. Both are
required for noninteractive publication. Validation and conflict checks still
apply. Without `--yes`, `--file` shows the diff and asks for confirmation in an
interactive terminal. `--from`, `--file`, and event-discovery options cannot be
combined as competing sources for a draft.

## Delete and regenerate

`delete` removes one definition after confirmation. It does not delete other
variants, derivatives, or logs, and does not change registry demand or workers.
Use `--yes` to skip confirmation, including in noninteractive scripts.

```bash
nro delete model spatialFIN/main
nro create model spatialFIN/main -P nptl
```

This recreates a model from current events and initializer defaults. Custom
contrasts, transformations, and model-set membership are not retained. Review
the draft and set membership before saving. Config, workflow, and markup
definitions use
the same staged pattern:

```bash
nro delete config clean/alternative
nro create config clean/alternative
nro delete workflow experiment
nro create workflow experiment
nro delete markup alternative
nro create markup alternative
```

Deleting an external class `main` override restores the packaged defaults.
Deleting a named config reports workflows that still select it. Full-store
validation prevents publication until those references are changed in the same
transaction or beforehand. Deleting workflow `main` warns that default
requests will need it recreated. Missing definitions can affect later artifact
assessments, so coordinate shared-store resets with other users.

Deletion uses the authoring lock and snapshot check, and reports the path of a
private recovery copy in the system temporary directory. Copy it elsewhere to
retain it beyond temporary-file cleanup. Only the definition file is removed;
parent directories and the small hidden authoring lock remain. No scientific
outputs are purged.

`python -m nro.bin.create`, `python -m nro.bin.edit`, and
`python -m nro.bin.delete` expose the same commands.
