# Manage definitions

Use `nro def` to create, edit, list, and remove the models, module
configurations, workflows, and markup that processing reads. The same command
also initializes, migrates, and validates whole
[definitions stores](../definitions.md). None of these operations open the
registry or request scientific work.

## Edit a definition

`nro def edit` opens an existing definition, or creates it if it is missing:

```bash
nro def edit model newtask
nro def edit model newtask alternative
nro def edit clean alternative
nro def edit workflow experiment
nro def edit markup
```

The arguments depend on the kind of definition:

* Models take `ANALYSIS [VARIANT]`. The variant defaults to `main`.
* Module configurations take `MODULE ID`. The modules are `anat`, `func`,
  `clean`, `microparcellation`, `dynconn`, `networks`, and `contrasts`.
* Workflows take one ID.
* Markup takes an optional ID that defaults to `main`.

nro copies the definition to a private draft and opens it with `$VISUAL`, or
`$EDITOR` if `$VISUAL` is unset. If neither is set, it tries `nano` and then
`vi`. The editor command may include options, which are split without shell
evaluation. A GUI editor must wait until you close the file, as with
`code --wait`.

When you save the draft and close the editor, nro validates it, prints the
diff, and publishes it. There is no separate save prompt. If you exit without
saving, or save without changes, the stored definition stays as it was.

Before publishing, nro checks that the stored file has not changed since you
started editing. It then validates the changed definition together with the
definitions that depend on it, in one transaction. For example, changing a
config rechecks every workflow that selects it. Run `nro def validate` for an
audit of the whole store.

If a draft is invalid, you can reopen it and fix it. Interrupted, conflicting,
failed, and invalid edits all keep a private draft in the definitions store,
and the next edit of that definition offers to recover it or start over.
Drafts are kept separately for each operating-system user, ignored by Git, and
excluded from store validation.

## Start a new definition

### Models

When a model is missing, nro can infer a starting model from source BIDS
events:

```bash
nro def edit model language --task sentences pictures -P myproject -p 01 02
nro def edit model newtask alternative --from newtask main
```

By default, nro finds source BOLD runs whose task matches the analysis label
and resolves their inherited event tables. Use `--task TASK ...` when the
analysis uses a differently named source task or several tasks. Omitted
project (`-P`) and participant (`-p`) selectors match everything under the
configured BIDS root. Both accept the glob and `re:` patterns described under
[shared selectors](index.md#shared-selectors). To use specific local tables instead of discovery, pass
`--events FILE ...`.

The initializer reads conditions from the `trial_type` column. Pass
`--conditions COLUMN` to use another column without being prompted.

When the event tables reference stimuli through `stim_id`, the initializer
lists the annotations present in each project's `stimuli/` directory. Pass
`--annotation LABEL` to draft from one of them. The tables are then expanded
from the annotation files, as `contrasts` does before fitting, and the draft
includes `annotation: LABEL`. An annotation draft does not default to
`trial_type`; pass `--conditions COLUMN` or choose a column when prompted.
`--annotation` requires event files inside a BIDS project.

The generated model contains the union of observed conditions, the SPM HRF,
and one contrast per condition against the implicit baseline. Review the
design before fitting it. Event discovery cannot tell whether a design is
scientifically suitable or estimable.

New models, whether inferred or copied with `--from`, have `model_set: []`.
Add `model_set: main` if the model should be included in default requests.

### Configs and workflows

A new config starts as an empty override, with the packaged `main` values shown
in comments. Add only the values that should differ. `--from ID` copies another
config of the same class instead:

```bash
nro def edit clean another --from alternative
nro def edit workflow another --from experiment
```

A new workflow draft lists every configuration class with `main` selected.
Creating a config does not select it in any workflow, so edit a workflow to use
it.

## Work with local files

`--output` writes the initial draft to a local file without registering it or
opening an editor:

```bash
nro def edit model newtask --events run1_events.tsv run2_events.tsv --output model.yml
nro def edit clean alternative --output clean.yml
```

The output path must be outside the definitions store and must not exist yet.

`--file` publishes a local file without opening an editor:

```bash
nro def edit model newtask --file model.yml
nro def edit clean alternative --file clean.yml
```

Passing `--file` is itself the publication request, so no confirmation flag is
needed. Validation and conflict checks still apply. `--from`, `--file`, and the
event-discovery options each supply a draft, so they cannot be combined.

Definitions that have no typed editor are addressed by their path relative to
the store:

```bash
nro def edit file hardware/gradient_unwarping.yml
nro def edit file bidsify/main.yml --file ./main.yml
```

An edit in this form validates the complete staged store before publishing.
Use `nro site set` instead of `file` to change the protected `site/site.yml`
settings.

## List definitions

`nro def ls` shows the active definitions of one kind and the inheritance layer
that supplies each one:

```bash
nro def ls anat
nro def ls clean
nro def ls model
nro def ls model langlocSN
nro def ls workflow
nro def ls markup
nro def ls file
```

For models, an optional analysis label narrows the list. Config listings
include the packaged `main` definitions even when the external store does not
override them. Add `--json` for machine-readable output.

## Remove a definition

`nro def rm` removes one definition after asking for confirmation. Pass
`-y`/`--yes` to skip the prompt. Removal does not touch derivatives, logs,
registry records, or demand.

```bash
nro def rm model spatialFIN main
nro def edit model spatialFIN -P nptl

nro def rm clean alternative
nro def rm workflow experiment
nro def rm markup alternative
```

The first pair of commands removes a model and then regenerates it from
events.

Removing an external `main` override for a class restores the packaged
defaults. If you remove a named config, nro reports the workflows that still
select it, and validation blocks the removal while those references remain.

Removal reports the location of a temporary recovery copy. Copy it elsewhere if
you need it to survive temporary-file cleanup.

## Store operations

The same command manages whole stores:

```bash
nro def init /data/lab/nro-definitions
nro def migrate
nro def validate
```

`nro def apply` publishes several local files in one validated transaction.
See [definitions stores](../definitions.md) for store selection, schema
migrations, and version control.
