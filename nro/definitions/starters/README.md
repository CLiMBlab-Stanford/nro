# nro definitions

This directory holds the scientific definitions that an nro installation uses:
module configurations, workflows, models, markup, and site policy. Version it
separately from nro's code. Keep credentials, imaging data, registry databases,
logs, and generated derivatives out of it.

## Layout

* `configs/CLASS/ID_CLASS.yml`: optional module-configuration overrides.
* `site/site.yml`: protected storage, execution, resource, and ingestion-source
  settings. Only the shared site store may contain this file.
* `workflows/ID_workflow.yml`: the configuration selected for each
  configuration class.
* `models/ANALYSIS/VARIANT.yml`: source tasks, predictors, contrasts, and
  model-set membership.
* `markup/ID_markup.yml`: optional anatomical selections and source
  exclusions, grouped by BIDS project and then by participant.
* `hardware/gradient_unwarping.yml`: acquisition matching, gradient-unwarping
  policy, and optional scanner metadata assertions.
* `events/TASK/`: standard event tables and an `index.yml` listing their IDs.
* `bidsify/PROFILE.yml`: conversion rules and ingestion worker settings.
* `scanplans/parser.py`: optional site parser for arbitrary scan-plan files.

## What a new store contains

A new store inherits the packaged `main` configurations. Adding
`configs/CLASS/main_CLASS.yml` overrides selected defaults for that class, and
omitted settings still follow the package.

New shared stores have no first-level models, event tables, or configured
Flywheel servers. The starter hardware catalog is empty, so automatic gradient
unwarping does nothing until the site adds a matching profile and coefficient
resource. Add these definitions before you request work that needs them.

Development stores inherit `site/site.yml` from the shared store and are
rejected if they supply their own copy.

## Editing definitions

Change definitions through nro commands. Managed YAML and Python files carry a
warning against direct editing, and the store manifest detects changes made
outside those commands.

* `nro def edit`, `nro def rm`, and `nro def ls` manage configs, workflows,
  models, and source markup.
* `nro def apply --file RELATIVE_PATH=LOCAL_FILE` installs event catalogs,
  ingestion profiles, hardware policy, and scan-plan parsers.
* `nro def validate PATH` checks the store.
* `nro site set storage.definitions=PATH` selects this directory. Shared
  installations require `--maintain`.

Git initialization, commits, remotes, and branch changes are explicit
operations for the people who manage the store.
