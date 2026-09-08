# nro definitions

This directory holds the scientific definitions used by an nro installation.
Version it separately from nro's code. Keep credentials, imaging data, registry
databases, logs, and generated derivatives outside this repository.

* `configs/CLASS/ID_CLASS.yml`: module configurations; `main` supplies defaults.
* `workflows/ID_workflow.yml`: configuration selections for each derivative class.
* `models/TASK/VARIANT.yml`: task predictors, contrasts, and model-set membership.
* `events/TASK/`: standard event tables and an `index.yml` listing their IDs.
* `bidsify/PROFILE.yml`: Flywheel servers, protocol rules, and ingestion settings.

Run `nro definitions validate PATH` before adopting edits. Select this directory
with `nro paths set definitions=PATH`; shared installations require `--maintain`.
Use `nro create`, `nro edit`, and `nro delete` for configs, workflows, and models.
Edit event indexes and ingestion profiles directly, then validate the store.

New stores have no task models, event tables, or configured Flywheel servers.
Populate those definitions for your experiments before requesting that work.
The configuration values copied at creation remain under your control: updating
nro does not overwrite them. Git initialization, commits, remotes, and branch
changes are explicit human-managed operations.
