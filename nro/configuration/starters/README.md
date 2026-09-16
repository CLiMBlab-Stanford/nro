# nro definitions

This directory holds the scientific definitions used by an nro installation.
Version it separately from nro's code. Keep credentials, imaging data, registry
databases, logs, and generated derivatives outside this repository.

* `configs/CLASS/ID_CLASS.yml`: optional module-configuration overrides. Packaged
  `main` configurations supply defaults when the store has no matching file.
* `site/site.yml`: protected storage, execution, resource, and ingestion-source
  settings. Only the shared site store may contain this file.
* `workflows/ID_workflow.yml`: configuration selections for each configuration class.
* `models/TASK/VARIANT.yml`: task predictors, contrasts, and model-set membership.
* `markup/ID_markup.yml`: optional anatomical selections and source exclusions,
  grouped first by BIDS project and then by participant.
* `hardware/gradient_unwarping.yml`: acquisition matching and gradient-correction
  policy for scanners and coils known to the site.
* `events/TASK/`: standard event tables and an `index.yml` listing their IDs.
* `bidsify/PROFILE.yml`: conversion rules and ingestion worker settings.
* `scanplans/parser.py`: optional site parser for arbitrary scan-plan files.

Run `nro definitions validate PATH` before adopting edits. Select this directory
with `nro paths set definitions=PATH`; shared installations require `--maintain`.
Use `nro create`, `nro edit`, and `nro delete` for configs, workflows, models,
and source markup.
Edit event indexes and ingestion profiles directly, then validate the store.

New shared stores have no task models, event tables, or configured Flywheel
servers. Development stores inherit `site/site.yml` from the shared store and
are rejected if they try to supply their own copy.
The starter hardware catalog is empty, so automatic gradient unwarping is off
until the site adds a matching profile and coefficient resource.
Populate those definitions for your experiments before requesting that work.
New stores inherit packaged `main` configurations. Adding
`configs/CLASS/main_CLASS.yml` overrides selected defaults for that class; omitted
settings still follow the package. Git initialization, commits, remotes, and
branch changes are explicit human-managed operations.
