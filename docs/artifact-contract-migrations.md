# Artifact contract migrations

This page is for developers who add, rename, or remove settings in a module's
artifact contract. It explains how nro compares contracts written under older
schemas and what a contract migration must declare.

Artifact contracts record the scientific meaning of completed work. Their schema
changes independently of the private SQLite registries that store orchestration
state, so the two use separate migration systems. Registry changes follow the
[registry migration](registry-migrations.md) process.

Each module's artifact-contract schema starts at version 1, and a contract with
no explicit version is read as version 1. Before comparing freshness, nro
migrates both the recorded contract and the requested contract to the module's
current schema. The version number itself does not affect scientific
equivalence.

A development-branch module that is not part of the central installation uses
the unchanged version-1 baseline. The branch's own source must compile its
scientific contract before scheduler admission. The central scheduler does not
import the branch's module catalog or invent migrations for it.

## Adding a field

A new field must declare both its current default and its historical value.
They answer different questions:

* The current default controls newly compiled work when an author omits the field.
* The historical value describes work created before the field existed.

For example, lesion-aware anatomy adds a Boolean field to captured source
markup. Its current default and historical value are both false, so existing
anatomy stays equivalent to newly compiled ordinary anatomy. Setting the value
to true selects a different anatomical graph and makes that participant's
dependent work stale.

When historical behavior cannot be inferred, the migration records an
indeterminate value. An artifact with an indeterminate value cannot prove that
it matches current work, so it becomes stale. nro never substitutes today's
default for unknown historical behavior.

## Supported changes

Contract migrations use restricted operations to add, rename, remove, or map
fields. Removing a field requires a declaration that its value no longer
contributes to scientific identity. A value mapping must cover every historical
value it encounters. Arbitrary migration callbacks are not supported.

A migration produces a view used only for comparison. It does not rewrite the
original completion record or the public derivative. nro accepts that evidence
only if the saved fingerprint still matches the original contract bytes.

## Development requirements

When a contract changes, append the next migration to the affected module's
chain. Do not edit a migration after release. Tests must cover the historical
representation, the new default, nondefault current values, and any
indeterminate state.
