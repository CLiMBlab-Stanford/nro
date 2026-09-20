# Artifact contract migrations

Artifact contracts record the scientific meaning of completed work. Their schema
can change independently from the private SQLite registries that store orchestration
state. nro therefore uses separate migration systems for these two kinds of change.

Each module has an artifact-contract schema beginning at version 1. Older contracts
without an explicit version are interpreted as version 1. Before freshness is
compared, nro migrates the recorded contract and the requested contract to the
module's current schema. The version number itself does not affect scientific
equivalence.

A development-branch module that is not part of the central installation uses
the unchanged version-1 baseline. Its own source must compile its scientific
contract before scheduler admission; the central scheduler does not import the
branch's module catalog or invent migrations for it.

## Adding a field

A new field must declare both its current default and its historical meaning. These
values answer different questions:

* The current default controls newly compiled work when an author omits the field.
* The historical value describes work created before the field existed.

For example, lesion-aware anatomy adds a Boolean field to captured source markup.
Its current default and historical value are both false. Existing anatomy therefore
remains equivalent to newly compiled ordinary anatomy. Setting the value to true
selects a different anatomical graph and makes that participant's dependent work
stale.

When historical behavior cannot be inferred, the migration records an indeterminate
value. Such an artifact cannot prove equivalence to current work and becomes stale.
The system never substitutes today's default for unknown historical behavior.

## Supported changes

Contract migrations use restricted operations to add, rename, remove, or map fields.
Removing a field requires a declaration that the value no longer contributes to
scientific identity. Value mappings must cover every historical value encountered.
Arbitrary migration callbacks are not supported.

Migrations create a comparison view. They do not rewrite the original completion
record or public derivative. The original saved fingerprint must continue to match
the original contract bytes before nro accepts that evidence.

## Development requirements

Append the next migration in the affected module's chain when a contract changes.
Do not edit a migration after release. Tests must cover the historical representation,
the new default, nondefault current values, and any indeterminate state. Changes to
SQLite structure follow the separate [registry migration](registry-migrations.md)
process.
