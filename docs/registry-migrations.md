# Registry migrations

nro derives each private SQLite schema from an immutable baseline and an ordered
chain of migrations. Scheduler schema 21 and branch-scientific schema 6 are the
first supported baselines. Scheduler schema 21 stores completion evidence directly
in SQLite; it has no private completion-manifest tree. Older private registries are
reconstructed from durable ownership records and public artifacts.

Runtime registry access does not replay the chain. Installation applies a required
migration once, validates the replacement, and atomically activates it. It retains a
copy of the preceding database. Scientific artifacts are not registry rows and are
never made stale merely because a registry schema changed.

Migration snapshots use SQLite's backup interface rather than copying the database
file. This preserves committed data that may still reside in a write-ahead log, even
though nro's shared registries normally use rollback journals.

## Changing persisted state

Any change to stored registry structure or meaning requires one migration to the
immediately following schema version. Add one file named for its destination, such as
`nro/orchestration/migrations/scheduler/v0022.py` or
`nro/orchestration/migrations/scientific/v0007.py`. The file defines one `MIGRATION`
with a concise summary and restricted operations. Do not edit the baseline SQL, its
manifest, or a migration that has already reached the target branch.

The current operation set supports checked column additions and renames, table
renames, index creation and removal, exhaustive value mappings, and explicit removal
of reconstructible columns. Extend the migration language when a new transformation
cannot be expressed. Do not use arbitrary SQL or Python as an unreviewed escape
hatch. A transformation that cannot assign an unambiguous valid value must reject the
old state or leave it for reconstruction.

Migration tests must cover populated and empty state, boundary values, failure before
activation, and equivalence between migrated and freshly created schemas. Add semantic
invariant checks for meanings that SQLite constraints cannot express.

## Inspecting schemas

Generated schemas are build products and are not committed. Developers can inspect or
write any supported state:

```bash
nro dev schema show --family scheduler
nro dev schema show --family scientific --version 6
nro dev schema build --family scheduler --output .nro-cache/scheduler.sql
nro dev schema diff --family scheduler --from-version 21 --to-version 22
nro dev schema check
```

The main-branch PR check also passes its base commit to `schema check`. This rejects a
rewritten baseline, an incomplete migration chain, or a generated schema that fails
structural, foreign-key, or SQLite integrity checks. Advancing the earliest supported
baseline is a separate release-policy operation.
