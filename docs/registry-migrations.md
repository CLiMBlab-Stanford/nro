# Registry migrations

This page is for developers who change the structure or meaning of nro's private
SQLite registries: the central scheduler database and each branch's scientific
database. It explains how those schemas are versioned, how upgrades apply, and
what a new migration must include.

## How registry schemas evolve

nro derives each registry schema from an immutable baseline followed by an
ordered chain of migrations. The first supported baselines are scheduler schema
21 and branch-scientific schema 6. Older private registries are not migrated.
nro reconstructs them from durable ownership records and public artifacts.

Scheduler schema 21 stores completion evidence directly in SQLite and has no
private completion-manifest tree. Scientific schema 7 adds compiled-plan and
source-fingerprint caches, which can be reconstructed. Upgrading from scientific
schema 6 keeps the scientific records and starts with empty caches. No
derivative contract or freshness meaning changes.

Runtime registry access does not replay the chain. When a migration is
required, installation applies it once, validates the result, and activates it
atomically, keeping a copy of the previous database. Scientific artifacts are
not registry rows, so a registry schema change never makes them stale.

Migration snapshots use SQLite's backup interface instead of copying the
database file. The backup includes committed data that may still sit in a
write-ahead log, even though nro's shared registries normally use rollback
journals.

## Changing persisted state

Any change to stored registry structure or meaning needs one migration to the
next schema version. Add one file named for its destination version, such as
`nro/orchestration/migrations/scheduler/v0022.py` or
`nro/orchestration/migrations/scientific/v0007.py`. The file defines one `MIGRATION`
with a concise summary and a list of restricted operations. Do not edit the
baseline SQL, its manifest, or any migration that has already reached the target
branch.

The available operations are checked column additions and renames, table
renames, index creation and removal, exhaustive value mappings, and explicit
removal of reconstructible columns. If a transformation cannot be expressed with
these, extend the migration language. Do not fall back to arbitrary SQL or
Python that bypasses review. When a transformation cannot assign an unambiguous
valid value, it must reject the old state or leave it for reconstruction.

Migration tests must cover populated and empty state, boundary values, failure before
activation, and equivalence between migrated and freshly created schemas. Add semantic
invariant checks for meanings that SQLite constraints cannot express.

## Inspecting schemas

Generated schemas are build products and are not committed. Use `nro dev schema`
to print, write, or compare any supported version, and to validate the chains:

```bash
nro dev schema show --family scheduler
nro dev schema show --family scientific --version 6
nro dev schema build --family scheduler --output .nro-cache/scheduler.sql
nro dev schema diff --family scheduler --from-version 21 --to-version 22
nro dev schema check
```

For pull requests into `main`, the check also passes the base commit to
`schema check`. The check rejects a rewritten baseline, an incomplete migration
chain, or a generated schema that fails structural, foreign-key, or SQLite
integrity checks. Moving the earliest supported baseline forward is a separate
release-policy decision.
