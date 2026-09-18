# Orchestration agent guidance

This package owns scientific registration, artifact selection, demand, scheduling,
worker coordination, and publication. It consumes compiled `WorkItemSpec` contracts.
It must not import module DAG builders or interpret neuroimaging formats.

Preserve these boundaries:

* `WorkflowRegistry` owns module-lineage identity and runtime snapshots.
* `BranchRegistry` owns branch scientific contracts and observations.
* `Registry` and the coordinator own shared demand and execution state.
* `Runner` alone owns module-internal traversal and step freshness.
* Public ownership records are durable recovery inputs. SQLite registries and caches
  are reconstructable.
* Private completion evidence belongs in the scheduler database. Do not create a
  parallel completion certificate or manifest tree.

External control files are limited to state that must cross a filesystem or process
boundary: public ownership receipts, immutable execution inputs, logs and runner
ledgers, crash-safe filesystem transaction journals, service election records, and
the durable scheduler request log. A convenience cache or transport file must not
become a second authority for data already owned by SQLite.

Focused registry operation modules receive an existing SQLite connection. They do
not connect, commit, roll back, or acquire locks. `Registry` methods remain the named
transaction boundaries and the only public authority that owns database access.

Run `nro dev test --sphere orchestration` for focused validation. Changes to schemas,
artifact identity, publication, or shared execution also require the full suite and
the upgrade rehearsal before release.

Persisted scheduler or branch-scientific changes must use the restricted chain under
`nro.orchestration.migrations`. Do not edit a baseline or define a current schema in
parallel. Add migration fixtures, semantic invariant tests, and fresh-versus-migrated
schema comparison, then run `nro dev schema check`.
