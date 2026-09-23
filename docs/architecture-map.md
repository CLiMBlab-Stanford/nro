# Development architecture map

This page identifies the owner of each shared contract. Use it to keep a narrow
change inside one development sphere and to find the boundary tests required by
that change. The detailed scientific graph is described in the
[module guides](modules/index.md); the runtime lifecycle is described in
[work-item planning and execution](work-item-lifecycle.md).

```text
definitions and source BIDS
          |
          v
configuration compiler -----> module planners and static Runner DAGs
          |                               |
          v                               v
compiled work-item contracts -----> branch scientific registry
          |                               |
          +-------------------------------+
                          |
                          v
                 central scheduler
                          |
                          v
               pinned worker execution
                          |
                          v
       public artifacts and ownership receipts
```

## Boundaries

- `nro.configuration` owns definitions loading, validation, inheritance, and
  compiled scientific values. Scientific modules consume compiled values.
- `nro.modules.<name>` owns one module's planning, DAG, algorithms, artifact
  contract, and output metadata. It may use public helpers from `nro.engine`.
- `nro.engine` owns reusable scientific and user-interface primitives. It does
  not own scheduling policy or a module's processing sequence.
- `nro.orchestration` owns module lineages, work-item contracts, demand,
  attempts, workers, publication, and recovery. Its catalog imports the narrow
  planning and artifact-contract interfaces exposed by each module. The
  scheduler does not import a module's DAG constructor or scientific execution
  code. A runner may yield a declared resource-specific step through the
  scheduler, but the step remains owned by the module DAG and parent work item.
- `nro.bin` owns user-facing command parsing and presentation. Commands call the
  configuration, orchestration, or engine interfaces; they do not implement a
  second scheduler or scientific pipeline.
- `nro.bidsify` owns staged source-data ingestion. Published BIDS data is an
  input to planning, not a derivative work item.
- `install` and `nro.engine.shared_installation` own environment setup and
  transactional shared maintenance.

Artifact producers declare their inventory and metadata once. Completion,
repair, downstream resolution, scenes, rendering, and purge consume those
contracts through shared orchestration APIs. A consumer must not infer a
producer's current layout from a private helper or reconstruct filenames when a
manifest supplies the path.

## Scoped validation

Run `nro dev test --dry-run` to inspect the tests selected from the current Git
diff. Run `nro dev test --sphere NAME` for a known sphere, or
`nro dev test --full` before publication. The tracked map is
`development/test-scopes.toml`. An implementation path absent from that map
fails closed to full validation.
