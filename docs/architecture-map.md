# Architecture and validation scopes

This page is for developers changing nro. It shows which package owns each
shared contract and how the repository selects tests for a change, so you can
keep an edit inside one area and run the checks it needs. The
[module guides](modules/index.md) describe the scientific graph, and
[work-item planning and execution](work-item-lifecycle.md) describes the
runtime lifecycle.

## How the parts connect

Definitions and source BIDS data feed the configuration compiler. Its compiled
values reach both the module planners, which build static Runner DAGs, and the
compiled work-item contracts. Both are recorded in the branch's scientific
registry. The central scheduler reads those records and runs work on pinned
workers, which publish public artifacts and ownership receipts.

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

## Package boundaries

Each package below owns one part of that flow.

`nro.definitions` owns every declarative record managed by `nro def`, including
module configurations. It loads, validates, and inherits those records and
produces compiled scientific values. Scientific modules consume the compiled
values. User definitions compile before planning. Runtime settings stay outside
scientific freshness.

`nro.modules.<name>` owns one module's planning, DAG, algorithms, artifact
contract, and output metadata. A module may use public helpers from
`nro.engine`. Modules build static DAGs and publish the artifacts they declare.

`nro.engine` owns reusable scientific and user-interface primitives. It does not
own scheduling policy or any module's processing sequence.

`nro.orchestration` owns module lineages, work-item contracts, demand, attempts,
workers, publication, and recovery. Its catalog imports only the narrow
planning and artifact-contract interfaces that each module exposes. The
scheduler works from serialized work-item and artifact contracts. It does not
import a module's DAG constructor or scientific execution code. A runner may
yield a declared resource-specific step through the scheduler, but the module
DAG and parent work item still own that step.

`nro.bin` owns command parsing and presentation. Commands call the
configuration, orchestration, or engine interfaces. They do not implement a
second scheduler or scientific pipeline.

`nro.bidsify` owns staged source-data ingestion. Published BIDS data is an input
to planning, not a derivative work item.

`nro.site` owns protected site settings, environment setup, and transactional
shared maintenance. These settings are operational inputs, not user-authored
definitions. Installation goes through this public maintenance boundary and does
not use scheduler internals.

## Artifact contracts

An artifact producer declares its inventory and metadata once. Completion,
repair, downstream resolution, scenes, rendering, and purge read those
declarations through shared orchestration APIs. A consumer must use public
manifests and shared readers. It must not infer a producer's current layout
from a private helper or rebuild a filename when a manifest supplies the path.

Static boundary tests enforce these rules. They keep scientific modules out of
scheduler state, prevent one module from importing another module's
implementation, restrict orchestration to the planner-facing module interfaces,
and keep the planning catalog independent of array and image libraries.

## Scoped validation

A development sphere is a validation scope: a named group of source paths and
the tests that cover them. Spheres let you test a narrow change without running
unrelated suites on every edit. They are not runtime namespaces and do not
affect artifact identity.

The tracked map in `development/test-scopes.toml` assigns source paths, focused
tests, boundary tests, and downstream consumers to each sphere. The spheres
cover each scientific module, the shared scientific engine, orchestration,
definitions, commands, installation, BIDSification, documentation, and
development tooling.

```console
nro dev test
nro dev test --dry-run
nro dev test --sphere networks
nro dev test --full
```

With no options, `nro dev test` reads the Git working tree and selects tests for
every changed path. When a changed path belongs to a shared contract, the
selection also includes the downstream spheres that consume it. A path missing
from the map selects the full suite. `--sphere NAME` runs only that sphere and
its boundary tests, and may be repeated. `--dry-run` prints the selection
without running it. Use `--full` before publication.

Infrastructure spheres use synthetic contracts and small boundary cases instead
of the scientific module suites, which keeps iteration fast. Focused validation
does not replace the release requirements: the full suite and the upgrade
rehearsal must still pass.
