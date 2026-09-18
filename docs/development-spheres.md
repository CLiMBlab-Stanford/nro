# Development spheres

Development spheres keep a narrow change from requiring unrelated repository context
and tests during every edit. They are validation scopes, not runtime namespaces and
do not affect artifact identity.

The tracked map in `development/test-scopes.toml` assigns source paths, focused tests,
boundary tests, and downstream consumers to each sphere. Run:

```console
nro dev test
nro dev test --sphere networks
nro dev test --full
```

The default command reads the Git working tree and selects tests for every changed
path. It includes downstream spheres where a shared contract can affect consumers.
An unmapped path selects the full suite. `--dry-run` prints the selection without
running it.

The spheres separate scientific modules, the shared scientific engine,
orchestration, configuration, commands, installation, BIDSification, documentation,
and development tooling. Directory placement is only part of the boundary. The
important rules are:

* scientific modules build static DAGs and publish declared artifacts;
* orchestration consumes serialized work-item and artifact contracts without loading
  module implementations;
* consumers use public manifests and shared readers instead of producer internals;
* user definitions compile before planning, while runtime settings stay outside
  scientific freshness; and
* installation uses the public maintenance boundary instead of scheduler internals.

Static boundary tests also keep scientific modules out of scheduler state, prevent
cross-module implementation imports, restrict orchestration to planner-facing module
interfaces, and keep the planning catalog independent of array and image libraries.

Focused validation speeds iteration. The full suite and upgrade rehearsal remain
release requirements.
