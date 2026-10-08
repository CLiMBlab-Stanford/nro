# Shared engine agent guidance

This package contains reusable scientific and data-processing primitives. It does not
own a scientific module's DAG and must not read scheduler tables. Keep helpers typed,
deterministic, and independent of active worker state. Installation, site maintenance,
and definition-store operations belong in `nro.site` or `nro.definitions`.

Place a helper here when its contract is useful across modules or defines a shared
format such as manifests, CIFTI indexes, paths, or atomic I/O. Keep module-specific
algorithms in `nro.modules`.

Run `nro dev test --sphere shared_engine` for focused validation. A change to a public
engine contract requires its boundary tests and every affected module sphere.
