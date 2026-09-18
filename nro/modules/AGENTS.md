# Scientific module agent guidance

Each child package owns one module's configuration use, static DAG construction,
scientific methods, output contract, and focused tests. Step factories return `Step`
objects and do not mutate a `Runner`. A larger construction stage may return an
immutable `StagePlan`: an ordered tuple of steps plus typed downstream products.
Only the module entry point adds those declarations to its runner, and it builds the
complete DAG before freshness decisions or execution begin.

Use shared readers, writers, image helpers, and contract utilities from `nro.engine`.
Do not read scheduler tables, branch topology, or worker state from a scientific
module. Dependencies arrive through compiled paths and `ExecutionContext` bindings.

Run `nro dev test --sphere MODULE` for focused validation. Update the module guide
and methods pages when observable processing or artifact metadata changes.
