# Python API and extension

This page is for developers who write or change scientific modules or call nro
from Python. It shows how to declare module work with the runner and points to
the main planning, execution, and ingestion classes.

The [generated API reference](autoapi/nro/index.rst) lists public classes,
functions, signatures, and source docstrings. It is built by static analysis, so
building the documentation does not import nro, open a registry, or need
neuroimaging containers. The reference also lists implementation helpers so you
can find them. Only documented public interfaces may receive compatibility
support, and a private helper is not part of the supported API just because it
appears there.

## Declaring module steps

A module declares its work as immutable `Step` objects and registers them with
one shared `Runner`. A step factory declares inputs, outputs, and an action
without running anything. The runner infers dependencies between producers and
consumers from their paths, and it also accepts explicit `after` edges. Register
every step before you call `execute()`.

```python
from pathlib import Path
from itertools import count
import logging
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from nro.engine.io import atomic_write_text

output = Path("result.txt")
runner = Runner(
    module_name="Example",
    container=None,
    binds=(),
    logger=logging.getLogger("example"),
    next_step=count(1).__next__,
)
runner.add_step(
    Step.python(
        name="Write result",
        outputs=(output,),
        action=lambda: atomic_write_text(output, "complete\n"),
    )
)
with runner.run_context():
    runner.execute()
```

`execute()` must run inside `run_context()`, which scopes execution state and
the final report. The [Runner signature](autoapi/nro/orchestration/runner/index.rst)
documents the container, logging, and graph-path arguments. Do not create a
separate graph and change it on its own.

Besides `Step.python`, two factories cover other kinds of work.
`Step.command_step` declares an external command, with optional preparation and
finalization. `Step.directory_step` declares a directory written by a tool,
together with a validator and a completion breadcrumb. A directory that merely
exists is not complete. File outputs must exist and be nonempty.

An action may close over resolved module inputs, but it must not add steps that
depend on runtime results. When a helper builds work, it should return the steps
themselves. A helper that changes the graph and returns only a description of
its outputs hides that change from the caller.

## Planning and execution records

The planner and workers exchange a small set of typed records.
[Work-item planning and execution](work-item-lifecycle.md) describes each one in
detail.

- `WorkItemIdentity` identifies one work item.
- `WorkItemContract` holds the substantive configuration, dependencies, inputs,
  processing details, and `OutputContract`.
- `ExecutionRecipe` holds the command and runtime configuration.
- `ResourceRequest` holds the worker resources.
- `WorkItemSpec` combines the four records above for registration.
- `ExecutionEnvelope` is the immutable record a worker receives when it claims
  work.

`Planner` builds these records from module descriptors and subject planning
contexts. `Registry` runs inside the controller and handles transactions,
demand, attempts, leases, and scheduler submissions. User commands send direct
scheduler requests that are backed by durable registry rows.
`WorkerSchedulerClient` gives workers claims and lifecycle operations without a
database handle, and `Worker` supervises an `ExecutionLauncher`. Scientific
modules must not open the registry.

The generated reference covers
[contracts](autoapi/nro/orchestration/contracts/index.rst),
[planner](autoapi/nro/orchestration/planner/index.rst),
[registry](autoapi/nro/orchestration/registry/index.rst), and
[execution](autoapi/nro/orchestration/execution/index.rst).

## Adding a module

A new module needs configuration validation, a deterministic constructor,
work-item construction for the planner, public output contracts, and validation
of completion metadata. Register it in the explicit module catalog. Use the
shared engine helpers for paths, BIDS metadata, atomic I/O, container execution,
and admission of cleaned data.

Add tests for topology, selectors, public output signatures, freshness, and
resumption after interruption.

A method change that alters what an output means must update the substantive
contract. A docstring change alone should not change the declared processing
signature. [Work-item planning and execution](work-item-lifecycle.md) explains
how output contracts, completion generations, and ownership receipts interact.

## Ingestion

`nro.bidsify` handles pre-BIDS work separately from the scientific modules. Do
not represent ingestion as a synthetic derivative work item.

`IngestionStore` saves interactive review state under its own short-lived lock.
`FlywheelSource` keeps read-only cloud access in one place. The `nro.bin.bidsify`
wizard makes every user decision, and `run_stage` then executes one fixed,
noninteractive stage. The scheduler controller claims stages and counts their
worker capacity. Workers supervise ingestion with the same execution launcher
and include it in shared concurrency accounting. See
[bidsification](commands/bidsify.md) for the privacy boundary and the
publication protocol.
