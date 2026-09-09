# Running derivative workflows

The orchestration layer turns user selections into scientific work, reuses
compatible artifacts, and runs ready instances through one shared worker pool.
It supports `anat`, `func`, `clean`, `dynconn`, `microparcellation`, `networks`,
and `firstlevels`.

The lab default keeps scheduler state and branch scientific records beneath one
private control root:

```text
/juice6/u/nlp/climblab/.nro/
  shared/scheduler/registry.sqlite3
  branches/<branch-id>/registry.sqlite3
```

The scheduler database is the sole authority for requests, attempts, workers,
Slurm submissions, and the concurrency limit. A branch database records that
branch's compiled scientific contracts, workflow lineages, and artifact
observations. All projects and branches therefore share worker capacity without
requiring the scheduler to import development code.

## Requests

The common selector interface narrows projects, participants, modules,
workflows, runs, spaces, smoothing levels, tasks, and models. Omitted selectors
usually mean all applicable values. For example:

```bash
nro run -p t20 -P nptl -m networks -w main -s fsnative -S 2
```

The planner compiles the requested terminal instances and their dependency
closure. Selecting an endpoint together with an upstream dependency does not
duplicate demand. A bare invocation selects every endpoint of the default
workflow, currently `dynconn`, `networks`, and `firstlevels`; firstlevels uses
model set `main` unless the request selects another model or set.

From `clean` onward, space and smoothing are independent instance entities.
Multiple values request their cross-product. The planner does not register
unused combinations in advance. Exact run selectors use
`--run ENTITY=VALUE[,VALUE...]`; alternatives for one entity are OR choices,
while different entities are combined with AND.

See [work commands](commands/work.md) for every selector, execution resource,
status mode, cancellation rule, and repair boundary.

## Planning and execution

The planner registers complete instance specifications before submitting work.
Workers claim ready instances transactionally and run each one in a separate
subprocess. A module constructs its complete runner graph before freshness is
checked. The shared runner then executes or skips each declared step and writes
the completion record.

Development requests send their compiled graph and execution recipe to the
approved central scheduler. The scheduler resolves compatible ancestor
artifacts and branch-owned output paths. The attempt runs from captured source
and site settings chosen at submission; a worker can therefore serve different
branches in succession without retaining a branch's imports. See
[development](development.md#requests-and-execution) for inheritance and
cross-branch invalidation.

## Configuration and freshness

Workflows select one configuration ID per derivative class from the external
[definitions store](definitions.md). Resolution produces a complete scientific
configuration and a separate execution snapshot. Equivalent compiled
definitions share work even when their source YAML differs in formatting.

Instance contracts record substantive configuration, direct inputs, dependency
topology and generations, processing policy, and required outputs. Command
formatting, scheduler resources, Git identity, and source-capture paths are not
freshness inputs. The filesystem remains authoritative for whether outputs
exist and satisfy their contracts.

`nro status --cached` reports saved state. The default status previews
inexpensive reassessment without storing it. `nro status --verify` performs the
authoritative assessment and updates the registry; it may cancel an active
attempt whose contract is obsolete. See [instance planning and
execution](instance-lifecycle.md) for the complete state model.

## Direct module execution

The installed `nro` interface is the normal entry point. Direct scientific
entry points remain available for diagnosis:

```bash
python -m nro.modules.anat -p t20 -P nptl -w main
python -m nro.modules.func -p t20 -P nptl -w main \
  --run task=Rest run=01
python -m nro.modules.clean -p t20 -P nptl -w main \
  --space fsnative --smoothing 2 --run task=Rest run=01
python -m nro.modules.dynconn -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.microparcellation -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.networks -p t20 -P nptl -w main \
  --space fsnative --smoothing 2
python -m nro.modules.firstlevels -p t20 -P nptl -w main \
  --task langlocSN --model main --space fsnative --smoothing 2
```

These commands use the same configuration compiler, execution context, runner,
and artifact contracts as scheduled work. They do not define alternate
scientific or freshness behavior.
