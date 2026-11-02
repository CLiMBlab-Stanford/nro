# Processing modules

These pages describe what each nro processing module does, what it needs, which
settings change its results, and what it publishes. Read them when you choose a
configuration or interpret a module's outputs.

```{toctree}
:maxdepth: 1

anat
func
clean
dynconn
microparcellation
networks
contrasts
```

## How the modules fit together

`anat` runs once per participant and uses every available anatomical
acquisition. `func` runs once per BOLD acquisition and depends on the
participant's anatomy. `clean` runs once per acquisition, space, and smoothing
level.

`dynconn`, `microparcellation`, and `networks` each combine one participant's
cleaned runs for one space and smoothing level. The planner records source
selections that it excludes because they are unusable. Exclusions based on run
quality happen later and are recorded in artifact metadata.

`contrasts` takes a separate path from `func` and does not use `clean`. It fits
first-level models to each run and summarizes the effects within each
participant. Its work items are distinguished by model, space, and smoothing,
and its run, session, and subject results go in separate output directories.

## Defaults and outputs

Each page includes the packaged starter YAML for its module, so the defaults
shown match the installed package. A site's active
[definitions store](../definitions.md) can override those values.

Module outputs go under
`BIDS/PROJECT/derivatives/nro/MODULE/CONFIG_ID-LINEAGE_DIGEST/`. The registry
derives this label from the selected configuration and its upstream lineage.
The label is not the workflow ID. The WORK directory uses the same layout for
private files.

All seven modules are packages under `nro.modules`, so each one can be run
directly with `python -m nro.modules.NAME`. Read
[configuration](../configuration.md) and [the API](../api.md) before you add or
change a module.
