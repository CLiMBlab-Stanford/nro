# Processing modules

Each guide describes the default method, configuration-controlled paths,
public artifacts, and dependencies. Default values are included from the
packaged starter YAML files. A site's active [definitions store](../definitions.md)
may override those values.

All seven scientific packages live under `nro.modules`. Their direct module
entry points therefore use `python -m nro.modules.NAME`.

```{toctree}
:maxdepth: 1

anat
func
clean
dynconn
microparcellation
networks
firstlevels
```

`anat` runs per participant across available anatomical acquisitions. `func`
runs per BOLD acquisition, and `clean` runs per acquisition, space, and smoothing.
`dynconn`, `microparcellation`, and `networks` aggregate per participant,
space, and smoothing. The planner records exclusions of unusable source
selections. Artifact metadata records downstream run-quality exclusions.

`firstlevels` follows its own dependency path from `func`. It fits task models
per run and summarizes effects within each participant. Its instances
distinguish model, space, and smoothing, with separate run/session/subject
output directories.

The ordinary output root is `BIDS/PROJECT/derivatives/nro/MODULE/MODULE_ID/`.
`MODULE_ID` is the registry-assigned lineage label, usually the selected
configuration ID. It is not the workflow ID. WORK mirrors the derivative
organization for private files. See [configuration](../configuration.md) and
[the API](../api.md) before adding or changing a module.
