# Processing modules

Each guide describes the default method, configuration-controlled branches,
public artifacts, and dependencies. Default values are included from the
packaged starter YAML files. A lab's active [definitions store](../definitions.md)
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

`anat` runs per participant across available anatomical acquisitions; `func`
runs per BOLD acquisition. `clean` runs per acquisition, space, and smoothing.
`dynconn`, `microparcellation`, and `networks` aggregate per participant,
space, and smoothing. The planner records exclusions of unusable source selections;
downstream run-quality exclusions are recorded in artifact metadata.

`firstlevels` is a separate branch from `func`. It fits task models per run and
summarizes effects within each participant. Its instances distinguish model,
space, and smoothing, with separate run/session/subject output directories.

The ordinary output root is `BIDS/PROJECT/derivatives/CLASS/LINEAGE/`.
`LINEAGE` is the registry-assigned directory label, often `main`, not a promise
that every workflow name gets its own directory. WORK mirrors the derivative
organization for private files. See [configuration](../configuration.md) and
[the API](../api.md) before adding or changing a module.
