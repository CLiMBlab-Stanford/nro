# nro documentation

nro turns anatomical and functional MRI in BIDS datasets into preprocessed
images, cleaned time courses, dynamic-connectivity series, small brain parcels,
individualized functional networks, and within-participant task-effect maps.
You request results. nro finds their dependencies, shares work between
requests, and runs ready work through a cluster worker pool.

Where to start depends on what you need:

- To set up nro at a new site or on your own machine, follow the
  [new-installation quickstart](quickstart.md). The
  [installation guide](installation.md) covers shared deployments and
  maintenance.
- CLIMBLAB members connecting to the lab installation should follow the
  [CLIMBLAB quickstart](climblab-quickstart.md).
- To understand how requests become work and outputs, read
  [How nro works](design.md) and then the [concepts](concepts.md) page. The
  [command reference](commands/index.md) lists every command.
- To choose processing settings or interpret outputs, use the
  [module guides](modules/index.md) and the methods pages.
- To change nro itself, start with the [architecture map](architecture-map.md)
  and the [development guide](development.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

quickstart
climblab-quickstart
installation
```

```{toctree}
:maxdepth: 2
:caption: Using nro

design
concepts
orchestration
commands/index
configuration
definitions
task-models
event-files
```

```{toctree}
:maxdepth: 2
:caption: Processing reference

modules/index
methods/denoising
methods/estimation
methods/contrasts
methods/software
methods/reporting
```

```{toctree}
:maxdepth: 2
:caption: Development

architecture-map
orchestration-design
work-item-lifecycle
api
development
branching-tutorial
registry-migrations
artifact-contract-migrations
portable-provenance
autoapi/nro/index
```
