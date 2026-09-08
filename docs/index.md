# nro documentation

nro turns anatomical and functional MRI in BIDS datasets into preprocessed
images, cleaned time courses, small brain parcels, individualized networks,
and within-participant task-effect maps.
Users request results; nro finds their dependencies, shares work between
requests, and runs ready work through a reusable cluster worker pool.

Start with the [quickstart](quickstart.md). The [design guide](design.md)
explains how work and outputs are organized. The module guides describe
processing, branch selection, output files, and configuration. The command
reference covers the user interface; the API guide covers Python extension.

```{toctree}
:maxdepth: 2
:caption: Using nro

quickstart
installation
commands/index
configuration
definitions
task-models
event-files
```

```{toctree}
:maxdepth: 2
:caption: Design and methods

design
concepts
instance-lifecycle
modules/index
methods/denoising
methods/estimation
methods/firstlevels
methods/software
orchestration
orchestration-design
```

```{toctree}
:maxdepth: 2
:caption: Development

api
development
autoapi/nro/index
```
