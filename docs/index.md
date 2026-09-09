# nro documentation

nro turns anatomical and functional MRI in BIDS datasets into preprocessed
images, cleaned time courses, small brain parcels, individualized networks,
and within-participant task-effect maps.
Users request results; nro finds their dependencies, shares work between
requests, and runs ready work through a reusable cluster worker pool.

Outside users should start with the [new-installation quickstart](quickstart.md).
CLIMBLAB members should use the
[internal installation quickstart](climblab-quickstart.md). The
[design guide](design.md) explains how work and outputs are organized. The
module guides describe processing, branch selection, output files, and
configuration. The command reference covers the user interface; the API guide
covers Python extension.

```{toctree}
:maxdepth: 2
:caption: Using nro

quickstart
climblab-quickstart
installation
commands/index
configuration
definitions
task-models
event-files
```

```{toctree}
:maxdepth: 2
:caption: Processing and system model

design
concepts
orchestration
modules/index
methods/denoising
methods/estimation
methods/firstlevels
methods/software
```

```{toctree}
:maxdepth: 2
:caption: Development

instance-lifecycle
orchestration-design
api
development
autoapi/nro/index
```
