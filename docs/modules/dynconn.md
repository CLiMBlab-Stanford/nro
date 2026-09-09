# Dynamic connectivity

`dynconn` packages a participant's admissible cleaned runs for interactive
vertex- or voxel-level dynamic connectivity in Connectome Workbench. It does
not precompute a connectivity matrix. Workbench calculates correlations when
the user selects a location in the saved scene.

## Inputs and admission

One instance selects a participant, space, and smoothing level. `input_filter`
uses the same BIDS-entity matching rules as `microparcellation`. The module
reads the clean sidecars to exclude runs with undefined cleaning, too few
retained frames, too little residual design freedom, low effective rank, or
excessive temporal concentration. It also enforces the configured minimum
usable-run count and aggregate retained-frame count.

The admitted runs must have one repetition time. Censored frames are omitted
from the packaged series. The manifest records each included run's half-open
`start_frame` and `stop_frame` range, so the original run boundaries remain
available after concatenation. It also records excluded runs and their reasons.

## Public artifacts

Outputs live under
`derivatives/dynconn/LINEAGE/space-SPACE_smoothing-Nmm/sub-ID/`.

Surface targets contain one `_desc-dynamicConnectivity_bold.dtseries.nii`
file. Its brain-model axis joins the complete left and right surface meshes;
its series axis records the common repetition time. The directory also contains
pial, midthickness, white, and inflated surfaces for each hemisphere.

Volume targets contain one uncompressed
`_desc-dynamicConnectivity_bold.nii`. The uncompressed 4D NIfTI supports
random access while Workbench calculates voxelwise correlations. Volume scenes
do not copy anatomical images or surfaces.

Both forms include a relocatable `_desc-dynamicConnectivity_scene.scene`, a
YAML publication manifest, and a JSON publication index. Open completed scenes
with:

```bash
nro wb_view dynconn -p PARTICIPANT -P PROJECT -s SPACE -S MM
```

## Configuration

`surface` chooses the source geometry used to identify the surface family;
the scene always packages all four display surfaces and opens with midthickness.
`output_dir`, `prefix`, and `overwrite` control direct execution and publication.
The `inclusion` mapping contains the sidecar-based run and aggregate thresholds.

```{literalinclude} ../../nro/configuration/starters/configs/dynconn/main_dynconn.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/dynconn/module/index.rst).
