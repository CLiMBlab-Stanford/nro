# Functional-connectivity gradients

`fcgradients` maps where a participant's functional connectivity changes
sharply across cortex or gray matter. High values mark locations where
neighboring seeds have different connectivity maps, which makes them candidate
areal boundaries. The module reads one `dynconn` series per participant, space,
and smoothing level, and supports both surface and volume spaces.

## Method

The `backend` setting chooses how the map is computed. Every backend reads the
same inputs and writes the same outputs. Compare backends by defining separate
configuration IDs.

The only backend is `workbench_correlation_gradient`. It runs
`wb_command -cifti-correlation-gradient`, which follows the approach of
Glasser et al. (2016). Workbench correlates every seed's series with every
other location, takes the gradient magnitude of each seed's correlation map,
and averages those magnitudes over seeds. Each local gradient is a linear fit
to the neighbors: mesh neighbors on a surface and face neighbors in a volume.

Workbench correlates within each CIFTI structure only. On the surface, left
and right cortex are separate structures, so a seed never contributes a map of
the other hemisphere. In the volume, nro puts the whole gray-matter mask in one
structure so that every voxel sees every gray-matter seed.

Because the map averages nonnegative magnitudes, noise that has no common
direction across seeds still raises the value. The averaging also discards each
gradient's direction.

## Inputs

A surface target uses the `dynconn` `.dtseries.nii` directly. nro passes the
participant's midthickness surfaces for `fsnative`, or the template
midthickness surfaces that match each hemisphere's vertex count. The `dynconn`
series covers whole meshes, including the medial wall. Its zero-valued
locations correlate with nothing, so the medial-wall border can show a
gradient that does not reflect cortex.

A volume target starts from the `dynconn` NIfTI series. nro restricts it to
gray matter and converts it to a volume-only dense CIFTI series in the work
directory. The mask comes from `mask` when it is set. Otherwise nro uses the
participant's published anatomical gray-matter mask in native spaces or the
TemplateFlow gray-matter probability map in MNI spaces, as `microparcellation`
does. A mask on another grid is resampled to the series grid, and voxels above
`mask_threshold` are kept.

When `dynconn` publishes a low-rank series, row correlations equal the
low-rank connectome, so the map shows only the detail that rank retains. The
manifest copies the `dynconn` low-rank summary.

## Outputs

Outputs share `derivatives/nro/fcgradients/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`,
and filenames include the space and smoothing entities.

* `_desc-fcGradient_stat.dscalar.nii` holds one map, the mean gradient
  magnitude.
* Volume targets also publish `_desc-fcGradient_stat.nii.gz` on the `dynconn`
  grid. Voxels outside the mask are zero.
* `_desc-fcGradient_manifest.yaml` records the backend, the Workbench version
  and options, the source `dynconn` manifest and representation, the geometry,
  and the CIFTI structures with their location counts.
* `_desc-fcGradientIndex_manifest.json` is the publication index.

`nro scene` and `nro viz` show the surface map as a metric layer under
`fcgradients`. Both viewers display cortical maps only, so they do not show
volume targets.

## Configuration

`markup` and `overwrite` behave as in the other modules. `backend` selects the
method. `mask` and `mask_threshold` apply only to volume targets.

`workbench` holds the options for the Workbench backend:

* `fisher_z` applies the Fisher z transform to correlations before taking
  gradients.
* `double_correlation` correlates the correlation maps again before taking
  gradients, so the map measures change in connectivity-profile similarity,
  as in Gordon et al. (2016).
* `fisher_z_first` applies the Fisher z transform to the first correlation when
  `double_correlation` is on. It has no effect, and does not affect freshness,
  otherwise.
* `presmooth_fwhm_mm` smooths each correlation map before taking gradients.
  Zero disables it.
* `surface_exclude_mm` and `volume_exclude_mm` drop locations within that
  distance of each seed, where local autocorrelation dominates the map. Zero
  disables exclusion.
* `memory_limit_gb` bounds Workbench memory use. It does not change results and
  does not affect freshness. `null` uses 75% of the memory allocated to the
  work item, or of a Slurm job's `--mem` when the module runs directly. With no
  stated allocation, Workbench runs without a limit.

`wb_command` names the Workbench executable. It defaults to the site's
Workbench and does not affect freshness. The manifest records its version.

```{literalinclude} ../../nro/definitions/starters/configs/fcgradients/main_fcgradients.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/fcgradients/module/index.rst).
