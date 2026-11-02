# Microparcellation and connectivity

`microparcellation` divides a participant's cortex or brain volume into many
small, functionally coherent parcels and estimates the parcel-by-parcel
connectivity matrix. It pools the participant's usable cleaned runs in one
space and smoothing level. Use it when you want a compact connectome, or as
the input to OSLOM network estimation.

Surface and volume spaces are parcellated independently. Parcels are not
projected from one space to another.

## Run admission

The module reads quality metadata from each `clean` sidecar and excludes
undefined or inadmissible runs before it streams any signal. The thresholds
cover retained frames and retained fraction, residual design degrees of
freedom, participation effective rank, dominant temporal variance, the number
of usable runs, and the total retained frames. Rejected runs are recorded with
their reasons. Missing required metadata is a contract error. The module does
not recompute the metadata from the data.

When it loads a run, the module still checks shape, pairing, and finite
values. Censored frames are dropped. The module does not reconstruct them or
repeat the `clean` quality analysis.

## Spatial graph

Surface neighbors come from triangle edges on the configured surface geometry
and mask. Volume neighbors use the selected mask and 6-, 18-, or
26-connectivity. Template surfaces and masks come from configured local
resources. Direct `anat` dependencies supply native geometry and gray-matter
support.

## Estimation

The estimator streams the data and does not keep a connectome for each run.

1. **Prepare runs.** Each admitted run is standardized over its retained
   frames. Global signal regression can be applied again for connectivity. The
   default `clean` configuration already removes global signal, and this
   connectivity switch is set separately.
2. **Estimate local similarity.** nro computes correlations between spatial
   neighbors. Runs are centered and standardized independently, then pooled
   with equal or effective-DOF weights.
3. **Coarsen.** Local similarities are transformed with the configured
   exponential temperature. Loukas edge-based local-variation coarsening then
   contracts spatial neighbors over several levels while approximating a
   low-frequency graph-Laplacian eigenspace, followed by the configured number
   of refinement passes. `target_vertices` is the target size of the coarse
   graph. Disconnected support and the contractions allowed can keep the result
   from reaching it.
4. **Accumulate connectivity.** nro streams parcel-mean signals and
   accumulates their correlation matrix. In the same pass it accumulates how
   much original variance the parcels retain, along with quality summaries.
5. **Build null partitions.** Spatial region growing generates five null
   partitions by default. Seeds and growth targets approximate the observed
   parcel-size distribution, though topology can prevent an exact match. nro
   records variance-retention and size-match statistics but does not publish
   the null label arrays.
6. **Summarize.** nro summarizes connectome weights, node strength, spectral
   concentration, run contributions, and split-half reproducibility. Splits
   use complete runs. Only a single-run input falls back to temporal blocks.
   Half matrices are working accumulators and are not published. Treat these
   summaries as diagnostics, not calibrated uncertainty intervals.

See [estimation equations](../methods/estimation.md) for run weighting,
variance-retention denominators, quantization, and split allocation.

## Outputs

Outputs for every space and smoothing level share
`derivatives/nro/microparcellation/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`.
`PREFIX` includes participant, space, and smoothing, which keeps targets
separate within that directory. The scientific outputs are:

| Suffix after `PREFIX` | Meaning |
| --- | --- |
| `_desc-microparcellation_dseg.dlabel.nii` | Parcel labels in CIFTI brain-model coordinates. |
| `_connectivity.pconn.nii` | Parcel-by-parcel correlation matrix and parcel axis. |
| `_desc-microparcellationQuality_metrics.json` | Variance retention, null statistics, and connectome diagnostics. |
| `_desc-microparcellation_manifest.yaml` | Method, sources, admission, outputs, and completion. |
| `_desc-microparcellationIndex_manifest.json` | Paths used by downstream discovery and viewing. |
| `_desc-microparcellation_dseg.nii.gz` | Additional volume labels for volume targets. |

Use `nro scene -m microparcellation` to create a Workbench view. The linked
form reuses the source geometry and outputs without copying them. Use
`nro render -m microparcellation --seeds FILE` to create static parcel
connectivity maps from T1w coordinates. See the
[viewing guide](../commands/viewing.md#nro-render).

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.
nro sets the output root and filename prefix from the project, module ID,
participant, space, and smoothing request. `overwrite` requests rebuilding the
outputs.

**Inputs and spatial support.** `input_filter` restricts the cleaned inputs.
`surface`, `mask`, `mask_threshold`, and `volume_connectivity` define spatial
support.

**Coarsening.** `coarsening.target_vertices` sets the target graph size and
`iterations` the refinement passes. `exponential_temperature` scales
similarities. `eigenvectors`, `max_levels`, and `eigensolver_tolerance` govern
the spectral approximation.

**Connectivity.** The `connectivity` thresholds are admission rules.
`temporal_block_size` bounds streaming chunks. `weighting` selects `precision`
or `equal` run weights. `global_signal_regression` changes the estimator and
reduces each run's connectivity degrees of freedom by one.

**Quality.** `quality.null_parcellations` sets the number of null partitions.
`random_seed` makes null generation repeatable, and `region_growing_attempts`
bounds retries during null construction. `connectome_power_iterations`
controls the spectral diagnostic approximation. `split_half_block_frames` sets
the single-run split block size in retained frames. It defaults to 128 and is
capped at half the run length. This setting is scientific and independent of
the execution-only streaming chunk sizes.

```{literalinclude} ../../nro/definitions/starters/configs/microparcellation/main_microparcellation.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/microparcellation/module/index.rst),
[statistics](../autoapi/nro/modules/microparcellation/statistics/index.rst),
[coarsening](../autoapi/nro/modules/microparcellation/coarsen/index.rst),
[quality](../autoapi/nro/modules/microparcellation/quality/index.rst).
