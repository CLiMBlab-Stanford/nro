# Microparcellation and connectivity

`microparcellation` aggregates a participant's admissible cleaned runs in one
space and smoothing level. Surface and volume spaces are parcellated
independently; tiny parcels are not projected between spaces.

## Admission and geometry

The module reads required quality metadata from clean sidecars and excludes
undefined or inadmissible runs before streaming signals. Thresholds cover
retained frames/fraction, residual design degrees of freedom, participation
effective rank, dominant temporal variance, usable-run count, and aggregate
retained frames. It records rejected runs and reasons. Missing required metadata
is a contract problem, not an invitation to recompute it from data.

On load, inexpensive shape, pairing, and finite-value checks remain. Censored
frames already exist in the input and are dropped; microparcellation does not
reconstruct them or redo the clean quality analysis.

Surface neighborhoods come from triangle edges and the configured surface
geometry/mask. Volume neighborhoods use the selected mask and 6-, 18-, or
26-connectivity. Template surfaces/masks resolve from configured local resources.
Direct anatomy dependencies supply native geometry and gray-matter support.

## Streaming estimation

1. Standardize admitted runs over retained frames. Optional additional global
   signal regression is applied for connectivity. Default clean already
   includes global signal; the connectivity switch is separately configurable.
2. Estimate local neighbor correlations. Runs are centered and standardized
   independently, then pooled with equal or effective-DOF weights.
3. Transform local similarities with the configured exponential temperature.
   Apply Loukas edge-based local-variation coarsening, contracting spatial
   neighbors through multiple levels while approximating a low-frequency
   graph-Laplacian eigenspace. Iterative refinement uses the configured number
   of passes. `target_vertices` is a target coarse graph size; disconnected
   support and allowable contractions constrain what can be reached.
4. Stream parcel-mean signals and accumulate their correlation matrix. Accumulate
   original-versus-parcel variance retention and quality summaries in the same
   pass. The estimator does not retain a connectome for every run.
5. Generate five spatial region-growing null partitions by default. Seeds and
   growth targets approximate the observed parcel-size distribution; topology
   can prevent exact matching. Record variance-retention and size-match
   statistics, not the null label arrays as public outputs.
6. Summarize connectome weights, node strength, spectral concentration, run
   contributions, and split-half reproducibility. Splits use complete runs;
   only a single-run input falls back to temporal blocks. Half matrices are
   working accumulators, not per-run public connectomes. Interpret these
   summaries as diagnostics, not calibrated uncertainty intervals.

See [estimation equations](../methods/estimation.md) for run weighting,
variance-retention denominators, quantization, and split allocation.

## Public artifacts

Outputs from every space and smoothing target share
`derivatives/microparcellation/LINEAGE/sub-ID/`. `PREFIX` includes subject,
space, and smoothing, which keeps targets distinct within that directory. The
fixed scientific outputs are:

| Suffix after `PREFIX` | Meaning |
| --- | --- |
| `_desc-microparcellation_dseg.dlabel.nii` | Parcel labels in CIFTI brain-model coordinates. |
| `_connectivity.pconn.nii` | Parcel-by-parcel correlation matrix and parcel axis. |
| `_desc-microparcellationQuality_metrics.json` | Variance retention, null statistics, and connectome diagnostics. |
| `_desc-microparcellation_manifest.yaml` | Method, sources, admission, outputs, and completion. |
| `_desc-microparcellationIndex_manifest.json` | Paths used by downstream discovery/viewing. |
| `_desc-microparcellation_dseg.nii.gz` | Additional volume labels for volume targets. |

Use `nro scene -m microparcellation` to create a Workbench view. The linked
form reuses the source geometry and scientific outputs without copying them.

## Configuration

`input_filter` restricts selected cleaned inputs; `surface`, `mask`,
`mask_threshold`, and `volume_connectivity` define spatial support. `output_dir`
and `prefix` override direct invocation destinations; orchestrated execution
constructs owned paths. `overwrite` requests rebuilding module outputs.

`coarsening.iterations` controls refinement; `exponential_temperature` scales
similarities; `eigenvectors`, `max_levels`, and `eigensolver_tolerance` govern
the spectral approximation. The connectivity thresholds are admission rules;
`temporal_block_size` bounds streaming chunks. `weighting` selects `precision`
or `equal` run weights. `global_signal_regression` changes the estimator and
reduces each run's connectivity DOF by one.
`quality.random_seed` makes null generation repeatable;
`region_growing_attempts` bounds null construction retries;
`connectome_power_iterations` controls the spectral diagnostic approximation.
`quality.split_half_block_frames` sets the single-run split blocks in retained
frames (default 128, capped at half the run length). This scientific setting
is independent of the execution-only streaming chunk sizes.

```{literalinclude} ../../nro/configuration/starters/configs/microparcellation/main_microparcellation.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/microparcellation/module/index.rst),
[statistics](../autoapi/nro/modules/microparcellation/statistics/index.rst),
[coarsening](../autoapi/nro/modules/microparcellation/coarsen/index.rst),
[quality](../autoapi/nro/modules/microparcellation/quality/index.rst).
