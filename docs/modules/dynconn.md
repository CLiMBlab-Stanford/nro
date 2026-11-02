# Dynamic connectivity

`dynconn` publishes a participant's usable cleaned runs so you can explore
vertex- or voxel-level connectivity interactively in Connectome Workbench.
It does not precompute a connectivity matrix. Workbench computes correlations
when you select a location in a scene built by `nro scene`. The `networks`
module also uses `dynconn` output as its default input.

## Run admission

One work item covers a participant, space, and smoothing level. `input_filter`
restricts the cleaned inputs with the same BIDS-entity matching rules as
`microparcellation`.

The module reads each `clean` sidecar and excludes runs with undefined
cleaning, too few retained frames, too little residual design freedom, low
effective rank, or too much temporal concentration. It also requires a minimum
number of usable runs and a minimum total of retained frames. The `inclusion`
settings hold these thresholds.

Censored frames are omitted. Each admitted run is centered and standardized on
its own before pooling. With the default `weighting: precision`, a run's
covariance contribution is proportional to the algebraic temporal rank that
`clean` recorded. `weighting: equal` gives every admitted run the same weight.

The manifest records each included run's half-open `start_frame` and
`stop_frame` range, so the original run boundaries stay recoverable. It also
records each run's repetition time, the distinct source repetition times, and
every excluded run with its reason.

## Time-series representation

`low_rank` chooses what the published series contains.

With the default `low_rank: true`, nro publishes a short synthetic series
whose correlations approximate the pooled correlation matrix at limited rank.
A streamed randomized decomposition estimates the leading spatial eigenspace of
the weighted run covariances without forming a vertex-by-vertex or
voxel-by-voxel matrix. Admitted runs may have different repetition times.

The rank ceiling defaults to 1,000. If the centered input has lower rank, the
fit uses the rank available. For `D` fitted dimensions the output has `D + 1`
samples, built from orthonormal, zero-mean Helmert contrasts. Their sample
covariance reproduces the fitted rank-`D` matrix, and their Pearson
correlations reproduce its normalized correlation matrix. These samples are
coordinates for Workbench seed correlation, not acquisition times. The manifest
records the requested and fitted dimensions, fitted eigenvalues, realized rank,
retained variance, locations with zero temporal variance, and the effective
random seed.

With `low_rank: false`, nro publishes the retained acquisition frames in their
original order. Because the series axis then represents acquisition time, all
admitted runs must share one repetition time.

## Outputs

Outputs for every space and smoothing level share
`derivatives/nro/dynconn/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`, and filenames
include both entities.

A surface target has one `_desc-dynamicConnectivity_bold.dtseries.nii` file.
Its brain-model axis joins the complete left and right surface meshes. A full
series records the common repetition time on its series axis. A low-rank series
uses a nominal one-second step because its axis is synthetic.

A volume target has one uncompressed `_desc-dynamicConnectivity_bold.nii`. The
file is left uncompressed so Workbench can read it randomly while computing
voxelwise correlations.

Both forms include a YAML publication manifest and a JSON publication index.

## Viewing

Build and open a combined scene with:

```bash
nro scene -m dynconn -p PARTICIPANT -P PROJECT -s SPACE -S MM --open
```

Use `nro render --seeds FILE` to make static seed-correlation images. Seed
coordinates are in the participant's T1w world coordinates. See the
[viewing guide](../commands/viewing.md#nro-render).

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.
nro sets output placement from the project, module ID, participant, space, and
smoothing request. `overwrite` requests rebuilding the outputs.

`inclusion` holds the per-run and aggregate admission thresholds from the
sidecars. `weighting` selects `precision` or `equal` covariance pooling.
`low_rank` turns compression on or off.

When `low_rank` is true, `low_rank_options` tunes the randomized fit:

* `dimensions` sets the output-rank ceiling.
* `oversampling` widens the trial subspace beyond that ceiling before
  truncation. This lowers the chance of missing components near the cutoff and
  costs memory and computation in proportion to the wider subspace. It does not
  add dimensions to the published result.
* `power_iterations` applies the correlation operator to the trial subspace
  repeatedly. Each iteration separates strong components from weak ones, which
  helps when the spectrum decays slowly, and costs one more streaming pass.

The random seed is drawn from the host system and recorded in the manifest so
that the fit can be reproduced. It is not a configuration value. Low-rank
options do not affect scientific freshness while `low_rank` is false.

```{literalinclude} ../../nro/definitions/starters/configs/dynconn/main_dynconn.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/dynconn/module/index.rst).
