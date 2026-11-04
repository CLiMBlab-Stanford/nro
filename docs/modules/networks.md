# Individualized networks

`networks` estimates a participant's individualized brain networks as a set of
continuous membership maps, one per network, and gives them heuristic labels
from population reference maps. Read this page to choose an input source and
estimation strategy and to interpret the published maps.

## Choosing a source and strategy

`networks` reads one connectivity source, chosen by `connectivity_source`:

* `dynconn`, the default, supplies vertex- or voxel-level time series.
* `microparcellation` supplies continuous parcel-connectivity profiles.

The source is a static workflow dependency, so requesting `networks` plans only
the selected source. Anatomy is always a direct dependency because it is
needed to project the reference atlases.

`parcellation_strategy` selects the estimator: `ica`, the default, `clustering`,
or `oslom`. OSLOM is valid only with a `microparcellation` source. The same
input preparation, labeling, and publication stages surround every strategy.

The packaged workflows cover the common combinations. `main` uses dynconn with
ICA, and `clustering` uses dynconn with repeated clustering. `oslom` uses
microparcellation with OSLOM. `dynconn-networks` is an explicit name for
the default dynconn source.

## Input preparation

Both sources become a matrix with one row per spatial location.

A `dynconn` source supplies its dtseries or 4D volume directly. nro never
builds a dense vertex-by-vertex connectome from it.

For a `microparcellation` source, the loader checks the CIFTI parcel axis
against the dlabel and applies `connectivity.transform`. It then removes
weights at or below `minimum_weight` and applies `percentile_cutoff`. The
default clips negative weights and keeps the top decile of transformed
off-diagonal weights. Ties can make the retained fraction differ slightly from
10%.

`feature_reduction.maximum_dimensions` caps the number of feature columns
passed to ICA and clustering. Narrower inputs pass through unchanged. Wider
inputs are projected onto their leading randomized-SVD scores, controlled by
`oversampling`, `power_iterations`, and `random_seed`. For low-rank dynconn,
nro uses the rank declared in the dynconn manifest and does not treat the extra
synthetic frame as an independent dimension. The publication manifest records
the source, input and fitted dimensions, seed, and retained variance.

## ICA

scikit-learn FastICA fits the reduced feature matrix with unit-variance
whitening. For microparcellation this is ICA of parcel-connectivity profiles.
For dynconn it is spatial ICA, with locations as observations and retained
time points or low-rank dimensions as features.

nro converts each component to a bounded membership score. It flips the sign
when the median loading is positive, clips negative loadings to zero, clips
positive loadings at the `upper_quantile` (default 0.99), and divides by that
upper value. This assumes a network's positive tail covers a minority of the
spatial domain. The score is not a calibrated probability, and scores are not
normalized across networks at a location. A component with no positive finite
tail fails validation. Reaching the iteration limit logs a convergence warning.

`n_networks` defaults to 100 and must be smaller than both the number of
locations and the number of fitted features. `random_seed`, `max_iterations`,
and `tolerance` control FastICA.

## Repeated clustering

This strategy adapts the [Shain and Fedorenko procedure](../methods/software.md)
to either source. MiniBatchKMeans is fitted repeatedly to the bounded,
continuous feature matrix, so microparcellation input keeps its connectivity
magnitudes instead of being binarized. Each repetition uses an offset random
seed.

nro sorts the fits by inertia, and the lowest-inertia fit defines the initial
network identities. Each later hard partition is matched one-to-one to the
accumulated memberships by spatial correlation with the Hungarian algorithm.
The assignments are averaged and each network map is min-max normalized. After
that normalization, scores at a location do not necessarily sum to one.

`n_networks` and `repetitions` set the network count and number of fits.
`random_seed`, `n_init`, `max_iterations`, `batch_size`, `max_no_improvement`,
and `reassignment_ratio` are scikit-learn fitting controls.

## OSLOM

For microparcellation input, nro writes the retained graph for the configured
OSLOM `executable`. `initialization` can be Leiden, an explicit
`initial_partition`, or no seed partition, as the configuration validator
allows. `leiden_resolution`, `leiden_iterations`, and `leiden_seed` affect
only the initialization and do not produce a separate result. `weighted`,
`directed`, and `significance` control how OSLOM reads the graph and tests
significance.

`repetitions` counts complete nro-level OSLOM fits and defaults to one.
`internal_runs` and `hierarchical_runs` are OSLOM's own repeated searches and
default to 10 and 50, so a single nro fit still performs many searches.
`extra_args` passes additional OSLOM options, and `timeout_seconds` limits run
time. OSLOM output is streamed to the log. The OSLOM binary is sequential, so
more worker CPUs do not speed it up.

Replicates are matched by Jaccard overlap. In the `consensus` block,
`minimum_match_jaccard` sets the threshold for a valid match,
`assignment_threshold` decides membership, and `homeless_threshold` classifies
locations left unassigned. With one replicate, stability says nothing about
reproducibility between fits.

## Labels

When `labeling.enabled` is true, nro projects the 15 DU15 maps and the LanA
reference into the target geometry, using anatomical transforms and surface
sampling as needed. It averages reference values within microparcels and
computes spatial correlations with each candidate network map. For each
reference it keeps the `candidates_per_reference` most similar networks
(default three), in order. One network can receive labels from several
references, and networks that no reference selects keep numeric identifiers.
The correlations and ranks are retained.

These labels are spatial heuristics. They are not functional localizer results
or ground truth.

## Outputs

Outputs for every space and smoothing level share
`derivatives/nro/networks/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`, and filenames
include both entities. Public outputs are:

* a multi-map `desc-networks_stat.dscalar.nii`;
* stability, homelessness, and overlap CIFTIs;
* label TSV and JSON files;
* a YAML publication manifest and a JSON index.

The network artifact does not copy the upstream feature source. See the
[path contract](../autoapi/nro/modules/networks/paths/index.rst) for exact
names.

Maps have human-readable names, so you can step through networks in Workbench.
Each CIFTI has a JSON sidecar that maps zero-based map indices to metadata and
provides reverse lookups. A network map can be found by numeric network ID,
numeric label, heuristic label, or display name, so one map may answer to
`network005`, `lana002`, and `dna003`. Downstream code can call
`nro.engine.cifti.load_indexed_cifti_map` instead of parsing display names.

Use `nro scene -m networks` to build a linked or portable view.
`nro render -m networks` writes one static image for each named map. See the
[viewing guide](../commands/viewing.md#nro-render).

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.
nro sets output placement from the project, module ID, participant, space, and
smoothing request. `overwrite` requests rebuilding the outputs. The other keys
are described in the sections above.

```{literalinclude} ../../nro/definitions/starters/configs/networks/main_networks.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/networks/module/index.rst),
[ICA](../autoapi/nro/modules/networks/ica/index.rst),
[clustering](../autoapi/nro/modules/networks/clustering/index.rst),
[OSLOM](../autoapi/nro/modules/networks/oslom/index.rst),
[labeling](../autoapi/nro/modules/networks/labeling/index.rst).
