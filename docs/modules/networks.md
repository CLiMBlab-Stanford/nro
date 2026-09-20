# Individualized networks

`networks` estimates individualized maps from one configured source. The default
uses `microparcellation` connectivity. Set `connectivity_source: dynconn` to use
vertex- or voxel-level dynamic-connectivity time series instead. The selected
source is a static workflow dependency, so requesting `networks` plans only that
source. Anatomy remains a direct dependency for reference-atlas projection.

## Common input preparation

Both sources become a matrix with spatial locations on rows. A
`microparcellation` source supplies continuous parcel-connectivity profiles.
The loader validates the CIFTI parcel axis against the dlabel, applies
`connectivity.transform`, removes weights at or below `minimum_weight`, and
applies `percentile_cutoff`. The default clips negative weights and retains the
top decile of transformed off-diagonal weights. Ties can make the retained
fraction differ from exactly 10%. A `dynconn` source supplies its dtseries or
four-dimensional volume directly; it never creates a dense vertex-by-vertex
connectome.

`feature_reduction.maximum_dimensions` bounds the feature columns used by ICA
and clustering. Inputs at or below the bound pass through unchanged. Wider
inputs are projected onto leading randomized-SVD scores. Low-rank dynconn uses
the rank declared in its manifest rather than treating the extra synthetic
frame as an independent dimension. `oversampling`, `power_iterations`, and
`random_seed` control this reduction. The publication manifest records source,
input and fitted dimensions, seed, and retained variance.

The same output and labeling stages surround every `parcellation_strategy`.
`main` selects ICA from microparcellation. `clustering` and `oslom` select the
corresponding estimator, while `dynconn-networks` selects dynconn-backed ICA.
OSLOM is valid only with a microparcellation source.

## ICA

scikit-learn FastICA fits the bounded feature matrix with unit-variance
whitening. For microparcellation, this is ICA of parcel-connectivity profiles.
For dynconn, it is spatial ICA: locations are observations and retained time or
low-rank dimensions are features.

For each component, flip the sign if its median loading is positive. Clip
negative loadings to zero and positive loadings at the `upper_quantile`
(default 0.99); divide by that upper value. The result is a bounded membership
score. It is neither a calibrated probability nor normalized across networks
at a vertex. The convention assumes a network's positive tail occupies a
minority of the spatial domain. Components without a positive finite tail fail
validation. Reaching the iteration limit logs a convergence warning.

`n_networks` defaults to 50 and must be smaller than both the spatial and fitted
feature dimensions. `random_seed`, `max_iterations`, and `tolerance` control
FastICA.

## Repeated clustering

Fit MiniBatchKMeans repeatedly to the continuous bounded feature matrix. This
preserves retained connectivity magnitudes for microparcellation instead of
binarizing edges. Each repetition uses an offset random seed. Sort fits by inertia;
the lowest-inertia fit defines initial network identities. Align subsequent
hard partitions one-to-one to accumulated memberships using spatial correlation
and the Hungarian assignment algorithm. Average assignments, then min-max
normalize each network map. Final scores no longer necessarily sum to one
across networks after that normalization.

This adapts the [Shain and Fedorenko procedure](../methods/software.md) to either
source representation. `n_networks`, `repetitions`, `random_seed`, `n_init`,
`max_iterations`, `batch_size`, `max_no_improvement`, and `reassignment_ratio`
are the network count, repeat policy, and scikit-learn fitting controls.

## OSLOM

For microparcellation input, write the retained graph for the configured OSLOM executable. Initialization
can use Leiden, an explicit partition, or no seed partition, as validated by
the config. Leiden resolution, iterations, and seed affect initialization,
not a separate network result. `weighted`, `directed`, and `significance`
configure graph interpretation and significance testing.

`repetitions` counts complete nro-level OSLOM fits and defaults to one.
`internal_runs` and `hierarchical_runs` are OSLOM's own repeated searches and
default to 10 and 50. One nro fit therefore still performs many internal
searches. `extra_args` forwards additional OSLOM options; `timeout_seconds`
bounds execution. OSLOM output is streamed to the log. More worker CPUs do not
make the current sequential OSLOM binary parallel.

Replicate matching uses Jaccard overlap. `minimum_match_jaccard` controls valid
matches, `assignment_threshold` determines assigned memberships, and
`homeless_threshold` classifies unassigned locations. With one replicate,
stability is not evidence of between-fit reproducibility.

## Labels and publication

When `labeling.enabled`, project the 15 DU15 maps and LanA reference into the
target geometry using anatomical transforms and surface sampling as needed.
Average reference values within microparcels and compute spatial correlations
with candidate network maps. For each reference, retain
`candidates_per_reference` candidates (default three), ordered by similarity.
A network can receive labels from several references. Unselected networks keep
numeric identifiers. These are spatial heuristics, not functional localizer
results or ground-truth labels. Correlations and ranks are retained.

Outputs from every space and smoothing target share
`derivatives/nro/networks/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`. Their filenames include both
entities.
Public outputs include a multi-map `desc-networks_stat.dscalar.nii`, stability,
homelessness, and overlap CIFTIs; label TSV/JSON; YAML publication manifest; and
JSON index. Human-readable map names permit stepping through networks in
Workbench. Each CIFTI has a JSON sidecar that maps zero-based map indices to
metadata and provides reverse lookups. Network membership maps can be selected
by their numeric network ID, numeric label, heuristic label, or display name.
One map can therefore resolve from several labels such as `network005`,
`lana002`, and `dna003`. Downstream code can use
`nro.engine.cifti.load_indexed_cifti_map` instead of parsing display names. Use
`nro scene -m networks` to build a linked or portable view. The network artifact
does not copy the upstream feature source. See the
[path contract](../autoapi/nro/modules/networks/paths/index.rst).

`nro render -m networks` writes one static image for each named map; see the
[viewing guide](../commands/viewing.md#nro-render).

## Shipped configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup); `null` ignores markup.
nro fixes output placement from the project, module ID, participant, space, and
smoothing request. `overwrite` requests rebuilding module outputs.

```{literalinclude} ../../nro/configuration/starters/configs/networks/main_networks.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/networks/module/index.rst),
[ICA](../autoapi/nro/modules/networks/ica/index.rst),
[clustering](../autoapi/nro/modules/networks/clustering/index.rst),
[OSLOM](../autoapi/nro/modules/networks/oslom/index.rst),
[labeling](../autoapi/nro/modules/networks/labeling/index.rst).
