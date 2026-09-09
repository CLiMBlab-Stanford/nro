# Individualized networks

`networks` consumes matching CIFTI microparcel labels, parcel connectivity, and
their manifest. It also depends directly on anatomy for reference-atlas
projection. It does not copy anatomical volumes into the scene.

## Common input preparation

Load the CIFTI parcel axis and validate it against the microparcellation labels.
Transform connectivity weights with `connectivity.transform`, remove weights
at or below `minimum_weight`, and apply `percentile_cutoff`. The default clips
negative weights and applies the 90th percentile of transformed off-diagonal
weights, including zeros. Ties at the threshold are retained, so the resulting
edge fraction is not necessarily exactly 10%.
The sparse adjacency stores one triangle; algorithms symmetrize where needed.
Reported edge counts must be interpreted according to that representation.

The same input/output and labeling stages surround all `parcellation_strategy`
branches. `main` selects `ica`; `clustering` and `oslom` workflows select the
corresponding configuration overrides.

## ICA

Randomized SVD reduces the symmetric sparse adjacency to `ica.n_networks`
dimensions. scikit-learn FastICA fits the resulting spatial scores with
unit-variance whitening. This is ICA of microparcel connectivity profiles,
not temporal ICA of the original BOLD frames.

For each component, flip the sign if its median loading is positive. Clip
negative loadings to zero and positive loadings at the `upper_quantile`
(default 0.99); divide by that upper value. The result is a bounded membership
score. It is neither a calibrated probability nor normalized across networks
at a vertex. The convention assumes a network's positive tail occupies a
minority of the spatial domain. Components without a positive finite tail fail
validation. Reaching the iteration limit logs a convergence warning.

`n_networks` defaults to 50 and must be smaller than the number of microparcels.
`random_seed`, `max_iterations`, and `tolerance` control FastICA;
`svd_oversamples` and `svd_power_iterations` control the approximation.

## Repeated clustering

Binarize the symmetric retained connectivity profiles and fit MiniBatchKMeans
repeatedly. Each repetition uses an offset random seed. Sort fits by inertia;
the lowest-inertia fit defines initial network identities. Align subsequent
hard partitions one-to-one to accumulated memberships using spatial correlation
and the Hungarian assignment algorithm. Average assignments, then min-max
normalize each network map. Final scores no longer necessarily sum to one
across networks after that normalization.

This adapts the [Shain and Fedorenko procedure](../methods/software.md) to sparse
microparcel profiles. `n_networks`, `repetitions`, `random_seed`, `n_init`,
`max_iterations`, `batch_size`, `max_no_improvement`, and `reassignment_ratio`
are the network count, repeat policy, and scikit-learn fitting controls.

## OSLOM

Write the retained graph for the configured OSLOM executable. Initialization
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

The root is `derivatives/networks/LINEAGE/space-SPACE_smoothing-Nmm/sub-ID/`.
Public outputs include a multi-map `desc-networks_stat.dscalar.nii`, stability,
homelessness, and overlap CIFTIs; per-network maps; label TSV/JSON; YAML
publication manifest; JSON index; and a relocatable scene. Human-readable map
names permit stepping through networks in Workbench. Surface geometry is copied
from the source scene, but anatomical volumes and registration dependencies are
not scene dependencies. See the [path contract](../autoapi/nro/modules/networks/paths/index.rst).

## Shipped configuration

`output_dir`, `prefix`, and `overwrite` control direct output placement and
rebuilding. The planner supplies instance-specific paths during orchestration.

```{literalinclude} ../../nro/configuration/starters/configs/networks/main_networks.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/networks/module/index.rst),
[ICA](../autoapi/nro/modules/networks/ica/index.rst),
[clustering](../autoapi/nro/modules/networks/clustering/index.rst),
[OSLOM](../autoapi/nro/modules/networks/oslom/index.rst),
[labeling](../autoapi/nro/modules/networks/labeling/index.rst).
