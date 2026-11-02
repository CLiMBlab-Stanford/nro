# First-level models

`contrasts` estimates task effects within one participant. It fits each run
separately, then builds session and subject summaries from those runs, and
publishes effect, variance, t, and degrees-of-freedom maps in volume or surface
space. It never combines participants or fits population-level models.

## Inputs

`contrasts` depends on `func` and does not use `clean`. Its nuisance model
excludes global signal. It reads the canonical `desc-preproc` series from the
selected functional workflow, whether or not that workflow applied ICA-AROMA.
To compare denoising choices, use separate workflows.

Models live in `DEFINITIONS/models/ANALYSIS/VARIANT.yml` in the selected
external store. A model defines source tasks, event predictors, and contrasts.
The `contrasts` configuration defines denoising, estimation, and run
aggregation. A single-task model takes its source task from `ANALYSIS`, and a
multi-task model lists `tasks` explicitly. New definitions stores start with no
models. See [first-level models](../task-models.md) for the YAML format,
transformations, and model sets, and [model commands](../commands/models.md)
for registration.

## Requesting fits

A request without a model selector uses model set `main`. This example assumes
the site has registered a `mytask/main` model in that set:

```bash
nro models list
nro models show mytask/main
nro run -P example -p 01 -m contrasts -s fsnative -S 2
```

A bare `nro run` requests `dynconn`, `networks`, and `contrasts`, including
GLMs for every matching task in model set `main`. Use `-m` to restrict the
requested endpoints. Analysis, task, and model-set selectors narrow the models:

```bash
nro run -m contrasts --analysis language --model main
nro run -m contrasts --task langlocSN
nro run -m contrasts --model-set development
```

One work item covers a participant, model variant, space, and smoothing value,
so several spaces or smoothing values create independent work items. The run
fits are steps within the work item, not separate worker jobs.

Model selection is independent of workflow configuration. To fit one model
variant under different denoising configurations, select different workflows.

A work item includes every run of its source tasks for that participant that
matches the configuration's `input_filter` and has an applicable events table.
Runs without an events table are skipped. CLI run selectors other than task
are rejected, because they would change the subject-level artifact without
changing the work item's identity.

## Processing

1. **Match runs and events.** nro matches the registered source tasks and
   resolves events through BIDS inheritance. A more specific events table
   replaces a general one, and rows are not merged. `Run` grouping must
   identify each BOLD run.
2. **Smooth.** At positive FWHM, volumes are smoothed with SciPy's Gaussian
   filter, using voxel dimensions to convert mm to standard deviations. The
   grid axes must be orthogonal, and boundaries are zero-padded. Surfaces are
   smoothed with Workbench `-metric-smoothing -fwhm` on midthickness geometry.
   Native geometry comes directly from `anat`, which is an explicit dependency
   and is not copied. fsaverage and fsaverage6 geometry come from TemplateFlow.
   Zero FWHM skips smoothing.
3. **Build the design.** The model YAML and module configuration are compiled
   into Stats Models. `confounds_regex` selects continuous nuisance columns
   from the confounds. Events are convolved with Nilearn's SPM or Glover
   canonical HRF, normalized to unit sum over discrete samples.
   Positive-duration events keep their source-scale height and duration, and
   zero-duration events are unit-area impulses. Frames marked in any
   `temporal_mask_regex` column are excluded. Those outlier columns do not
   enter the continuous design.
4. **Scale and reduce nuisance.** At each location, the response is divided by
   its mean over retained frames, multiplied by 100, and centered at zero.
   Nuisance principal components are selected within the residual-rank budget
   of the retained frames, then task and nuisance terms are fitted together.
   Contrasts stay in the original task parameterization. No temporal filtering
   or unit-variance standardization is applied.
5. **Fit.** nro fits OLS or grouped AR(1) GLS. The AR covariance respects the
   original gaps between retained frames. Each run saves compact coefficients
   and covariance information, the requested maps, a labeled design SVG, a
   numerical design, and design and rank metadata.
6. **Aggregate.** Effects are pooled across the original runs before the
   requested linear combinations are evaluated. Session summaries are built
   where sessions exist. Subject summaries are built directly from runs, not
   from session summaries. Each contribution is traced to its original run so
   that covariance between effects that share data is kept. t-values are never
   averaged.
7. **Publish.** nro publishes node inventories and commits the work-item
   completion record.

Some outcomes produce omission records instead of maps. Missing conditions and
non-estimable contrasts get omission records, not zero-valued effect maps. If
censoring makes the temporal model unidentifiable or leaves no residual
dimensions, the whole run's estimates are omitted with a reason. These
decisions do not change the runner graph, which is built in advance. Invalid
inputs and unsupported model instructions still raise errors.

If the rank after task terms is below the absolute nuisance-protection floor,
nuisance regression is skipped. That alone does not exclude the run, and the
fit keeps its actual residual degrees of freedom. Locations with zero variance
have NaN t and DOF values and must be excluded from inference.

## Units

Effect maps are percentage-point BOLD changes per unit increase in the
source-scale predictor. Standard errors use the same units, and variances are
in squared percentage points. The mean over retained frames sets a run- and
location-specific baseline, so censored frames do not affect it. Canonical HRF
normalization keeps the predictor's declared amplitude scale. For a
unit-height block, a longer duration prolongs the convolved response and
increases its area, but one predictor unit means the same thing.

See [estimation and inference](../methods/contrasts.md) for equations,
assumptions, weighting, and the limits of conditional AR inference.

## Outputs

```text
derivatives/nro/contrasts/<CONFIG_ID>-<LINEAGE_DIGEST>/
  sub-ID/
    ses-SESSION/task-TASK/node-run/
    ses-SESSION/task-TASK/node-session/
    task-TASK/
      node-session/
      node-subject/
```

The configuration directory follows nro's lineage rules: the configuration ID
followed by a deterministic digest of the complete upstream lineage. Run
outputs and session maps use `ses-SESSION` when the source data have sessions.
The task-level `node-session` directory holds the aggregate node's manifest,
and datasets without sessions also put their session maps there.

Filenames include model variant, task, space, smoothing, Stats Models node
name, and any run or session entities. Run filenames from a multi-task model
also keep the source `task-` entity. These names are BIDS-like, but nro does not
claim validator compliance.

**Statistic maps.** Each node target has one dense-scalar CIFTI for each
available statistic: effect, variance, t, and degrees of freedom. Contrasts are
named maps on the CIFTI scalar axis, so Workbench can page through them. A
pass-through contrast has no t map and is left out of the t-statistic CIFTI.
Each CIFTI has a JSON sidecar with zero-based map indices, per-map contrast
metadata, and a reverse lookup from contrast name to map index. Downstream code
can call `nro.engine.cifti.load_indexed_cifti_map` to select a contrast without
parsing its display name. Internal primitive-effect records support
aggregation without publishing maps for effects you did not request.

**Compact fits.** Each fitted run saves a coefficient array, a residual
variance map, a noise-group map, small group covariance matrices, and the AR
grid as NumPy arrays. These allow later covariance-aware summaries without
loading time courses or storing a map for every coefficient pair. They are
public artifacts and must be kept with the contrast maps.

**Design records.** `_design.tsv` contains the fitted columns on retained
frames, with zero placeholders on excluded rows, and the SVG marks those rows.
`_design.json` records the mask, column names, coefficient mapping, the absence
of temporal filtering, the rank budget, and the residual DOF. Together they
describe the realized PCA fit. GLS whitening can use a different array for
each noise group. The shared design and the group covariance records describe
that fit without a separate plot per group.

Each fitted run also saves `_statsmodel.json` with its resolved event HRFs,
selected nuisance columns, exact outlier columns, and estimator rules. The work
item saves the scientific model YAML, the resolved configuration, and the
compiled model template. Model-set membership is not part of these scientific
records.

**Manifests and completion.** Node manifests list maps, compact fits, design
files, contributing records, and omissions. The fixed
`_desc-contrasts_manifest.json` under `task-TASK/node-run/` collects the
completion evidence for the work item. Deleting a declared output invalidates
completion. Resumption checks the model, the configuration, and the selected
run set as well as file freshness. During purge, artifacts from different
variants or targets are kept apart by their complete entity-decorated filename
prefix.

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.

| Key | Effect |
| --- | --- |
| `input_filter` | Restrict source runs by stable BIDS entity values. Alternatives for one entity form a set; different entities intersect. |
| `confounds_regex` | Select continuous nuisance candidates from `confounds.tsv`. Selected global signal columns are rejected. |
| `temporal_mask_regex` | One-hot columns whose union excludes frames from estimation. |
| `nuisance_variance_explained` | Desired fraction of standardized nuisance variance, subject to the rank cap. |
| `minimum_temporal_rank` | Absolute number of post-task dimensions protected from nuisance removal. |
| `minimum_temporal_rank_fraction` | Protected fraction of post-task dimensions. |
| `aggregation_weighting` | Pool runs by inverse estimated marginal variance (`precision`) or arithmetic mean (`equal`). |
| `noise_model` | `ar1` or `ols`. |
| `ar_grid` | Finite AR(1) correlation candidates strictly between -1 and 1. Ignored for OLS. |
| `spatial_block_size` | Locations evaluated together during fitting and summary calculation. One run's input array is loaded at a time. |
| `container` | QuNex container settings used for Workbench surface smoothing. |
| `wb_command` | Workbench command inside the configured container. |

```{literalinclude} ../../nro/definitions/starters/configs/contrasts/main_contrasts.yml
:language: yaml
```

## Stats Models execution

The [first-level model compiler](../task-models.md) generates the Stats Models
document used for fitting. The estimator implements a bounded subset of
[BIDS Stats Models 1.0.0](https://bids-standard.github.io/stats-models/) and
does not depend on FitLins. Unsupported task controls fail validation.

nro records its adaptive nuisance PCA, exact outlier handling, rank
safeguards, noise estimation, and aggregation weighting under
`Model.Software.nro`. Another engine must honor these rules to reproduce the
estimates. A schema-valid document alone does not guarantee identical results.

Dataset nodes, cross-subject aggregation, summary-level GLMs, formulas, F
tests, random effects, HRF derivatives, and temporal filtering are not
supported.

Implementation: [compiler](../autoapi/nro/modules/contrasts/compiler/index.rst),
[design](../autoapi/nro/modules/contrasts/design/index.rst),
[statistics](../autoapi/nro/modules/contrasts/statistics/index.rst),
[module](../autoapi/nro/modules/contrasts/module/index.rst).
