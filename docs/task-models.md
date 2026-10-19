# First-level models

A first-level model defines event predictors and contrasts for
[firstlevels](modules/firstlevels.md). Store it as
`DEFINITIONS/models/ANALYSIS/VARIANT.yml` in the selected
[definitions store](definitions.md); its identifier is
`ANALYSIS/VARIANT`. For an ordinary single-task model, the analysis label is
also the source BIDS task and the YAML does not repeat it. A model that pools
several tasks declares them explicitly:

```yaml
tasks: [ActPassBlock, ActPassEvent]
conditions: trial_type
contrasts:
  plausibility:
    trial_type.activeImplausible: 0.25
    trial_type.passiveImplausible: 0.25
    trial_type.activePlausible: -0.25
    trial_type.passivePlausible: -0.25
```

`Input.task` in the compiled Stats Models document contains the declared source
tasks. Each run is still fitted independently. Session and subject summaries
pool the available primitive effects across tasks, so a contrast may become
estimable even when no individual run contains all of its conditions.

The current derivative layout predates multi-task models and encodes the
analysis label as `task-ANALYSIS`. This is an on-disk compatibility convention;
it does not mean that the analysis label is a source BIDS task. The public model
snapshot and compiled Stats Models document retain the actual source tasks.

`nro def edit model ANALYSIS [VARIANT]` opens an existing model or drafts a missing
one from matching BIDS events. See
[definition authoring](commands/authoring.md) for discovery selectors, local
drafts, and save behavior.

The firstlevels configuration supplies nuisance regression, temporal outlier
handling, rank protection, run aggregation, and the noise model. First-level
models cannot override those settings or introduce confounds. This lets the
same scientific model run under different firstlevels configurations.

## Basic model

```{literalinclude} examples/task-model.yml
:language: yaml
```

`conditions` names a categorical column in `events.tsv`. nro includes every
observed category, including categories absent from the contrast definitions.
In this short form, contrast keys such as `S` resolve to `trial_type.S`.
Contrast weights are finite numbers or fraction strings such as `"1/2"`.
Output contrast names must use letters, digits, dots, underscores, or hyphens.

`hrf` defaults to `spm`; `glover` and `null` are also supported. The selected
HRF applies to event-derived predictors. Confounds and the intercept are never
convolved. Timing comes from `onset` and `duration`, adjusted for the functional
input's `StartTime`; these timing columns are not automatic predictors.

The firstlevels configuration's `aggregation_weighting` setting controls how
available original-run estimates are pooled. The default, `precision`, uses
inverse estimated marginal variance; `equal` uses an arithmetic mean. See
[inference](methods/firstlevels.md) for covariance and DOF calculations.

## Selection and model sets

`model_set` is a name or list of names. Omission means no set membership.
Assign development models to a separate set, or leave them unassigned, to keep
them out of default requests.

```yaml
model_set: [development, language-experiment]
```

```bash
nro run -m firstlevels                         # set main
nro run -m firstlevels --analysis language     # set main within this analysis
nro run -m firstlevels --task langlocSN        # models that consume this source task
nro run -m firstlevels --model alternative    # explicit variants, any set
nro run -m firstlevels --model language/main  # one qualified model
nro run -m firstlevels --model-set development language-experiment
```

Values within a selector are alternatives; different selectors intersect.
`--analysis` matches the first component of `ANALYSIS/VARIANT`. `--task` matches
a model's declared source BIDS tasks. Either selector selects whole models; a
task selection does not remove the model's other declared source tasks. Use a
qualified `--model ANALYSIS/VARIANT` to select one model without ambiguity.
`--model main` matches that variant across analyses. Explicit `--model` bypasses
the default set restriction; an explicit `--model-set` still intersects it.
`--task` and `--run task=...` constrain the same source-task entity.

Registration creates no demand. Bare `nro run` requests `dynconn`, `networks`,
and `firstlevels`, using model set `main` for matching tasks. Use `-m firstlevels`
to request only the modeling module and its upstream dependencies. Existing
registered work remains independent of later changes to model-set membership.

Model sets are execution metadata. Membership changes do not alter work item
identity, scientific fingerprints, or completion checks. Model file timestamps,
comments, and `description` do not affect freshness either. Changing predictors,
contrasts, transforms, or HRFs invalidates the corresponding model's artifacts,
not every model using that firstlevels configuration. Changing a scientific
firstlevels setting affects every model fitted with that configuration.

Status, logs, stop, and purge accept the same explicit model/set filters. Omitted
selectors on those commands mean all registered work, including development
models. Registry discovery considers existing artifacts from all model sets.

## Predictors and transformations

Use `predictors` instead of `conditions` when the design includes numeric
variables, interactions, or selected category indicators. Predictors and
transformation inputs refer only to events, never to confounds. Wildcards match
event variable names; transformations cannot access arbitrary Python code.

```yaml
model_set: development
predictors: [trial_type.*, sentence_difficulty]
hrf: spm
transformations:
  - Name: Factor
    Input: trial_type
  - Name: Demean
    Input: difficulty
    Groupby: [trial_type]
    Output: centered_difficulty
  - Name: Product
    Input: [trial_type.S, centered_difficulty]
    Output: sentence_difficulty
contrasts:
  sentences: {trial_type.S: 1}
  difficulty: {sentence_difficulty: 1}
```

Instructions run in order on event amplitudes, before HRF convolution and
sampling. Multiplication therefore constructs event-level modulation; it does
not multiply convolved predictors. Grouping for centering is explicit. No
automatic orthogonalization of modulators is applied.

| Instruction | Supported options and behavior |
| --- | --- |
| `Factor` | Expand categories as `COLUMN.VALUE`. `Constraint: drop_one` requires an explicit `RefLevel`; default `none` keeps all indicators. |
| `Demean` | Subtract the event mean, within each `Groupby` cell if supplied. Optional `Output` creates a new variable. |
| `Scale` | `Demean` and `Rescale` default to true. Rescaling uses sample SD. Constant groups raise an error. Optional `Groupby` and `Output`. |
| `Sum` | Sum aligned event variables into required `Output`; optional `Weights` must match inputs. |
| `Product` | Multiply aligned event variables into required `Output`. |
| `Select` | Keep only the named variables. Select indicators after `Factor` to restrict modeled categories. |
| `Convolve` | Apply explicit `Model: spm` or `glover`. No amplitude transformations may follow convolution. |

`Input` accepts one variable or a list. `Output` accepts one name; for Demean
and Scale with Output, Input must resolve to one variable. `Groupby` is the
transformation option; Stats Models node grouping uses the separate `GroupBy`
field. Unsupported options are errors, not ignored hints.

Override the default HRF for individual event predictors with `hrf_overrides`:

```yaml
hrf_overrides:
  sentence_difficulty: glover
  already_modeled_response: null
```

Each override must match an event predictor. Overlapping overrides, repeated
convolution, and an override conflicting with explicit convolution are errors.
Absent category indicators contribute no estimable effect; within event algebra,
an absent explicitly named category has zero amplitude. Selected nonfinite
predictors are errors.

## Advanced Stats Models blocks

`statsmodels` accepts `Transformations` and `Contrasts`. These replace the
corresponding shorthand sections; they do not patch the generated model.
Advanced Transformations also replaces automatic `conditions` expansion, so
use an explicit `predictors` list with it.

```yaml
predictors: [trial_type.*]
hrf: spm
statsmodels:
  Transformations:
    Transformer: pybids-transforms-v1
    Instructions:
      - {Name: Factor, Input: trial_type}
  Contrasts:
    - Name: sentences
      ConditionList: [trial_type.S]
      Weights: [1]
      Test: t
```

The same supported transformation semantics apply to both forms. Advanced
contrasts must be linear t contrasts. Raw Nodes, Edges, Model, Software, and
Options blocks are not accepted. Supporting a Stats Models field requires an
estimator implementation and tests, not just a schema-valid JSON document.

## Levels, missing effects, and saved definitions

nro always plans run and subject outputs, plus session outputs when sessions
exist. Sessions group original run estimates; subject summaries also pool
original runs directly. Saving session results does not give a session with
one run the same weight as a session with four runs. A summary with one
contributing run retains that run's uncertainty.

Missing conditions produce omission metadata instead of zero-valued maps.
Contrasts can become estimable after pooling conditions from different runs.
Shared-run covariance is retained. Numerical estimability changes the emitted
contrast inventory, not the execution graph.

The compiler creates the Stats Models document used by the estimator. At run
time it resolves selected nuisance/outlier columns and event HRFs. The saved
run document records those choices; numerical design files record the fitted
PCs, retained rows, parameter mapping, and rank budget. The work item also saves
its scientific model YAML, resolved firstlevels configuration, and compiled
template. Execution-only set membership is excluded.

## Model edits and freshness

Freshness compares canonical compiled task definitions, not YAML text or file
timestamps. Short condition names and their qualified forms, shorthand and
equivalent advanced blocks, omitted defaults and their explicit values, and
equivalent numeric contrast weights produce the same contract. Mapping key
order, comments, `description`, and `model_set` do not change it.

Compilation for this comparison uses metadata only. It does not read event
tables or images, build design matrices, or fit a model. A bounded in-process
cache reuses compilation for identical model contents. Edited contents get a
new cache entry; file timestamps are not cache keys.

Predictor lists, transformation sequences, and explicit advanced contrast lists
remain ordered. Changes to those sequences, HRFs, contrast weights, or
aggregation rules can change the contract. nro does not attempt to prove
arbitrary algebraic equivalence or equivalence that depends on event values.

Planning, status assessment, public-output adoption, and execution use this
normalization. Recorded contracts are normalized as well as current ones, so
an authoring-style change alone does not invalidate an existing derivative.
Resumed steps also compare the compiled model and configuration, preserving
column and internal contrast identifiers. Summary completion checks retain
the selected run set. Input, dependency, and output checks still apply.

Adaptive denoising and covariance-aware estimation are described under
[`Model.Software.nro`](https://bids-standard.github.io/stats-models/_autosummary/bsmschema.models.Model.html).
Another Stats Models engine must implement those rules to reproduce the fit.
Transformation instructions follow the supported subset of the
[named transformer interface](https://bids-standard.github.io/stats-models/_autosummary/bsmschema.models.Transformations.html);
the generated format does not claim unrestricted PyBIDS or FitLins compatibility.
