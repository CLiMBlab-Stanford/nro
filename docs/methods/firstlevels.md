# First-level estimation and inference

The [firstlevels guide](../modules/firstlevels.md) describes inputs, model
registration, processing, and outputs. These equations describe nro's estimator,
not a requirement of the Stats Models interchange format.

The [task-model compiler](../task-models.md) combines event predictors with the
firstlevels configuration's nuisance design. Exact outlier indicators do not
enter nuisance PCA. The estimator excludes their marked rows, preserving gaps
in the temporal covariance. For a fixed covariance model, this gives the same
task estimates and residual DOF as fitting one unconstrained spike coefficient
per excluded observation in the full design. Interpolated observations are not
introduced.

## Retained observations and temporal noise

For one run, let `Y` be the full time-by-location response and `X` the full
declared design. Subscript `R` selects uncensored frames. Fit `Y_R` using the
independent task columns and nuisance PCs constructed from `X_R`. No temporal
filter or frequency-basis projection is applied. The number of observations
for residual-rank accounting is the number of retained frames. A run with no
retained frames or no residual dimensions has no fit.

For AR(1) candidate `rho`, let `i` and `j` be original frame indices:

```text
V_R[i, j] = rho ** abs(i - j)
```

Thus a censoring gap never becomes an artificial one-TR interval. OLS uses
`rho = 0`. Each AR candidate is fitted by Cholesky-whitened GLS. Restricted
likelihood selects a group independently at each spatial location. The
normalized coefficient covariance is shared by locations in that group;
residual variance supplies the location-specific scale.

Task columns retain their original interpretation through a recorded mapping
from independent fit columns. Nuisance candidates are RMS-scaled and reduced
by SVD, targeting `nuisance_variance_explained`. If `d` dimensions remain after
the exact task model, at most

```text
max(0, d - max(minimum_temporal_rank,
               ceil(minimum_temporal_rank_fraction * d)))
```

nuisance PCs may enter. PCs redundant with task columns are excluded. The PCs
are fitted jointly with the task predictors; they are not first made
task-orthogonal. Making nuisance predictors task-orthogonal would change the
meaning of adjusted task coefficients.

For joint independent design `D`, the conditional residual DOF and covariance
are:

```text
nu = number_of_retained_frames - rank(D)
sigma_squared = whitened_residual_sum_of_squares / nu
Cov(beta) = sigma_squared * inverse(D.T @ inverse(V_R) @ D)
```

The stored coefficient mapping converts this covariance back to the named
scientific effects. Non-estimable linear combinations are omitted. Nuisance
rank protection does not impose a minimum fit DOF: an otherwise identifiable
short run can be fitted with few remaining dimensions.

## Contrasts and within-participant summaries

Every summary retains its linear coefficients on original run estimates.
Equal weighting means an arithmetic mean over available inputs for each
effect. Precision weighting instead normalizes inverse estimated marginal
variances at each location. Zero/nonfinite variances do not receive precision
weight; a location with no valid contribution has undefined t/DOF.

Both session and subject summaries pool original runs directly. Session
summaries are not inputs to subject summaries. The presence of sessions
therefore does not change the default relative weights of the runs.

When a condition is absent, its run supplies no estimate and receives no weight
for that condition. A subtraction at the run level therefore uses only runs
where both effects are estimable. A subtraction after pooling conditions can
use different run sets for its two terms. These are different estimands.

For a final contrast, let `g_r` collect all its coefficients on original run
`r`, including pooling weights. Assuming independent errors between runs:

```text
effect = sum_r(g_r @ beta_r)
v_r = g_r @ Cov(beta_r) @ g_r.T
variance = sum_r(v_r)
nu_summary = variance**2 / sum_r(v_r**2 / nu_r)
t = effect / sqrt(variance)
```

Combining coefficients before computing `v_r` retains covariance between
conditions sharing that run. A run is one independent variance contribution,
even when it reaches a summary through several effects. Pooling the same source
run twice as independent inputs is rejected.

The DOF formula is a
[Satterthwaite approximation](https://doi.org/10.2307/3002019).
For one run it reduces to that run's residual DOF. Disjoint sets of runs have
no cross-covariance under the independent-runs assumption. This is within-subject
inference about pooled effects; it does not estimate between-run random effects
or support population claims.

If all conditions occur in all runs and share the same fixed run weights,
linear contrasts commute with aggregation. nro tests equality of effects,
variances, t-values, and DOF for general linear combinations, including unequal
fixed run weights. Condition-specific precision weights need not commute.
Nested equal-weight session summaries give equal weight to sessions, not
necessarily to every original run. Connect a Subject node directly to Run
outputs to request equal run weighting, as the example in the
[task-model guide](../task-models.md) does.

## Inference limits and validation

GLS t inference is exact under a correctly specified, fixed Gaussian noise
covariance. Estimated AR groups make the reference distribution approximate;
precision-weighted summaries also condition on estimated weights. The code
does not claim that subtracting one extra DOF makes estimated AR inference
exact. Voxelwise AR(1) may also miss temporal structure in real fMRI data.

In the synthetic regression check with 160 frames, censoring every thirteenth
frame, true AR coefficient 0.6, and 3,000 independent null locations,
the test checks both fixed-covariance and estimated-AR inference. The
estimated-AR test's wider tolerance detects large regressions; it does not
certify nominal error control. More extensive simulation and real-data residual
diagnostics are needed before confirmatory use.

Tests also compare OLS and gap-aware fixed-covariance GLS against direct matrix
calculations; check nuisance-adjusted coefficients, rank safeguards, missing
conditions, shared-run covariance, nested summaries, zero variance, and
volume/surface publication; and exercise resumption after an output disappears.
No spatial multiple-comparison correction is performed.
