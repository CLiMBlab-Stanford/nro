# First-level estimation and inference

These equations define how nro estimates first-level task effects and combines
them within a participant. They describe nro's own estimator. The Stats Models
interchange format does not require them. For inputs, model registration,
processing, and outputs, see the [contrasts guide](../modules/contrasts.md).

The [first-level model compiler](../task-models.md) combines event predictors
with the nuisance design from the contrasts configuration. Exact outlier
indicators do not enter nuisance PCA. Instead, the estimator drops the rows they
mark and keeps the resulting gaps in the temporal covariance. Under a fixed
covariance model, this gives the same task estimates and residual DOF as
fitting one unconstrained spike coefficient per excluded observation in the
full design. No interpolated observations are introduced.

## Retained observations and temporal noise

For one run, let `Y` be the full time-by-location response and `X` the full
declared design. The subscript `R` selects uncensored frames.

### Response scaling

Before fitting, the response at location `v` is converted to percent signal
change:

```text
Y_PSC[:, v] = 100 * (Y[:, v] / mean(Y_R[:, v]) - 1)
```

A location without a finite, positive retained-frame mean is invalid. Apart
from the constant offset the scaling is linear, so it scales a run's
coefficients and standard errors together and leaves t statistics unchanged.
Each effect is the estimated percentage-point BOLD change per unit increase in
the predictor on its source scale. Variances are in squared percentage points.

### Design

nro fits `Y_PSC,R` with the independent task columns and nuisance PCs built
from `X_R`. It applies no temporal filter or frequency-basis projection. The
observation count used for residual-rank accounting is the number of retained
frames. A run with no retained frames or no residual dimensions has no fit.

Task columns keep their original meaning through a recorded mapping from the
independent fit columns. Nuisance candidates are RMS-scaled and reduced by SVD
toward the `nuisance_variance_explained` target. If `d` dimensions remain after
the exact first-level model, at most

```text
max(0, d - max(minimum_temporal_rank,
               ceil(minimum_temporal_rank_fraction * d)))
```

nuisance PCs may enter. PCs that are redundant with task columns are excluded.
The PCs are fitted jointly with the task predictors and are not first made
orthogonal to them, because doing so would change the meaning of the adjusted
task coefficients.

The SPM and Glover canonical HRFs have unit discrete sum. A positive-duration
event keeps its declared height and duration before convolution. A
zero-duration event is a unit-area impulse at the oversampled resolution, so
its scale does not depend on TR or on the internal oversampling factor.

### Noise model

For an AR(1) candidate `rho`, with `i` and `j` the original frame indices:

```text
V_R[i, j] = rho ** abs(i - j)
```

Because the indices are original frame numbers, a censoring gap never becomes
an artificial one-TR interval. OLS is the case `rho = 0`. Each AR candidate is
fitted by Cholesky-whitened GLS. At each spatial location, restricted
likelihood selects a group independently. Locations in a group share the
normalized coefficient covariance, and the residual variance supplies the
location-specific scale.

For the joint independent design `D`, the conditional residual DOF and the
coefficient covariance are:

```text
nu = number_of_retained_frames - rank(D)
sigma_squared = whitened_residual_sum_of_squares / nu
Cov(beta) = sigma_squared * inverse(D.T @ inverse(V_R) @ D)
```

The stored coefficient mapping converts this covariance back to the named
scientific effects. Linear combinations that are not estimable are omitted.
Nuisance rank protection does not impose a minimum fit DOF. A short run that
is otherwise identifiable can be fitted with few remaining dimensions.

## Contrasts and within-participant summaries

Every summary keeps its linear coefficients on the original run estimates.
Session and subject summaries both pool original runs directly, and session
summaries are not inputs to subject summaries. The presence of sessions
therefore does not change the default relative weights of the runs.

Two weightings are available. Equal weighting takes the arithmetic mean of the
available inputs for each effect. Precision weighting normalizes the inverse
estimated marginal variances at each location. Zero or nonfinite variances
receive no precision weight, and a location with no valid contribution has
undefined t and DOF.

When a condition is absent from a run, that run supplies no estimate and no
weight for the condition. A subtraction computed at the run level therefore
uses only runs where both effects are estimable. A subtraction computed after
pooling each condition can use different run sets for its two terms. These are
different estimands.

For a final contrast, let `g_r` collect all of its coefficients on original run
`r`, including pooling weights. Assuming errors are independent between runs:

```text
effect = sum_r(g_r @ beta_r)
v_r = g_r @ Cov(beta_r) @ g_r.T
variance = sum_r(v_r)
nu_summary = variance**2 / sum_r(v_r**2 / nu_r)
t = effect / sqrt(variance)
```

Combining coefficients before computing `v_r` keeps the covariance between
conditions that share a run. A run counts as one independent variance
contribution, even when it reaches a summary through several effects. Pooling
the same source run twice as independent inputs is rejected.

The DOF formula is a
[Satterthwaite approximation](https://doi.org/10.2307/3002019). For one run it
reduces to that run's residual DOF. Disjoint sets of runs have no
cross-covariance under the independent-runs assumption. This is within-subject
inference about pooled effects. It does not estimate between-run random effects
and does not support population claims.

If every condition occurs in every run with the same fixed run weights, linear
contrasts commute with aggregation. nro tests that effects, variances,
t-values, and DOF agree for general linear combinations, including unequal
fixed run weights. Condition-specific precision weights need not commute.

Nested equal-weight session summaries weight sessions equally, which need not
weight every original run equally. To request equal run weighting, connect a
Subject node directly to Run outputs. See the
[first-level model guide](../task-models.md) for how nodes and edges are
declared.

## Inference limits and validation

GLS t inference is exact when the Gaussian noise covariance is fixed and
correctly specified. Estimated AR groups make the reference distribution
approximate, and precision-weighted summaries also condition on estimated
weights. The code does not claim that subtracting one extra DOF makes
estimated-AR inference exact. Voxelwise AR(1) may also miss temporal structure
in real fMRI data.

A synthetic regression check uses 160 frames, censors every thirteenth frame,
sets the true AR coefficient to 0.6, and fits 3,000 independent null
locations. It checks both fixed-covariance and estimated-AR inference. The
estimated-AR test has a wider tolerance that catches large regressions but does
not certify nominal error control. Confirmatory use needs more extensive
simulation and residual diagnostics on real data.

Other tests compare OLS and gap-aware fixed-covariance GLS against direct
matrix calculations. They also cover nuisance-adjusted coefficients, rank
safeguards, missing conditions, shared-run covariance, nested summaries, zero
variance, volume and surface publication, and resumption after an output
disappears.

nro performs no spatial multiple-comparison correction.
