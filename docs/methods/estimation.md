# Temporal projection and streaming estimators

This page specifies the estimators used by [clean](../modules/clean.md) and
[microparcellation](../modules/microparcellation.md). Symbols without a row
subscript contain all original frames. Subscript $R$ selects retained frames.
No equation treats censored reconstructions as observed data.

## Cleaning as a constrained fit

Let $Y$ be the frame-by-location input matrix and $B$ the real Fourier basis
inside the requested passband. Frequencies lie on the grid determined by the
original frame count and TR. Censoring selects rows $B_R$; it does not construct
a new evenly sampled time axis. With filtering enabled, require that $B_R$
identify all requested basis coefficients. A rank-deficient retained basis
produces an undefined-cleaning sentinel.

Construct a full-length basis $U$ spanning $B$ whose retained rows $U_R$ are
orthonormal. Projection of any full-length regressor matrix $X$ into the
estimable passband is then:

$$X_{\mathrm{BP}} = U U_R^T X_R.$$

Exact terms include an intercept, optional linear detrending, and task
regressors. Task regressors use Nilearn's SPM HRF on event onset/duration and
one indicator per `trial_type`; absent trial types become `task`. Events missing
onset/duration do not generate task regressors. The output sidecar records the
actual design, so do not infer successful task removal merely from a config flag.

Make projected nuisance regressors orthogonal to the independent exact terms
on retained frames. Scale each by retained RMS rather than subtracting another
full-length constant, which could leave the passband. SVD supplies nuisance
PCs ordered by squared singular values. Choose enough components to reach the
configured variance fraction, subject to:

$$k \le \max\{0,\ d-\max(a,\lceil f d\rceil)\},$$

where $d$ is rank after exact-term removal, $a$ the minimum absolute rank, and
$f$ the protected fraction. Defaults are variance target 0.99, $a=30$, and
$f=0.5$. When $d<a$, no nuisance PC is removed. Metadata reports whether the
variance target was reached or the rank cap prevented it.

Let $D$ contain the orthonormal exact and selected nuisance directions. Let
$Q$ span the complement of $U_R^T D_R$ in passband coordinates, and set
$C=UQ$. The cleaned signal before optional standardization is:

$$Y_{\mathrm{clean}}=C C_R^T Y_R.$$

This computes the clean result directly; it does not require materializing an
intermediate bandpassed image. $C_R$ is orthonormal, so its column count is the
model's final algebraic temporal rank. It is not the observed signal's effective
rank: actual data can concentrate variance in fewer directions.

With filtering disabled, the implementation uses retained-frame nuisance
regression and evaluates full-length residuals instead of Fourier
reconstruction. With standardization enabled, the retained output supplies
location/scale for every output frame.

Some invalid inputs still raise errors rather than producing sentinels: an
all-censored run, mismatched frame counts, or an original frame count below
`min_trs` cannot follow the normal cleaning path. Undefined-model sentinels
specifically describe the unidentifiable passband/exact-design cases handled
by the projection constructor.

## Pooled connectivity

For run $r$, let $Z_r$ contain retained signals after optional
connectivity-stage global-signal regression. Center and scale each valid
location so that its temporal sum of squares is one. Let $\nu_r$ be the
algebraic temporal rank reported by `clean`, minus one when connectivity-stage
global-signal regression is used. Define normalized run weights:

$$
\alpha_r=
\begin{cases}
1/R & \text{for equal weighting},\\
\nu_r/\sum_s\nu_s & \text{for precision weighting}.
\end{cases}
$$

Scale each run as $A_r=\sqrt{\alpha_r}Z_r$ and accumulate:

$$G=\sum_r A_r^T A_r,\qquad
\rho_{ij}=\frac{G_{ij}}{\sqrt{G_{ii}G_{jj}}}.$$

Local coarsening computes this only for spatial-neighbor edges. Final parcel
connectivity computes the full Gram matrix in a fixed number of accumulators.
Row-blocked matrix products avoid materializing a separate connectome for each
run. Only one triangle is computed, and its values are mirrored so the
accumulator is exactly symmetric. The two split-half accumulators remain in
memory at float32 precision.
Runs are standardized independently. Precision weighting is linear in
effective DOF because covariance sampling variance scales inversely with that
quantity. It describes temporal information, not freedom from artifact or
measurement noise. Run admission handles known quality failures before
weighting. Undefined spatial variance receives zero support; the saved diagonal
is zeroed for downstream graph construction. The estimator pools correlations
directly and does not apply a Fisher transform.

The public pconn uses int8 quantization with a scale of $1/127$. Thus finite
correlations are represented approximately; percentile threshold ties can
retain slightly more than the requested tail fraction. Network loading handles
this quantized representation directly.

## Variance preserved by parcels

For standardized source signal $Z_{ti}$ and parcel mean $M_{t,p(i)}$, the
source-resolution score is:

$$1-\frac{\sum_{r,t,i}(Z_{rti}-M_{rt,p(i)})^2}
{\sum_{r,t,i}Z_{rti}^2}.$$

Means use valid locations in each parcel. The implementation accumulates
source sum of squares and preserved parcel-mean sums of squares, avoiding a
reconstructed source-resolution array. This score concerns temporal signal
variance, not graph spectral approximation error or network-label accuracy.
Null partitions use the same source data and scoring; only spatial assignments
change. Five nulls provide a descriptive baseline, not a well-resolved tail
probability. Seeds, growth attempts, and size mismatch are recorded.

## Connectome diagnostics

Run contributions include retained frames, effective DOF, normalized run
weight, Gram trace/Frobenius norm, and relative diagonal weight. Parcel-support
summaries report the number and weight-equivalent number of contributing runs.
Histogram summaries describe unique off-diagonal weights, while power iteration
approximates the dominant eigenvalue fraction. Participation rank is based on
trace squared over squared Frobenius norm, accounting for valid
self-correlations.
Per-run Gram trace and Frobenius norm are computed through the temporal dual
matrix. This gives the same quantities without retaining a run-level parcel
matrix.

For split-half comparison, assign complete runs greedily to balance retained
frame counts in input order. A single run uses alternating contiguous temporal
blocks instead, each containing `quality.split_half_block_frames` retained
frames (default 128), capped at half the run length. A final shorter block
stays in its assigned half. Streaming chunk sizes do not determine this split.
The two accumulated matrices yield edge correlation,
Spearman–Brown correction $2r/(1+r)$, mean absolute difference, and RMS difference.
These are a single diagnostic split, not bootstrap confidence intervals.
High reliability can reflect shared artifact; interpret it together with
motion, effective rank, and dominant-variance metadata.
