# Temporal projection and streaming estimators

These are the estimators used by [clean](../modules/clean.md) and
[microparcellation](../modules/microparcellation.md), written out so that
analyses can be reported and checked. [Denoising and temporal
censoring](denoising.md) gives the motivation for the cleaning choices.

Two notation rules apply throughout. A symbol without a row subscript contains
all original frames, and the subscript $R$ selects retained frames. No equation
treats censored reconstructions as observed data.

## Cleaning as a constrained fit

Let $Y$ be the frame-by-location input matrix and $B$ the real Fourier basis
inside the requested passband. The frequencies lie on the grid set by the
original frame count and TR. Censoring selects the rows $B_R$. It does not
build a new evenly sampled time axis. When filtering is enabled, $B_R$ must
identify every requested basis coefficient. A rank-deficient retained basis
produces an undefined-cleaning sentinel.

Construct a full-length basis $U$ that spans $B$ and whose retained rows $U_R$
are orthonormal. Any full-length regressor matrix $X$ is then projected into
the estimable passband as:

$$X_{\mathrm{BP}} = U U_R^T X_R.$$

The exact terms are an intercept, optional linear detrending, and task
regressors. Task regressors use Nilearn's SPM HRF on event onset and duration,
with one indicator per `trial_type`. When the events table has no
`trial_type` column, every event is treated as the single type `task`. Events missing onset or duration do not generate task regressors. The
output sidecar records the design that was actually used, so check it rather
than inferring task removal from a configuration flag.

Projected nuisance regressors are made orthogonal to the independent exact
terms on retained frames. Each is then scaled by its retained RMS. nro does not
subtract another full-length constant, because that constant could fall
outside the passband. An SVD supplies nuisance PCs ordered by squared singular
values. nro chooses enough components to reach the configured variance
fraction, subject to:

$$k \le \max\{0,\ d-\max(a,\lceil f d\rceil)\},$$

where $d$ is the rank after exact-term removal, $a$ is the minimum absolute
rank, and $f$ is the protected fraction. The defaults are a variance target of
0.99, $a=30$, and $f=0.5$. When $d<a$, no nuisance PC is removed. The metadata
reports whether the variance target was reached or the rank cap stopped it.

Let $D$ contain the orthonormal exact directions and the selected nuisance
directions. Let $Q$ span the complement of $U_R^T D_R$ in passband
coordinates, and set $C=UQ$. Before optional standardization, the cleaned
signal is:

$$Y_{\mathrm{clean}}=C C_R^T Y_R.$$

This gives the clean result directly, without writing an intermediate
bandpassed image. Because $C_R$ is orthonormal, its column count is the model's
final algebraic temporal rank. That rank is a property of the model. The
observed signal's effective rank can be lower, because real data can
concentrate variance in fewer directions.

When filtering is disabled, nro runs nuisance regression on retained frames and
evaluates full-length residuals instead of a Fourier reconstruction. When
standardization is enabled, location and scale estimated from retained frames
are applied to every output frame.

Some invalid inputs raise errors instead of producing sentinels. These are an
all-censored run, mismatched frame counts, and an original frame count below
`min_trs`. Sentinels are reserved for the cases the projection handles: an
unidentifiable passband, or an exact design that fills it.

## Pooled connectivity

For run $r$, let $Z_r$ contain the retained signals after optional
connectivity-stage global-signal regression. Center and scale each valid
location so that its temporal sum of squares is one. Let $\nu_r$ be the
algebraic temporal rank reported by `clean`, minus one when connectivity-stage
global-signal regression is used. The normalized run weights are:

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

Each run is standardized on its own. Precision weighting is linear in effective
degrees of freedom (DOF), because the sampling variance of a covariance scales
inversely with that quantity. The weight measures temporal information. It
says nothing about artifact or measurement noise; run admission removes known
quality failures before weighting. The estimator pools correlations directly
and does not apply a Fisher transform.

Local coarsening computes $\rho$ only for spatial-neighbor edges. Final parcel
connectivity computes the full Gram matrix in a fixed number of accumulators.
Row-blocked matrix products avoid building a separate connectome for each run.
Only one triangle is computed, and it is mirrored so the accumulator is exactly
symmetric. The two split-half accumulators are held in memory at float32
precision. A location with undefined spatial variance receives zero support,
and the saved diagonal is set to zero for downstream graph construction.

The public pconn is quantized to int8 with a scale of $1/127$, so finite
correlations are stored approximately. Ties at a percentile threshold can
therefore retain slightly more than the requested tail fraction. Network
loading reads this quantized form directly.

## Variance preserved by parcels

For standardized source signal $Z_{ti}$ and parcel mean $M_{t,p(i)}$, the
source-resolution score is:

$$1-\frac{\sum_{r,t,i}(Z_{rti}-M_{rt,p(i)})^2}
{\sum_{r,t,i}Z_{rti}^2}.$$

Parcel means use the valid locations in each parcel. The implementation
accumulates the source sum of squares and the sums of squares preserved by
parcel means, so it never reconstructs a source-resolution array. The score
measures temporal signal variance. It does not measure graph spectral
approximation error or network-label accuracy.

Null partitions use the same source data and scoring, and change only the
spatial assignments. Five nulls give a descriptive baseline. They are too few
to resolve a tail probability. Seeds, growth attempts, and size mismatch are
recorded.

## Connectome diagnostics

The per-run contributions are retained frames, effective DOF, normalized run
weight, Gram trace and Frobenius norm, and relative diagonal weight. The Gram
trace and Frobenius norm are computed through the temporal dual matrix, which
gives the same values without keeping a run-level parcel matrix.

Parcel-support summaries report the number of contributing runs and their
weight-equivalent number. Histogram summaries describe the unique off-diagonal
weights. Power iteration approximates the dominant eigenvalue fraction.
Participation rank is the trace squared over the squared Frobenius norm,
accounting for valid self-correlations.

### Split-half reliability

When there are several runs, complete runs are assigned greedily to the two
halves in input order so that retained frame counts balance. A single run is
split into alternating contiguous temporal blocks instead. Each block contains
`quality.split_half_block_frames` retained frames (default 128), capped at half
the run length, and a final shorter block stays in its assigned half. Streaming
chunk sizes do not affect the split.

The two accumulated matrices give the edge correlation, the Spearman–Brown
correction $2r/(1+r)$, the mean absolute difference, and the RMS difference.
These come from a single diagnostic split and are not bootstrap confidence
intervals. Shared artifact can inflate reliability, so interpret it together
with the motion, effective-rank, and dominant-variance metadata.
