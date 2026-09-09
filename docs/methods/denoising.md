# Denoising and temporal censoring

This note records methodological decisions that should remain traceable to their
motivation and sources.

## Motion outliers

Each run contains three independently numbered families of one-hot columns:

- `extreme_fd_outlierNN` marks frames whose Power framewise displacement is
  greater than 1 mm.
- `dvars_outlierNN` marks frames that pass both Afyouni and Nichols' statistical
  and practical criteria: an upper-tail DVARS test significant at a 5%
  Bonferroni family-wise error rate within the run, and a change in percent
  D-var greater than 5%.
- `motion_outlierNN` marks the unique union of the preceding two families.

The two-part DVARS rule is intentional. Statistical significance alone can flag
negligible changes in high-dimensional data, while an unstandardized fixed
DVARS cutoff is sensitive to signal scaling and the run's baseline variability.
Requiring both statistical and practical significance follows the proposal of
Afyouni and Nichols ([2018](https://doi.org/10.1016/j.neuroimage.2017.12.098)). They noted that the 5% practical cutoff
worked adequately in the HCP data they studied and might require recalibration
for other data sources. These cutoffs should be validated on this project's data.

For connectivity, the temporal mask is the row-wise union of
`non_steady_state_outlierNN` and `motion_outlierNN`. The reason-specific FD and
DVARS columns are retained for provenance and quality control. This mask is
logically identical to the row-wise union of every `*_outlierNN` column; the
reason-specific columns cannot add a frame that is absent from
`motion_outlierNN`. First-level modeling uses the same union to select retained
rows. It does not fit the reason-specific and union columns as duplicate
scan-nulling regressors.

## Confound regression

Cleaning begins with the 36-parameter model described by Satterthwaite et al.
([2013](https://doi.org/10.1016/j.neuroimage.2012.08.052)). Its nine base signals are the six rigid-body
motion parameters, mean white-matter signal, mean cerebrospinal-fluid signal,
and global signal. The model contains those nine signals, their first temporal
derivatives, the squares of the nine signals, and the squares of their
derivatives. Framewise displacement is used to identify extreme motion but is
not an additional continuous nuisance regressor.

Cleaning constructs a real Fourier basis for frequencies inside the configured
passband. Task regressors and nuisance regressors are projected into that same
basis using only retained frames. Task and detrending terms remain explicit.
The nuisance design is first made orthogonal to those exact terms, standardized,
and reduced by principal-component analysis to the smallest number of components
that reaches the configured variance target (99% by default). Component retention
is capped so that at least 30 dimensions and at least 50% of the post-task
passband rank remain. Both protections are configurable. If fewer than 30
dimensions remain before nuisance regression, nuisance regression is skipped.

The task and retained nuisance-PC directions are removed from the passband basis.
Each BOLD series is fitted directly to the remaining clean-passband basis using
only retained frames, and that fitted model is evaluated at every original
frame. This avoids the rank saturation that can occur when a large stopband
basis is fitted on an irregular, censored grid. It also guarantees the stated
passband and exact task removal rather than allowing PCA to trade either away.

This censored-fit, full-length reconstruction preserves the original time axis.
Values at censored frames are finite model reconstructions, not observed or
repaired data, and downstream analyses remain responsible for applying the
temporal mask. Post-cleaning location and scale are estimated from retained
frames and applied to the complete output.

Each cleaned time-course sidecar reports the temporal-mask burden; retained
duration and longest censored interval; passband dimension, rank, and condition
number; exact-design rank; nuisance columns and retained PCA components; and the
joint regression rank, residual design degrees of freedom, and final algebraic
temporal rank. It also reports numerical and effective ranks of the cleaned
data, the dominant temporal variance fraction, and the number and selection
method of spatial locations used for those data-derived estimates. These are
descriptive measurements. The cleaning module does not itself assign a
run-level pass/fail status from them.

If censoring makes the passband basis unidentifiable, or if the exact task and
detrending design consumes the complete estimable passband, cleaning is
mathematically undefined. The module still preserves its output contract: it
writes an all-zero time course and marks `CleaningDefined` false in its sidecar,
with a machine-readable `CleaningUndefinedReason`. Such an output is a sentinel,
not usable functional data. Downstream modules must exclude it and record that
exclusion in their own metadata.

## Run usability

A run is usable for connectivity only if it retains at least 100 frames, at
least 70% of its original frames, and at least 50 residual design degrees of
freedom reported by the clean model. The clean sidecar must also report
participation effective rank of at least 10, with no single component
accounting for more than 50% of the variance.

The `main` microparcellation configuration additionally requires at least two
usable runs and 1,100 retained frames in aggregate. The `main` dynconn
configuration requires one usable run and 100 retained frames in aggregate.
Module configuration is the authority for these thresholds.

## References

The citation records used here are in [`../references.bib`](../references.bib).
