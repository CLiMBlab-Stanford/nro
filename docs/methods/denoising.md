# Denoising and temporal censoring

This note records methodological decisions that should remain traceable to their
motivation and sources. It describes the intended implementation; until the
corresponding code change lands, it must not be treated as a description of
existing outputs.

## Motion outliers

Each run will contain three independently numbered families of one-hot columns:

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
Afyouni and Nichols [@afyouni2018dvars]. They noted that the 5% practical cutoff
worked adequately in the HCP data they studied and might require recalibration
for other data sources, so the threshold must remain configurable and should be
validated on this project's data.

For connectivity, the temporal mask is the row-wise union of
`non_steady_state_outlierNN` and `motion_outlierNN`. The reason-specific FD and
DVARS columns are retained for provenance and quality control. This mask is
logically identical to the row-wise union of every `*_outlierNN` column; the
reason-specific columns cannot add a frame that is absent from
`motion_outlierNN`. A GLM should use the union columns, rather than all three
motion families together, to avoid duplicating scan-nulling regressors when a
frame meets both criteria.

## Confound regression

Cleaning uses the 36-parameter model described by Satterthwaite et al.
[@satterthwaite2013confound]. Its nine base signals are the six rigid-body
motion parameters, mean white-matter signal, mean cerebrospinal-fluid signal,
and global signal. The model contains those nine signals, their first temporal
derivatives, the squares of the nine signals, and the squares of their
derivatives. Framewise displacement is used to identify extreme motion but is
not an additional continuous nuisance regressor.

For connectivity, nuisance regression and bandpass filtering are performed
together as a linear projection. The design contains the configured confounds,
task regressors, detrending terms, and real Fourier bases for frequencies
outside the configured passband. Coefficients are fitted only to retained
frames, following the censor-aware projection used by AFNI's default
`3dTproject -cenmode KILL` approach [@afni3dtproject]. Model predictions are
then evaluated at every original frame from that frame's design values and the
retained-frame coefficients. Subtracting these predictions produces a finite,
full-length time series without treating censored BOLD measurements as evidence
during model fitting.

This masked-fit, full-length evaluation preserves the original time axis and
avoids interpolation. Retained values match the corresponding censor-aware
projection; values at censored frames are finite residuals, not repaired data.
Post-projection location and scale are estimated from retained frames and
applied to the complete output. Downstream analyses remain responsible for
applying the temporal mask. Only retained frames count as observations or
degrees of freedom.

## Run usability

A run is usable for connectivity only if it retains at least 100 frames, at
least 70% of its original frames, and at least 50 residual design degrees of
freedom before temporal filtering. Its retained parcel time series must also
have participation effective rank of at least 10, with no single component
accounting for more than 50% of the variance. Aggregated connectivity requires
at least 1,100 retained frames and at least two usable runs.

## References

The citation records used here are in [`../references.bib`](../references.bib).
