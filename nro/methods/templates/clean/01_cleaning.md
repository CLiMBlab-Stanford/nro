---
section: Nuisance regression and filtering
context_version: 1
requires: [observed, runs, participants, spaces, smoothing_mm, min_trs, gm_mask_threshold, default_temporal_mask, default_confounds, high_pass_hz, low_pass_hz, regress_out_task, detrend, nuisance_variance_explained, minimum_temporal_rank, minimum_temporal_rank_fraction, standardize]
---
{% if observed %}
Denoising was applied to {{ runs }} {{ runs | plural("run") }} from
{{ participants }} {{ participants | plural("participant") }} in the
{{ spaces | join_and }} {{ spaces | length | plural("space") }} at
{{ smoothing_mm | join_and }} mm full width at half maximum (FWHM) smoothing.
{% endif %}
Runs with fewer than {{ min_trs }} frames were excluded. When the smoothing FWHM
was positive, data were smoothed with Connectome Workbench
[@marcus2011workbench] in the surface or volume geometry before temporal
modeling, and volumetric smoothing was restricted to a gray-matter mask
thresholded at {{ gm_mask_threshold | num }}.

Frames marked by {% if default_temporal_mask %}any outlier column{% else %}the configured outlier columns{% endif %}
were censored: they were excluded from model fitting, and their output values
are model reconstructions. Nuisance regressors were
{% if default_confounds %}the mean cerebral-cortex signal, averaged over FreeSurfer cortical voxels after scaling each voxel to unit variance, FD, the six rigid-body motion parameters, the first five anatomical CompCor components, and the cosine drift terms{% else %}a configured selection of confound columns, recorded in the methods trace{% endif %}.
A real Fourier basis spanning {{ high_pass_hz | num }} to {{ low_pass_hz | num }} Hz
was constructed{% if regress_out_task %}, together with task regressors where events were available{% endif %}{% if detrend %} and detrending terms{% endif %}.
On retained frames, nuisance regressors were projected into this passband,
orthogonalized with respect to the task and detrending terms, standardized, and
reduced by principal component analysis to the fewest components explaining
{{ nuisance_variance_explained | pct }} of their variance. At least
{{ minimum_temporal_rank }} dimensions and {{ minimum_temporal_rank_fraction | pct }}
of the passband rank were left after nuisance removal, and nuisance regression
was skipped when fewer than {{ minimum_temporal_rank }} dimensions remained
before it. Each BOLD series was fitted to the remaining passband basis on
retained frames and reconstructed at every frame{% if standardize %}, then standardized with the mean and standard deviation of retained frames{% endif %}.
