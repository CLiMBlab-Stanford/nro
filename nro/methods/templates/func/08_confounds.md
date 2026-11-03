---
section: Confounds and outlier frames
context_version: 1
requires: [confounds.fd_radius_mm, confounds.n_acompcor, confounds.cosine_high_pass_hz, confounds.cosine_period_s, confounds.fd_threshold_mm, confounds.dvars_alpha, confounds.dvars_practical_percent, confounds.nonsteady_max_vols]
---
Confound time series included the six rigid-body motion parameters; global,
white-matter, and cerebrospinal-fluid signals; the temporal derivatives and
quadratic terms of these signals [@satterthwaite2013confound]; framewise
displacement (FD) [@power2012spurious] computed with a
{{ confounds.fd_radius_mm | num }} mm head radius; {{ confounds.n_acompcor }}
anatomical CompCor components from combined white-matter and
cerebrospinal-fluid voxels [@behzadi2007compcor]; DVARS; and discrete cosine
drift regressors with a {{ confounds.cosine_high_pass_hz | num(7) }} Hz
({{ confounds.cosine_period_s | num(0) }} s) cutoff.

Frames were marked as outliers when FD exceeded
{{ confounds.fd_threshold_mm | num }} mm or when DVARS met both the statistical
criterion (one-sided test at a {{ confounds.dvars_alpha | pct }}
Bonferroni-corrected family-wise error rate) and the practical criterion
(change in percent D-var greater than
{{ confounds.dvars_practical_percent | num }}%) of @afyouni2018dvars. Up to
{{ confounds.nonsteady_max_vols }} initial non-steady-state volumes were also
marked.
