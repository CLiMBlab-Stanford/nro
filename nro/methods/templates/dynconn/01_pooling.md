---
section: Connectivity pooling
context_version: 1
requires: [observed, participants, included_runs, excluded_runs, fitted_dimensions, input_filter, minimum_retained_frames, minimum_retained_fraction, minimum_residual_design_dof, minimum_participation_effective_rank, maximum_dominant_temporal_variance_fraction, minimum_usable_runs, minimum_aggregate_retained_frames, precision_weighting, low_rank, dimensions, oversampling, power_iterations]
---
Cleaned runs{% if input_filter %} that matched a configured entity filter{% endif %}
were pooled within each participant, space, and smoothing level. A run was
included if cleaning was defined, at least {{ minimum_retained_frames }} frames
and {{ minimum_retained_fraction | pct }} of its frames were retained, the
cleaning model left at least {{ minimum_residual_design_dof }} residual degrees
of freedom, the participation effective rank of the cleaned data was at least
{{ minimum_participation_effective_rank | num }}, and no single temporal
component explained more than
{{ maximum_dominant_temporal_variance_fraction | pct }} of its variance. Each
participant required at least {{ minimum_usable_runs }} usable
{{ minimum_usable_runs | plural("run") }} and {{ minimum_aggregate_retained_frames }}
retained frames in total.
{% if observed %}
These criteria admitted {{ included_runs }} {{ included_runs | plural("run") }}
from {{ participants }} {{ participants | plural("participant") }} and excluded
{{ excluded_runs }}.
{% endif %}

Censored frames were omitted, and each admitted run was centered and
standardized at every spatial location before pooling.
{% if precision_weighting %}
Each run's contribution to the pooled covariance was weighted in proportion to
the temporal degrees of freedom remaining after cleaning.
{% else %}
All admitted runs contributed equally to the pooled covariance.
{% endif %}
{% if low_rank %}
The leading eigenspace of the weighted, pooled correlation matrix was estimated
with a streamed randomized decomposition [@halko2011randomized] of at most
{{ dimensions }} dimensions, with an oversampling of {{ oversampling }} and
{{ power_iterations }} power {{ power_iterations | plural("iteration") }}, without
forming the full location-by-location matrix.
{% if observed and fitted_dimensions %}
The fitted rank was {{ fitted_dimensions | join_and("or") }}.
{% endif %}
The fit was represented as a short synthetic series whose sample correlations
reproduce the fitted low-rank correlation matrix, for seed-based connectivity
in Connectome Workbench [@marcus2011workbench].
{% else %}
The retained frames of all admitted runs were concatenated in acquisition order
for seed-based connectivity in Connectome Workbench [@marcus2011workbench].
{% endif %}
