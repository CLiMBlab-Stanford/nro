---
section: Individualized network estimation
context_version: 1
requires: [observed, participants, spaces, smoothing_mm, low_rank_source, strategy, n_networks, maximum_dimensions, reduction_oversampling, reduction_power_iterations, ica_max_iterations, ica_tolerance, ica_upper_quantile, clustering_repetitions, clustering_n_init, clustering_max_iterations, clustering_batch_size, clustering_max_no_improvement, clustering_reassignment_ratio]
---
{% if observed %}
Individualized networks were estimated for {{ participants }}
{{ participants | plural("participant") }} in the {{ spaces | join_and }}
{{ spaces | length | plural("space") }} at {{ smoothing_mm | join_and }} mm FWHM
smoothing.
{% endif %}
Each participant's pooled connectivity series formed a matrix with one row per
spatial location{% if low_rank_source %} and one column per fitted low-rank dimension{% endif %}.
When it had more than {{ maximum_dimensions }} columns, it was projected onto its
leading {{ maximum_dimensions }} randomized singular vectors
[@halko2011randomized] with an oversampling of {{ reduction_oversampling }} and
{{ reduction_power_iterations }} power
{{ reduction_power_iterations | plural("iteration") }}.

{% if strategy in ("ica", "ica_correlation") %}
Spatial independent component analysis with FastICA [@hyvarinen1999fastica],
as implemented in scikit-learn [@pedregosa2011sklearn], decomposed this matrix
into {{ n_networks }} components, treating locations as observations, with
unit-variance whitening, at most {{ ica_max_iterations }} iterations, and a
tolerance of {{ ica_tolerance | num(6) }}. Each component was oriented so that
its median loading was nonpositive.
{% if strategy == "ica_correlation" %}
Each network map was the correlation between every location and the oriented
component's time course, signed to agree with the component's loadings. The
correlations were not rescaled or thresholded.
{% else %}
Its loadings were then clipped between
zero and their {{ ica_upper_quantile | num }} quantile across locations and
divided by that quantile, giving a membership map bounded by zero and one.
{% endif %}
These maps are not calibrated probabilities and are not normalized across
networks.
{% else %}
Following @shain2026language, mini-batch k-means [@sculley2010minibatch], as
implemented in scikit-learn [@pedregosa2011sklearn], partitioned locations into
{{ n_networks }} networks in {{ clustering_repetitions }} independent fits with
different random seeds ({{ clustering_n_init }}
{{ clustering_n_init | plural("initialization") }} per fit, batch size
{{ clustering_batch_size }}, at most {{ clustering_max_iterations }} iterations,
early stopping after {{ clustering_max_no_improvement }} batches without
improvement, and a reassignment ratio of
{{ clustering_reassignment_ratio | num }}). The fit with the lowest inertia
defined the network identities. Every other partition was matched one-to-one to
the accumulated memberships by spatial correlation with the Hungarian algorithm
[@kuhn1955hungarian]. The matched assignments were averaged across fits, and
each network's map was min-max normalized.
{% endif %}
