---
section: Gradient correction and head motion
context_version: 1
requires: [observed, runs, gradient_unwarping.mode, gradient_unwarping.applied_runs, sbref.selected_runs]
---
{% if gradient_unwarping.mode != "off" %}
BOLD, single-band reference (SBRef), and spin-echo images were corrected for
gradient nonlinearity with the HCP gradient-unwarping procedure
[@glasser2013minimal] when a site gradient profile matched the acquisition and
its metadata did not report prior correction{% if gradient_unwarping.applied_runs %} ({{ gradient_unwarping.applied_runs | count_of(runs) }} runs){% endif %}.
The BOLD displacement field was applied within the single final resampling.
{% endif %}
Head motion was estimated with FSL MCFLIRT [@jenkinson2002mcflirt].
{% if observed and not sbref.selected_runs %}
All runs used a BOLD-derived registration reference, because no matched SBRef
was available or passed the metadata, pose, overlap, and similarity checks.
{% elif observed %}
A matched SBRef that passed metadata, pose, overlap, and similarity checks served
as the registration reference for {{ sbref.selected_runs | count_of(runs) }} runs;
the remaining runs used a BOLD-derived reference.
{% else %}
A matched SBRef served as the registration reference when it passed metadata,
pose, overlap, and similarity checks; otherwise a BOLD-derived reference was used.
{% endif %}
