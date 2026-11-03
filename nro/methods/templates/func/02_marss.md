---
section: Simultaneous-slice artifact correction
context_version: 1
when: "marss.mode != 'off'"
requires: [observed, runs, marss.mode, marss.min_multiband_factor, marss.applied_runs]
---
{% if marss.mode == "diagnose" %}
Excess correlation among simultaneously acquired slices was measured
[@tubiolo2024marss] but not corrected.
{% else %}
Signal shared among simultaneously acquired slices was estimated and removed in
native scanner space with Multiband Artifact Regression in Simultaneous Slices
(MARSS) [@tubiolo2024marss] for runs with a multiband acceleration factor of
{{ marss.min_multiband_factor }} or greater.
{% if observed and not marss.applied_runs %}
No run met this criterion, so no run was corrected.
{% else %}
{% if observed %}
This applied to {{ marss.applied_runs | count_of(runs) }} runs, and the
{% else %}
The
{% endif %}
correction preceded all other processing.
{% endif %}
{% endif %}
