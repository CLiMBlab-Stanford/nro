---
section: Anatomical references
context_version: 1
requires: [observed, participants, selection_strategy, t1w_fallback, counts.multiple_t1w, counts.synthetic_t1w]
---
{% if observed %}
Anatomical data were processed for {{ participants }} {{ participants | plural("participant") }}.
{% endif %}
For each participant, one T1-weighted (T1w) and one T2-weighted (T2w) reference
were selected from all available sessions.
{% if selection_strategy == "first" %}
The earliest acquisition of each modality was used.
{% else %}
When a participant had several acquisitions of a modality, they were combined
into an unbiased within-participant template with FreeSurfer
*mri_robust_template* [@reuter2010robust]{% if counts.multiple_t1w %} ({{ counts.multiple_t1w | count_of(participants) }} participants had more than one T1w image){% endif %}.
{% endif %}
{% if t1w_fallback == "synthesize_from_t2w" %}
When no T1w image was available, a 1 mm T1w-like reference was synthesized from
the T2w image with SynthSR [@iglesias2023synthsr]{% if counts.synthetic_t1w %}; this applied to {{ counts.synthetic_t1w | count_of(participants) }} participants{% endif %}.
{% else %}
Participants without a T1w image were not processed.
{% endif %}
