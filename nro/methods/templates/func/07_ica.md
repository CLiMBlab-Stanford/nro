---
section: ICA-based denoising
context_version: 1
when: "ica.classifier != 'none'"
requires: [ica.classifier, ica.regression]
---
Spatial independent component analysis was computed with FSL MELODIC
[@beckmann2004melodic] on the spatially smoothed T1w-space series. Noise
components were classified with
{% if ica.classifier == "ica_aroma" %}ICA-AROMA [@pruim2015aroma]{% else %}CICADA [@dodd2025cicada]{% endif %}
and removed from every output-space series by {{ ica.regression }} regression.
