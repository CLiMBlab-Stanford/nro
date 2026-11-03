---
section: Susceptibility distortion correction
context_version: 1
requires: [observed, runs, sdc.requested, sdc.from_sbref_pair, sdc.fieldmap_syn_refine, sdc.topup, sdc.synbold, sdc.syn, sdc.fallback_runs]
---
Susceptibility distortion was estimated with the first applicable method: FSL
TOPUP [@andersson2003topup] with an opposite-phase-encoding
{% if sdc.from_sbref_pair %}SBRef{% else %}spin-echo fieldmap{% endif %} pair;
{% if sdc.requested == "synbold_disco" %}
otherwise SynBOLD-DisCo [@yu2023synbold], which synthesizes an undistorted BOLD
reference from the T1w image for TOPUP;
{% endif %}
and otherwise nonlinear registration of the BOLD reference to the T1w anatomy
with ANTs SyN [@avants2008syn].
{% if sdc.fieldmap_syn_refine %}
Fieldmap-based corrections were further refined with ANTs SyN.
{% endif %}
{% if observed %}
Of the {{ runs }} runs, {{ sdc.topup }} used TOPUP with acquired references,
{{ sdc.synbold }} used SynBOLD-DisCo, and {{ sdc.syn }} used anatomical SyN.
{% if sdc.fallback_runs %}
For {{ sdc.fallback_runs }} {{ sdc.fallback_runs | plural("run") }}, the requested
method could not be used, and the recorded reason is given in Supplementary
Table S1.
{% endif %}
{% endif %}
