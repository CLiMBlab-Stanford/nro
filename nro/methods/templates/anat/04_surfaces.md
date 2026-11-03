---
section: Surface reconstruction
context_version: 1
requires: [engine, fsaverage_template, fsaverage_vertices]
---
{% if engine == "fastsurfer" %}
Cortical surfaces were reconstructed from the T1w reference with FastSurfer
[@henschel2020fastsurfer], using FastSurferVINN segmentation at 1 mm
[@henschel2022fastsurfervinn] followed by FreeSurfer-compatible surface
reconstruction.
{% else %}
Cortical surfaces were reconstructed from the T1w reference with FreeSurfer
*recon-all* [@dale1999cortical; @fischl2012freesurfer]. FreeSurfer skull
stripping was replaced by the SynthStrip brain mask, resampled to FreeSurfer's
conformed grid.
{% endif %}
Gray-matter, cortical-ribbon, and subcortical masks were derived from the
FreeSurfer segmentation labels. White, pial, midthickness, and inflated surfaces
and spherical registrations were exported to GIFTI with Connectome Workbench
[@marcus2011workbench]. The {{ fsaverage_template }} template geometry
{% if fsaverage_vertices %}({{ fsaverage_vertices | num }} vertices per hemisphere) {% endif %}was
obtained from TemplateFlow [@ciric2022templateflow].
