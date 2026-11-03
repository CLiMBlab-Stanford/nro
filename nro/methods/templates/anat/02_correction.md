---
section: Intensity and gradient correction
context_version: 1
requires: [gradient_unwarping.mode, gradient_unwarping.applied_images, gradient_unwarping.total_images]
---
{% if gradient_unwarping.mode == "off" %}
Gradient-nonlinearity correction was not applied.
{% else %}
Each source image was corrected for gradient nonlinearity with the Human
Connectome Project (HCP) gradient-unwarping procedure [@glasser2013minimal]
when a site gradient profile matched the acquisition and its metadata did not
report prior correction{% if gradient_unwarping.applied_images %} ({{ gradient_unwarping.applied_images | count_of(gradient_unwarping.total_images) }} source images were corrected){% endif %}.
{% endif %}
A brain mask was estimated on each image with SynthStrip [@hoopes2022synthstrip],
and intensity nonuniformity was corrected within that mask with ANTs
N4BiasFieldCorrection [@tustison2010n4itk].
