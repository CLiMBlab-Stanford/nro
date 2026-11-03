---
section: Resampling
context_version: 1
requires: [resampling.tool, resampling.data_interpolation, resampling.warp_interpolation, resampling.output_grid, resampling.use_jacobian]
---
Per-frame motion, distortion, and anatomical transforms were composed and
applied in a single interpolation with AFNI *3dNwarpApply* [@cox1996afni],
using {{ resampling.data_interpolation }} interpolation for the signal and
{{ resampling.warp_interpolation }} interpolation for the warps. Template-space
outputs used the T1w-to-template transforms from anatomical preprocessing.
Volumes were written on
{% if resampling.output_grid == "t1_epi_vox" %}an anatomically oriented grid at the BOLD voxel size{% else %}the anatomical grid{% endif %}{% if resampling.use_jacobian %} with Jacobian intensity modulation{% endif %}.
Surface time series were sampled within the cortical ribbon between the white
and pial surfaces with Connectome Workbench [@marcus2011workbench].
