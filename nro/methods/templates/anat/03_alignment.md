---
section: Reference alignment
context_version: 1
requires: [observed, participants, mni_space, counts.t2w]
---
The T1w reference was rigidly aligned to the pose of the {{ mni_space }}
template [@fonov2011mni] with ANTs mutual-information registration
[@avants2011ants]. It was resampled once with cubic B-spline interpolation onto
a template-oriented grid at the source resolution, and its brain mask was
resampled with nearest-neighbor interpolation.
When a T2w reference was available{% if observed %} ({{ counts.t2w | count_of(participants) }} participants){% endif %},
it was aligned to the T1w reference with six-degree-of-freedom FSL FLIRT
[@jenkinson2001flirt].
