---
section: Spatial normalization
context_version: 1
requires: [mni_space]
---
The T1w reference was registered nonlinearly to the {{ mni_space }} template
[@fonov2011mni] with ANTs [@avants2011ants]. Rigid and affine stages used a
mutual-information metric, and a final symmetric normalization (SyN) stage
[@avants2008syn] used a cross-correlation metric. All stages used brain masks.
