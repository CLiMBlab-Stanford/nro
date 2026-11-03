---
section: Anatomical registration
context_version: 1
requires: [bbregister.surface, bbregister.init, bbregister.dof]
---
The distortion-corrected BOLD reference was aligned to the T1w anatomy with
FreeSurfer boundary-based registration [@greve2009bbr], using the
{{ bbregister.surface }} surface, {{ bbregister.init }} initialization, and
{{ bbregister.dof }} degrees of freedom.
