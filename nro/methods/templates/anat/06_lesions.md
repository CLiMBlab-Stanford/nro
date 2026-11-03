---
section: Lesion-aware anatomy
context_version: 1
when: "counts.lesion > 0"
requires: [counts.lesion]
---
For the {{ counts.lesion }} {{ counts.lesion | plural("participant") }} marked as
having a structural lesion, a lesion mask was estimated on the bias-corrected
T1w image with SynthStroke [@chalcroft2025synthstroke]. Surfaces were
reconstructed from a complete synthetic T1w scaffold generated with SynthSR
[@iglesias2023synthsr]. The lesion mask was projected through the cortical
ribbon, and every surface triangle touching the lesion was removed from the
published surfaces.
