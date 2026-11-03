---
section: Multimodal surface matching
context_version: 1
when: "counts.msmall > 0"
requires: [counts.msmall]
---
For {{ counts.msmall }} {{ counts.msmall | plural("participant") }}, MSMAll
surface registration [@robinson2014msm; @robinson2018msmall] was computed with
the HCP Pipelines [@glasser2013minimal] from resting-state runs selected for
calibration, then transferred to the participant's native surface topology.
