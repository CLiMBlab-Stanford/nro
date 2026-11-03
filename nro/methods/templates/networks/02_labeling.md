---
section: Network labeling
context_version: 1
when: "labeling"
requires: [labeling, candidates_per_reference]
---
Networks were given heuristic labels by spatial correspondence with population
reference maps: the 15 networks of @du2024organization and the probabilistic
language atlas of @lipkin2022probabilistic. Each reference map was projected
into the participant's target geometry and correlated spatially with every
network map. For each reference, the {{ candidates_per_reference }} most
correlated networks received its label, in rank order, so a network could
receive several labels. Networks that no reference selected kept numeric
identifiers. These labels describe spatial similarity and are not functional
localizer results.
