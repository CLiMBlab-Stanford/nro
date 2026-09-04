# Network-labeling reference maps

These 16 population reference maps are runtime resources for heuristic labels
on individualized networks. They were copied byte-for-byte from
`climbprep/climbprep/resources/`, where they support the reference-correlation
implementation in `climbprep.parcellate`:

- `LanA_n806.nii`
- the 15 `DU15_*.nii.gz` maps

`nro.networks.labeling` projects each map into the target CIFTI space and ranks
the top configured number of individualized networks independently for each
reference. These resources are scientific inputs and must not be modified or
silently replaced; changing one changes the networks instance inputs.
