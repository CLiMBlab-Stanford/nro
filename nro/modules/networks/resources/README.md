# Network-labeling reference maps

These 16 population reference maps let `networks` attach heuristic labels to
individualized networks:

- `LanA_n806.nii`
- the 15 `DU15_*.nii.gz` maps

`nro.modules.networks.labeling` projects each map into the target CIFTI space.
For each reference separately, it ranks the individualized networks and keeps
the configured number of top candidates.

The files were copied byte-for-byte from `climbprep/climbprep/resources/`,
where they support the reference-correlation code in `climbprep.parcellate`.
They are scientific inputs. Do not modify or silently replace them, because
changing one changes the inputs of every networks work item.
