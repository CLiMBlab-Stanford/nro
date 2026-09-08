# Viewing and quality control

## `nro wb_view`

```bash
nro wb_view microparcellation -P nptl -p t20 -s fsnative -S 2
nro wb_view networks -P nptl -p t20 --no-open
```

The positional derivative type is `microparcellation` or `networks`. Common
selectors choose matching artifacts. The command opens scenes already stored
with completed artifacts; it does not rerun processing. `--no-open` prints
scene paths. `--wb-command` overrides the Workbench command path; `wb_view`
is found beside it. Opening a GUI requires a working display.

Surface scenes provide white, pial, midthickness, and inflated geometry.
Midthickness is the default view. Network CIFTIs contain named maps so users
can step through networks. Volume scenes use volume data without carrying
surface files. See each [module's output guide](../modules/index.md).

## `nro qc registration`

Registration QC retains its own parser:

```bash
nro qc registration t20 -p nptl -w main
```

Here the participant is positional and `-p` means **project**, unlike the
common selectors used by `run` and `status`. `-w`/`--workflow` chooses the
preprocessing lineage; the default is `main`. `--output-dir` overrides the QC
destination. `--sagittal-coordinate MM` selects the displayed world-space
left/right coordinate (default −20 mm); `--slab-thickness VOXELS` defaults to 3.

The command assembles registration views from completed preprocessing outputs
for visual inspection. It is ad hoc QC, not an automatically scheduled sixth
scientific module and not a numerical pass/fail classifier. Outputs normally
live under the preprocessing lineage's nested
`derivatives/qc/registration/sub-ID/` directory, not as a BIDS datatype beneath
the original subject. Inspect both anatomy and EPI alignment; a plausible image
alone does not validate every transform or motion estimate.

The engine form `python -m nro.qc registration ...` is also supported.
