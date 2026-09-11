# Software and methodological sources

The module guides describe what nro executes. Citing a method does not imply
that every option in its original publication is used. Retain resolved
configurations, manifests, container identities, reference maps, and installation
receipts when reporting an analysis.

[First-level inference](firstlevels.md) uses an nro implementation of grouped
GLS and covariance-preserving summaries. Nilearn supplies canonical HRF
convolution; Workbench supplies geodesic smoothing. BIDS Stats Models supplies
the model format. FitLins is not a dependency.

## Processing dependencies

| Component | Role | Primary documentation |
| --- | --- | --- |
| QuNex image | Runtime bundle; nro defines its own processing graphs. | [QuNex](https://qunex.readthedocs.io/) |
| ANTs | N4 correction, SyN registration, transform application. | [ANTs](https://github.com/ANTsX/ANTs) |
| FreeSurfer | Recon-all, segmentations, surfaces, boundary registration. | [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki) |
| SynthStrip | Anatomical brain extraction. | [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) |
| FSL | MCFLIRT, TOPUP, FLIRT, warp composition, MELODIC. | [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) |
| AFNI | Composed 4D warping with frame-specific affine transforms. | [3dNwarpApply](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNwarpApply.html) |
| ICA-AROMA | Motion-component classification and denoising. | [Upstream implementation](https://github.com/maartenmennes/ICA-AROMA) |
| MARSS 1.0.2 | Native-space estimation and removal of signal shared by simultaneous slices. | [Official implementation](https://github.com/CNaP-Lab/MARSS) |
| Workbench | Smoothing, surfaces, CIFTI, scenes. | [Workbench](https://www.humanconnectome.org/software/connectome-workbench) |
| TemplateFlow | Local MNI references and fsaverage geometry. | [TemplateFlow](https://www.templateflow.org/) |
| NumPy/SciPy | Streaming moments, sparse graphs, eigensolvers, projection. | [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/) |
| NiBabel/NiTransforms | Image formats and coordinate transforms. | [NiBabel](https://nipy.org/nibabel/), [NiTransforms](https://nitransforms.readthedocs.io/) |
| Nilearn | Task-design construction. | [Nilearn](https://nilearn.github.io/stable/) |
| scikit-learn | Randomized SVD, FastICA, MiniBatchKMeans. | [scikit-learn](https://scikit-learn.org/stable/) |
| igraph/leidenalg | Leiden initialization of OSLOM. | [leidenalg](https://leidenalg.readthedocs.io/) |
| OSLOM 2.5 | Overlapping graph communities. | [Official source](http://www.oslom.org/software.htm) |

SynBOLD-DisCo runs from its separate image for synthetic-reference distortion
correction. Its configured image and the wrapper policy are recorded in the
functional manifest. Installation pins QuNex 1.5.1, SynthStrip 1.7,
SynBOLD-DisCo 1.4, and Workbench 2.2.1 acquisition sources. Existing configured
resources are reused, so those pins do not establish the versions of every
executable already present at a site. Python versions are locked in `uv.lock`;
TemplateFlow object versions/checksums are in the installed resource catalog.
See [installation](../installation.md) for acquisition and verification.

MARSS is an optional GPLv3 package invoked through a process boundary. Nro does
not copy or modify its implementation. The integration follows Tubiolo,
Williams, and Van Snellenberg (2024),
[*Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact*](https://doi.org/10.1002/hbm.70066).
The official package estimates and subtracts the artifact. Nro validates its
outputs and replaces the full 4D artifact file with a slice-wise rank-one
factorization whose reconstruction error is checked and recorded.

## Methods

[Denoising](denoising.md) adapts the following methods:

- Afyouni and Nichols (2018), *Insight and inference for DVARS*,
  [doi:10.1016/j.neuroimage.2017.12.098](https://doi.org/10.1016/j.neuroimage.2017.12.098).
- Satterthwaite et al. (2013), *An improved framework for confound regression
  and filtering for control of motion artifact in the preprocessing of
  resting-state functional connectivity data*,
  [doi:10.1016/j.neuroimage.2012.08.052](https://doi.org/10.1016/j.neuroimage.2012.08.052).

Microparcellation uses edge-based local-variation coarsening from Loukas (2019),
[*Graph Reduction with Spectral and Cut Guarantees*](https://jmlr.org/papers/v20/18-680.html).
Spatial-neighbor restrictions, streaming reliability weights, iterative targets,
and null diagnostics are nro implementation choices; theoretical guarantees
should not be assumed to transfer unchanged to every processed dataset.

Network clustering and reference labeling adapt Shain and Fedorenko (2026),
[*A language network in the individualized functional connectomes of 1199 human
brains doing arbitrary tasks*](https://www.nature.com/articles/s41467-026-75745-8).
nro uses sparse microparcel connectivity profiles, mini-batch fitting, and
multiple ranked reference candidates. It is not the study's complete voxel-level
protocol. OSLOM follows Lancichinetti et al. (2011),
[*Finding statistically significant communities in networks*](https://doi.org/10.1371/journal.pone.0018961).
Record sparsification, initialization, and internal search counts when reporting it.

The 16 bundled references are scientific inputs, not interchangeable labels.
Their resource provenance note is:

```{literalinclude} ../../nro/modules/networks/resources/README.md
:language: text
```
