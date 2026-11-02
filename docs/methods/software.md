# Software and methodological sources

Use this page to cite the software and published methods behind an nro
analysis. The module guides describe what nro actually executes. Citing a
method does not mean that nro uses every option from its original publication.
When reporting an analysis, keep the resolved configurations, manifests,
container identities, reference maps, and installation receipts.

## Processing dependencies

| Component | Role | Primary documentation |
| --- | --- | --- |
| QuNex image | Runtime bundle; nro defines its own processing graphs. | [QuNex](https://qunex.readthedocs.io/) |
| ANTs | N4 correction, SyN registration, transform application. | [ANTs](https://github.com/ANTsX/ANTs) |
| FreeSurfer | Recon-all, segmentations, surfaces, boundary registration. | [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki) |
| SynthStrip | Anatomical brain extraction. | [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) |
| SynthStroke | Automatic stroke-lesion probability maps. | [SynthStroke](https://github.com/liamchalcroft/SynthStroke) |
| FastSurfer | Optional cortical reconstruction. | [FastSurfer](https://github.com/Deep-MI/FastSurfer) |
| SynthSR | Whole-brain synthetic T1w scaffolds for lesion-aware reconstruction. | [SynthSR](https://doi.org/10.1126/sciadv.add3607) |
| FSL | MCFLIRT, TOPUP, FLIRT, warp composition, MELODIC. | [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) |
| AFNI | Composed 4D warping with frame-specific affine transforms. | [3dNwarpApply](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNwarpApply.html) |
| HCP Pipelines | Hardware-specific gradient correction and optional MSMAll surface registration. | [HCP Pipelines](https://github.com/Washington-University/HCPpipelines) |
| ICA-AROMA | Motion-component classification and denoising. | [Upstream implementation](https://github.com/maartenmennes/ICA-AROMA) |
| CICADA | Alternative ICA component classification. | [Method paper](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.114/132108/CICADA-An-automated-and-flexible-tool-for) |
| MARSS 1.0.2 | Native-space estimation and removal of signal shared by simultaneous slices. | [Official implementation](https://github.com/CNaP-Lab/MARSS) |
| Workbench | Smoothing, surfaces, CIFTI, scenes. | [Workbench](https://www.humanconnectome.org/software/connectome-workbench) |
| TemplateFlow | Local MNI references and fsaverage geometry. | [TemplateFlow](https://www.templateflow.org/) |
| NumPy/SciPy | Streaming moments, sparse graphs, eigensolvers, projection. | [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/) |
| NiBabel/NiTransforms | Image formats and coordinate transforms. | [NiBabel](https://nipy.org/nibabel/), [NiTransforms](https://nitransforms.readthedocs.io/) |
| Nilearn | Task-design construction. | [Nilearn](https://nilearn.github.io/stable/) |
| scikit-learn | Randomized SVD, FastICA, MiniBatchKMeans. | [scikit-learn](https://scikit-learn.org/stable/) |
| igraph/leidenalg | Leiden initialization of OSLOM. | [leidenalg](https://leidenalg.readthedocs.io/) |
| OSLOM 2.5 | Overlapping graph communities. | [Official source](http://www.oslom.org/software.htm) |

SynBOLD-DisCo runs from its own image to provide synthetic-reference distortion
correction. The functional manifest records its configured image and the
wrapper policy.

### Versions

Installation pins the acquisition sources for QuNex 1.5.1, SynthStrip 1.7, the
official FreeSurfer 7.4.1 image, SynBOLD-DisCo 1.4, and Workbench 2.2.1. If a
site already has configured resources, nro reuses them, so these pins do not
establish the version of every executable present at that site. Python package
versions are locked in `uv.lock`. TemplateFlow object versions and checksums are
in the installed resource catalog. See [installation](../installation.md) for
acquisition and verification.

## Component notes

### First-level inference

[First-level inference](contrasts.md) uses nro's own implementation of grouped
GLS and covariance-preserving summaries. Nilearn supplies canonical HRF
convolution, Workbench supplies geodesic smoothing, and BIDS Stats Models
supplies the model format. FitLins is not a dependency.

### Anatomical reconstruction

Ordinary anatomy can use FastSurfer 2.5.4 instead of FreeSurfer.
FastSurferVINN segmentation runs at 1 mm on an on-demand GPU worker. Surface
reconstruction runs on a general CPU worker and produces the
FreeSurfer-compatible files that later anatomical steps consume.

Optional MSMAll anatomy uses the HCP Pipelines, MSM, pyFIX, FSL, FreeSurfer,
Workbench, Octave, the HCP dedrifting assets, and the
`HCP_Style_Single_Multirun_Dedrift` FIX model bundled in the pinned QuNex image.
The resting-state calibration runs are selected in source markup. nro adapts
the HCP route to its skull-stripped source contract with explicit FNIRT masks.
Each major HCP operation runs as an ordinary anatomical runner step. nro
publishes only the surface registrations, atlas geometry, QC, and version
records needed to use the resulting space.

### Lesion-aware anatomy

Lesion-aware anatomy uses the SynthStroke SynthPlus model at the revision
recorded in the anatomical contract. nro implements the inference adapter
itself, following the architecture and preprocessing policy of the recorded
upstream source revision. It checks the downloaded model files against pinned
SHA-256 digests. Inference requires the optional `lesion` Python extra. nro
does not copy the upstream training implementation.

Before N4 correction, FreeSurfer's `mri_synthsr` all-purpose model synthesizes
a complete 1 mm T1w scaffold from the pose-normalized source image. The model
reconstructs the whole brain without seeing the SynthStroke lesion mask.
Negative synthesized intensities are clipped to zero. The configured FreeSurfer
or FastSurfer backend then reconstructs the synthetic image through the
ordinary reconstruction interface. SynthStroke remains the independent source
of the lesion mask and of the later surface cut.

### MARSS

MARSS is an optional GPLv3 package that nro calls as a separate process. nro
does not copy or modify its implementation. The integration follows Tubiolo,
Williams, and Van Snellenberg (2024),
[*Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact*](https://doi.org/10.1002/hbm.70066).
The official package estimates and subtracts the artifact. nro validates its
outputs and replaces the full 4D artifact file with a slice-wise rank-one
factorization, checking and recording the reconstruction error.

### CICADA

pycicada is also called as a separate process. The optional `--with-cicada`
installation extra pins its package separately. nro supplies prepared images
and an adapted MELODIC directory, then reads the one-based signal and noise
labels that pycicada returns. nro applies the configured component regression
itself, so AROMA and CICADA comparisons share the same MELODIC estimation
policy and regression implementation.

## Methods

[Denoising](denoising.md) adapts these methods:

- Afyouni and Nichols (2018), *Insight and inference for DVARS*,
  [doi:10.1016/j.neuroimage.2017.12.098](https://doi.org/10.1016/j.neuroimage.2017.12.098).
- Satterthwaite et al. (2013), *An improved framework for confound regression
  and filtering for control of motion artifact in the preprocessing of
  resting-state functional connectivity data*,
  [doi:10.1016/j.neuroimage.2012.08.052](https://doi.org/10.1016/j.neuroimage.2012.08.052).

Microparcellation uses edge-based local-variation coarsening from Loukas (2019),
[*Graph Reduction with Spectral and Cut Guarantees*](https://jmlr.org/papers/v20/18-680.html).
The spatial-neighbor restrictions, streaming reliability weights, iterative
targets, and null diagnostics are nro implementation choices. Do not assume
that the paper's theoretical guarantees carry over unchanged to every processed
dataset.

Network clustering and reference labeling adapt Shain and Fedorenko (2026),
[*A language network in the individualized functional connectomes of 1199 human
brains doing arbitrary tasks*](https://www.nature.com/articles/s41467-026-75745-8).
nro uses sparse microparcel connectivity profiles, mini-batch fitting, and
several ranked reference candidates. It does not reproduce the study's full
voxel-level protocol. OSLOM follows Lancichinetti et al. (2011),
[*Finding statistically significant communities in networks*](https://doi.org/10.1371/journal.pone.0018961).
When reporting OSLOM results, record the sparsification, initialization, and
internal search counts.

The 16 bundled network references are scientific inputs and cannot be swapped
for one another as if they were labels. Their resource provenance note follows.

```{literalinclude} ../../nro/modules/networks/resources/README.md
:language: text
```
