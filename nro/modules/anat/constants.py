"""FreeSurfer labels and metadata constants used by anatomical processing."""

_FS_GIFTI_VOLGEOM_META_PREFIXES = ("VolGeom", "VolGeomC_")

_FREESURFER_ASEG_LABELS = {
    "Left-Cerebral-Cortex": 3,
    "Left-Cerebellum-White-Matter": 7,
    "Left-Cerebellum-Cortex": 8,
    "Left-Thalamus-Proper": 10,
    "Left-Caudate": 11,
    "Left-Putamen": 12,
    "Left-Pallidum": 13,
    "Brain-Stem": 16,
    "Left-Hippocampus": 17,
    "Left-Amygdala": 18,
    "Left-Accumbens-area": 26,
    "Left-VentralDC": 28,
    "Right-Cerebral-Cortex": 42,
    "Right-Cerebellum-White-Matter": 46,
    "Right-Cerebellum-Cortex": 47,
    "Right-Thalamus-Proper": 49,
    "Right-Caudate": 50,
    "Right-Putamen": 51,
    "Right-Pallidum": 52,
    "Right-Hippocampus": 53,
    "Right-Amygdala": 54,
    "Right-Accumbens-area": 58,
    "Right-VentralDC": 60,
}

_FREESURFER_GRAY_MATTER_SEGMENTATIONS = (
    "Left-Cerebral-Cortex",
    "Right-Cerebral-Cortex",
    "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex",
    "Left-Thalamus-Proper",
    "Right-Thalamus-Proper",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Pallidum",
    "Right-Pallidum",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Accumbens-area",
    "Right-Accumbens-area",
    "Left-VentralDC",
    "Right-VentralDC",
)

_FREESURFER_SUBCORTICAL_SEGMENTATIONS = {
    "cerebellum": (
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
    ),
    "thalamus": ("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
    "caudate": ("Left-Caudate", "Right-Caudate"),
    "putamen": ("Left-Putamen", "Right-Putamen"),
    "pallidum": ("Left-Pallidum", "Right-Pallidum"),
    "brainStem": ("Brain-Stem",),
    "hippocampus": ("Left-Hippocampus", "Right-Hippocampus"),
    "amygdala": ("Left-Amygdala", "Right-Amygdala"),
}
