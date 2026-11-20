"""The stimulus annotation levels that nro defines.

A level means the same for every stimulus; each annotation's sidecar
describes its own columns and sources. ``nro stimuli publish`` writes a
level's description to the catalog's ``annotations.tsv``. This module has no
dependencies, so configuration schemas can name the levels.
"""

ANNOTATION_LEVELS = {
    "words": (
        "Words of the stimulus, one row per word at its onset in seconds from the start "
        "of the stimulus"
    ),
    "phonemes": (
        "Phonemes of the stimulus, one row per phoneme at its onset in seconds from the "
        "start of the stimulus"
    ),
}
