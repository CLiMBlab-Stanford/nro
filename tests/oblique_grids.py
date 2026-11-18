"""BOLD affines that separate AFNI's cardinal-axis choice from nibabel's."""

import numpy as np

# Two BOLD headers from one session: the first is tilted close enough to 45
# degrees that AFNI and nibabel choose different cardinal axes.
NEAR_DIAGONAL_BOLD_AFFINE = np.array(
    [
        [-1.997658e00, 8.618139e-02, 4.840655e-02, 1.036175e02],
        [2.983866e-02, 1.413818e00, -1.555724e00, -1.599374e01],
        [9.205113e-02, 1.411981e00, 1.554793e00, -1.359081e02],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CLEARLY_OBLIQUE_BOLD_AFFINE = np.array(
    [
        [-1.999088e00, 5.955050e-02, -1.099369e-02, 1.060899e02],
        [4.787040e-02, 1.361202e00, -1.610979e00, -2.437929e01],
        [3.680452e-02, 1.464098e00, 1.498208e00, -1.374230e02],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
