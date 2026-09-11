"""Phase 4: open-source benchmarking.

Purpose (see CLAUDE.md): sanity-check that the CNR/SNR metrics behave
consistently on real clinical/reference images (JSRT + BSE-JSRT, IEEE
DataPort bone-shadow-suppression, NIH ChestX-ray14/RSNA, BS-LDM, hand/
forearm atlases), independent of the custom phantom.

These datasets require external download/registration and are not bundled
in this repo. `evaluate_benchmark_pair` works on any (original,
bone-suppressed) image pair once you point it at real files via
`pipeline.load.load_image_pair_generic`; until then,
`pipeline.synthetic.generate_synthetic_benchmark_pair` provides a
structural stand-in so this code path can be exercised and checked.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def evaluate_benchmark_pair(
    original: np.ndarray,
    suppressed: np.ndarray,
    bone_center: tuple[float, float],
    bone_semi_axes: tuple[float, float],
    tissue_semi_axes: tuple[float, float],
) -> dict:
    """Compute CNR/SNR on an (original, bone-suppressed) benchmark pair
    using the same ROI-based metrics as the phantom pipeline (Phase 5),
    so results are directly comparable across datasets.

    ROIs: a bone-region ellipse, a surrounding soft-tissue annulus, and an
    outside-the-phantom background region used as the noise reference.
    """
    shape = original.shape
    bone_roi = metrics.elliptical_mask(shape, bone_center, bone_semi_axes)
    tissue_roi = metrics.annulus_mask(shape, bone_center, bone_semi_axes, tissue_semi_axes)
    background_roi = ~metrics.elliptical_mask(shape, bone_center, tissue_semi_axes)

    def summarize(img: np.ndarray) -> dict:
        return {
            "snr": metrics.snr(img[tissue_roi], img[background_roi]),
            "cnr": metrics.cnr(img[bone_roi], img[tissue_roi], img[background_roi]),
        }

    return {
        "original": summarize(original),
        "suppressed": summarize(suppressed),
    }
