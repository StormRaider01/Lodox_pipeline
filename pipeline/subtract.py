"""Phase 3: weighted log-subtraction bone suppression (SS3).

Core algorithm (Brody et al. 1981): for co-registered low-kVp and high-kVp
images of the same subject, bone signal cancels out of

    S = log(I_low) - w * log(I_high)

when w is chosen so that w = mu_bone_low / mu_bone_high (the ratio of bone
attenuation coefficients between the two energies). The residual image S
then contains soft-tissue contrast with the bone signal suppressed.

w is not known a priori for a real acquisition, so it is calibrated from a
bone step-wedge scan (Brody et al. 1981): a series of known bone
thicknesses over a *constant* soft-tissue-equivalent background. Because
the background is constant, the slope of log(I_low) against log(I_high)
across the steps is exactly mu_bone_low / mu_bone_high. Fitting this
directly on an anatomical bone ROI instead is tempting but wrong here --
in the mid-forearm phantom (and in real anatomy) the soft-tissue thickness
also varies across that ROI, which confounds the fit. Lemacks et al. 2002
note this weighting also sets the noise tradeoff in the subtracted image
-- a weight tuned purely to null bone may not minimise noise, which is
worth revisiting once real acquisitions are available.
"""

from __future__ import annotations

import numpy as np


def log_transform(img: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Log-transform an image, clipping non-positive values so log is
    defined. `epsilon` should be small relative to the image's dynamic
    range.
    """
    return np.log(np.clip(img, a_min=epsilon, a_max=None))


def weighted_log_subtract(low: np.ndarray, high: np.ndarray, weight: float) -> np.ndarray:
    """Produce the soft-tissue (bone-suppressed) image for a given weight."""
    return log_transform(low) - weight * log_transform(high)


def estimate_weight_from_stepwedge(low_wedge: np.ndarray, high_wedge: np.ndarray,
                                    step_mask: np.ndarray) -> float:
    """Calibrate the bone-cancelling weight from a step-wedge scan.

    `step_mask` labels each pixel with its step index (1..n_steps; 0 =
    ignored). Per-step mean log intensities are fit with a line; the slope
    is the bone-cancelling weight (see module docstring for the physics).
    """
    log_low = log_transform(low_wedge)
    log_high = log_transform(high_wedge)

    step_ids = np.unique(step_mask)
    step_ids = step_ids[step_ids != 0]
    mean_log_low = np.array([log_low[step_mask == i].mean() for i in step_ids])
    mean_log_high = np.array([log_high[step_mask == i].mean() for i in step_ids])

    slope, _intercept = np.polyfit(mean_log_high, mean_log_low, deg=1)
    return float(slope)


def suppress_bone(low: np.ndarray, high: np.ndarray, weight: float) -> dict:
    """Full Phase-3 pipeline: apply the (separately-calibrated) bone-
    cancelling weight to produce the soft-tissue image. See
    `estimate_weight_from_stepwedge` for how `weight` should be obtained.
    """
    soft_tissue_image = weighted_log_subtract(low, high, weight)
    return {"soft_tissue_image": soft_tissue_image, "weight": weight}
