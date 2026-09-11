"""Phase 5: CNR/SNR validation.

SNR: mean signal in an ROI relative to the noise (std dev) of a background
region. CNR: contrast between two ROIs relative to noise. ROIs are tied to
the phantom's known material geometry (bone core / soft-tissue sleeve),
not picked ad hoc from the image.
"""

from __future__ import annotations

import numpy as np


def elliptical_mask(shape: tuple[int, int], center: tuple[float, float],
                     semi_axes: tuple[float, float]) -> np.ndarray:
    """Boolean mask selecting pixels inside an ellipse -- used to select
    ROIs from known phantom geometry (bone core, tissue sleeve, background).
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy = (yy - center[0]) / semi_axes[0]
    dx = (xx - center[1]) / semi_axes[1]
    return (dy**2 + dx**2) < 1.0


def annulus_mask(shape: tuple[int, int], center: tuple[float, float],
                  inner_semi_axes: tuple[float, float],
                  outer_semi_axes: tuple[float, float]) -> np.ndarray:
    """Boolean mask selecting the ring between two concentric ellipses --
    used e.g. to isolate the soft-tissue sleeve around a bone core.
    """
    outer = elliptical_mask(shape, center, outer_semi_axes)
    inner = elliptical_mask(shape, center, inner_semi_axes)
    return outer & ~inner


def snr(signal_roi: np.ndarray, noise_roi: np.ndarray) -> float:
    """SNR = mean(signal ROI) / std(noise ROI)."""
    noise_std = np.std(noise_roi)
    if noise_std == 0:
        return float("inf")
    return float(np.mean(signal_roi) / noise_std)


def cnr(roi_a: np.ndarray, roi_b: np.ndarray, noise_roi: np.ndarray) -> float:
    """CNR = |mean(ROI a) - mean(ROI b)| / std(noise ROI)."""
    noise_std = np.std(noise_roi)
    if noise_std == 0:
        return float("inf")
    return float(np.abs(np.mean(roi_a) - np.mean(roi_b)) / noise_std)


def theoretical_weighted_signal(mu_low: float, mu_high: float, thickness: float, weight: float) -> float:
    """Theoretical contribution of a material to the weighted log-subtracted
    image: thickness * (mu_low - weight * mu_high). Comparing this between
    two materials gives the expected (noise-free) contrast, to check
    against the measured CNR.
    """
    return thickness * (mu_low - weight * mu_high)


def theoretical_contrast(coeffs, weight: float, thickness_a: float, thickness_b: float,
                          material_a: str = "bone", material_b: str = "soft") -> float:
    """Expected |signal_a - signal_b| in the weighted log-subtracted image,
    from the phantom's known attenuation coefficients and material
    thicknesses, for comparison against the measured CNR.
    """
    mu_low = {"bone": coeffs.mu_bone_low, "soft": coeffs.mu_soft_low}
    mu_high = {"bone": coeffs.mu_bone_high, "soft": coeffs.mu_soft_high}
    signal_a = theoretical_weighted_signal(mu_low[material_a], mu_high[material_a], thickness_a, weight)
    signal_b = theoretical_weighted_signal(mu_low[material_b], mu_high[material_b], thickness_b, weight)
    return abs(signal_a - signal_b)
