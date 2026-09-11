"""Phase 2: preprocessing / alignment.

Acquisition is sequential (two separate exposures at different kVp), so the
pair may be offset by subject or detector motion between shots. For a
static phantom this should be near-zero, but it is measured and corrected
rather than assumed away.

Strategy: estimate a translation via FFT phase correlation (robust, no
initial guess needed), then optionally refine with OpenCV's ECC optimiser
for sub-pixel intensity-based alignment.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation


def estimate_shift(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the (dy, dx) translation of `moving` relative to `fixed`
    via FFT phase correlation. Returns (shift, error) where shift is
    applied as: aligned = shift(moving, shift).
    """
    shift, error, _phasediff = phase_cross_correlation(fixed, moving, upsample_factor=10)
    return shift, error


def needs_registration(shift: np.ndarray, threshold_px: float = 0.5) -> bool:
    """Whether the estimated shift is large enough to bother correcting."""
    return float(np.hypot(*shift)) > threshold_px


def apply_shift(moving: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Translate `moving` by (dy, dx) using an affine warp (bilinear)."""
    dy, dx = shift
    h, w = moving.shape
    transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    aligned = cv2.warpAffine(
        moving.astype(np.float32), transform, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned.astype(moving.dtype)


def refine_ecc(fixed: np.ndarray, moving: np.ndarray, warp_mode: int = cv2.MOTION_EUCLIDEAN,
               n_iterations: int = 200, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel intensity-based refinement using OpenCV's ECC algorithm.

    Returns (aligned_moving, warp_matrix). Falls back to the identity warp
    if ECC fails to converge (e.g. insufficient texture).
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations, eps)

    def _norm(img):
        img = img.astype(np.float32)
        rng = img.max() - img.min()
        return (img - img.min()) / rng if rng > 0 else img

    try:
        _, warp_matrix = cv2.findTransformECC(
            _norm(fixed), _norm(moving), warp_matrix, warp_mode, criteria,
        )
    except cv2.error:
        return moving, warp_matrix

    h, w = moving.shape
    aligned = cv2.warpAffine(
        moving.astype(np.float32), warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned.astype(moving.dtype), warp_matrix


def align_pair(fixed: np.ndarray, moving: np.ndarray, shift_threshold_px: float = 0.5,
               refine: bool = True) -> dict:
    """Full Phase-2 pipeline: check whether registration is needed, apply a
    coarse phase-correlation translation if so, then optionally refine with
    ECC. Returns a dict with the aligned image and a record of what was
    applied (for the notebook to display/log).
    """
    shift, error = estimate_shift(fixed, moving)
    registered = needs_registration(shift, shift_threshold_px)

    aligned = moving
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    if registered:
        aligned = apply_shift(moving, shift)
        if refine:
            aligned, warp_matrix = refine_ecc(fixed, aligned)

    return {
        "aligned": aligned,
        "shift": shift,
        "shift_error": error,
        "registration_applied": registered,
        "warp_matrix": warp_matrix,
    }
