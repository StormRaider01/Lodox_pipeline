"""Synthetic test-harness data.

No real Lodox acquisitions exist in this repo yet, so this module generates
physically-plausible dual-energy image pairs from a Beer-Lambert projection
model, purely so every phase of the pipeline can be exercised end-to-end
before real phantom/patient DICOMs are available. Attenuation coefficients
used here are representative order-of-magnitude values (loosely based on
the general shape of NIST XCOM soft-tissue/cortical-bone data) — they are
NOT calibrated phantom-design values and must not be cited as such.

Replace calls to these functions with `pipeline.load.load_dicom_pair` once
real acquisitions exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhantomCoefficients:
    """Approximate linear attenuation coefficients (arbitrary units) for a
    two-material (bone / soft-tissue) cross-section at two simulated kVp
    settings. Bone attenuates more strongly than soft tissue at both
    energies, but the bone:soft-tissue ratio is larger at low kVp than at
    high kVp -- this is the physical property weighted log-subtraction
    exploits to cancel bone.
    """

    mu_soft_low: float = 0.20
    mu_bone_low: float = 0.50
    mu_soft_high: float = 0.15
    mu_bone_high: float = 0.25


def _ellipsoid_thickness(shape: tuple[int, int], center: tuple[float, float],
                          semi_axes: tuple[float, float], peak_thickness: float) -> np.ndarray:
    """Projected thickness map of a solid ellipsoid (extruded along the beam
    axis), as seen by a planar X-ray projection: thickness = peak * 2*sqrt(1 - (dy/a)^2 - (dx/b)^2)
    inside the ellipse footprint, 0 outside.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy = (yy - center[0]) / semi_axes[0]
    dx = (xx - center[1]) / semi_axes[1]
    r2 = dy**2 + dx**2
    thickness = np.zeros(shape, dtype=np.float64)
    inside = r2 < 1.0
    thickness[inside] = peak_thickness * np.sqrt(1.0 - r2[inside])
    return thickness


def generate_forearm_thickness_maps(
    shape: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray]:
    """Build thickness maps for a mid-forearm-shaft phantom cross-section:
    an outer soft-tissue ellipsoid with a concentric bone-core ellipsoid
    inside it (matching the CLAUDE.md phantom description: bone core +
    soft-tissue sleeve).

    Returns (t_soft, t_bone) thickness maps in arbitrary length units.
    Soft-tissue thickness excludes the volume displaced by the bone core.
    """
    center = (shape[0] / 2, shape[1] / 2)
    t_soft_total = _ellipsoid_thickness(shape, center, (shape[0] * 0.28, shape[1] * 0.16), peak_thickness=8.0)
    t_bone = _ellipsoid_thickness(shape, center, (shape[0] * 0.10, shape[1] * 0.06), peak_thickness=3.0)
    t_bone = np.minimum(t_bone, t_soft_total)
    t_soft = t_soft_total - t_bone
    return t_soft, t_bone


def project(t_soft: np.ndarray, t_bone: np.ndarray, mu_soft: float, mu_bone: float,
            i0: float = 1.0, photon_noise_scale: float = 20_000.0,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Beer-Lambert projection through the two-material thickness maps at a
    single simulated energy, with Poisson-like quantum noise added.

    `photon_noise_scale` stands in for unattenuated photon count (detector
    dose): it must stay large enough that even the most attenuated pixel
    keeps a non-trivial expected count, otherwise Poisson sampling drives
    it to zero and `subtract.log_transform`'s clipping floor dominates the
    result (silent photon starvation, not a meaningful signal).
    """
    if rng is None:
        rng = np.random.default_rng()
    intensity = i0 * np.exp(-(mu_soft * t_soft + mu_bone * t_bone))
    counts = np.clip(intensity * photon_noise_scale, a_min=1e-6, a_max=None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    return noisy_counts / photon_noise_scale


def generate_synthetic_phantom_pair(
    shape: tuple[int, int] = (512, 512),
    coeffs: PhantomCoefficients | None = None,
    seed: int | None = 0,
) -> dict:
    """Generate a synthetic low-kVp / high-kVp dual-energy image pair for a
    static forearm-shaft phantom, along with the ground-truth thickness
    maps and coefficients used, so downstream phases (alignment, weighted
    log-subtraction, CNR/SNR) can be exercised and checked against a known
    answer.

    Returns a dict with keys: low, high, t_soft, t_bone, coeffs,
    bone_center, bone_radius_px, tissue_outer_semi_axes.
    """
    if coeffs is None:
        coeffs = PhantomCoefficients()
    rng = np.random.default_rng(seed)

    t_soft, t_bone = generate_forearm_thickness_maps(shape)
    low = project(t_soft, t_bone, coeffs.mu_soft_low, coeffs.mu_bone_low, rng=rng)
    high = project(t_soft, t_bone, coeffs.mu_soft_high, coeffs.mu_bone_high, rng=rng)

    return {
        "low": low,
        "high": high,
        "t_soft": t_soft,
        "t_bone": t_bone,
        "coeffs": coeffs,
        "bone_center": (shape[0] / 2, shape[1] / 2),
        "bone_semi_axes": (shape[0] * 0.10, shape[1] * 0.06),
        "tissue_semi_axes": (shape[0] * 0.28, shape[1] * 0.16),
    }


def generate_bone_stepwedge(
    n_steps: int = 6,
    shape: tuple[int, int] = (120, 360),
    min_thickness: float = 0.5,
    max_thickness: float = 4.0,
    soft_background_thickness: float = 5.0,
    coeffs: PhantomCoefficients | None = None,
    seed: int | None = 2,
) -> dict:
    """Generate a synthetic bone step-wedge calibration pair: a series of
    bone-equivalent steps of known, increasing thickness over a *constant*
    soft-tissue-equivalent background.

    This is the standard calibration acquisition used to determine the
    bone-cancelling weight (Brody et al. 1981): because the background
    thickness is held constant, the slope of log(I_low) against
    log(I_high) across the steps is exactly mu_bone_low / mu_bone_high,
    uncontaminated by any soft-tissue thickness variation (unlike fitting
    directly on an anatomical bone ROI, where soft-tissue thickness also
    varies).

    Returns a dict with keys: low, high, step_mask (int array, 0 =
    background-only margin, 1..n_steps = step index), step_thickness
    (array of bone thickness per step), coeffs.
    """
    if coeffs is None:
        coeffs = PhantomCoefficients()
    rng = np.random.default_rng(seed)

    margin = shape[1] // (n_steps + 2)
    step_width = (shape[1] - 2 * margin) // n_steps
    step_thickness = np.linspace(min_thickness, max_thickness, n_steps)

    t_bone = np.zeros(shape, dtype=np.float64)
    step_mask = np.zeros(shape, dtype=np.int32)
    for i in range(n_steps):
        x0 = margin + i * step_width
        x1 = x0 + step_width
        t_bone[:, x0:x1] = step_thickness[i]
        step_mask[:, x0:x1] = i + 1

    t_soft = np.full(shape, soft_background_thickness, dtype=np.float64)

    low = project(t_soft, t_bone, coeffs.mu_soft_low, coeffs.mu_bone_low, rng=rng)
    high = project(t_soft, t_bone, coeffs.mu_soft_high, coeffs.mu_bone_high, rng=rng)

    return {
        "low": low,
        "high": high,
        "step_mask": step_mask,
        "step_thickness": step_thickness,
        "coeffs": coeffs,
    }


def generate_synthetic_benchmark_pair(shape: tuple[int, int] = (256, 256),
                                       seed: int | None = 1) -> dict:
    """Generate a synthetic stand-in for an open-source bone-suppression
    benchmark pair (e.g. JSRT/BSE-JSRT style: a single-exposure "original"
    image containing a bone-like structure, plus a "reference" image with
    that structure suppressed).

    This is a structural stand-in only -- it validates that the Phase 4
    loader/metric code path works, it is not a substitute for evaluating
    against the real published datasets.
    """
    rng = np.random.default_rng(seed)
    t_soft, t_bone = generate_forearm_thickness_maps(shape)
    coeffs = PhantomCoefficients()

    original = project(t_soft, t_bone, coeffs.mu_soft_low, coeffs.mu_bone_low, rng=rng)
    reference_suppressed = project(t_soft, np.zeros_like(t_bone), coeffs.mu_soft_low, coeffs.mu_bone_low, rng=rng)

    return {
        "original": original,
        "reference_suppressed": reference_suppressed,
        "bone_center": (shape[0] / 2, shape[1] / 2),
        "bone_semi_axes": (shape[0] * 0.10, shape[1] * 0.06),
        "tissue_semi_axes": (shape[0] * 0.28, shape[1] * 0.16),
    }
