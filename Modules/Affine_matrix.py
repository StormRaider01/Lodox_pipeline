"""Rigid registration of a misaligned scan onto a target scan (rotation about the image centre + translation).

The optimal parameters minimise the mean squared error (MSE) between the target and the warped image. The search is
run coarse-to-fine: first on block-averaged copies of the images (fast, finds the large shifts), then refined at full
resolution starting from the coarse answer. Both images are smoothed for the search: otherwise linear interpolation
at a half-pixel shift averages away pixel noise, lowering the MSE, and the optimiser drifts to a spurious ~0.5 px shift.
"""
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import minimize


def downsample(image, factor):
    """Average factor x factor blocks of pixels (the image is trimmed so it divides evenly)."""
    if factor == 1:
        return image
    h = (image.shape[0] // factor) * factor
    w = (image.shape[1] // factor) * factor
    return image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def rotation_offset(shape, theta, tx, ty):
    """Matrix R and offset so that input pixel = R @ output pixel + offset, in (row, col) coordinates."""
    theta_rad = np.deg2rad(theta)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cy, cx = shape[0] / 2, shape[1] / 2         # so that centre of image is the pivot point for rotation

    # Rotational matrix
    R = np.array([[cos_t,  sin_t],
                  [-sin_t, cos_t]])

    # Offset to ensure rotation around the centre and translation
    offset = np.array([cy, cx]) - R @ np.array([cy, cx]) + np.array([ty, tx])
    return R, offset


def transformation(image, theta, tx, ty, cval=0):
    """Rotate the image by theta degrees about its centre and shift it by (tx, ty) pixels (cval fills the edges)."""
    R, offset = rotation_offset(image.shape, theta, tx, ty)
    return affine_transform(image, R, offset=offset, order=1, mode='constant', cval=cval)


def affine_matrix(theta, tx, ty, shape):
    """3 x 3 homogeneous matrix applied by transformation(): [row, col, 1] of the aligned image -> misaligned image."""
    R, offset = rotation_offset(shape, theta, tx, ty)
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = offset
    return M


def mse(params, target, misaligned):
    """Similarity metric minimised by the optimiser: MSE between the target and the warped misaligned image.

    Only the overlap counts: pixels warped in from outside the image are NaN and left out, so the empty border
    created by a rotation/shift does not pull the solution.
    """
    theta, tx, ty = params                                              # Retrieve parameters
    warped = transformation(misaligned, theta, tx, ty, cval=np.nan)     # Perform transformation
    return np.nanmean((target - warped) ** 2)


def register(target, misaligned, x0=(0, 0, 0), factors=(4, 1), sigma=2.0):
    """Find [theta (deg), tx (px), ty (px)] that align misaligned to target, coarse-to-fine over the given factors.

    sigma: Gaussian smoothing (pixels of each level) applied to both images before the search.
    Returns the parameters (full-resolution pixels) and the scipy result of the last (full-resolution) search.
    """
    params = np.asarray(x0, float)
    for factor in factors:
        guess = [params[0], params[1] / factor, params[2] / factor]       # translations in this level's pixels
        images = (gaussian_filter(downsample(target, factor), sigma),
                  gaussian_filter(downsample(misaligned, factor), sigma))
        result = minimize(mse, guess, args=images, method='Powell')     # mse continuously called internally until optimised parameters are found
        params = np.array([result.x[0], result.x[1] * factor, result.x[2] * factor])
    return params, result
