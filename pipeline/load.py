"""Phase 1: image loading.

Reads Lodox DICOM image pairs (low-kVp, high-kVp) and generic paired
images used for open-source benchmarking (Phase 4).
"""

from __future__ import annotations

import numpy as np
import pydicom
from skimage import io as skio


def load_dicom(path: str) -> tuple[np.ndarray, pydicom.Dataset]:
    """Read a single DICOM file, applying rescale slope/intercept.

    Returns the pixel data as float64 (in the DICOM's stored units, e.g.
    Hounsfield-like or raw detector counts depending on modality) plus the
    parsed dataset for metadata (kVp, exposure, etc).
    """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return arr, ds


def load_dicom_pair(
    low_kvp_path: str, high_kvp_path: str
) -> tuple[np.ndarray, np.ndarray, pydicom.Dataset, pydicom.Dataset]:
    """Read the low-kVp / high-kVp DICOM pair for a single acquisition.

    Returns (low_arr, high_arr, low_ds, high_ds).
    """
    low_arr, low_ds = load_dicom(low_kvp_path)
    high_arr, high_ds = load_dicom(high_kvp_path)
    return low_arr, high_arr, low_ds, high_ds


def load_image_pair_generic(path_a: str, path_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a generic (non-DICOM) paired image, e.g. PNG/TIFF benchmark data
    such as JSRT/BSE-JSRT (original + bone-suppressed reference).

    Images are converted to float64 and left in their native intensity
    range (no normalisation), since downstream metrics only care about
    relative contrast within a single image.
    """
    img_a = skio.imread(path_a, as_gray=True).astype(np.float64)
    img_b = skio.imread(path_b, as_gray=True).astype(np.float64)
    return img_a, img_b
