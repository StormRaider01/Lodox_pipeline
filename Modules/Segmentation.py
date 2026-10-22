"""Otsu thresholding and the helpers shared by the ROI mask modules (Ice_tray_mask, Wedge_mask, Phantom_mask).

Every mask class stores its ROIs as one integer label image (0 = no ROI, k = names[k - 1]) so the ROIs of the tray,
wedge and phantom can be combined, plotted and saved in the same way.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import regionprops


def otsu_threshold(values, nbins=256, passes=1):
    """Otsu's method: the grey level that maximises the between-class variance of the histogram of values.

    passes > 1 repeats the split on the pixels below the previous threshold. When dense objects (bone samples,
    phantoms) take up much of the image, the first split separates them from everything else and the faint objects
    (fat cells, the thin wedge steps) fall below it; the second split separates those from the background.
    """
    values = np.asarray(values, float).ravel()
    for _ in range(passes - 1):
        values = values[values < otsu_threshold(values, nbins)]
    hist, edges = np.histogram(values, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2

    w0 = np.cumsum(hist)                        # pixels in class 0 (at or below each level)
    w1 = w0[-1] - w0                            # pixels in class 1 (above it)
    s0 = np.cumsum(hist * centres)
    m0 = s0 / np.maximum(w0, 1)                 # class means
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2          # between-class variance (up to a constant)
    return centres[np.argmax(between)]


def smooth(image, pixel_spacing, sigma_mm=1.0):
    """Gaussian smoothing so the threshold follows the objects rather than the pixel noise."""
    return ndi.gaussian_filter(image, sigma_mm / pixel_spacing)


def label_objects(image, threshold, pixel_spacing, min_area_mm2=50):
    """Label the objects brighter than threshold (smoothed image), ignoring specks and anything touching the border.

    Returns the label image and a list of skimage regionprops, one per kept object.
    """
    fg = smooth(image, pixel_spacing) > threshold
    fg = ndi.binary_opening(fg, iterations=max(1, int(round(1 / pixel_spacing))))     # remove 1 mm specks
    fg = ndi.binary_fill_holes(fg)
    labels, _ = ndi.label(fg)

    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    regions = []
    for r in regionprops(labels):
        if r.label in border or r.area * pixel_spacing ** 2 < min_area_mm2:
            labels[labels == r.label] = 0
        else:
            regions.append(r)
    return labels, regions


def background_noise(image):
    """Median and robust standard deviation (1.4826 x MAD) of an image that is mostly background."""
    med = np.median(image)
    return med, 1.4826 * np.median(np.abs(image - med))


def principal_axis(mask):
    """Centre (row, col) and unit vector (d_row, d_col) along the long axis of a mask."""
    yy, xx = np.nonzero(mask)
    centre = np.array([yy.mean(), xx.mean()])
    evals, evecs = np.linalg.eigh(np.cov(np.c_[yy - centre[0], xx - centre[1]].T))
    return centre, evecs[:, np.argmax(evals)]


def background_band(footprint, exclude, pixel_spacing, inner_mm, outer_mm):
    """Pixels inner_mm to outer_mm from the footprint that are at least inner_mm away from every excluded object."""
    d_obj = ndi.distance_transform_edt(~footprint) * pixel_spacing
    d_any = ndi.distance_transform_edt(~(exclude | footprint)) * pixel_spacing
    band = (d_obj > inner_mm) & (d_obj < outer_mm) & (d_any > inner_mm)
    edge = int(round(5 / pixel_spacing))                        # keep clear of the image border
    band[:edge], band[-edge:], band[:, :edge], band[:, -edge:] = False, False, False, False
    return band


class RoiMask:
    """Behaviour shared by the mask classes: the ROIs are held in one label image (0 = no ROI)."""

    short_labels = None                                         # {name: text} drawn on the overlay instead of the name

    def __init__(self, shape, pixel_spacing):
        self.mask = np.zeros(shape, np.int16)
        self.names = []
        self.pixel_spacing = pixel_spacing

    def add(self, name, roi):
        """Add an ROI (pixels already claimed by an earlier ROI are left alone)."""
        if not roi.any():
            print(f"Warning: ROI '{name}' is empty")
        self.names.append(name)
        self.mask[roi & (self.mask == 0)] = len(self.names)

    def roi(self, name):
        """Boolean mask of one ROI."""
        return self.mask == self.names.index(name) + 1

    @property
    def rois(self):
        """{name: boolean mask} of every ROI."""
        return {name: self.mask == k for k, name in enumerate(self.names, start=1)}

    def show(self, image, ax=None, title=None, crop=True):
        return show_rois(image, self.mask, self.names, ax=ax, title=title, crop=crop, pixel_spacing=self.pixel_spacing,
                         labels=self.short_labels)


def combine_masks(masks):
    """Merge several mask objects into one label image + list of names (later masks never overwrite earlier ones)."""
    combined = np.zeros(masks[0].mask.shape, np.int16)
    names = []
    for m in masks:
        for k, name in enumerate(m.names, start=1):
            roi = (m.mask == k) & (combined == 0)
            if roi.sum() < (m.mask == k).sum():
                print(f"Warning: ROI '{name}' overlaps an earlier ROI; the shared pixels keep the earlier label")
            names.append(name)
            combined[roi] = len(names)
    return combined, names


def show_rois(image, mask, names, ax=None, title=None, crop=True, pad_mm=15, pixel_spacing=0.2, labels=None):
    """Draw the ROIs of a label image over a greyscale image (filled, semi-transparent, with their names).

    labels: optional {name: shorter text to draw} for crowded ROIs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, cmap='gray')

    colours = plt.get_cmap('tab20')
    rgba = np.zeros(mask.shape + (4,))
    for k in range(1, len(names) + 1):
        roi = mask == k
        if not roi.any():
            continue
        rgba[roi] = colours((k - 1) % 20)
        rgba[roi, 3] = 0.45
        rows, cols = np.nonzero(roi)                            # label on the ROI pixel nearest its centroid
        i = np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2)
        text = (labels or {}).get(names[k - 1], names[k - 1])
        ax.text(cols[i], rows[i], text, color='white', fontsize=7, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
    ax.imshow(rgba)

    if crop and mask.any():                                     # zoom onto the ROIs
        rows, cols = np.nonzero(mask)
        pad = pad_mm / pixel_spacing
        ax.set_xlim(max(cols.min() - pad, 0), min(cols.max() + pad, mask.shape[1] - 1))
        ax.set_ylim(min(rows.max() + pad, mask.shape[0] - 1), max(rows.min() - pad, 0))
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax
