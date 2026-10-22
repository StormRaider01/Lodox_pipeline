"""ROI mask of the PLA step wedge (five steps, 1-5 cm thick).

Only the thicker steps of the wedge pass the global Otsu threshold, so the Otsu object is used to find the wedge's
position, direction and width, and the full length is then found from the contrast profile along the wedge's long
axis. The length is split into equal steps, the thin end is the end with the lower contrast, and each step ROI is
the middle part of its step (away from the step edges and the wedge sides).
"""
import numpy as np
from scipy import ndimage as ndi

from .Segmentation import RoiMask, smooth, principal_axis, background_band, background_noise

STEP_THICKNESS_CM = [1, 2, 3, 4, 5]


def find_wedge(regions, pixel_spacing):
    """The wedge is the long, thin object (elongation > 2, 4-20 mm wide, 100-1500 mm^2 above the threshold)."""
    mm = pixel_spacing
    candidates = [r for r in regions if r.axis_minor_length > 0
                  and r.axis_major_length / r.axis_minor_length > 2
                  and 4 < r.axis_minor_length * mm < 20
                  and 100 < r.area * mm ** 2 < 1500]
    if not candidates:
        raise RuntimeError("Step wedge not found among the Otsu objects")
    return max(candidates, key=lambda r: r.area)


class Wedge_Mask(RoiMask):
    """A class to represent the step wedge mask: one ROI per step ('Wedge 1 cm' ... 'Wedge 5 cm') + 'Wedge background'."""

    def __init__(self, image, labels, regions, pixel_spacing=0.2, step_thickness_cm=STEP_THICKNESS_CM,
                 step_fraction=0.5, width_fraction=0.6, search_mm=40, bg_inner_mm=5, bg_outer_mm=15):
        super().__init__(image.shape, pixel_spacing)
        mm = pixel_spacing
        n_steps = len(step_thickness_cm)

        region = find_wedge(regions, mm)
        blob = labels == region.label
        centre, axis = principal_axis(blob)
        normal = np.array([-axis[1], axis[0]])

        # Window around the wedge, with s = distance along the wedge axis and n = distance across it (pixels)
        half = region.axis_major_length / 2 + search_mm / mm
        r0, r1 = max(int(centre[0] - half), 0), min(int(centre[0] + half) + 1, image.shape[0])
        c0, c1 = max(int(centre[1] - half), 0), min(int(centre[1] + half) + 1, image.shape[1])
        yy, xx = np.mgrid[r0:r1, c0:c1]
        s = (yy - centre[0]) * axis[0] + (xx - centre[1]) * axis[1]
        n = (yy - centre[0]) * normal[0] + (xx - centre[1]) * normal[1]
        half_width = np.percentile(np.abs(n[blob[r0:r1, c0:c1]]), 99)

        # Contrast profile along the axis: middle of the wedge minus two strips beside it (local background)
        sm = smooth(image[r0:r1, c0:c1], mm, 0.5)
        bins = np.round(s).astype(int)
        b0 = bins.min()                                  # s of bin 0
        bins -= b0

        def profile(sel):
            total = np.bincount(bins[sel], sm[sel], minlength=bins.max() + 1)
            return total / np.maximum(np.bincount(bins[sel], minlength=bins.max() + 1), 1)

        inside = profile(np.abs(n) < 0.5 * half_width)
        beside = profile((np.abs(n) > 1.5 * half_width) & (np.abs(n) < 2.5 * half_width))
        contrast = ndi.uniform_filter1d(inside - beside, max(1, int(round(1 / mm))))
        _, noise = background_noise(np.diff(contrast))

        # The wedge is the run of bins above the noise that contains the Otsu object (s = 0)
        above = contrast > 5 * noise
        runs, _ = ndi.label(above)
        run = runs[-b0]
        if run == 0:
            raise RuntimeError("Step wedge contrast profile not found")
        i0, i1 = np.nonzero(runs == run)[0][[0, -1]]

        # On low-noise (unfiltered) scans the run also takes in the blurred tails a few mm past each end, so the ends
        # are moved to where the contrast reaches half the level of the end step (the edge of a blurred step)
        step = (i1 - i0) / n_steps
        lo = np.median(contrast[int(i0 + 0.25 * step):int(i0 + 0.75 * step) + 1])
        hi = np.median(contrast[int(i1 - 0.75 * step):int(i1 - 0.25 * step) + 1])
        run_contrast = contrast[i0:i1 + 1]
        i0, i1 = i0 + np.argmax(run_contrast > 0.5 * lo), i1 - np.argmax(run_contrast[::-1] > 0.5 * hi)
        s0, s1 = i0 + b0, i1 + b0
        step = (s1 - s0) / n_steps

        thin_first = contrast[i0:i0 + int(step)].mean() < contrast[i1 - int(step):i1 + 1].mean()
        self.length_mm = (s1 - s0) * mm
        self.width_mm = 2 * half_width * mm
        thin_dir = axis if not thin_first else -axis     # image direction from the wedge centre to its thin end
        if abs(thin_dir[1]) >= abs(thin_dir[0]):
            self.thin_end = 'right' if thin_dir[1] > 0 else 'left'
        else:
            self.thin_end = 'bottom' if thin_dir[0] > 0 else 'top'
        self.axis = axis

        def full(sel):                                   # window selection -> full-size image mask
            out = np.zeros(image.shape, bool)
            out[r0:r1, c0:c1] = sel
            return out

        footprint = full((s >= s0) & (s <= s1) & (np.abs(n) <= half_width)) | blob
        self.footprint = footprint
        for k, t in enumerate(step_thickness_cm):
            j = k if thin_first else n_steps - 1 - k
            a = s0 + j * step
            sel = ((s > a + (1 - step_fraction) / 2 * step) & (s < a + (1 + step_fraction) / 2 * step)
                   & (np.abs(n) < width_fraction * half_width))
            self.add(f'Wedge {t:g} cm', full(sel))
        self.short_labels = {f'Wedge {t:g} cm': f'{t:g}' for t in step_thickness_cm}

        exclude = (labels > 0) & ~blob
        self.add('Wedge background', background_band(footprint, exclude, mm, bg_inner_mm, bg_outer_mm))

        # ROI validation: the wedge length and a signal that rises with step thickness
        means = [image[self.mask == k].mean() for k in range(1, n_steps + 1)]
        self.step_means = dict(zip(step_thickness_cm, means))
        self.valid = bool(np.all(np.diff(means) > 0)) and 40 < self.length_mm < 80
        if not self.valid:
            print(f"Warning: wedge ROI check failed (length {self.length_mm:.0f} mm, step means {np.round(means, 3)})")
