"""ROI masks of the phantoms.

Phantom_Mask: a phantom with a visible bone (or a bare bone, soft_tissue=False): bone (shaft + head), muscle and fat
(prototype 2 and the final phantom). soft_tissue_trend gives the soft tissue under a bone cast inside a cylinder.
Final_Phantom_Mask: prototype 3 (the 'final phantom' of 21 October), where the bone only shows where the CaCO3 settled.

Phantom_Mask

Layers, from the inside out: bone -> grey muscle layer -> thin fat layer -> background.
* Phantom outline: every pixel above the background noise that is connected to the bone (the Otsu object).
* Bone: with soft tissue (soft_tissue=True), Otsu's threshold of the pixels inside the outline separates bone from
  soft tissue; for a bare bone (soft_tissue=False) the Otsu object is the bone and the muscle/fat ROIs stay empty. The bone is split along its long axis into the shaft and the head
  at the point where it widens (head_width_ratio x the shaft width).
* Fat: the band of thickness fat_thickness_mm just inside the outline. In projection, rays this close to the edge
  pass through the fat layer only, so the band width equals the fat layer thickness. If the thickness is not given
  it is estimated from the image (see estimate_fat_thickness).
* Muscle: everything between the fat band and the bone.
Every ROI is kept margin_mm away from the boundaries between layers (edge blur / edge enhancement).
"""
import numpy as np
from scipy import ndimage as ndi

from .Segmentation import RoiMask, smooth, otsu_threshold, principal_axis, background_band, background_noise


def phantom_outline(image, labels, regions, pixel_spacing, outline_sigma=6):
    """Seed (the largest Otsu object, the phantom), outline (every pixel above the background noise connected to it)
    and the smoothed image."""
    mm = pixel_spacing
    region = max(regions, key=lambda r: r.area)
    seed = ndi.binary_fill_holes(labels == region.label)
    sm = smooth(image, mm)
    bg, noise = background_noise(sm)
    low = ndi.binary_opening(sm > bg + outline_sigma * noise, iterations=max(1, int(round(1 / mm))))
    low_labels, _ = ndi.label(low)
    ids = np.unique(low_labels[seed])
    outline = ndi.binary_fill_holes(np.isin(low_labels, ids[ids > 0]) | seed)
    return seed, outline, sm


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def estimate_fat_thickness(sm, soft, d_out, pixel_spacing, max_mm=20):
    """Distance from the outline at which the soft-tissue profile first rises above Otsu's fat/muscle threshold."""
    threshold = otsu_threshold(sm[soft])
    bins = np.round(d_out[soft] / pixel_spacing).astype(int)
    total = np.bincount(bins, sm[soft])
    count = np.bincount(bins)
    profile = total / np.maximum(count, 1)
    for i in range(1, min(len(profile), int(max_mm / pixel_spacing))):
        if count[i] > 0 and profile[i] > threshold:
            return i * pixel_spacing
    raise RuntimeError("Could not estimate the fat layer thickness; pass fat_thickness_mm instead")


def fat_and_muscle(contrast, soft, d_out, pixel_spacing, edge_mm=1.0, levels=(0.25, 0.75)):
    """Fat and muscle pixels from a contrast in which fat reads lower than muscle whatever the path length (e.g. the low
    over the high kVp contrast, which follows the spectral ratio of the material). Otsu's threshold splits the soft-tissue
    pixels (more than edge_mm inside the outline) into a fat and a muscle class; a pixel is fat when its contrast lies less
    than levels[0] of the way from the fat to the muscle class median, muscle when it lies more than levels[1] of the way.
    The rays in between cross both layers and are left out. The test is pixel by pixel, so the ROIs follow a fat layer of
    uneven thickness. Returns the fat and muscle masks."""
    inside = soft & (d_out > edge_mm)
    v = contrast[inside]
    t = otsu_threshold(v)
    fat_level, muscle_level = np.median(v[v <= t]), np.median(v[v > t])
    lo, hi = (fat_level + f * (muscle_level - fat_level) for f in levels)
    speck = max(1, int(round(1 / pixel_spacing)))
    return (ndi.binary_opening(inside & (contrast < lo), iterations=speck),
            ndi.binary_opening(inside & (contrast > hi), iterations=speck))


class Phantom_Mask(RoiMask):
    """A class to represent the phantom mask: 'Bone shaft', 'Bone head', 'Muscle', 'Fat' + 'Phantom background'."""

    def __init__(self, image, labels, regions, pixel_spacing=0.2, soft_tissue=True, fat_thickness_mm=None,
                 margin_mm=2.0, head_width_ratio=1.3, outline_sigma=6, bg_inner_mm=25, bg_outer_mm=35,
                 fat_contrast=None):
        """fat_contrast: an image in which fat reads lower than muscle whatever the path length (e.g. the low over the
        high kVp contrast). When given, the fat and muscle ROIs are found pixel by pixel from it (fat_and_muscle) instead
        of a band of fixed thickness under the outline: in a cylinder the grey level falls towards the edge for any
        material, and a fat layer of uneven thickness has no single band width."""
        super().__init__(image.shape, pixel_spacing)
        mm = pixel_spacing
        m = margin_mm / mm

        seed, outline, sm = phantom_outline(image, labels, regions, mm, outline_sigma)
        self.footprint = outline

        self.soft_tissue = soft_tissue

        # Bone
        if self.soft_tissue:
            bone = largest_component((sm > otsu_threshold(sm[outline])) & outline)
            bone = ndi.binary_fill_holes(bone)
        else:
            bone = seed
        self.bone = bone

        # Shaft / head split: width profile along the bone's long axis, head at the wider end
        centre, axis = principal_axis(bone)
        yy, xx = np.nonzero(bone)
        s = (yy - centre[0]) * axis[0] + (xx - centre[1]) * axis[1]
        bins = np.round(s - s.min()).astype(int)
        width = ndi.uniform_filter1d(np.bincount(bins).astype(float), max(1, int(round(2 / mm))))
        third = len(width) // 3
        if width[:third].mean() > width[-third:].mean():           # head at the low-s end: flip the axis
            axis, s = -axis, -s
            bins = np.round(s - s.min()).astype(int)
            width = ndi.uniform_filter1d(np.bincount(bins).astype(float), max(1, int(round(2 / mm))))
        n_len = len(width)
        shaft_width = np.median(width[int(0.05 * n_len):int(0.35 * n_len)])
        peak = np.argmax(width)                                     # widest point, in the head
        if width[peak] < head_width_ratio * shaft_width:
            raise RuntimeError("Bone head not found: the bone does not widen along its axis")
        # the head is the run of wide bins that ends at the widest point; it starts after the last narrow bin
        narrow = np.nonzero(width[:peak] <= head_width_ratio * shaft_width)[0]
        split = s.min() + (narrow[-1] + 1 if len(narrow) else 0)    # s position where the head starts
        self.shaft_width_mm = shaft_width * mm
        self.bone_length_mm = n_len * mm
        self.shaft_length_mm = (split - s.min()) * mm
        self.axis = axis
        self.split_s = split                                        # along_axis() value where the head starts

        s_img = np.zeros(image.shape)                               # s for every bone pixel
        s_img[yy, xx] = s
        d_in_bone = ndi.distance_transform_edt(bone)
        core = bone & (d_in_bone > m)
        self.add('Bone shaft', core & (s_img < split - m))
        self.add('Bone head', core & (s_img > split + m))

        # Muscle and fat
        if self.soft_tissue:
            d_out = ndi.distance_transform_edt(outline) * mm        # distance from the outside of the phantom
            d_bone = ndi.distance_transform_edt(~bone) * mm         # distance from the bone
            soft = outline & ~bone & (d_bone > margin_mm)
            if fat_contrast is not None:
                fat, muscle = fat_and_muscle(smooth(fat_contrast, mm), soft, d_out, mm)
                fat_thickness_mm = float(np.percentile(d_out[fat], 95))        # depth reached by the pure-fat rays
                print(f"Fat and muscle from the spectral contrast: pure-fat rays reach up to {fat_thickness_mm:.1f} mm "
                      f"inside the outline (95th percentile)")
                self.add('Muscle', muscle)
                self.add('Fat', fat)
            else:
                if fat_thickness_mm is None:
                    fat_thickness_mm = estimate_fat_thickness(sm, soft, d_out, mm)
                    print(f"Estimated fat layer thickness: {fat_thickness_mm:.1f} mm (pass fat_thickness_mm to override)")
                fat_margin = min(margin_mm, fat_thickness_mm / 4)
                self.add('Muscle', soft & (d_out > fat_thickness_mm + margin_mm))
                self.add('Fat', soft & (d_out > fat_margin) & (d_out < fat_thickness_mm - fat_margin))
        else:
            print("Bare bone (soft_tissue=False): the muscle and fat ROIs are left empty")
            self.names += ['Muscle', 'Fat']
        self.fat_thickness_mm = fat_thickness_mm

        exclude = (labels > 0) & ~outline
        self.add('Phantom background', background_band(outline, exclude, mm, bg_inner_mm, bg_outer_mm))

        # ROI validation: bone brighter than muscle, muscle brighter than fat
        means = {n: image[self.roi(n)].mean() for n in self.names if self.roi(n).any()}
        self.roi_means = means
        order = [n for n in ['Bone shaft', 'Muscle', 'Fat', 'Phantom background'] if n in means]
        self.valid = all(means[a] > means[b] for a, b in zip(order, order[1:]))
        if not self.valid:
            print(f"Warning: phantom ROI check failed, mean values {({n: round(v, 3) for n, v in means.items()})}")
        self.centre = centre

    def along_axis(self):
        """Position of every pixel along the bone's long axis (px from the bone's centre, increasing towards the head;
        the head starts at split_s)."""
        yy, xx = np.indices(self.mask.shape)
        return (yy - self.centre[0]) * self.axis[0] + (xx - self.centre[1]) * self.axis[1]

    def soft_tissue_trend(self, image, **kwargs):
        """The soft-tissue level under the bone, interpolated from the muscle ROI on both sides of it (see
        soft_tissue_trend below)."""
        return soft_tissue_trend(image, self.roi('Muscle'), self.centre, self.axis, self.pixel_spacing, **kwargs)


def soft_tissue_trend(image, muscle, centre, axis, pixel_spacing, degree=2, window_mm=2.0, min_pixels=50):
    """The soft-tissue level under a bone, interpolated from the muscle pixels on both sides of it.

    For a bone cast inside a cylinder of soft tissue (the final phantom) the muscle next to the bone has a shorter path
    than the soft tissue the bone sits in, so the muscle ROI is not a fair reference. Across the bone (n, the distance from
    the line through centre along axis, the bone's long axis) a polynomial of the given degree is fitted to the muscle
    pixels of every 1-px slice along the axis (pooled over window_mm) and evaluated at every pixel of that slice. NaN where
    a slice has fewer than min_pixels muscle pixels on either side of the line. The fit is linear in the image, so the trend
    of T_H - w T_L is trend(T_H) - w trend(T_L)."""
    yy, xx = np.indices(image.shape)
    normal = np.array([-axis[1], axis[0]])
    s = (yy - centre[0]) * axis[0] + (xx - centre[1]) * axis[1]
    n = ((yy - centre[0]) * normal[0] + (xx - centre[1]) * normal[1]) / 100     # scaled for conditioning
    bins = np.round(s - s.min()).astype(int)
    n_bins = bins.max() + 1
    b, x, v = bins[muscle], n[muscle], image[muscle]
    window = max(1, int(round(window_mm / pixel_spacing)))

    def pooled(weights):
        return ndi.uniform_filter1d(np.bincount(b, weights, minlength=n_bins), window, mode='constant')

    powers = [pooled(x ** k) for k in range(2 * degree + 1)]
    rhs = [pooled(v * x ** k) for k in range(degree + 1)]
    M = np.stack([np.stack([powers[j + k] for k in range(degree + 1)], -1) for j in range(degree + 1)], -2)
    y = np.stack(rhs, -1)
    support = np.minimum(pooled((x > 0).astype(float)), pooled((x < 0).astype(float))) * window >= min_pixels
    coef = np.full((n_bins, degree + 1), np.nan)
    coef[support] = np.linalg.solve(M[support], y[support][..., None])[..., 0]
    c = coef[bins]
    return sum(c[..., k] * n ** k for k in range(degree + 1))


class Final_Phantom_Mask(RoiMask):
    """A class to represent the final phantom mask: 'Bone head', 'Bone joint', a soft-tissue reference for each
    ('Soft tissue (head)', 'Soft tissue (joint)') + 'Phantom background'.

    Bone and muscle are both resin, and the CaCO3 settled at the head and at the head-shaft joint, so only those two
    regions show bone. The phantom is an upright cylinder, so the path length (and the grey level) changes from column
    to column: the median of each column inside the phantom is the soft-tissue trend, and the bone is the part of the
    residual more than bone_sigma robust standard deviations above it. Bone regions are named from the top of the image
    down (bone_names). Each soft-tissue reference is the bone ROI moved straight down (or up, if there is no room
    below) by its own height + gap_mm: the same columns, so the same path length through the cylinder. Dark voids
    (air bubbles) are kept out of the soft-tissue ROIs.
    axis: the direction of the cylinder's long axis in the image. 'vertical' (21 October) as above; 'horizontal'
    (25 October, head on the left) works on the transposed image, so the trend is taken along each row, the bone
    regions are named from the left and the references move along the row.
    """

    def __init__(self, image, labels, regions, pixel_spacing=0.24, bone_names=('head', 'joint'), interior_mm=5.0,
                 bone_sigma=5.0, min_bone_mm2=100, margin_mm=2.0, gap_mm=3.0, outline_sigma=6, bg_inner_mm=25,
                 bg_outer_mm=35, axis='vertical'):
        if axis not in ('vertical', 'horizontal'):
            raise ValueError(f"axis must be 'vertical' or 'horizontal', not {axis!r}")
        if axis == 'horizontal':
            image, labels = image.T, labels.T
        super().__init__(image.shape, pixel_spacing)
        mm = pixel_spacing
        m = max(1, int(round(margin_mm / mm)))

        _, outline, sm = phantom_outline(image, labels, regions, mm, outline_sigma)
        self.footprint = outline
        interior = ndi.distance_transform_edt(outline) * mm > interior_mm

        # Soft-tissue trend per column (columns that cross the interior only) and the residual
        trend = np.zeros(image.shape[1])
        inside = interior.any(axis=0)
        trend[inside] = np.nanmedian(np.where(interior[:, inside], sm[:, inside], np.nan), axis=0)
        residual = np.where(interior, sm - trend[None, :], 0.0)
        self.residual_sigma = 1.4826 * np.median(np.abs(residual[interior]))
        self.bone_threshold = bone_sigma * self.residual_sigma

        # Bone regions: above the threshold, at least min_bone_mm2, ordered from the top of the image down
        bone = ndi.binary_opening(interior & (residual > self.bone_threshold), iterations=max(1, int(round(1 / mm))))
        lab, n = ndi.label(bone)
        parts = [lab == k for k in range(1, n + 1) if (lab == k).sum() * mm ** 2 >= min_bone_mm2]
        parts.sort(key=lambda p: np.nonzero(p)[0].mean())
        names = list(bone_names) if len(parts) == len(bone_names) else [str(k) for k in range(1, len(parts) + 1)]
        if len(parts) != len(bone_names):
            print(f"Warning: {len(parts)} bone regions found instead of {len(bone_names)}; named by position instead")
        self.bone = np.any(parts, axis=0) if parts else np.zeros(image.shape, bool)

        voids = ndi.binary_dilation(interior & (residual < -self.bone_threshold), iterations=m)
        allowed = interior & ~ndi.binary_dilation(self.bone, iterations=int(round(gap_mm / mm))) & ~voids

        soft = {}
        for name, part in zip(names, parts):
            roi = ndi.binary_erosion(part, iterations=m)
            self.add(f'Bone {name}', roi)
            rows = np.nonzero(roi)[0]
            step = rows.max() - rows.min() + 1 + int(round(gap_mm / mm))
            down, up = np.zeros_like(roi), np.zeros_like(roi)
            down[step:] = roi[:-step]
            up[:-step] = roi[step:]
            soft[name] = max(down & allowed, up & allowed, key=np.sum)
        for name in names:
            self.add(f'Soft tissue ({name})', soft[name])

        exclude = (labels > 0) & ~outline
        self.add('Phantom background', background_band(outline, exclude, mm, bg_inner_mm, bg_outer_mm))

        # ROI validation: every bone region brighter than its soft-tissue reference, which is brighter than background
        means = {n: image[self.roi(n)].mean() for n in self.names if self.roi(n).any()}
        self.roi_means = means
        self.valid = bool(parts) and all(
            means.get(f'Bone {n}', 0) > means.get(f'Soft tissue ({n})', np.inf) > means['Phantom background']
            for n in names)
        if not self.valid:
            print(f"Warning: final phantom ROI check failed, mean values {({n: round(v, 3) for n, v in means.items()})}")
        if axis == 'horizontal':                                    # back to the orientation of the image
            self.mask, self.footprint, self.bone = (np.ascontiguousarray(a.T) for a in (self.mask, self.footprint,
                                                                                        self.bone))
