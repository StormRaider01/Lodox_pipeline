"""ROI masks of the phantoms.

Phantom_Mask: a phantom with a visible bone (or a bare bone, soft_tissue=False): bone (shaft + head), muscle and fat.
Final_Phantom_Mask: the final phantom of 21 October, where the bone only shows where the CaCO3 settled.

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


class Phantom_Mask(RoiMask):
    """A class to represent the phantom mask: 'Bone shaft', 'Bone head', 'Muscle', 'Fat' + 'Phantom background'."""

    def __init__(self, image, labels, regions, pixel_spacing=0.2, soft_tissue=True, fat_thickness_mm=None,
                 margin_mm=2.0, head_width_ratio=1.3, outline_sigma=6, bg_inner_mm=25, bg_outer_mm=35):
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
    """

    def __init__(self, image, labels, regions, pixel_spacing=0.24, bone_names=('head', 'joint'), interior_mm=5.0,
                 bone_sigma=5.0, min_bone_mm2=100, margin_mm=2.0, gap_mm=3.0, outline_sigma=6, bg_inner_mm=25,
                 bg_outer_mm=35):
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
