"""Region split of the scans: on 21 October every frame holds the ice trays with the step wedge, the plan B phantom and the
final phantom; on 25 October prototype 3, the wedge and the final phantom (SESSION_BOXES_MM). The objects are always in the
same places, so each one gets a fixed box and is processed on its own.

The boxes are in mm from the top-left corner of the image (top, left, bottom, right). They leave about 30 mm around
each object, which covers the shifts between scans (the scan start moves the objects by up to 15 mm). The wedge lies
above the red tray, so it is in the tray box; the phantoms are on the left, well clear of it.

Before a box is used its contents are checked: an object cut by the box edge is dropped by the segmentation, so a
moved object shows up as a missing one rather than as a wrong measurement.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .Segmentation import label_objects
from .Ice_tray_mask import find_cells
from .Wedge_mask import find_wedge

BOXES_MM = {
    'Trays': (80, 245, 440, 555),               # red + blue tray and the step wedge above them
    'Plan B phantom': (55, 70, 280, 245),       # upper left
    'Final phantom': (280, 60, 525, 245),       # lower left
}
PHANTOM_MIN_AREA_MM2 = 3000                     # a phantom is far larger than a tray cell (~800 mm^2)

# The boxes of each scan session, with the phantoms under their final names: prototype 2 = the plan B phantom (silicone
# muscle), prototype 3 = the 21 October 'final phantom' (sedimented CaCO3) and the final phantom (eggshell bone in resin
# and candle wax). On 25 October the phantoms lie side by side (axes along the rows) with the wedge, upright, between them.
SESSION_BOXES_MM = {
    '21 Oct': {'Trays': BOXES_MM['Trays'], 'Prototype 2': BOXES_MM['Plan B phantom'],
               'Prototype 3 (21 Oct)': BOXES_MM['Final phantom']},
    '25 Oct': {'Wedge': (285, 285, 405, 375),            # wedge alone (no trays in these frames)
               'Prototype 3 (25 Oct)': (245, 20, 440, 306),     # left
               'Final phantom': (270, 345, 460, 600)},          # right
}


class Region:
    """One box of the frame: .slices crops an image to it, .labels / .objects are its Otsu objects (box coordinates)."""

    def __init__(self, name, box_mm, image, threshold, pixel_spacing):
        self.name = name
        self.box_mm = box_mm
        top, left, bottom, right = (int(round(v / pixel_spacing)) for v in box_mm)
        if bottom > image.shape[0] or right > image.shape[1]:
            raise RuntimeError(f"{name}: box {box_mm} mm reaches past the image ({np.array(image.shape) * pixel_spacing} mm)")
        self.slices = (slice(top, bottom), slice(left, right))
        self.labels, self.objects = label_objects(self.crop(image), threshold, pixel_spacing)
        self.check(pixel_spacing)

    def crop(self, image):
        return image[self.slices]

    def check(self, pixel_spacing):
        """The box must hold what it is meant to: 20 tray cells (21 with water) and the wedge, the wedge alone (a box
        named 'Wedge', for frames without trays) or one phantom."""
        mm = pixel_spacing
        if self.name == 'Trays':
            n = len(find_cells(self.objects, mm))
            find_wedge(self.objects, mm)                                    # raises if the wedge is missing
            if n not in (20, 21):
                raise RuntimeError(f"Trays box holds {n} tray cells instead of 20 (21 with water): has a tray moved?")
        elif self.name == 'Wedge':
            find_wedge(self.objects, mm)
        else:
            n = sum(r.area * mm ** 2 > PHANTOM_MIN_AREA_MM2 for r in self.objects)
            if n != 1:
                raise RuntimeError(f"{self.name} box holds {n} phantoms instead of 1: has the phantom moved out of its box?")


def split(image, threshold, pixel_spacing, boxes=BOXES_MM):
    """{name: Region} for every box, after checking its contents."""
    return {name: Region(name, box, image, threshold, pixel_spacing) for name, box in boxes.items()}


def show_regions(image, regions, pixel_spacing, ax=None, title=None):
    """The frame with the region boxes drawn on it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap='gray')
    for k, region in enumerate(regions.values()):
        top, left, bottom, right = (v / pixel_spacing for v in region.box_mm)
        colour = plt.get_cmap('tab10')(k)
        ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=colour, linewidth=2))
        ax.text(left + 10, top + 40, region.name, color=colour, fontsize=9, fontweight='bold')
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax
