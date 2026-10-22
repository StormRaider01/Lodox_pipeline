"""ROI mask of the two ice trays (2 x 6 cells each) in the 21 October scans.

Both trays stand upright side by side: the red tray (fat and muscle substitutes) on the left, the blue tray (bone
substitutes) on the right. Cell IDs follow the notebook sketches: rows 1-6 from top to bottom, columns L/R, e.g. 'Red 3L'.
The filled cells that pass the threshold are detected by their size and grouped into the two trays (the widest gap
between cells separates them). A grid is fitted to each tray, so every cell (filled, water or empty) gets an ROI, and
the detected cells are matched to the tray layout to number the rows.
"""
import numpy as np
from skimage.draw import polygon

from .Segmentation import RoiMask, principal_axis, background_band

ROWS = ['1', '2', '3', '4', '5', '6']
COLS = ['L', 'R']

# Contents of every cell. 'optional' cells are only filled on some scans: a detected cell there is filled (the water
# cell, filled from scan I112 on), otherwise the cell is empty. Cells in neither are empty.
TRAYS = {
    'red': dict(name='Red', optional={'Red 1R': 'Water reference'}, water='Red 1R', layout={
        'Red 3L': 'Resin', 'Red 3R': 'Resin + red food colouring',
        'Red 4L': 'Candle wax', 'Red 4R': 'Silicone putty',
        'Red 5L': 'Coconut oil + beeswax (1:1)', 'Red 5R': 'Beeswax + coconut oil (2:1)',
        'Red 6L': 'Coconut oil', 'Red 6R': 'Beeswax'}),
    'blue': dict(name='Blue', optional={}, water=None, layout={
        'Blue 1L': 'CaCO3 + resin (4:25)', 'Blue 1R': 'CaCO3 + resin (6:25)',
        'Blue 2L': 'CaCO3 + resin (1:25)', 'Blue 2R': 'CaCO3 + resin (2:25)',
        'Blue 3L': 'CaCO3 + PoP + water (1:1:2.5)', 'Blue 3R': 'CaCO3 + PoP + water (3:1:7.5)',
        'Blue 4L': 'CaCO3 + PoP + water (2:1:5)', 'Blue 4R': 'CaCO3 + PoP + water (1:1:10)',
        'Blue 5L': 'Eggshell + PoP + water (3:1:10)', 'Blue 5R': 'Eggshell + PoP + water (1:1:125)',
        'Blue 6L': 'Eggshell + PoP + water (1:1:2.5)', 'Blue 6R': 'Eggshell + PoP + water (2:1:7.5)'}),
}


def cell_id(tray, row, col):
    return f"{TRAYS[tray]['name']} {ROWS[row]}{COLS[col]}"


def find_cells(regions, pixel_spacing):
    """Tray cells above the threshold: about 23 x 44 mm, so 15-35 mm wide, 28-50 mm long and 400-1500 mm^2."""
    mm = pixel_spacing
    return [r for r in regions if 15 < r.axis_minor_length * mm < 35 and 28 < r.axis_major_length * mm < 50
            and 400 < r.area * mm ** 2 < 1500]


def group_trays(cells, n_trays):
    """Split the cells into n_trays trays side by side (left to right) at the widest gaps between cell centres."""
    cells = sorted(cells, key=lambda r: r.centroid[1])
    x = np.array([r.centroid[1] for r in cells])
    cuts = np.sort(np.argsort(np.diff(x))[len(x) - n_trays:]) + 1
    return [list(g) for g in np.split(np.array(cells, dtype=object), cuts)]


def grid_index(p, size):
    """Grid index (0, 1, ...) of the positions p along one tray axis; size = cell size along that axis.

    Cells closer than half a cell size belong to the same row/column; the pitch is the smallest step between them.
    """
    order = np.argsort(p)
    group = np.concatenate([[0], np.cumsum(np.diff(p[order]) > 0.5 * size)])
    centres = np.array([p[order][group == g].mean() for g in range(group[-1] + 1)])
    if len(centres) < 2:
        raise RuntimeError("The detected tray cells are all in one row or column; the grid cannot be fitted")
    pitch = np.min(np.diff(centres))
    index = np.empty(len(p), int)
    index[order] = np.round((centres[group] - centres[0]) / pitch).astype(int)
    return index


def fit_grid(centres, rows, cols):
    """Least-squares grid centre = origin + col * U + row * V; returns origin, U, V and the largest residual."""
    A = np.zeros((2 * len(centres), 6))
    A[0::2, 0], A[1::2, 1] = 1, 1
    A[0::2, 2], A[1::2, 3] = cols, cols
    A[0::2, 4], A[1::2, 5] = rows, rows
    p, *_ = np.linalg.lstsq(A, centres.ravel(), rcond=None)
    residual = np.max(np.linalg.norm((A @ p - centres.ravel()).reshape(-1, 2), axis=1))
    return p[0:2], p[2:4], p[4:6], residual


def rectangle(centre, u, v, half_u, half_v, shape):
    """Pixel indices (rows, cols) of a rectangle with half-sizes half_u, half_v along the unit vectors u, v."""
    corners = np.array([centre + su * half_u * u + sv * half_v * v
                        for su, sv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    return polygon(corners[:, 0], corners[:, 1], shape)


class Tray_Mask(RoiMask):

    """A class to represent one ice tray: one ROI per cell ('Red 1L' ... 'Red 6R') + '<Red/Blue> tray background'.

    cells: the detected cells of this tray (from find_cells/group_trays).
    """

    def __init__(self, image, labels, cells, pixel_spacing=0.24, tray='red', margin_mm=3.0, rim_mm=12.0,
                 bg_inner_mm=10.0, bg_outer_mm=20.0, max_residual=0.15):
        super().__init__(image.shape, pixel_spacing)
        mm = pixel_spacing
        spec = TRAYS[tray]
        self.tray = tray
        if len(cells) < 3:
            raise RuntimeError(f"{tray} tray: only {len(cells)} cells found above the threshold")

        # Orientation correction: the tray tilt is the cells' long-axis angle (the cells lie across the upright tray),
        # u runs along the image columns and v down the rows
        angles = []
        for r in cells:
            _, a = principal_axis(labels == r.label)
            angles.append((np.degrees(np.arctan2(a[0], a[1])) + 90) % 180 - 90)
        tilt = np.radians(np.median(angles))
        u = np.array([np.sin(tilt), np.cos(tilt)])
        v = np.array([np.cos(tilt), -np.sin(tilt)])
        self.rotation_deg = -np.degrees(tilt)                      # anticlockwise in the image is positive

        # Cell size from the detected cells (extent along u and v)
        half_u = np.median([np.ptp(np.argwhere(labels == r.label) @ u) for r in cells]) / 2
        half_v = np.median([np.ptp(np.argwhere(labels == r.label) @ v) for r in cells]) / 2

        # Row and column of each detected cell relative to the first one, then shifted down the tray until every
        # detected cell sits on a filled cell of the layout (the red tray's top rows are empty)
        centres = np.array([r.centroid for r in cells])
        rows = grid_index(centres @ v, 2 * half_v)
        cols = grid_index(centres @ u, 2 * half_u)
        filled = {(i, j) for i in range(len(ROWS)) for j in range(len(COLS))
                  if cell_id(tray, i, j) in spec['layout'] or cell_id(tray, i, j) in spec['optional']}
        shifts = [dr for dr in range(len(ROWS) - rows.max())
                  if all((i + dr, j) in filled for i, j in zip(rows, cols))]
        if len(shifts) != 1:
            raise RuntimeError(f"{tray} tray: the detected cells fit the layout in {len(shifts)} ways instead of one")
        rows = rows + shifts[0]
        origin, U, V, residual = fit_grid(centres, rows, cols)
        self.pitch_mm = (np.linalg.norm(U) * mm, np.linalg.norm(V) * mm)
        self.cell_size_mm = (2 * half_u * mm, 2 * half_v * mm)
        m = margin_mm / mm

        # Contents: the layout plus the optional cells that were detected (filled) on this scan
        detected = {cell_id(tray, i, j) for i, j in zip(rows, cols)}
        self.layout = dict(spec['layout'])
        self.layout.update({cid: text for cid, text in spec['optional'].items() if cid in detected})
        self.water_cell = spec['water'] if spec['water'] in self.layout else None

        self.centres = {}
        for i in range(len(ROWS)):
            for j in range(len(COLS)):
                cid = cell_id(tray, i, j)
                centre = origin + j * U + i * V
                self.centres[cid] = centre
                self.add(cid, self._region(centre, u, v, half_u - m, half_v - m))
        self.empty_cells = [cid for cid in self.centres if cid not in self.layout]

        # Tray footprint (all cells + rim) and the background band around it
        mid = origin + (len(COLS) - 1) / 2 * U + (len(ROWS) - 1) / 2 * V
        rim = rim_mm / mm
        self.footprint = self._region(mid, u, v, (len(COLS) - 1) / 2 * np.linalg.norm(U) + half_u + rim,
                                      (len(ROWS) - 1) / 2 * np.linalg.norm(V) + half_v + rim)
        exclude = (labels > 0) & ~self.footprint
        self.background_name = f"{spec['name']} tray background"
        self.add(self.background_name, background_band(self.footprint, exclude, mm, bg_inner_mm, bg_outer_mm))

        # ROI validation: the detected cells must sit on the fitted grid
        self.residual_px = residual
        self.valid = residual < max_residual * min(np.linalg.norm(U), np.linalg.norm(V))
        if not self.valid:
            print(f"Warning: {tray} tray cells do not fit a regular grid (residual {residual:.1f} px)")

    def _region(self, centre, u, v, half_u, half_v):
        out = np.zeros(self.mask.shape, bool)
        out[rectangle(centre, u, v, half_u, half_v, out.shape)] = True
        return out


class Trays(RoiMask):
    """Both ice trays as one mask (red on the left, blue on the right); each tray is also kept in .trays.

    .layout: {cell: contents} of the filled cells, .empty_cells, .water_cell (None if not filled in this scan) and
    .background: {cell: name of its tray's background ROI}.
    """

    def __init__(self, image, labels, regions, pixel_spacing=0.24, trays=('red', 'blue'), **kwargs):
        super().__init__(image.shape, pixel_spacing)
        groups = group_trays(find_cells(regions, pixel_spacing), len(trays))
        self.trays = {t: Tray_Mask(image, labels, g, pixel_spacing, tray=t, **kwargs) for t, g in zip(trays, groups)}

        self.layout, self.empty_cells, self.background = {}, [], {}
        self.water_cell = None
        for tm in self.trays.values():
            for name, roi in tm.rois.items():
                self.add(name, roi)
            self.layout.update(tm.layout)
            self.empty_cells += tm.empty_cells
            self.background.update({cid: tm.background_name for cid in tm.centres})
            self.water_cell = self.water_cell or tm.water_cell
        self.valid = all(tm.valid for tm in self.trays.values())
