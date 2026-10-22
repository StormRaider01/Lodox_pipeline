"""Wedge calibration: grey level -> PLA-equivalent thickness (Implementation chapter, Wedge Calibration section).

Beam hardening, scatter and any processing (e.g. Lucid) are treated as one unknown increasing function f, so the
contrast of a ray is C = f(mu d). The wedge gives C at five known PLA thicknesses:
* step contrast C_k = median(step k) - median(wedge background), t_k = k cm;
* least-squares fit through the origin C(t) = a t + b t^2 (a straight line C = a t is fitted alongside it so the
  improvement from the quadratic term can be reported);
* inverse t_eq(C) = 2C / (a + sqrt(a^2 + 4bC)), the root that stays valid as b -> 0 (t_eq = C / a).
The fit is accepted only if it rises across the whole range (a + 2 b t_max > 0, i.e. a + 10b > 0 for 5 cm), and the
inverse is only trusted for 0 <= t_eq <= t_max: contrasts outside it are flagged rather than extrapolated.

Ice trays: every cell becomes a PLA-equivalent thickness, the tray floor (empty cells) is subtracted, and
R = mu_cell / mu_water = (t_cell - t_floor) / (t_water - t_floor) for equal fill depths. Cells beyond the wedge range
(most of the plaster-of-Paris bone cells, 6-10 cm of PLA) take R from Beer-Lambert instead: the 21 October images are
unfiltered log images, so R is the ratio of the floor-subtracted contrasts of the cell and the water cell.

Thickness maps: T(x, y) = t_eq(image - background) inside the wedge range. The phantoms are thicker than the wedge, so
their maps use the straight-line gain, T = (image - background) / a, which only puts the two presets on a common
scale (cm of PLA) for the comparison of the subtraction weight with theory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .Wedge_mask import STEP_THICKNESS_CM


class Wedge_Calibration:
    """Wedge calibration of one image (wedge: the Wedge_Mask whose ROIs line up with this image)."""

    def __init__(self, image, wedge, label=''):
        self.label = label
        self.t = np.array(STEP_THICKNESS_CM, float)
        self.t_max = self.t.max()

        # Step contrasts: medians, which are less sensitive than means to texture and edge enhancement
        self.background = np.median(image[wedge.roi('Wedge background')])
        self.contrast = np.array([np.median(image[wedge.roi(f'Wedge {t:g} cm')]) for t in self.t]) - self.background

        # Quadratic through the origin, C = a t + b t^2, with the parameter covariance s^2 (A^T A)^-1
        A = np.c_[self.t, self.t ** 2]
        (self.a, self.b), *_ = np.linalg.lstsq(A, self.contrast, rcond=None)
        resid = self.contrast - A @ np.array([self.a, self.b])
        self.cov = resid @ resid / (len(self.t) - 2) * np.linalg.inv(A.T @ A)
        self.rms_quadratic = np.sqrt(np.mean(resid ** 2))

        # Straight line through the origin, C = a t, for comparison
        self.a_linear = (self.t @ self.contrast) / (self.t @ self.t)
        self.rms_linear = np.sqrt(np.mean((self.contrast - self.a_linear * self.t) ** 2))

        # Fit check: the contrast must rise with thickness across the whole calibrated range
        self.valid = bool(self.a > 0 and self.a + 2 * self.b * self.t_max > 0)
        if not self.valid:
            print(f"Warning: {label} wedge fit does not rise across 0-{self.t_max:g} cm (a = {self.a:.4g}, "
                  f"b = {self.b:.4g}); the calibration cannot be used")

        # Top of the calibrated range: the thickest step, or what its own contrast reads if the fit puts it a little higher
        self.t_top = max(self.t_max, float(self._invert(self.contrast[-1])))

    def fit(self, t):
        """Contrast of PLA thickness t (cm) from the quadratic fit."""
        return self.a * t + self.b * t ** 2

    def thickness(self, contrast, extrapolate=False):
        """PLA-equivalent thickness (cm) of contrast values (scalar or array).

        Outside 0 <= t_eq <= t_max the result is NaN unless extrapolate=True; it is always NaN where the quadratic
        never reaches the contrast (b < 0, past the top of the parabola).
        """
        t = self._invert(contrast)
        if not extrapolate:
            t = np.where(self.in_range(t), t, np.nan)
        return t if t.ndim else float(t)

    def _invert(self, contrast):
        """Root of C = a t + b t^2 in the form that stays valid as b -> 0 (NaN past the top of the parabola)."""
        c = np.asarray(contrast, float)
        disc = self.a ** 2 + 4 * self.b * c
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(disc >= 0, 2 * c / (self.a + np.sqrt(np.maximum(disc, 0))), np.nan)

    def in_range(self, t):
        """True where a thickness lies inside the calibrated range (0 to the thickest step)."""
        with np.errstate(invalid='ignore'):
            return (np.asarray(t) >= 0) & (np.asarray(t) <= self.t_top)

    def thickness_map(self, image):
        """T(x, y): PLA-equivalent thickness of every pixel (cm), extrapolated outside the wedge range (check with
        in_range); proportional to the log attenuation of the image."""
        return self.thickness(image - self.background, extrapolate=True)

    def linear_thickness_map(self, image, background=None):
        """T(x, y) = (image - background) / a from the straight-line fit (cm of PLA): valid at any thickness for
        unfiltered log images, used for the phantoms. background: the open-field level next to the object (defaults
        to the wedge background)."""
        return (image - (self.background if background is None else background)) / self.a_linear

    def summary(self):
        return (f"{self.label}: C(t) = {self.a:.4g} t + {self.b:.4g} t^2 (rms residual {self.rms_quadratic:.2g}); "
                f"straight line C = {self.a_linear:.4g} t (rms residual {self.rms_linear:.2g}); "
                f"{'rises' if self.valid else 'does NOT rise'} across 0-{self.t_max:g} cm")

    def show(self, ax=None):
        """Step contrasts with the quadratic and straight-line fits."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        tt = np.linspace(0, self.t_max * 1.05, 100)
        ax.plot(self.t, self.contrast, 'ko', label='Wedge steps')
        ax.plot(tt, self.fit(tt), 'C0-', label=f'$C = at + bt^2$ (rms {self.rms_quadratic:.2g})')
        ax.plot(tt, self.a_linear * tt, 'C1--', label=f'$C = at$ (rms {self.rms_linear:.2g})')
        ax.set_xlabel('PLA thickness $t$ (cm)')
        ax.set_ylabel('Contrast $C$')
        ax.set_title(self.label)
        ax.legend(fontsize=8)
        return ax


def relative_attenuation(cal, image, trays, depth_cm=2.8, water_depth_cm=2.8):
    """Relative attenuation of every filled tray cell (one row per cell).

    trays: the Ice_tray_mask.Trays whose ROIs line up with image. Each cell's contrast is taken against the background
    band of its own tray. The tray floor is the mean of the empty cells of the same tray, or of all empty cells when a
    tray has none (the blue tray: assumes the same floor). Columns:
        contrast, net_contrast = contrast - floor contrast, t_eq_cm (cm of PLA, NaN beyond the wedge range), in_range,
        mu_over_mu_PLA = (t_eq - t_floor) / d,
        R_wedge = mu/mu_water from the wedge calibration (cells inside the wedge range),
        R_beer_lambert = (C - C_floor) / (C_water - C_floor) x d_water / d (every cell),
        R = R_wedge inside the wedge range, R_beer_lambert beyond it, and R_source saying which.
    The R columns are NaN when the scan has no water cell.
    """
    rois = trays.rois
    bg = {name: np.median(image[rois[name]]) for name in set(trays.background.values())}
    contrast = {cid: np.median(image[rois[cid]]) - bg[trays.background[cid]] for cid in trays.background}
    t_all = {cid: cal.thickness(c, extrapolate=True) for cid, c in contrast.items()}

    def floor(values, tray_bg):
        """Mean over the empty cells of the same tray, else over all empty cells."""
        own = [values[c] for c in trays.empty_cells if trays.background[c] == tray_bg]
        return np.mean(own) if own else np.mean([values[c] for c in trays.empty_cells])

    rows = []
    for cid, contents in trays.layout.items():
        tray_bg = trays.background[cid]
        t = t_all[cid]
        ok = bool(cal.in_range(t))
        rows.append(dict(ROI=cid, contents=contents, contrast=contrast[cid], t_eq_cm=t if ok else np.nan, in_range=ok,
                         floor_from='same tray' if any(trays.background[c] == tray_bg for c in trays.empty_cells)
                         else 'other tray',
                         mu_over_mu_PLA=(t - floor(t_all, tray_bg)) / depth_cm if ok else np.nan,
                         net_contrast=contrast[cid] - floor(contrast, tray_bg)))
    df = pd.DataFrame(rows).set_index('ROI')

    w = trays.water_cell
    if w is None:
        df['R_wedge'] = df['R_beer_lambert'] = df['R'] = np.nan
        df['R_source'] = 'no water cell'
    else:
        df['R_wedge'] = df['mu_over_mu_PLA'] / (df.loc[w, 'mu_over_mu_PLA'] * depth_cm / water_depth_cm)
        df['R_beer_lambert'] = (df['net_contrast'] / depth_cm) / (df.loc[w, 'net_contrast'] / water_depth_cm)
        df['R'] = df['R_wedge'].where(df['in_range'], df['R_beer_lambert'])
        df['R_source'] = np.where(df['in_range'], 'wedge', 'Beer-Lambert')
    return df
