"""Weighted log subtraction and the weight sweep (Methodology chapter, Bone Suppression section).

The subtraction runs on the PLA-equivalent thickness maps of Phase 4, S(x, y) = T_H(x, y) - w T_L(x, y) (eq. wls_T).

Signatures: the median of T in an ROI minus the median in its reference ROI, before any subtraction.
* bone signature b_i = bone ROI - the soft tissue next to it (final phantom: same columns, so the same path length
  through the cylinder; plan B phantom: the silicone muscle around the bone) or - the background for a bare bone. For
  a bone cast along the axis of a soft-tissue cylinder (the final phantom) the muscle beside it has a shorter path, so
  the reference is the soft-tissue trend interpolated under the bone (TREND), pixel by pixel;
* soft-tissue signature s_i = soft-tissue ROI - background.
w_bone = b_H / b_L is the weight that cancels the bone signature if the subtraction is exactly linear.

Weight sweep: S is computed for w from 0 to 2 w_bone and three quantities are recorded at each step (eq. residuals),
all measured on S itself with the same ROIs as the signatures:
* bone residual = bone signature of S / b_H x 100 % (100 % at w = 0, 0 % once the bone has gone);
* soft retention = soft-tissue signature of S / s_H x 100 %;
* background noise sigma_S = standard deviation of S in the background, with the prediction sqrt(sigma_H^2 + w^2 sigma_L^2)
  (the two scans have independent noise).
w_opt is where the bone residual crosses zero, found by linear interpolation between the two sweep steps either side,
so the step size sets the smoothness of the curves rather than the accuracy of w_opt.

Theory: w_wedge = mu_b(E_H)/mu_b(E_L) x mu_PLA(E_L)/mu_PLA(E_H) (eq. w_wedge). Only mass attenuation coefficients are
needed (the densities cancel). The effective energies are unknown, so w_wedge is evaluated for E = f x kVp with the same
fraction f for both presets over f = 0.4-0.6 and given as a band.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xraydb

# Cured bone materials as {chemical formula: mass in g}. Eggshell is taken as CaCO3. Plaster of Paris (CaSO4.0.5H2O,
# 145.15 g/mol) sets to gypsum (CaSO4.2H2O, 172.17 g/mol) and binds 1.5 H2O per formula unit; the rest of the mixing water
# is assumed to stay in the cast.
POP, GYPSUM, WATER = 145.15, 172.17, 18.015
BONE_MATERIALS = {
    'Eggshell + PoP + water (1:1:2.5)': {'CaCO3': 10, 'CaSO4H4O2': 10 * GYPSUM / POP,
                                         'H2O': 2.5 - 10 * 1.5 * WATER / POP},
    'CaCO3': {'CaCO3': 1},
}
PLA = {'C3H4O2': 1}
TREND = 'Soft-tissue trend'         # bone reference: the soft tissue interpolated under the bone (Weight_Sweep trend)


def mass_attenuation(material, energy_kev):
    """mu/rho (cm^2/g) of a material {formula: mass} by the mixture rule (mass fractions x mu/rho of each component)."""
    total = sum(material.values())
    return sum(m / total * xraydb.material_mu(formula, energy_kev * 1000, density=1.0) for formula, m in material.items())


def w_wedge(material, kvp_low, kvp_high, fractions=np.linspace(0.4, 0.6, 21)):
    """Theoretical weight of eq. w_wedge for effective energies E = f x kVp; returns (fractions, w for each f)."""
    w = np.array([mass_attenuation(material, f * kvp_high) / mass_attenuation(material, f * kvp_low)
                  * mass_attenuation(PLA, f * kvp_low) / mass_attenuation(PLA, f * kvp_high) for f in fractions])
    return fractions, w


def tray_weight(cell_results, calibration, cell, displaced=None):
    """Weight that cancels a 28 mm slab of a tray cell in the thickness maps: (C_H / a_H) / (C_L / a_L), with C the
    floor-subtracted contrast of the cell and a the straight-line wedge gain of each preset. displaced: a second cell
    whose material the first one replaces along the path (a bone cast inside the muscle); the weight then cancels the
    difference of the two slabs, (t_H - t'_H) / (t_L - t'_L)."""
    def slab(c, k):
        return cell_results[k].loc[c, 'net_contrast'] / calibration[k].a_linear

    t = {k: slab(cell, k) - (0 if displaced is None else slab(displaced, k)) for k in ('low', 'high')}
    return t['high'] / t['low']


def zero_crossing(x, y):
    """First x where y crosses zero (linear interpolation between samples); NaN if it never does."""
    k = np.nonzero(np.diff(np.sign(y)) != 0)[0]
    if not len(k):
        return np.nan
    k = k[0]
    return x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k])


class Weight_Sweep:
    """Weight sweep of one phantom.

    T_low, T_high: thickness maps (cm of PLA) of the low and the registered high kVp image; mask: the phantom's RoiMask.
    bones: {bone ROI: its reference ROI}; soft: {soft-tissue ROI: its reference ROI} (empty for a bare bone).
    weights: the w values to sweep (default: n_steps values from 0 to 2 w_bone, w_bone the largest of the bone ROIs).
    trend: a function giving the soft-tissue level under the bone from an image, linear in the image (e.g.
    Phantom_Mask.soft_tissue_trend); a bone whose reference is TREND is measured against it, pixel by pixel.
    """

    def __init__(self, T_low, T_high, mask, bones, soft=None, background='Phantom background', weights=None,
                 n_steps=201, label='', trend=None):
        self.T_low, self.T_high = T_low, T_high
        self.label = label
        self.bones, self.soft = bones, soft or {}
        self.rois = mask.rois
        bg = self.rois[background]
        self.trend = trend
        self.trend_low, self.trend_high = (None, None) if trend is None else (trend(T_low), trend(T_high))

        # Signatures before subtraction and w_bone = b_H / b_L
        self.b_low = {r: self.signature(T_low, r, ref, self.trend_low) for r, ref in bones.items()}
        self.b_high = {r: self.signature(T_high, r, ref, self.trend_high) for r, ref in bones.items()}
        self.s_low = {r: self.signature(T_low, r, ref) for r, ref in self.soft.items()}
        self.s_high = {r: self.signature(T_high, r, ref) for r, ref in self.soft.items()}
        self.w_bone = {r: self.b_high[r] / self.b_low[r] for r in bones}
        self.sigma_low, self.sigma_high = np.std(T_low[bg]), np.std(T_high[bg])

        # The sweep: subtract at every weight and measure the subtracted image S
        self.weights = np.linspace(0, 2 * max(self.w_bone.values()), n_steps) if weights is None else np.asarray(weights)
        self.bone_residual = {r: np.zeros(len(self.weights)) for r in bones}
        self.soft_retention = {r: np.zeros(len(self.weights)) for r in self.soft}
        self.noise = np.zeros(len(self.weights))
        for j, w in enumerate(self.weights):
            S = self.subtract(w)
            S_trend = None if trend is None else self.trend_high - w * self.trend_low
            for r, ref in bones.items():
                self.bone_residual[r][j] = 100 * self.signature(S, r, ref, S_trend) / self.b_high[r]
            for r, ref in self.soft.items():
                self.soft_retention[r][j] = 100 * self.signature(S, r, ref) / self.s_high[r]
            self.noise[j] = np.std(S[bg])
        self.noise_predicted = np.sqrt(self.sigma_high ** 2 + self.weights ** 2 * self.sigma_low ** 2)

        # w_opt: the bone residual crosses zero
        self.w_opt = {r: zero_crossing(self.weights, self.bone_residual[r]) for r in bones}
        for r, w in self.w_opt.items():
            if np.isnan(w):
                print(f"Warning: {label} {r}: the bone residual does not cross zero between w = 0 and {self.weights[-1]:.3g}")

    def signature(self, image, roi, reference, image_trend=None):
        """Median of an ROI minus the median of its reference ROI, or, for the reference TREND, the median of the ROI
        minus the soft-tissue trend of the same image (image_trend)."""
        if reference == TREND:
            return np.nanmedian((image - image_trend)[self.rois[roi]])
        return np.median(image[self.rois[roi]]) - np.median(image[self.rois[reference]])

    def bone_contrast(self, image, roi):
        """The bone signature of any image (e.g. a denoised S) measured as in the sweep."""
        ref = self.bones[roi]
        return self.signature(image, roi, ref, None if ref != TREND else self.trend(image))

    def subtract(self, w):
        """S = T_H - w T_L (eq. wls_T)."""
        return self.T_high - w * self.T_low

    def subtract_parts(self, w, w_map, bone):
        """S with every part of the bone removed at its own weight: S = T_H - w T_L - (w_map - w) B_L inside bone, with
        B_L = T_L - trend(T_L) the bone's own contribution to T_L (needs the trend). The soft tissue is subtracted at w
        everywhere, so its retention stays uniform, while the bone of each part cancels at the local weight w_map
        (e.g. the head and the shaft, if they cancel at different weights)."""
        B_L = np.where(bone, self.T_low - self.trend_low, 0.0)
        return self.subtract(w) - np.nan_to_num((w_map - w) * B_L)

    def at(self, w, values):
        """A swept quantity at weight w (linear interpolation between the sweep steps)."""
        return float(np.interp(w, self.weights, values))

    def summary(self, retention_of=None):
        """One row per bone ROI: signatures, w_bone, w_opt and, at w_opt, the noise and the soft retention of the bone's
        reference ROI (the soft tissue next to it; NaN for a bare bone). retention_of: {bone ROI: soft-tissue ROI} to report
        instead, e.g. the muscle for a bone measured against the TREND."""
        rows = []
        for r, ref in self.bones.items():
            w = self.w_opt[r]
            ref = (retention_of or {}).get(r, ref)
            rows.append(dict(ROI=r, b_L=self.b_low[r], b_H=self.b_high[r], w_bone=self.w_bone[r], w_opt=w,
                             soft_retention_pct=self.at(w, self.soft_retention[ref]) if ref in self.soft else np.nan,
                             sigma_S=self.at(w, self.noise), sigma_S_predicted=self.at(w, self.noise_predicted)))
        return pd.DataFrame(rows).set_index('ROI')

    def show(self, axes=None, theory=None, marks=None):
        """Bone residual and soft retention (left) and background noise (right) against w.

        theory: (w_min, w_max) of the theoretical band; marks: {label: w} drawn as vertical lines (e.g. the tray weights).
        A single axes gets the bone residual and soft retention only.
        """
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(12, 4))
        if isinstance(axes, plt.Axes):
            axes = [axes, None]
        ax = axes[0]
        for k, (r, y) in enumerate(self.bone_residual.items()):
            ax.plot(self.weights, y, color=f'C{k}', label=f'{r}: bone residual')
            ax.axvline(self.w_opt[r], color=f'C{k}', ls=':', lw=1)
        for k, (r, y) in enumerate(self.soft_retention.items(), start=len(self.bone_residual)):
            ax.plot(self.weights, y, color=f'C{k}', ls='--', label=f'{r}: soft retention')
        ax.axhline(0, color='k', lw=0.8)
        if theory is not None:
            ax.axvspan(*theory, color='0.85', label='$w_{wedge}$ (theory, $E = 0.4$-$0.6$ kVp)')
        for k, (name, w) in enumerate((marks or {}).items()):
            ax.axvline(w, color='k', ls=['-.', (0, (5, 2, 1, 2)), (0, (1, 1))][k % 3], lw=1, label=f'{name} ({w:.3f})')
        ax.set_xlabel('Weight $w$')
        ax.set_ylabel('% of the high kVp signature')
        ax.set_title(f'{self.label}: weight sweep')
        ax.legend(fontsize=7)

        ax = axes[1]
        if ax is None:
            return axes
        ax.plot(self.weights, self.noise, 'k-', label=r'measured $\sigma_S$ (background)')
        ax.plot(self.weights, self.noise_predicted, 'r--', label=r'$\sqrt{\sigma_H^2 + w^2\sigma_L^2}$')
        ax.set_xlabel('Weight $w$')
        ax.set_ylabel(r'$\sigma_S$ (cm of PLA)')
        ax.set_title(f'{self.label}: background noise')
        ax.legend(fontsize=7)
        return axes
