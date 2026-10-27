"""Validation stage (Implementation chapter, Validation section): CNR and SNR of the phantom before and after bone
suppression, comparison with theory and comparison with conventional radiographs of a real humerus.

Metrics (eq. metrics), the same functions for every image: SNR_k = median_k / sigma_bg and
CNR_kj = |median_k - median_j| / sigma_bg, with sigma_bg the standard deviation of the background ROI of that image.
The thickness maps are zero at the background band, so there the SNR of a region equals its CNR against the background.

Theory: the tray cells give the spectral ratio kappa = R_L / R_H, the relative attenuation (mu / mu_water) at the low kVp
over that at the high kVp. The density cancels in kappa, so it can be predicted from the recipe alone (mixture rule with
xraydb mass attenuation coefficients at E = f x kVp); the cured densities were not measured, so R itself cannot be predicted.
kappa > 1 for materials with a higher effective atomic number than water (bone), < 1 for lower (fats). The effective
energies are not known (the filtration is not published), so besides the assumed 0.4-0.6 x kVp they are also fitted: the
single pair (E_L, E_H) that best reproduces kappa of every tray cell of known composition. The subtraction weight of eq. 3.3
evaluated at that pair is then a prediction for the phantoms made from the trays alone.

Open-source data: conventional humerus radiographs (Kaggle humerus fracture dataset, non-fractured originals) show how
realistic the phantom looks. The contrast ratio K = |bone - soft tissue| / |soft tissue - background| does not depend on the
gain, offset or polarity of the image, so a processed clinical image can be compared with the phantom's thickness maps (only
a non-linear display curve distorts it). ROIs are circles {name, row, col, radius} saved per image in a JSON file, so every
run uses the same regions. They are found from two clicked points per image (click_rois, needs an interactive matplotlib
window, then line_rois) and checked on the QC overlay.
"""
import io
import json
import os
import re
import urllib.parse
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize

from Modules.Bone_suppression import GYPSUM, PLA, POP, TREND, WATER, mass_attenuation

# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
def snr(image, roi, background):
    """SNR_k = median of the ROI / standard deviation of the background."""
    return float(np.median(image[roi]) / np.std(image[background]))


def cnr(image, roi_a, roi_b, background):
    """CNR_kj = |median_a - median_b| / standard deviation of the background."""
    return float(abs(np.median(image[roi_a]) - np.median(image[roi_b])) / np.std(image[background]))


def metrics_table(images, rois, pairs, background, trend=None):
    """One row per image ({label: image}): sigma_bg, the SNR of every ROI in pairs and the CNR of every pair. A pair whose
    reference is TREND is measured against trend(image), the soft tissue interpolated under the ROI."""
    names = list(dict.fromkeys(n for pair in pairs for n in pair if n != TREND))
    rows = []
    for label, image in images.items():
        sigma = float(np.std(image[rois[background]]))
        row = dict(image=label, sigma_bg=sigma)
        row.update({f'SNR {n}': snr(image, rois[n], rois[background]) for n in names})
        under = None if trend is None else image - trend(image)
        row.update({f'CNR {a} / {b}': abs(np.nanmedian(under[rois[a]])) / sigma if b == TREND
                    else cnr(image, rois[a], rois[b], rois[background]) for a, b in pairs})
        rows.append(row)
    return pd.DataFrame(rows).set_index('image')


def noise_check(T_low, T_high, background, w):
    """Measured background noise of S = T_H - w T_L against the prediction for independent noise,
    sqrt(sigma_H^2 + w^2 sigma_L^2), and the correlation rho_LH of the two maps' background noise. With correlated noise
    sigma_S^2 = sigma_H^2 + w^2 sigma_L^2 - 2 w rho_LH sigma_L sigma_H (an identity on the same pixels, so rho_LH is the
    quantity that explains a measured sigma_S below the independent prediction)."""
    a, b = T_low[background], T_high[background]
    measured = float(np.std(b - w * a))
    independent = float(np.sqrt(b.std() ** 2 + w ** 2 * a.std() ** 2))
    return dict(sigma_S_measured=measured, sigma_S_independent=independent,
                measured_over_independent=measured / independent, rho_LH=float(np.corrcoef(a, b)[0, 1]))


# --------------------------------------------------------------------------------------------------------------------
# Theory: spectral ratio of the tray cells
# --------------------------------------------------------------------------------------------------------------------
# Component -> chemical formula. Eggshell is taken as CaCO3; resin as the epoxy base (DGEBA); coconut oil as trilaurin;
# beeswax as myricyl palmitate; candle wax as paraffin. Food colouring is neglected. Silicone putty is left out: its filler
# is unknown (it reads twice water, far above silicone rubber).
FORMULAS = {'Eggshell': 'CaCO3', 'CaCO3': 'CaCO3', 'Water': 'H2O', 'Resin': 'C21H24O4', 'Coconut oil': 'C39H74O6',
            'Beeswax': 'C46H92O2', 'Candle wax': 'C25H52'}
ALIASES = {'Resin + red food colouring': 'Resin', 'Water reference': 'Water'}


def cured_material(amounts):
    """{component: g (water in ml)} -> {formula: g} after curing: plaster of Paris sets to gypsum and binds 1.5 H2O per
    formula unit; the rest of the mixing water is assumed to stay in the cast (as in Bone_suppression.BONE_MATERIALS)."""
    material = {}
    pop = amounts.get('PoP', 0.0)
    for component, mass in amounts.items():
        if component == 'PoP':
            material['CaSO4H4O2'] = material.get('CaSO4H4O2', 0.0) + mass * GYPSUM / POP
        elif component == 'Water':
            free = max(mass - pop * 1.5 * WATER / POP, 0.0)
            if free > 0:
                material['H2O'] = material.get('H2O', 0.0) + free
        else:
            material[FORMULAS[component]] = material.get(FORMULAS[component], 0.0) + mass
    return material


def recipe(contents, composition):
    """Amounts {component: g} of a tray cell from its contents label (e.g. 'CaCO3 + PoP + water (1:1:2.5)') and the
    composition table of attenuation/generated/oct21/composition.csv; None when the composition is unknown."""
    contents = ALIASES.get(contents, contents)
    if contents in FORMULAS:
        return {contents: 1.0}
    m = re.match(r'(.*?)\s*\((.*)\)$', contents)
    if not m:
        return None
    name, ratio = m.group(1), m.group(2).replace(' ', '')
    rows = composition[(composition['compound'].str.replace('$_3$', '3', regex=False) == name)
                       & (composition['ratio'].str.replace(' ', '') == ratio)]
    return dict(zip(rows['component'], rows['amount'])) if len(rows) else None


def kappa_at(material, e_low, e_high):
    """kappa = [mu/rho(E_L) / mu/rho_water(E_L)] / [mu/rho(E_H) / mu/rho_water(E_H)] at two effective energies (keV)."""
    water = {'H2O': 1.0}

    def r(e):
        return mass_attenuation(material, e) / mass_attenuation(water, e)

    return r(e_low) / r(e_high)


def spectral_ratio(material, kvp_low, kvp_high, fractions=np.linspace(0.4, 0.6, 21)):
    """kappa for E = f x kVp at each fraction f."""
    return np.array([kappa_at(material, f * kvp_low, f * kvp_high) for f in fractions])


def spectral_ratio_table(cell_results, composition, kvp_low, kvp_high):
    """Measured kappa = R_L / R_H of every tray cell against the recipe prediction (E = 0.5 x kVp, band 0.4-0.6). The
    cured composition of each cell is kept in the 'material' column (None when unknown)."""
    low, high = cell_results['low'], cell_results['high']
    rows = []
    for cell in low.index:
        contents = low.loc[cell, 'contents']
        amounts = recipe(contents, composition)
        row = dict(cell=cell, contents=contents, kappa_measured=low.loc[cell, 'R'] / high.loc[cell, 'R'],
                   material=None if amounts is None else cured_material(amounts))
        if amounts is None:
            row.update(kappa_theory=np.nan, kappa_min=np.nan, kappa_max=np.nan)
        else:
            k = spectral_ratio(row['material'], kvp_low, kvp_high)
            row.update(kappa_theory=float(np.interp(0.5, np.linspace(0.4, 0.6, len(k)), k)),
                       kappa_min=float(k.min()), kappa_max=float(k.max()))
        rows.append(row)
    df = pd.DataFrame(rows).set_index('cell')
    df['measured / theory'] = df['kappa_measured'] / df['kappa_theory']
    return df


def fit_effective_energies(df, starts=((35, 55), (45, 60), (30, 70))):
    """The pair of effective energies (E_L, E_H) in keV that best reproduces the measured kappa of the cells with a known
    composition (least squares on log kappa; the water cell is left out, its kappa is 1 by definition). Returns E_L, E_H,
    the rms error of log kappa and the prediction for every cell (NaN where the composition is unknown)."""
    d = df[df['material'].notna() & (df['contents'] != 'Water reference')]

    def cost(x):
        return sum((np.log(k) - np.log(kappa_at(m, *x))) ** 2 for m, k in zip(d['material'], d['kappa_measured']))

    best = min((minimize(cost, x0, method='Nelder-Mead') for x0 in starts), key=lambda r: r.fun)
    e_low, e_high = (float(e) for e in best.x)
    fitted = pd.Series([np.nan if m is None else kappa_at(m, e_low, e_high) for m in df['material']], index=df.index)
    return e_low, e_high, float(np.sqrt(best.fun / len(d))), fitted


def apparent_density(material, R, energy_kev):
    """Density (g/cm3) at which a material {formula: g} has the measured relative attenuation R = mu / mu_water at an
    effective energy (keV); for a cured mixture whose density was not measured."""
    return float(R * mass_attenuation({'H2O': 1.0}, energy_kev) / mass_attenuation(material, energy_kev))


def w_at(e_low, e_high, material, displaced=None):
    """Subtraction weight of eq. 3.3 at effective energies (keV): the attenuation of the bone signature relative to PLA at
    E_H over the same at E_L. material and displaced are (composition {formula: g}, density g/cm3); with displaced the
    signature is the difference of the two linear attenuation coefficients (the bone replaces that material along the
    path); without it the density cancels."""
    def signature(e):
        mu = mass_attenuation(material[0], e) * material[1]
        if displaced is not None:
            mu -= mass_attenuation(displaced[0], e) * displaced[1]
        return mu / mass_attenuation(PLA, e)

    return signature(e_high) / signature(e_low)


def parity_plot(df, ax=None, column='kappa_theory', band=True, xlabel=r'$\kappa$ from the recipe ($E = 0.5\times$kVp, bars 0.4-0.6)',
                title='Spectral ratio: measured against theory'):
    """Measured kappa against a prediction column; with band, horizontal bars = effective energy 0.4-0.6 x kVp."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    d = df.dropna(subset=[column])
    groups = {'Bone (plaster of Paris)': d['contents'].str.contains('PoP'),
              'Bone (CaCO3 + resin)': d['contents'].str.contains('CaCO3 \\+ resin'),
              'Fat and muscle': ~d['contents'].str.contains('PoP|CaCO3')}
    for k, (label, sel) in enumerate(groups.items()):
        g = d[sel]
        xerr = [g[column] - g['kappa_min'], g['kappa_max'] - g[column]] if band else None
        ax.errorbar(g[column], g['kappa_measured'], xerr=xerr, fmt='os^'[k], color=f'C{k}', ms=6, capsize=2, lw=1, label=label)
    values = pd.concat([d['kappa_measured'], d[column]] + ([d['kappa_min'], d['kappa_max']] if band else []))
    lim = [0.8 * values.min(), 1.05 * values.max()]
    ax.plot(lim, lim, 'k--', lw=1, label='measured = prediction')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\kappa$ measured, $R_L / R_H$')
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------------------------------------------------
# Open-source data: download and ROIs
# --------------------------------------------------------------------------------------------------------------------
KAGGLE = 'https://www.kaggle.com/api/v1/datasets/download/'
HUMERUS = 'emiliopaspuel/deeplearning-driven-diagnosis-of-humerus-fractures'   # CC BY 4.0


def kaggle_file(dataset, path, out):
    """Download one file of a public Kaggle dataset to out (skipped if it exists). Kaggle zips some files."""
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = urllib.request.urlopen(KAGGLE + dataset + '/' + urllib.parse.quote(path, safe=''), timeout=120).read()
    if data[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(z.namelist()[0])
    with open(out, 'wb') as f:
        f.write(data)
    return out


def load_grey(path):
    """Any radiograph file as a float grey-level array."""
    return np.array(Image.open(path).convert('L'), dtype=float)


def disc(shape, row, col, radius):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2


def roi_masks(shape, rois):
    """{name: boolean mask} from the saved ROI circles (other entries, e.g. the clicked line, are skipped)."""
    return {r['name']: disc(shape, r['row'], r['col'], r['radius']) for r in rois if 'radius' in r}


def line_rois(image, air, far, radii=(3, 4, 6), width=2):
    """Bone, soft-tissue and background ROIs of a limb radiograph from two points: one in the air next to the limb and one
    beyond the bone shaft, the line between them crossing the shaft. Along the profile (averaged over +-width px across
    the line) the skin edge is the steepest rise away from the air level, the bone the extreme value beyond it (in the
    same direction as the soft tissue from the air, so either display polarity works) and the soft tissue half way
    between the two. The background ROI is the air point itself."""
    (r0, c0), (r1, c1) = air, far
    n = int(np.hypot(r1 - r0, c1 - c0))
    t = np.linspace(0, 1, n)
    normal = np.array([c1 - c0, -(r1 - r0)]) / n
    profile = np.mean([ndi.map_coordinates(image, [r0 + t * (r1 - r0) + d * normal[0], c0 + t * (c1 - c0) + d * normal[1]],
                                           order=1) for d in range(-width, width + 1)], axis=0)
    profile = ndi.gaussian_filter1d(profile, 1.5)
    q = profile - np.median(profile[:8])
    q *= np.sign(np.median(q[n // 3:]))                       # tissue above the air level
    edge = int(np.argmax(np.gradient(q)[:int(0.7 * n)]))
    bone = edge + 8 + int(np.argmax(q[edge + 8:]))
    soft = (edge + bone) // 2

    def at(i):
        return int(round(r0 + t[i] * (r1 - r0))), int(round(c0 + t[i] * (c1 - c0)))

    (rb, cb), (rs, cs) = at(bone), at(soft)
    return [dict(name='Bone', row=rb, col=cb, radius=radii[0]), dict(name='Soft tissue', row=rs, col=cs, radius=radii[1]),
            dict(name='Background', row=int(r0), col=int(c0), radius=radii[2])]


def click_rois(image, names, radii, title=''):
    """Place one circle per name by clicking its centre (needs an interactive backend, e.g. %matplotlib qt)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image, cmap='gray')
    rois = []
    for name, radius in zip(names, radii):
        ax.set_title(f'{title}: click the centre of {name}')
        fig.canvas.draw()
        (c, r), = plt.ginput(1, timeout=0)
        ax.add_patch(plt.Circle((c, r), radius, fill=False, color='r'))
        rois.append(dict(name=name, row=int(round(r)), col=int(round(c)), radius=int(radius)))
    plt.close(fig)
    return rois


def roi_file(path, keys, place):
    """{key: [ROI, ...]} from the JSON file at path; keys missing from it are placed with place(key) and saved."""
    saved = json.load(open(path)) if os.path.exists(path) else {}
    missing = [k for k in keys if k not in saved]
    for k in missing:
        saved[k] = place(k)
    if missing:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(saved, open(path, 'w'), indent=1)
    return {k: saved[k] for k in keys}


# --------------------------------------------------------------------------------------------------------------------
# Open-source data: metrics
# --------------------------------------------------------------------------------------------------------------------
def contrast_ratio(image, bone, soft, background):
    """K = |bone - soft| / |soft - background| from the medians of three masks (independent of gain, offset and
    polarity), with the two contrasts."""
    b, s, g = (float(np.median(image[m])) for m in (bone, soft, background))
    return dict(K=abs(b - s) / abs(s - g), bone_soft=abs(b - s), soft_background=abs(s - g))


def limb_metrics(image, rois, min_bone=2.0, min_soft=5.0):
    """K of a limb radiograph from its saved Bone / Soft tissue / Background circles. Invalid (K = NaN) when the bone is
    less than min_bone grey levels from the soft tissue or the soft tissue less than min_soft from the background (the
    ROIs missed the shaft or the background is not air)."""
    m = roi_masks(image.shape, rois)
    row = contrast_ratio(image, m['Bone'], m['Soft tissue'], m['Background'])
    row['valid'] = row['bone_soft'] >= min_bone and row['soft_background'] >= min_soft
    if not row['valid']:
        row['K'] = np.nan
    return row


def show_rois(image, rois, ax, title='', colours=None):
    """QC overlay of the ROI circles on an image."""
    colours = colours or {}
    ax.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    for r in rois:
        if 'radius' not in r:                                   # the clicked line of a limb image
            ax.plot([r['air'][1], r['far'][1]], [r['air'][0], r['far'][0]], 'y-', lw=0.6)
            continue
        key = r['name'].split(' ')[0]
        ax.add_patch(plt.Circle((r['col'], r['row']), r['radius'], fill=False, lw=1.2, color=colours.get(key, 'y')))
    ax.set_title(title, fontsize=8)
    ax.axis('off')
