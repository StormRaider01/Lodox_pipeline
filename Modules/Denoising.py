"""Noise suppression of the bone-suppressed images (Methodology chapter, Classical and Learning-Based Filter Selection).

Every filter works on the subtracted image S (cm of PLA) and is judged with the same ROIs as the bone suppression:
* max |delta mu|: the largest change of an ROI median (cm of PLA); the filter must not move the attenuation values, so it
  should be close to zero;
* CNR = |median_k - median_j| / sigma_bg (Implementation chapter, eq. metrics) and its change in %, delta CNR;
* edge width: the 10-90 % width of the phantom's outer edge (mm), the blur cost.

Model selection (Learning-Based Filter Selection): the lag-one autocorrelation of the noise residuals (image minus a
Gaussian-smoothed copy, sigma = 10 px) in a flat background region, along the rows (rho_x) and the columns (rho_y):
both below 0.1 -> Noise2Void, one above -> StructN2V, both above -> Noise2Noise on same-kV pairs.

Noise2Noise: a compact U-Net (depth 2, 32 base filters) maps one noisy realisation of an image to a second, independent
one of the same scene; minimising the mean squared error then gives the mean of the noisy target, i.e. the clean image.
No clean image is needed. Training uses 64 x 64 patches normalised with the training-set mean and standard deviation,
flips as the only augmentation, and both realisations as input in turn. The inverse normalisation is applied after
inference. The network works on the fine detail of the image only and its correction is kept zero-mean (Noise2Noise class).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage import restoration
from skimage.morphology import disk

from . import Affine_matrix
from .Bone_suppression import TREND


# --------------------------------------------------------------------------------------------------------------------
# Noise model selection
# --------------------------------------------------------------------------------------------------------------------
def noise_autocorrelation(image, mask, sigma_px=10):
    """Lag-one autocorrelation (rho_x along the rows, rho_y down the columns) of the noise residuals inside mask."""
    residual = image - ndi.gaussian_filter(image, sigma_px)

    def lag_one(a, b, ma, mb):
        ok = ma & mb
        x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
        return float((x * y).mean() / np.sqrt((x * x).mean() * (y * y).mean()))

    rho_x = lag_one(residual[:, :-1], residual[:, 1:], mask[:, :-1], mask[:, 1:])
    rho_y = lag_one(residual[:-1], residual[1:], mask[:-1], mask[1:])
    return rho_x, rho_y


def select_model(rho_x, rho_y, limit=0.1):
    """The decision rule of the Learning-Based Filter Selection section."""
    above = (abs(rho_x) > limit, abs(rho_y) > limit)
    if not any(above):
        return 'Noise2Void'
    if all(above):
        return 'Noise2Noise'
    return 'StructN2V (mask along the ' + ('rows)' if above[0] else 'columns)')


# --------------------------------------------------------------------------------------------------------------------
# Classical filters
# --------------------------------------------------------------------------------------------------------------------
LEVELS = ('light', 'medium', 'strong')
# Strength of each filter at the three levels. Gaussian: sigma (px); median: disk radius (px); bilateral: spatial sigma
# (px), with the grey-level sigma at 2 x the noise; non-local means: h as a multiple of the noise sigma
FILTER_SETTINGS = {
    'Gaussian': (1, 2, 4),
    'Median': (1, 2, 3),
    'Bilateral': (1, 2, 4),
    'Non-local means': (0.6, 0.8, 1.2),
}


def nl_means(image, h, sigma, patch_size=5, patch_distance=6):
    """skimage.restoration.denoise_nl_means (fast_mode=True) of a 2-D grey image. If skimage's compiled extension cannot
    be loaded (Windows Application Control blocked it on this machine in October 2026), the numpy port below is used."""
    try:
        return restoration.denoise_nl_means(image, h=h, sigma=sigma, patch_size=patch_size,
                                            patch_distance=patch_distance, fast_mode=True)
    except ImportError:
        return _nl_means_numpy(image, h, sigma, patch_size, patch_distance)


def _fast_exp(y):
    """skimage's _fast_exp (Schraudolph's approximation with its minimum-RMS constants): the upper 32 bits of a double are
    set to int(EXP_A y) + EXP_BC and the lower 32 bits to zero."""
    upper = (1512775.3951951856938 * y).astype(np.int32).astype(np.int64) + 1072632447
    return (upper << 32).view(np.float64)


def _nl_means_numpy(image, h, sigma, patch_size=5, patch_distance=6):
    """numpy port of skimage 0.26 _fast_nl_means_denoising_2d (one channel), loop for loop: the outer loops run over the
    patch shifts (t_row, t_col >= 0), the inner pixel loops are array operations. Patch distances come from an integral
    image of the squared differences minus 2 sigma^2, with skimage's box (rows/columns -offset+1 .. +offset), its cut-off
    (distance > 5 gives no weight), its fast exponential and its weight of 0.5 for t_col = 0."""
    s = patch_size + 1 - patch_size % 2                     # odd patch size, as in skimage
    d, o = patch_distance, s // 2
    pad = o + d + 1
    P = np.pad(np.asarray(image, np.float64), pad, mode='reflect')
    n_row, n_col = P.shape
    weights, result = np.zeros_like(P), np.zeros_like(P)
    h2s2 = h * h * s * s
    var_diff = 2 * sigma * sigma
    for t_row in range(-d, d + 1):
        r0, r1 = max(o, o - t_row), min(n_row - o, n_row - o - t_row)
        a0, a1 = max(1, -t_row), min(n_row, n_row - t_row)
        for t_col in range(0, d + 1):
            alpha = 0.5 if t_col == 0 else 1.0
            c1 = n_col - t_col
            diff = np.zeros_like(P)
            diff[a0:a1, 1:c1] = (P[a0:a1, 1:c1] - P[a0 + t_row:a1 + t_row, 1 + t_col:c1 + t_col]) ** 2 - var_diff
            I = diff.cumsum(0).cumsum(1)
            hi, lo = slice(2 * o, n_col - t_col), slice(0, n_col - 2 * o - t_col)
            dist = (I[r0 + o:r1 + o, hi] + I[r0 - o:r1 - o, lo] - I[r0 - o:r1 - o, hi] - I[r0 + o:r1 + o, lo])
            dist = np.maximum(dist, 0.0) / h2s2
            w = np.where(dist > 5.0, 0.0, alpha * _fast_exp(-np.minimum(dist, 5.0)))
            here = (slice(r0, r1), slice(o, n_col - o - t_col))
            there = (slice(r0 + t_row, r1 + t_row), slice(o + t_col, n_col - o))
            weights[here] += w
            weights[there] += w
            result[here] += w * P[there]
            result[there] += w * P[here]
    return (result / np.where(weights > 0, weights, 1.0))[pad:-pad, pad:-pad]


def classical_filter(image, name, level, sigma_noise):
    """Apply one classical filter at level 0, 1 or 2 (light, medium, strong); sigma_noise: background noise of image.

    The image is mapped linearly onto 0-1 for the filter and back afterwards, so the result stays in cm of PLA.
    """
    lo, hi = float(image.min()), float(image.max())
    x = (image - lo) / (hi - lo)
    s = sigma_noise / (hi - lo)
    p = FILTER_SETTINGS[name][level]
    if name == 'Gaussian':
        y = ndi.gaussian_filter(x, p)
    elif name == 'Median':
        y = ndi.median_filter(x, footprint=disk(p))
    elif name == 'Bilateral':
        y = restoration.denoise_bilateral(x, sigma_color=2 * s, sigma_spatial=p)
    elif name == 'Non-local means':
        y = nl_means(x, h=p * s, sigma=s, patch_size=5, patch_distance=6)
    else:
        raise ValueError(f"Unknown filter '{name}'")
    return lo + y * (hi - lo)


# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
def edge_width(image, footprint, pixel_spacing, half_mm=5.0):
    """10-90 % width (mm) of the edge of footprint: the edge spread function is the mean grey level against the signed
    distance from the outline (negative inside), within half_mm of it."""
    d = (ndi.distance_transform_edt(~footprint) - ndi.distance_transform_edt(footprint)) * pixel_spacing
    near = np.abs(d) <= half_mm
    bins = np.round(d[near] / pixel_spacing).astype(int)
    bins -= bins.min()
    esf = np.bincount(bins, image[near]) / np.maximum(np.bincount(bins), 1)
    x = (np.arange(len(esf)) + np.round(-half_mm / pixel_spacing)) * pixel_spacing
    inside, outside = esf[:3].mean(), esf[-3:].mean()
    level = (esf - outside) / (inside - outside)            # 1 inside, 0 outside

    def crossing(v):
        below = np.nonzero(level < v)[0]                    # bins below v, going outwards
        if not len(below) or below[0] == 0:
            return np.nan
        k = below[0]
        return x[k - 1] + (level[k - 1] - v) / (level[k - 1] - level[k]) * (x[k] - x[k - 1])

    return float(crossing(0.1) - crossing(0.9))


def image_metrics(image, rois, pairs, background, footprint, pixel_spacing, trend=None):
    """ROI medians, background sigma, CNR of each (ROI, reference) pair and the edge width of one image. A pair whose
    reference is Bone_suppression.TREND is measured against trend(image), the soft tissue interpolated under the ROI."""
    medians = {name: float(np.median(image[roi])) for name, roi in rois.items() if roi.any()}
    sigma = float(np.std(image[rois[background]]))
    under = None if trend is None else image - trend(image)
    cnr = {(a, b): abs(np.nanmedian(under[rois[a]]) if b == TREND else medians[a] - medians[b]) / sigma for a, b in pairs}
    return dict(medians=medians, sigma_bg=sigma, cnr=cnr, edge_mm=edge_width(image, footprint, pixel_spacing))


def repeat_difference(denoised, repeat, region):
    """RMS difference between a denoised image and an independent repeat realisation of the same scene over a region:
    mean((D - S2)^2) = mean((D - X)^2) + the noise of S2 (+ any small mismatch of the two scans), so the filter whose output
    comes closest to the repeat is the one closest to the clean image X, counting the noise left and the detail lost (blur)
    alike. The noise of S2 sets a floor, so the reduction against the unfiltered image understates the true one. (S2 is
    interpolated during registration, which smooths its noise, so its noise cannot be taken as var(S1 - S2) / 2.)"""
    return float(np.sqrt(np.mean((denoised[region] - repeat[region]) ** 2)))


def compare(original, filtered, label):
    """One table row: the filtered image's metrics against the unfiltered one (both from image_metrics)."""
    row = dict(filter=label,
               max_abs_delta_mu=max(abs(filtered['medians'][k] - original['medians'][k]) for k in original['medians']),
               sigma_bg=filtered['sigma_bg'],
               noise_reduction_pct=100 * (1 - filtered['sigma_bg'] / original['sigma_bg']),
               edge_mm=filtered['edge_mm'],
               edge_change_mm=filtered['edge_mm'] - original['edge_mm'])
    for (a, b), c in filtered['cnr'].items():
        row[f'CNR {a} / {b}'] = c
        row[f'delta CNR {a} / {b} (%)'] = 100 * (c / original['cnr'][(a, b)] - 1)
    return row


# --------------------------------------------------------------------------------------------------------------------
# Noise2Noise
# --------------------------------------------------------------------------------------------------------------------
def _block(n_in, n_out):
    return nn.Sequential(nn.Conv2d(n_in, n_out, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(n_out, n_out, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Compact U-Net: two poolings (depth 2), base filters doubling at each level, skip connections."""

    def __init__(self, base=32):
        super().__init__()
        self.enc1, self.enc2, self.middle = _block(1, base), _block(base, 2 * base), _block(2 * base, 4 * base)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), _block(4 * base, 2 * base)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * base, base, 2, 2), _block(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m = self.middle(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def sample_patches(pairs, n, size, rng):
    """n random patch pairs (size x size) from [(image_a, image_b, valid_mask), ...], fully inside the valid masks."""
    a_out, b_out = np.empty((n, 1, size, size), np.float32), np.empty((n, 1, size, size), np.float32)
    k = 0
    while k < n:
        a, b, valid = pairs[rng.integers(len(pairs))]
        r, c = rng.integers(0, a.shape[0] - size), rng.integers(0, a.shape[1] - size)
        if not valid[r:r + size, c:c + size].all():
            continue
        a_out[k, 0], b_out[k, 0] = a[r:r + size, c:c + size], b[r:r + size, c:c + size]
        k += 1
    return a_out, b_out


def align_pair(a, b, valid, boxes=None, erode_px=5):
    """Register realisation b onto a inside each box (default: the whole image) and return the pairs
    [(a_box, b_box, valid_box), ...]; pixels warped in from outside the box are left out of valid_box."""
    pairs = []
    for box in boxes or [(slice(None), slice(None))]:
        a_box, b_box = a[box], b[box]
        (theta, tx, ty), _ = Affine_matrix.register(a_box, b_box, [0, 0, 0])
        b_box = Affine_matrix.transformation(b_box, theta, tx, ty, cval=np.nan)
        ok = ndi.binary_erosion(valid[box] & np.isfinite(b_box), iterations=erode_px)
        pairs.append((a_box, np.nan_to_num(b_box), ok))
    return pairs


class Noise2Noise:
    """Noise2Noise denoiser: fit() on pairs of independent noisy realisations, predict() on a single image.

    The network only sees the detail layer, d = image - G(image) with G a Gaussian of detail_sigma px, and learns the target
    realisation's detail layer, b - G(a). This keeps the input in the same range for every object (the phantoms reach far
    higher values than the training images). Only the fine-scale part of the correction is kept, c - G(c) with
    c = output - input, so the denoiser cannot shift the mean level of a region.
    """

    def __init__(self, base=32, seed=0, detail_sigma=8):
        torch.manual_seed(seed)
        self.seed = seed
        self.detail_sigma = detail_sigma
        self.model = UNet(base)
        self.mean, self.std = 0.0, 1.0
        self.history = []

    def _smooth(self, image):
        return ndi.gaussian_filter(image, self.detail_sigma)

    def _detail_pairs(self, pairs):
        """Both directions of every pair in the detail domain: (a - G(a), b - G(a)) and (b - G(b), a - G(b))."""
        out = []
        for a, b, valid in pairs:
            for x, y in ((a, b), (b, a)):
                base = self._smooth(x)
                out.append(((x - base).astype(np.float32), (y - base).astype(np.float32), valid))
        return out

    def fit(self, train_pairs, val_pairs, epochs=10, steps=60, batch=32, size=64, lr=3e-4, n_val=256):
        """train_pairs / val_pairs: [(realisation_1, realisation_2, valid_mask), ...] (registered, same units)."""
        rng = np.random.default_rng(self.seed)
        train_pairs, val_pairs = self._detail_pairs(train_pairs), self._detail_pairs(val_pairs)
        values = np.concatenate([np.r_[a[v], b[v]] for a, b, v in train_pairs])
        self.mean, self.std = float(values.mean()), float(values.std())
        norm = lambda x: torch.from_numpy((x - self.mean) / self.std)
        va, vb = (norm(x) for x in sample_patches(val_pairs, n_val, size, rng))

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        best = (np.inf, None)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for _ in range(steps):
                a, b = sample_patches(train_pairs, batch, size, rng)
                if rng.random() < 0.5:                                  # flips only (rotations swap the noise directions)
                    a, b = a[..., ::-1], b[..., ::-1]
                if rng.random() < 0.5:
                    a, b = a[..., ::-1, :], b[..., ::-1, :]
                a, b = norm(np.ascontiguousarray(a)), norm(np.ascontiguousarray(b))
                optimiser.zero_grad()
                loss = F.mse_loss(self.model(a), b)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() / steps
            self.model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(self.model(va), vb).item()
            self.history.append((epoch + 1, train_loss, val_loss))
            if val_loss < best[0]:
                best = (val_loss, {k: v.clone() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(best[1])                             # keep the epoch with the lowest validation loss
        return pd.DataFrame(self.history, columns=['epoch', 'train_loss', 'val_loss']).set_index('epoch')

    def predict(self, image):
        """Denoised image in the units of the input."""
        h, w = image.shape
        base = self._smooth(image)
        detail = np.pad(image - base, ((0, -h % 4), (0, -w % 4)), mode='reflect')
        x = torch.from_numpy(((detail - self.mean) / self.std).astype(np.float32))[None, None]
        self.model.eval()
        with torch.no_grad():
            y = self.model(x)[0, 0].numpy()[:h, :w] * self.std + self.mean
        correction = base + y - image
        return image + correction - self._smooth(correction)

    def save(self, path):
        torch.save(dict(state=self.model.state_dict(), mean=self.mean, std=self.std, history=self.history,
                        detail_sigma=self.detail_sigma), path)

    @classmethod
    def load(cls, path, base=32):
        saved = torch.load(path, weights_only=False)
        model = cls(base, detail_sigma=saved['detail_sigma'])
        model.model.load_state_dict(saved['state'])
        model.mean, model.std, model.history = saved['mean'], saved['std'], saved['history']
        return model
