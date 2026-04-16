"""
metrics.py

Residual and goodness-of-fit metrics for posterior analysis.
All functions accept lists of arrays (ragged channel shapes).
"""

# === Imports ======================================================================================

import numpy as np

# === Main =========================================================================================

def noise_weighted_residuals(predicted, observed, noise_std, mask=None):
    """
    Compute noise-weighted residuals: (observed - predicted) / noise_std.

    If residuals are unit Gaussian, the noise model is well-calibrated.

    Parameters
    ----------
    predicted : list of arrays
        Model predictions per channel.
    observed : list of arrays
        Observed data per channel.
    noise_std : list of arrays
        Noise standard deviation per channel.
    mask : list of bool arrays, optional
        If provided, only return residuals at True pixels (flattened).

    Returns
    -------
    list of arrays
        Residuals per channel (2D if no mask, 1D if masked).
    """
    residuals = []
    for i in range(len(predicted)):
        r = (np.asarray(observed[i]) - np.asarray(predicted[i])) / np.asarray(noise_std[i])
        if mask is not None:
            r = r[np.asarray(mask[i])]
        residuals.append(r)
    return residuals


def uncertainty_weighted_residuals(mean, std, observed, mask=None):
    """
    Compute uncertainty-weighted residuals: (observed - mean) / std.

    Uses posterior uncertainty (std across samples) rather than noise std.

    Parameters
    ----------
    mean : list of arrays
        Posterior mean per channel.
    std : list of arrays
        Posterior std per channel.
    observed : list of arrays
        Observed data per channel.
    mask : list of bool arrays, optional
        If provided, only return residuals at True pixels (flattened).

    Returns
    -------
    list of arrays
        Residuals per channel.
    """
    residuals = []
    for i in range(len(mean)):
        r = (np.asarray(observed[i]) - np.asarray(mean[i])) / np.asarray(std[i])
        if mask is not None:
            r = r[np.asarray(mask[i])]
        residuals.append(r)
    return residuals


def reduced_chi_squared(residuals):
    """
    Compute reduced chi-squared: mean(residuals^2) per channel.

    A value near 1.0 indicates a well-calibrated model.

    Parameters
    ----------
    residuals : list of arrays
        Residuals per channel (from noise_weighted_residuals or similar).

    Returns
    -------
    list of float
        Reduced chi-squared per channel.
    """
    return [float(np.mean(np.asarray(r) ** 2)) if len(r) > 0 else float("nan") for r in residuals]


def power_spectrum(images, distances=None, n_bins=50):
    """
    Compute radially averaged power spectrum per channel.

    Parameters
    ----------
    images : list of 2D arrays
        One image per channel.
    distances : tuple of (dy, dx), optional
        Pixel scales in arcsec. If provided, k is in 1/arcsec.
    n_bins : int
        Number of radial k-bins.

    Returns
    -------
    list of (k_centers, power) tuples
        One per channel.
    """
    results = []
    for img in images:
        img = np.asarray(img)
        ny, nx = img.shape
        fft = np.fft.fft2(img)
        power = np.abs(fft) ** 2

        if distances is not None:
            ky = np.fft.fftfreq(ny, d=distances[0])
            kx = np.fft.fftfreq(nx, d=distances[1])
        else:
            ky = np.fft.fftfreq(ny)
            kx = np.fft.fftfreq(nx)

        kx2d, ky2d = np.meshgrid(kx, ky)
        k = np.sqrt(kx2d ** 2 + ky2d ** 2)

        k_max = min(np.max(np.abs(ky)), np.max(np.abs(kx)))
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        ps = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
            if np.any(mask):
                ps[i] = np.mean(power[mask])

        results.append((k_centers[1:], ps[1:]))
    return results
