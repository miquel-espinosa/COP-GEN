"""
Stream 2 physical metrics for the COP-GEN benchmark.

These metrics operate on **spectral vectors** (per-image global mean pooling)
and quantify physical / radiometric realism.

Functions
---------
spectral_pool(imgs)
    Reduce (N, B, H, W) to (N, B) by mean pooling over spatial dims.

mmd(real, fake, sigma=None)
    Maximum Mean Discrepancy with RBF kernel. Uses median heuristic for
    bandwidth if ``sigma`` not given.

wasserstein_per_band(real, fake)
    1D Wasserstein distance per band between real and generated marginal
    distributions.

spectral_range_coverage(real, fake)
    Per-band: what fraction of the real [min, max] range is spanned by fake.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


# Band indices in the canonical 12-band stack
BAND_INDEX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5,
    "B07": 6, "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11,
}


def spectral_pool(imgs: np.ndarray) -> np.ndarray:
    """Reduce ``(N, B, H, W)`` -> ``(N, B)`` via spatial mean (over valid pixels).

    Pixels with value 0 (assumed nodata in S2 L2A) are excluded from the mean.
    """
    if imgs.ndim != 4:
        raise ValueError(f"Expected 4D array, got {imgs.shape}")

    n, b, _h, _w = imgs.shape
    out = np.zeros((n, b), dtype=np.float32)
    for i in range(n):
        for j in range(b):
            band = imgs[i, j]
            valid = band[band != 0]
            out[i, j] = float(valid.mean()) if valid.size > 0 else 0.0
    return out


def _median_heuristic_sigma(samples: np.ndarray) -> float:
    """Median pairwise distance / 2 as RBF bandwidth."""
    if len(samples) < 2:
        return 1.0
    d = cdist(samples, samples)
    iu = np.triu_indices(len(samples), k=1)
    return float(np.median(d[iu]) / 2 + 1e-8)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix."""
    d2 = cdist(x, y, metric="sqeuclidean")
    return np.exp(-d2 / (2 * sigma * sigma))


def mmd(real: np.ndarray, fake: np.ndarray, sigma: float | None = None) -> float:
    """Maximum Mean Discrepancy (squared) with RBF kernel.

    Parameters
    ----------
    real : (N, D)
    fake : (M, D)
    sigma : float, optional
        RBF bandwidth. If None, uses median heuristic over all samples.

    Returns
    -------
    mmd2 : float
        Squared MMD. ``nan`` if either set has <2 samples.
    """
    if len(real) < 2 or len(fake) < 2:
        return float("nan")

    if sigma is None:
        sigma = _median_heuristic_sigma(np.concatenate([real, fake], axis=0))

    k_xx = _rbf_kernel(real, real, sigma)
    k_yy = _rbf_kernel(fake, fake, sigma)
    k_xy = _rbf_kernel(real, fake, sigma)

    n = len(real)
    m = len(fake)
    # Unbiased MMD^2 estimator
    mmd2 = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return float(mmd2)


def wasserstein_per_band(real: np.ndarray, fake: np.ndarray) -> np.ndarray:
    """1D Wasserstein distance per band between marginal distributions.

    Parameters
    ----------
    real : (N, B) spectral vectors
    fake : (M, B)

    Returns
    -------
    distances : (B,) ndarray
    """
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError(f"Bad shapes: {real.shape} vs {fake.shape}")
    b = real.shape[1]
    return np.array([
        wasserstein_distance(real[:, j], fake[:, j]) for j in range(b)
    ], dtype=np.float32)


def spectral_range_coverage(real: np.ndarray, fake: np.ndarray) -> np.ndarray:
    """Per-band: fraction of the real [min, max] range covered by fake samples.

    1.0 = fake fully spans the real range.
    <1.0 = fake distribution is narrower than real (mode collapse / conservative).

    Parameters
    ----------
    real : (N, B) spectral vectors
    fake : (M, B)

    Returns
    -------
    coverage : (B,) ndarray in [0, 1]
    """
    if len(real) < 2 or len(fake) < 2:
        return np.full(real.shape[1], np.nan, dtype=np.float32)

    real_min = real.min(axis=0)
    real_max = real.max(axis=0)
    real_range = real_max - real_min

    fake_min = fake.min(axis=0)
    fake_max = fake.max(axis=0)

    # Intersection of [real_min, real_max] and [fake_min, fake_max]
    overlap_lo = np.maximum(real_min, fake_min)
    overlap_hi = np.minimum(real_max, fake_max)
    overlap = np.clip(overlap_hi - overlap_lo, 0, None)

    # Avoid div by zero where real has constant value in a band
    coverage = np.where(real_range > 1e-8, overlap / real_range, 1.0)
    return coverage.astype(np.float32)
