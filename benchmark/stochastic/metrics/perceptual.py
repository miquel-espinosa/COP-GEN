"""
Stream 1 perceptual metrics for the COP-GEN benchmark.

All metrics operate in embedding space (R^D) and quantify how well a set of
generated samples matches a set of real samples in distribution and diversity.

Functions
---------
nn_accuracy(real, fake)
    1-Nearest-Neighbour accuracy via leave-one-out over the combined pool.
    50% = indistinguishable, >>50% = generated samples are easy to spot.

precision_recall(real, fake, k=3)
    Manifold-based precision (realism) and recall (diversity) using k-NN
    radius balls. See Kynkäänniemi et al. 2019.

intra_set_distance(samples)
    Mean pairwise distance within a set — diversity proxy.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def nn_accuracy(real: np.ndarray, fake: np.ndarray) -> float:
    """1-NN leave-one-out accuracy on combined real+fake pool.

    Parameters
    ----------
    real : (N, D) ndarray
    fake : (M, D) ndarray

    Returns
    -------
    accuracy : float in [0, 1]
        Fraction of samples whose nearest neighbour in the combined pool
        (excluding itself) has the same label. 0.5 means perfectly mixed
        (ideal); higher means the two distributions are easy to separate.
    """
    n, m = len(real), len(fake)
    if n == 0 or m == 0:
        return float("nan")
    pool = np.concatenate([real, fake], axis=0)
    labels = np.concatenate([np.zeros(n, dtype=np.int8),
                             np.ones(m, dtype=np.int8)])

    dist = cdist(pool, pool)
    np.fill_diagonal(dist, np.inf)
    nn_idx = dist.argmin(axis=1)
    correct = (labels[nn_idx] == labels).sum()
    return float(correct) / len(pool)


def _knn_radii(samples: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour for each sample (excluding self)."""
    n = len(samples)
    if n <= k:
        # Not enough points; use distance to farthest available neighbour
        k = max(1, n - 1)
    dist = cdist(samples, samples)
    np.fill_diagonal(dist, np.inf)
    # k-th smallest along axis 1
    return np.partition(dist, k - 1, axis=1)[:, k - 1]


def precision_recall(
    real: np.ndarray,
    fake: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """Manifold precision and recall via k-NN radius balls.

    Implementation follows Kynkäänniemi et al. 2019, "Improved Precision and
    Recall Metric for Assessing Generative Models".

    Precision: fraction of fake samples that fall inside the union of real
    samples' k-NN balls. High = realistic.

    Recall: fraction of real samples that fall inside the union of fake
    samples' k-NN balls. High = diverse coverage.

    Parameters
    ----------
    real : (N, D) ndarray
    fake : (M, D) ndarray
    k : int
        Neighbour count for radius estimation. Default 3.

    Returns
    -------
    precision, recall : float in [0, 1]
    """
    if len(real) <= 1 or len(fake) <= 1:
        return float("nan"), float("nan")

    real_radii = _knn_radii(real, k)
    fake_radii = _knn_radii(fake, k)

    # Precision: each fake sample is "real-like" if it falls inside any real ball
    d_fr = cdist(fake, real)  # (M, N)
    fake_in_real_ball = (d_fr <= real_radii[None, :]).any(axis=1)
    precision = float(fake_in_real_ball.mean())

    # Recall: each real sample is "covered" if it falls inside any fake ball
    d_rf = cdist(real, fake)  # (N, M)
    real_in_fake_ball = (d_rf <= fake_radii[None, :]).any(axis=1)
    recall = float(real_in_fake_ball.mean())

    return precision, recall


def intra_set_distance(samples: np.ndarray) -> float:
    """Mean pairwise Euclidean distance within a set.

    Parameters
    ----------
    samples : (N, D) ndarray

    Returns
    -------
    mean_distance : float
        ``nan`` if N < 2.
    """
    n = len(samples)
    if n < 2:
        return float("nan")
    dist = cdist(samples, samples)
    # Take upper triangle (exclude diagonal and duplicates)
    iu = np.triu_indices(n, k=1)
    return float(dist[iu].mean())
