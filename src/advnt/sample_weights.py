"""Covariate-shift sample-weight utilities."""

from __future__ import annotations

import numpy as np


def compute_density_ratio_weights(proba, eps=1e-6, clip=None, normalize=True):
    """Compute density-ratio-style sample weights from domain probabilities."""
    p = np.asarray(proba, dtype=float)

    if p.ndim != 1:
        raise ValueError("proba must be 1-dimensional.")

    p = np.clip(p, eps, 1.0 - eps)
    weights = p / (1.0 - p)

    if clip is not None:
        lo, hi = clip
        weights = np.clip(weights, lo, hi)

    if normalize:
        weights = weights / np.mean(weights)

    return weights
