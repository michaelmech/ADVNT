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



def compute_density_ratio_weights_from_train_test(
    X_train,
    X_test,
    *,
    model=None,
    cv=None,
    metric="roc_auc",
    random_state=42,
    clip=(0.01, 100.0),
    normalize=True,
    eps=1e-6,
):
    """Compute train sample weights directly from train/test features via AV."""
    from .workflows import run_adversarial_validation_workflow

    artifacts = run_adversarial_validation_workflow(
        X_train,
        X_test,
        model=model,
        cv=cv,
        metric=metric,
        random_state=random_state,
        compute_sample_weights=False,
        weight_clip=clip,
        normalize_weights=normalize,
        eps=eps,
    )

    return compute_density_ratio_weights(
        artifacts["oof_train_proba"],
        eps=eps,
        clip=clip,
        normalize=normalize,
    )
