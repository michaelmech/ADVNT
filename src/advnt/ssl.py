"""Self-supervised learning helpers (pseudo-labeling utilities)."""

from __future__ import annotations

import numpy as np


def select_safe_pseudo_labels(test_proba, threshold=0.1):
    """Select confident pseudo-label candidates from test probabilities."""
    proba = np.asarray(test_proba, dtype=float)

    if proba.ndim != 1:
        raise ValueError("test_proba must be 1-dimensional.")

    if not 0.0 <= threshold <= 0.5:
        raise ValueError("threshold must be in [0.0, 0.5].")

    low_mask = proba <= threshold
    high_mask = proba >= (1.0 - threshold)
    mask = low_mask | high_mask

    labels = (proba >= 0.5).astype(int)

    return {
        "mask": mask,
        "indices": np.flatnonzero(mask),
        "pseudo_labels": labels[mask],
        "confidence": np.maximum(proba[mask], 1.0 - proba[mask]),
    }



def select_safe_pseudo_labels_from_train_test(
    X_train,
    X_test,
    *,
    model=None,
    cv=None,
    metric="roc_auc",
    random_state=42,
    threshold=0.1,
):
    """Select pseudo-label candidates by running adversarial validation internally."""
    from .workflows import run_adversarial_validation_workflow

    artifacts = run_adversarial_validation_workflow(
        X_train,
        X_test,
        model=model,
        cv=cv,
        metric=metric,
        random_state=random_state,
        compute_sample_weights=False,
    )

    return select_safe_pseudo_labels(artifacts["test_proba"], threshold=threshold)
