"""SHAP diagnostics helpers."""

from __future__ import annotations


def compute_shap_values(model, X):
    """Compute SHAP values for the provided model and feature matrix."""
    import shap

    explainer = shap.Explainer(model, X)
    return explainer(X)
