"""Optional SHAP diagnostics helpers."""

from __future__ import annotations


def compute_shap_values(model, X):
    """Compute SHAP values if optional `shap` dependency is available."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("`shap` is optional. Install shap to use SHAP diagnostics.") from exc

    explainer = shap.Explainer(model, X)
    return explainer(X)
