"""Feature importance extraction and optional SHAP diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_model_importances(model, feature_names=None):
    """Extract model-based importances from fitted estimators."""
    values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(np.asarray(model.coef_, dtype=float)))

    if values is None:
        return None

    if feature_names is None:
        return values

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    )
    frame["rank"] = frame["importance"].rank(method="dense", ascending=False).astype(int)
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def compute_shap_values(model, X):
    """Compute SHAP values if optional `shap` dependency is available."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("`shap` is optional. Install shap to use SHAP diagnostics.") from exc

    explainer = shap.Explainer(model, X)
    return explainer(X)
