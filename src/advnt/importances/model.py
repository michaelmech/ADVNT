"""Model-based feature importance extraction."""

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
