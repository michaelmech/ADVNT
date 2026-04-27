"""Feature neutralization via residualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


def neutralize_features(X, exposures, *, model=None, fit_intercept=True):
    """Residualize features in ``X`` against ``exposures``.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Features to neutralize.
    exposures : pandas.DataFrame or array-like
        Exposure matrix to regress out from each column in ``X``.
    model : estimator, default=None
        User-supplied regressor implementing ``fit`` and ``predict``. If
        ``None``, uses ``LinearRegression(fit_intercept=fit_intercept)``.
    fit_intercept : bool, default=True
        Intercept flag for the default linear model.
    """
    X_df = X.copy(deep=False) if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    E_df = (
        exposures.copy(deep=False)
        if isinstance(exposures, pd.DataFrame)
        else pd.DataFrame(np.asarray(exposures))
    )

    if len(X_df) != len(E_df):
        raise ValueError("X and exposures must have the same number of rows.")

    base_model = model if model is not None else LinearRegression(fit_intercept=fit_intercept)
    neutralized = np.empty_like(X_df.to_numpy(dtype=float), dtype=float)

    E = E_df.to_numpy(dtype=float)
    X_values = X_df.to_numpy(dtype=float)
    for idx in range(X_values.shape[1]):
        y = X_values[:, idx]
        fitted_model = clone(base_model)
        fitted_model.fit(E, y)
        neutralized[:, idx] = y - fitted_model.predict(E)

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(neutralized, columns=X_df.columns, index=X_df.index)

    return neutralized
