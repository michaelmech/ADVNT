"""Feature neutralization via residualization."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


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


def neutralize_train_test_drift(
    X_train,
    X_test,
    features,
    *,
    model=None,
    model_features=None,
    cv=None,
    random_state=42,
    domain_feature_name="is_test",
):
    """Neutralize selected features against train/test domain drift.

    For each feature in ``features``, this helper concatenates ``X_train`` and
    ``X_test``, creates a binary domain indicator where train rows are ``0`` and
    test rows are ``1``, then uses cross-validated regressors to predict the
    feature from that domain indicator plus any ``model_features``. The returned
    values are residuals: ``original_feature - cross_validated_prediction``.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame or array-like
        Train and test feature matrices with matching columns.
    features : list-like
        Feature names to neutralize. For array inputs, use generated names such
        as ``"x0"`` or ``"x1"``.
    model : regressor, default=None
        User-supplied regressor implementing ``fit`` and ``predict``. If
        ``None``, uses ``lightgbm.LGBMRegressor``.
    model_features : list-like, optional
        Additional feature names used to predict each neutralized feature. The
        train/test indicator is always included. If the current target feature is
        present, it is removed from the predictors to avoid direct leakage.
    cv : cross-validator, default=None
        Cross-validator used to create out-of-fold predictions. Defaults to a
        shuffled 5-fold ``StratifiedKFold`` over the train/test indicator.
    random_state : int, default=42
        Random seed for the default model and CV splitter.
    domain_feature_name : str, default="is_test"
        Name used internally for the train/test indicator column.

    Returns
    -------
    train_residuals, test_residuals : pandas.DataFrame or numpy.ndarray
        Residualized values for ``features``, split back to the original train
        and test blocks. DataFrame inputs preserve indexes and column names.
    """
    X_train_df = _as_frame(X_train)
    X_test_df = _as_frame(X_test)

    if list(X_train_df.columns) != list(X_test_df.columns):
        raise ValueError("X_train and X_test columns must match exactly.")

    features = _validate_feature_list(features, X_train_df.columns, "features")
    if model_features is None:
        model_features = []
    model_features = _validate_feature_list(
        model_features,
        X_train_df.columns,
        "model_features",
    )

    X_all = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    domain = np.r_[
        np.zeros(len(X_train_df), dtype=float),
        np.ones(len(X_test_df), dtype=float),
    ]

    base_model = model if model is not None else _default_lgbm_regressor(random_state)
    splitter = cv if cv is not None else StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    residuals = pd.DataFrame(index=X_all.index)
    for feature in features:
        predictor_columns = [name for name in model_features if name != feature]
        design = pd.DataFrame({domain_feature_name: domain})
        if predictor_columns:
            design = pd.concat(
                [design, X_all[predictor_columns].reset_index(drop=True)],
                axis=1,
            )

        target = X_all[feature].to_numpy(dtype=float)
        predictions = np.full(len(X_all), np.nan, dtype=float)

        for train_idx, valid_idx in splitter.split(design, domain):
            fitted_model = clone(base_model)
            fitted_model.fit(design.iloc[train_idx], target[train_idx])
            predictions[valid_idx] = fitted_model.predict(design.iloc[valid_idx])

        if np.isnan(predictions).any():
            raise ValueError("cv did not produce predictions for every row.")

        residuals[feature] = target - predictions

    train_residuals = residuals.iloc[: len(X_train_df)].copy()
    test_residuals = residuals.iloc[len(X_train_df) :].copy()

    if isinstance(X_train, pd.DataFrame):
        train_residuals.index = X_train.index
        test_residuals.index = X_test.index
        return train_residuals, test_residuals

    return train_residuals.to_numpy(), test_residuals.to_numpy()


def _as_frame(X):
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=False)

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array or DataFrame.")

    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _validate_feature_list(features, columns, name):
    if isinstance(features, str):
        features = [features]

    features = list(features)
    missing = [feature for feature in features if feature not in columns]
    if missing:
        raise ValueError(f"Unknown {name}: {missing}")

    return features


def _default_lgbm_regressor(random_state):
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError(
            "lightgbm is required for the default regressor. "
            "Install lightgbm or pass `model=` explicitly."
        )

    lgbm = importlib.import_module("lightgbm")
    return lgbm.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )
