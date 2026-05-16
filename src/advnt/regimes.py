"""Regime similarity helpers."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import importlib.util

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


def _as_frame(X, *, name: str) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=False)

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array or DataFrame.")

    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _default_classifier(random_state):
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError(
            "lightgbm is required for the default classifier. "
            "Install lightgbm or pass `classifier=` explicitly."
        )

    lgbm = importlib.import_module("lightgbm")
    return lgbm.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )


def _resolve_splitter(cv, *, test_size, random_state):
    if cv is None:
        return StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

    if hasattr(cv, "split"):
        return cv

    raise TypeError("cv must be None or an object with split().")


def _resolve_metric(metric):
    if callable(metric):
        return metric

    if metric == "accuracy":
        return accuracy_score

    raise ValueError(f"Unsupported metric: {metric}")


def _predict_proba_aligned(model, X, classes):
    if not hasattr(model, "predict_proba"):
        raise TypeError("classifier must implement predict_proba().")

    raw_proba = model.predict_proba(X)
    proba = np.zeros((len(X), len(classes)), dtype=float)
    class_to_index = {label: idx for idx, label in enumerate(classes)}

    for source_idx, label in enumerate(model.classes_):
        target_idx = class_to_index[label]
        proba[:, target_idx] = raw_proba[:, source_idx]

    return proba


def infer_test_regimes(
    X_train,
    y_labels,
    X_test,
    *,
    classifier=None,
    cv=None,
    test_size=0.2,
    metric: str | Callable = "accuracy",
    random_state=42,
):
    """Infer which labeled training regime each test row most resembles.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Labeled reference rows.
    y_labels : array-like of shape (n_train,)
        Regime labels for the reference rows. These are not assumed to be the
        downstream prediction target.
    X_test : array-like of shape (n_test, n_features)
        Rows to assign to the closest known regime.
    classifier : estimator, optional
        Classifier implementing ``fit`` and ``predict_proba``. A LightGBM
        classifier is used by default.
    cv : CV splitter, optional
        Cross-validation splitter object with a ``split`` method. If None, use
        one stratified train/test split.
    test_size : float, default=0.2
        Validation fraction used when ``cv`` is None.
    metric : {"accuracy"} or callable, default="accuracy"
        Validation metric computed from validation labels and predicted labels.
    random_state : int, optional
        Random state for the default classifier and default splitters.

    Returns
    -------
    dict
        Regime assignments, probabilities, validation diagnostics, and the
        final fitted classifier.
    """
    X_train = _as_frame(X_train, name="X_train")
    X_test = _as_frame(X_test, name="X_test")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test must have matching columns.")

    y_labels = np.asarray(y_labels).reshape(-1)
    if len(X_train) != y_labels.shape[0]:
        raise ValueError("X_train and y_labels must contain the same number of rows.")

    classes = np.unique(y_labels)
    if classes.size < 2:
        raise ValueError("y_labels must contain at least two regimes.")

    splitter = _resolve_splitter(cv, test_size=test_size, random_state=random_state)
    estimator = classifier if classifier is not None else _default_classifier(random_state)
    scorer = _resolve_metric(metric)

    models = []
    fold_scores = []
    oof_proba = np.full((len(X_train), len(classes)), np.nan, dtype=float)
    oof_labels = np.full(len(X_train), None, dtype=object)

    for train_idx, valid_idx in splitter.split(X_train, y_labels):
        if (
            len(np.unique(y_labels[train_idx])) < classes.size
            or len(np.unique(y_labels[valid_idx])) < 2
        ):
            continue

        model = clone(estimator)
        model.fit(X_train.iloc[train_idx], y_labels[train_idx])

        valid_proba = _predict_proba_aligned(model, X_train.iloc[valid_idx], classes)
        valid_labels = classes[np.argmax(valid_proba, axis=1)]

        oof_proba[valid_idx] = valid_proba
        oof_labels[valid_idx] = valid_labels
        fold_scores.append(float(scorer(y_labels[valid_idx], valid_labels)))
        models.append(model)

    if not fold_scores:
        raise ValueError("No valid validation folds were available for regime scoring.")

    final_model = clone(estimator)
    final_model.fit(X_train, y_labels)

    test_proba_array = _predict_proba_aligned(final_model, X_test, classes)
    test_labels = classes[np.argmax(test_proba_array, axis=1)]

    test_proba = pd.DataFrame(test_proba_array, columns=classes, index=X_test.index)
    oof_proba_frame = pd.DataFrame(oof_proba, columns=classes, index=X_train.index)

    return {
        "labels": test_labels,
        "test_labels": test_labels,
        "proba": test_proba,
        "test_proba": test_proba,
        "classes": classes,
        "score": float(np.mean(fold_scores)),
        "fold_scores": fold_scores,
        "oof_labels": oof_labels,
        "oof_proba": oof_proba_frame,
        "model": final_model,
        "models": models,
        "feature_names": np.asarray(X_train.columns),
    }
