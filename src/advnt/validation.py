from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .importances import extract_model_importances
from .sample_weights import compute_density_ratio_weights


class AdversarialValidator(BaseEstimator):
    def __init__(
        self,
        model=None,
        cv=None,
        metric="roc_auc",
        random_state=42,
        compute_sample_weights=True,
        weight_clip=(0.01, 100.0),
        normalize_weights=True,
        eps=1e-6,
    ):
        self.model = model
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.compute_sample_weights = compute_sample_weights
        self.weight_clip = weight_clip
        self.normalize_weights = normalize_weights
        self.eps = eps

    def _default_model(self):
        if importlib.util.find_spec("lightgbm") is None:
            raise ImportError(
                "lightgbm is required for the default model. "
                "Install lightgbm or pass `model=` explicitly."
            )

        lgbm = importlib.import_module("lightgbm")
        return lgbm.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
        )

    def _default_cv(self):
        return StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )

    def _resolve_metric(self):
        if callable(self.metric):
            return self.metric

        if self.metric == "roc_auc":
            return roc_auc_score

        raise ValueError(f"Unsupported metric: {self.metric}")

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=False)

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input must be a 2D array or DataFrame.")

        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    @staticmethod
    def _proba_1(model, X):
        return model.predict_proba(X)[:, 1]

    def fit(self, X_train, X_test, y=None):
        del y

        X_train = self._as_frame(X_train)
        X_test = self._as_frame(X_test)

        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("X_train and X_test columns must match exactly.")

        X_adv = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_adv = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

        self.n_train_ = len(X_train)
        self.n_test_ = len(X_test)
        self.feature_names_in_ = np.asarray(X_adv.columns)

        model = self.model if self.model is not None else self._default_model()
        cv = self.cv if self.cv is not None else self._default_cv()
        scorer = self._resolve_metric()

        self.models_ = []
        self.fold_scores_ = []
        self.oof_proba_ = np.full(len(X_adv), np.nan, dtype=float)

        for train_idx, valid_idx in cv.split(X_adv, y_adv):
            m = clone(model)
            m.fit(X_adv.iloc[train_idx], y_adv[train_idx])

            valid_proba = self._proba_1(m, X_adv.iloc[valid_idx])
            self.oof_proba_[valid_idx] = valid_proba
            self.fold_scores_.append(float(scorer(y_adv[valid_idx], valid_proba)))
            self.models_.append(m)

        self.score_ = float(np.mean(self.fold_scores_))
        self.oof_train_proba_ = self.oof_proba_[: self.n_train_]
        self.oof_test_proba_ = self.oof_proba_[self.n_train_ :]

        self.model_ = clone(model)
        self.model_.fit(X_adv, y_adv)
        self.test_proba_ = self._proba_1(self.model_, X_test)

        self.feature_importances_ = extract_model_importances(
            self.model_, feature_names=self.feature_names_in_
        )

        if self.compute_sample_weights:
            self.sample_weights_ = compute_density_ratio_weights(
                self.oof_train_proba_,
                eps=self.eps,
                clip=self.weight_clip,
                normalize=self.normalize_weights,
            )
        else:
            self.sample_weights_ = None

        return self
