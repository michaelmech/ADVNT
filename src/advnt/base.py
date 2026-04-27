from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted


class AdversarialValidator(BaseEstimator):
    def __init__(
        self,
        estimator=None,
        cv=None,
        metric=roc_auc_score,
        random_state=42,
        refit=True,
        eps=1e-6,
    ):
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.refit = refit
        self.eps = eps

    def _default_estimator(self):
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def _default_cv(self):
        return StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=self.random_state,
        )

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")

        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    def _predict_scores(self, estimator, X):
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)[:, 1]

        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)

        raise TypeError(
            "estimator must implement either predict_proba or decision_function."
        )

    def _extract_feature_importance(self, estimator):
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_

        elif hasattr(estimator, "coef_"):
            values = np.ravel(estimator.coef_)

        else:
            return None

        return pd.Series(
            values,
            index=self.feature_names_in_,
            name="importance",
        ).sort_values(key=np.abs, ascending=False)

    def fit(self, X_train, X_test, y=None):
        X_train = self._as_frame(X_train)
        X_test = self._as_frame(X_test)

        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("X_train and X_test must have matching columns.")

        X_adv = pd.concat([X_train, X_test], axis=0, ignore_index=True)

        y_adv = np.r_[
            np.zeros(len(X_train), dtype=int),
            np.ones(len(X_test), dtype=int),
        ]

        self.n_train_ = len(X_train)
        self.n_test_ = len(X_test)
        self.feature_names_in_ = np.asarray(X_adv.columns)
        self.classes_ = np.array([0, 1])

        base_estimator = (
            self.estimator
            if self.estimator is not None
            else self._default_estimator()
        )

        cv = self.cv if self.cv is not None else self._default_cv()

        self.models_ = []
        self.fold_scores_ = []
        self.fold_indices_ = []
        self.oof_scores_ = np.full(len(X_adv), np.nan)
        self.train_oof_scores_ = np.full(len(X_train), np.nan)
        self.test_fold_scores_ = []
        self.feature_importances_by_fold_ = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_adv, y_adv)):
            model = clone(base_estimator)

            model.fit(X_adv.iloc[tr_idx], y_adv[tr_idx])

            va_scores = self._predict_scores(model, X_adv.iloc[va_idx])
            fold_score = float(self.metric(y_adv[va_idx], va_scores))

            self.models_.append(model)
            self.fold_scores_.append(fold_score)
            self.fold_indices_.append((tr_idx, va_idx))
            self.oof_scores_[va_idx] = va_scores

            train_mask = va_idx < self.n_train_
            train_idx = va_idx[train_mask]
            self.train_oof_scores_[train_idx] = va_scores[train_mask]

            self.test_fold_scores_.append(
                self._predict_scores(model, X_test)
            )

            fold_importance = self._extract_feature_importance(model)

            if fold_importance is not None:
                self.feature_importances_by_fold_.append(fold_importance)

        self.avg_score_ = float(np.mean(self.fold_scores_))
        self.std_score_ = float(np.std(self.fold_scores_))
        self.test_scores_ = np.mean(np.vstack(self.test_fold_scores_), axis=0)

        p = np.clip(self.train_oof_scores_, self.eps, 1.0 - self.eps)
        self.sample_weights_ = p / (1.0 - p)

        if self.feature_importances_by_fold_:
            self.feature_importances_ = (
                pd.concat(self.feature_importances_by_fold_, axis=1)
                .mean(axis=1)
                .sort_values(key=np.abs, ascending=False)
            )
        else:
            self.feature_importances_ = None

        if self.refit:
            self.estimator_ = clone(base_estimator)
            self.estimator_.fit(X_adv, y_adv)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "models_")

        X = self._as_frame(X)

        probs = np.mean(
            [self._predict_scores(model, X) for model in self.models_],
            axis=0,
        )

        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X=None, y=None):
        check_is_fitted(self, "avg_score_")
        return self.avg_score_

    def to_dict(self):
        check_is_fitted(self, "avg_score_")

        return {
            "avg_score": self.avg_score_,
            "std_score": self.std_score_,
            "fold_scores": self.fold_scores_,
            "cv_probs": self.train_oof_scores_,
            "test_probs": self.test_scores_,
            "sample_weights": self.sample_weights_,
            "models": self.models_,
            "feature_importances": self.feature_importances_,
            "feature_importances_by_fold": self.feature_importances_by_fold_,
        }