import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from advnt import infer_test_regimes


def _make_regime_data(seed=42):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["calm", "volatile", "trend"], 60)
    centers = {
        "calm": (0.0, 0.0),
        "volatile": (3.0, 0.0),
        "trend": (0.0, 3.0),
    }

    rows = []
    for label in labels:
        center = centers[label]
        rows.append(
            [
                rng.normal(center[0], 0.25),
                rng.normal(center[1], 0.25),
            ]
        )

    x_train = pd.DataFrame(rows, columns=["level", "slope"])
    x_test = pd.DataFrame(
        {
            "level": [0.05, 3.1, -0.1],
            "slope": [0.02, 0.1, 3.2],
        },
        index=["a", "b", "c"],
    )
    return x_train, labels, x_test


def test_infer_test_regimes_uses_single_split_by_default():
    x_train, labels, x_test = _make_regime_data()

    result = infer_test_regimes(
        x_train,
        labels,
        x_test,
        classifier=RandomForestClassifier(
            n_estimators=30,
            min_samples_leaf=2,
            random_state=42,
        ),
        random_state=42,
    )

    assert result["labels"].tolist() == ["calm", "volatile", "trend"]
    assert result["test_proba"].shape == (len(x_test), 3)
    assert result["test_proba"].index.tolist() == ["a", "b", "c"]
    assert result["test_proba"].columns.tolist() == ["calm", "trend", "volatile"]
    assert np.allclose(result["test_proba"].sum(axis=1), 1.0)
    assert len(result["fold_scores"]) == 1
    assert result["score"] > 0.9
    assert result["feature_names"].tolist() == ["level", "slope"]


def test_infer_test_regimes_uses_lgbm_default_classifier():
    lightgbm = pytest.importorskip("lightgbm")
    x_train, labels, x_test = _make_regime_data(seed=5)

    result = infer_test_regimes(
        x_train,
        labels,
        x_test,
        random_state=5,
    )

    assert isinstance(result["model"], lightgbm.LGBMClassifier)
    assert result["labels"].tolist() == ["calm", "volatile", "trend"]


def test_infer_test_regimes_accepts_cv_object_and_numpy_inputs():
    x_train, labels, x_test = _make_regime_data(seed=7)

    result = infer_test_regimes(
        x_train.to_numpy(),
        labels,
        x_test.to_numpy(),
        classifier=RandomForestClassifier(
            n_estimators=25,
            min_samples_leaf=2,
            random_state=7,
        ),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=7),
        random_state=7,
    )

    assert result["labels"].shape == (len(x_test),)
    assert result["oof_proba"].shape == (len(x_train), 3)
    assert np.isfinite(result["oof_proba"].to_numpy()).all()
    assert len(result["fold_scores"]) == 3
    assert result["score"] > 0.9
    assert result["feature_names"].tolist() == ["x0", "x1"]


def test_infer_test_regimes_rejects_non_cv_object():
    x_train, labels, x_test = _make_regime_data(seed=8)

    with pytest.raises(TypeError, match="object with split"):
        infer_test_regimes(x_train, labels, x_test, cv=3)


def test_infer_test_regimes_requires_predict_proba():
    x_train, labels, x_test = _make_regime_data(seed=11)

    with pytest.raises(TypeError, match="predict_proba"):
        infer_test_regimes(
            x_train,
            labels,
            x_test,
            classifier=LinearSVC(),
            random_state=11,
        )


def test_infer_test_regimes_rejects_mismatched_columns():
    x_train, labels, x_test = _make_regime_data(seed=12)
    x_test = x_test.rename(columns={"slope": "renamed"})

    with pytest.raises(ValueError, match="matching columns"):
        infer_test_regimes(x_train, labels, x_test)
