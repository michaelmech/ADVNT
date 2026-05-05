import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from advnt import neutralize_train_test_drift


def test_neutralize_train_test_drift_preserves_pandas_metadata():
    rng = np.random.default_rng(42)
    n_train = 80
    n_test = 80
    x_train = pd.DataFrame(
        {
            "stable": rng.normal(size=n_train),
            "shifted": rng.normal(loc=0.0, size=n_train),
            "helper": rng.normal(size=n_train),
        },
        index=[f"train_{idx}" for idx in range(n_train)],
    )
    x_test = pd.DataFrame(
        {
            "stable": rng.normal(size=n_test),
            "shifted": rng.normal(loc=5.0, size=n_test),
            "helper": rng.normal(size=n_test),
        },
        index=[f"test_{idx}" for idx in range(n_test)],
    )

    train_resid, test_resid = neutralize_train_test_drift(
        x_train,
        x_test,
        features=["shifted"],
        model=LinearRegression(),
        cv=KFold(n_splits=4, shuffle=True, random_state=42),
    )

    assert isinstance(train_resid, pd.DataFrame)
    assert isinstance(test_resid, pd.DataFrame)
    assert train_resid.index.equals(x_train.index)
    assert test_resid.index.equals(x_test.index)
    assert train_resid.columns.tolist() == ["shifted"]
    assert test_resid.columns.tolist() == ["shifted"]
    assert abs(train_resid["shifted"].mean() - test_resid["shifted"].mean()) < 0.5


def test_neutralize_train_test_drift_uses_optional_model_features():
    rng = np.random.default_rng(7)
    n = 120
    train_helper = rng.normal(size=n)
    test_helper = rng.normal(size=n)
    x_train = pd.DataFrame(
        {
            "helper": train_helper,
            "target": 3.0 * train_helper + rng.normal(scale=0.05, size=n),
        }
    )
    x_test = pd.DataFrame(
        {
            "helper": test_helper,
            "target": 8.0 + 3.0 * test_helper + rng.normal(scale=0.05, size=n),
        }
    )

    train_resid, test_resid = neutralize_train_test_drift(
        x_train,
        x_test,
        features=["target"],
        model_features=["helper"],
        model=LinearRegression(),
        cv=KFold(n_splits=4, shuffle=True, random_state=7),
    )

    assert abs(train_resid["target"].mean() - test_resid["target"].mean()) < 0.25
    assert train_resid.shape == (n, 1)
    assert test_resid.shape == (n, 1)


def test_neutralize_train_test_drift_supports_numpy_inputs():
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(60, 2))
    x_test = rng.normal(size=(40, 2))
    x_test[:, 1] += 4.0

    train_resid, test_resid = neutralize_train_test_drift(
        x_train,
        x_test,
        features=["x1"],
        model=LinearRegression(),
        cv=KFold(n_splits=5, shuffle=True, random_state=11),
    )

    assert isinstance(train_resid, np.ndarray)
    assert isinstance(test_resid, np.ndarray)
    assert train_resid.shape == (60, 1)
    assert test_resid.shape == (40, 1)
    assert np.isfinite(train_resid).all()
    assert np.isfinite(test_resid).all()


def test_neutralize_train_test_drift_rejects_unknown_features():
    x_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    x_test = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})

    with pytest.raises(ValueError, match="Unknown features"):
        neutralize_train_test_drift(
            x_train,
            x_test,
            features=["missing"],
            model=LinearRegression(),
            cv=KFold(n_splits=2),
        )
