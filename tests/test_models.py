import numpy as np
import pytest
from sklearn.base import clone

from advnt import (
    AdversarialValidationMLPClassifier,
    AdversarialValidationMLPRegressor,
    AdversarialValidator,
)


torch = pytest.importorskip("torch")


def _make_shifted_data(seed=42):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(0.0, 1.0, size=(120, 4))
    x_test = rng.normal(0.6, 1.0, size=(120, 4))
    return x_train, x_test


def test_mlp_classifier_sklearn_clone_and_predict_proba_shape():
    x_train, x_test = _make_shifted_data()
    x_adv = np.vstack([x_train, x_test])
    y_adv = np.r_[np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]

    clf = AdversarialValidationMLPClassifier(max_epochs=5, batch_size=64, random_state=7)
    cloned = clone(clf)
    cloned.fit(x_adv, y_adv)

    proba = cloned.predict_proba(x_adv)
    assert proba.shape == (len(x_adv), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_mlp_classifier_works_inside_adversarial_validator():
    x_train, x_test = _make_shifted_data(seed=123)

    av = AdversarialValidator(
        model=AdversarialValidationMLPClassifier(
            hidden_dims=(32,),
            max_epochs=8,
            batch_size=64,
            random_state=13,
        ),
        random_state=13,
    )
    av.fit(x_train, x_test)

    assert 0.5 <= av.score_ <= 1.0
    assert av.oof_train_proba_.shape[0] == len(x_train)
    assert av.test_proba_.shape[0] == len(x_test)


def test_mlp_regressor_sklearn_clone_and_predict_shape():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(180, 5))
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.1, size=180)

    reg = AdversarialValidationMLPRegressor(max_epochs=12, batch_size=64, random_state=11)
    cloned = clone(reg)
    cloned.fit(x, y)

    pred = cloned.predict(x)
    assert pred.shape == (len(x),)
    assert np.isfinite(pred).all()
