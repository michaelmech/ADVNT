import numpy as np
import pytest
from sklearn.base import clone

from advnt import (
    AdversarialValidationMLPClassifier,
    AdversarialValidationMLPRegressor,
    AdversarialValidator,
)


torch = pytest.importorskip("torch")


def _make_binary_data(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 4))
    logits = 1.2 * x[:, 0] - 0.8 * x[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    x_test = rng.normal(loc=0.4, size=(120, 4))
    return x, y, x_test


def test_classifier_accepts_eval_set_with_x_only_tuple():
    x, y, x_test = _make_binary_data(seed=7)

    clf = AdversarialValidationMLPClassifier(max_epochs=5, batch_size=64, random_state=7)
    cloned = clone(clf)
    cloned.fit(x, y, eval_set=[(x_test,)])

    proba = cloned.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_works_inside_adversarial_validator_without_eval_set():
    rng = np.random.default_rng(123)
    x_train = rng.normal(0.0, 1.0, size=(120, 4))
    x_test = rng.normal(0.6, 1.0, size=(120, 4))

    av = AdversarialValidator(
        model=AdversarialValidationMLPClassifier(
            hidden_dims=(32,),
            max_epochs=6,
            batch_size=64,
            random_state=13,
        ),
        random_state=13,
    )
    av.fit(x_train, x_test)

    assert 0.5 <= av.score_ <= 1.0
    assert av.oof_train_proba_.shape[0] == len(x_train)


def test_regressor_accepts_eval_set_with_x_only_tuple():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(180, 5))
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.1, size=180)
    x_test = rng.normal(loc=0.3, size=(100, 5))

    reg = AdversarialValidationMLPRegressor(max_epochs=8, batch_size=64, random_state=11)
    cloned = clone(reg)
    cloned.fit(x, y, eval_set=[(x_test,)])

    pred = cloned.predict(x)
    assert pred.shape == (len(x),)
    assert np.isfinite(pred).all()
