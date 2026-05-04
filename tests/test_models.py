import numpy as np
import pytest
from sklearn.base import clone

from advnt import (
    ADVMLPClassifier,
    ADVMLPRegressor,
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

    clf = ADVMLPClassifier(max_epochs=5, batch_size=64, random_state=7)
    cloned = clone(clf)
    cloned.fit(x, y, eval_set=[(x_test,)])

    proba = cloned.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_accepts_multiclass_labels():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(180, 5))
    logits = np.column_stack(
        [
            1.0 * x[:, 0],
            -0.7 * x[:, 0] + 0.9 * x[:, 1],
            0.5 * x[:, 2] - 0.4 * x[:, 3],
        ]
    )
    y = np.argmax(logits, axis=1)
    x_test = rng.normal(loc=0.2, size=(90, 5))

    clf = ADVMLPClassifier(max_epochs=5, batch_size=64, random_state=9)
    clf.fit(x, y, eval_set=[(x_test,)])

    proba = clf.predict_proba(x)
    pred = clf.predict(x)

    assert clf.classes_.tolist() == [0, 1, 2]
    assert proba.shape == (len(x), 3)
    assert pred.shape == (len(x),)
    assert set(np.unique(pred)).issubset(set(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_preserves_string_labels():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(90, 4))
    labels = np.array(["low", "medium", "high"])
    y = labels[np.argmax(np.column_stack([x[:, 0], x[:, 1], -x[:, 0] - x[:, 1]]), axis=1)]

    clf = ADVMLPClassifier(max_epochs=3, batch_size=32, random_state=10)
    clf.fit(x, y)

    proba = clf.predict_proba(x)
    pred = clf.predict(x)

    assert clf.classes_.tolist() == ["high", "low", "medium"]
    assert proba.shape == (len(x), 3)
    assert set(np.unique(pred)).issubset(set(labels))


def test_classifier_works_inside_adversarial_validator_without_eval_set():
    rng = np.random.default_rng(123)
    x_train = rng.normal(0.0, 1.0, size=(120, 4))
    x_test = rng.normal(0.6, 1.0, size=(120, 4))

    av = AdversarialValidator(
        model=ADVMLPClassifier(
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

    reg = ADVMLPRegressor(max_epochs=8, batch_size=64, random_state=11)
    cloned = clone(reg)
    cloned.fit(x, y, eval_set=[(x_test,)])

    pred = cloned.predict(x)
    assert pred.shape == (len(x),)
    assert np.isfinite(pred).all()
