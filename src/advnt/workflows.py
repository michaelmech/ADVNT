"""High-level workflow helpers built on top of adversarial validation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from .sample_weights import compute_density_ratio_weights
from .ssl import select_safe_pseudo_labels
from .validation import AdversarialValidator


def _resolve_default_model(model, random_state):
    if model is not None:
        return model

    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )


def run_adversarial_validation_workflow(
    X_train,
    X_test,
    *,
    model=None,
    cv=None,
    metric="roc_auc",
    random_state=42,
    compute_sample_weights=True,
    weight_clip=(0.01, 100.0),
    normalize_weights=True,
    eps=1e-6,
):
    """Fit :class:`AdversarialValidator` and return commonly used artifacts."""
    resolved_model = _resolve_default_model(model, random_state)

    av = AdversarialValidator(
        model=resolved_model,
        cv=cv,
        metric=metric,
        random_state=random_state,
        compute_sample_weights=compute_sample_weights,
        weight_clip=weight_clip,
        normalize_weights=normalize_weights,
        eps=eps,
    )
    av.fit(X_train, X_test)

    return {
        "validator": av,
        "score": av.score_,
        "fold_scores": av.fold_scores_,
        "oof_train_proba": av.oof_train_proba_,
        "oof_test_proba": av.oof_test_proba_,
        "test_proba": av.test_proba_,
        "sample_weights": av.sample_weights_,
        "feature_importances": av.feature_importances_,
    }


def run_shift_preparation_workflow(
    X_train,
    X_test,
    *,
    model=None,
    cv=None,
    metric="roc_auc",
    random_state=42,
    pseudo_label_threshold=0.1,
    weight_clip=(0.01, 100.0),
    normalize_weights=True,
    eps=1e-6,
):
    """Generate sample weights and safe pseudo-label candidates from AV outputs."""
    resolved_model = _resolve_default_model(model, random_state)

    av = AdversarialValidator(
        model=resolved_model,
        cv=cv,
        metric=metric,
        random_state=random_state,
        compute_sample_weights=False,
        weight_clip=weight_clip,
        normalize_weights=normalize_weights,
        eps=eps,
    )
    av.fit(X_train, X_test)

    sample_weights = compute_density_ratio_weights(
        av.oof_train_proba_,
        eps=eps,
        clip=weight_clip,
        normalize=normalize_weights,
    )
    safe_pseudo_labels = select_safe_pseudo_labels(
        av.test_proba_,
        threshold=pseudo_label_threshold,
    )

    return {
        "validator": av,
        "score": av.score_,
        "sample_weights": sample_weights,
        "safe_pseudo_labels": safe_pseudo_labels,
        "test_proba": av.test_proba_,
        "feature_importances": av.feature_importances_,
    }
