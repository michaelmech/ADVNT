# ADVNT
Adversarial Data Validation Navigational Toolkit

## Convenience workflows

ADVNT exposes high-level helpers for common shift-aware workflows.

```python
from advnt import run_adversarial_validation_workflow

artifacts = run_adversarial_validation_workflow(X_train, X_test)
print(artifacts["score"])
print(artifacts["feature_importances"])
```

```python
from advnt import run_shift_preparation_workflow

bundle = run_shift_preparation_workflow(
    X_train,
    X_test,
    pseudo_label_threshold=0.1,
)

sample_weights = bundle["sample_weights"]
safe = bundle["safe_pseudo_labels"]
```


## One-call helpers from train/test inputs

These helpers run `AdversarialValidator` internally and return outputs for common downstream tasks:

```python
from advnt import (
    compute_density_ratio_weights_from_train_test,
    select_safe_pseudo_labels_from_train_test,
    extract_model_importances_from_train_test,
)

weights = compute_density_ratio_weights_from_train_test(X_train, X_test)
safe = select_safe_pseudo_labels_from_train_test(X_train, X_test, threshold=0.1)
importances = extract_model_importances_from_train_test(X_train, X_test)
```

## Streamlit app

An interactive Streamlit app is included for adversarial validation diagnostics.

### Features
- Upload train and test datasets (CSV/Parquet).
- Strict schema validation (train/test must have identical columns).
- Choose adversarial model: `LGBMClassifier`, `RandomForestClassifier`, or `LogisticRegression`.
- Run adversarial validation and inspect CV AV AUC and fold-level scores.
- View top-N model-based feature importances.
- Optional SHAP force plots for train-domain (`class=0`) and test-domain (`class=1`) rows.
- Optional SHAP ranking via mean absolute SHAP values.

### Run

```bash
streamlit run streamlit_app.py
```

> Note: `shap` is optional. Install it only if you want the SHAP diagnostics panel.


## Neural adversarial-validation model (PyTorch)

You can use a sklearn-style neural model with a dedicated adversarial-validation head:

```python
from advnt import AdversarialValidator, AdversarialValidationMLPClassifier

av = AdversarialValidator(
    model=AdversarialValidationMLPClassifier(
        hidden_dims=(64, 32),
        max_epochs=20,
        batch_size=256,
        random_state=42,
    ),
    random_state=42,
)

av.fit(X_train, X_test)
print(av.score_)
```

> Note: this model requires `torch` to be installed.

You can also train the neural estimators as regular models and optionally pass
`eval_set=[(X_test,)]` so the adversarial head treats that block as target-domain
data while the main head learns from your normal `y` labels:

```python
from advnt import AdversarialValidationMLPClassifier, AdversarialValidationMLPRegressor

clf = AdversarialValidationMLPClassifier(max_epochs=20, random_state=42)
clf.fit(X_train, y_train_binary, eval_set=[(X_test,)])

reg = AdversarialValidationMLPRegressor(max_epochs=20, random_state=42)
reg.fit(X_train, y_train, eval_set=[(X_test,)])
```
