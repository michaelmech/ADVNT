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
