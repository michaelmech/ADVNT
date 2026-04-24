# AGENTS.md

## Project: ADVNT

ADVNT is a Python package for adversarial validation and drift-aware machine learning workflows.

The core idea is to treat adversarial validation as more than a one-off diagnostic. The foundational estimator should expose train-vs-test separability, out-of-fold adversarial probabilities, model-based feature importances, sample weights for covariate shift, and reusable artifacts that downstream modules can build on.

## Primary Goal

Build a clean, sklearn-style library around adversarial validation.

The base class should be the backbone for:

- train/test drift detection
- covariate shift sample weighting
- model-based drift feature attribution
- safe-zone pseudo-label candidate selection
- rolling time-series changepoint detection
- optional SHAP-based root-cause analysis
- future deep-learning domain-adversarial utilities

Keep the initial implementation practical and tabular-ML focused. Do not over-engineer the first version around neural networks.

## Foundational Class

The base estimator should live in something like:

```text
src/advnt/validation.py
```

Preferred public API:

```python
from advnt import AdversarialValidator

av = AdversarialValidator(
    model=None,
    cv=None,
    metric="roc_auc",
    random_state=42,
)

av.fit(X_train, X_test)

av.score_
av.fold_scores_
av.oof_train_proba_
av.test_proba_
av.feature_importances_
av.sample_weights_
```

Follow sklearn conventions:

- `fit(X_train, X_test, y=None)` returns `self`
- learned attributes end with `_`
- constructor should only store parameters, not perform work
- avoid mutating user inputs
- support pandas DataFrames and numpy arrays
- preserve feature names when available
- use sklearn utilities where they make sense
- implement `get_params` / `set_params` through `BaseEstimator`

## Base Class Requirements

The base adversarial validator should:

1. Combine `X_train` and `X_test` into one feature matrix.
2. Create a binary domain target:
   - `0` for train rows
   - `1` for test rows
3. Run cross-validation on the combined matrix.
4. Store out-of-fold probabilities for all rows where possible.
5. Store train-row adversarial probabilities separately.
6. Fit a final adversarial model on the full combined matrix.
7. Predict probabilities for the full test block.
8. Compute fold scores and aggregate score using the configured binary metric.
9. Compute model-based feature importances when supported.
10. Compute density-ratio-style sample weights for train rows.

## Defaults

Reasonable defaults:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=random_state,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=random_state,
)
```

Default metric should be ROC AUC. Accept either:

- string metric names like `"roc_auc"`
- callables like `roc_auc_score`

## Feature Importances

Use model-based importances only.

Supported extraction order:

1. `model.feature_importances_`
2. `model.coef_`
3. no importances if neither exists

For coefficients:

- flatten binary classifier coefficients
- use absolute values by default
- preserve signs only if an explicit option is added later

Return feature importances as a pandas DataFrame when feature names are available:

```text
feature | importance | rank
```

Do not implement permutation importances in the base class.

## Sample Weights

Implement covariate-shift weights using adversarial probabilities:

```python
p = np.clip(p_test_domain, eps, 1 - eps)
w = p / (1 - p)
```

The weights should be calculated for original train rows only.

Recommended options:

```python
compute_sample_weights=True
weight_clip=(0.01, 100.0)
normalize_weights=True
```

If normalized, scale weights to have mean `1.0`.

## Expected Attributes

At minimum, `AdversarialValidator.fit(...)` should populate:

```python
self.model_
self.models_
self.score_
self.fold_scores_
self.oof_proba_
self.oof_train_proba_
self.oof_test_proba_
self.test_proba_
self.sample_weights_
self.feature_importances_
self.n_train_
self.n_test_
self.feature_names_in_
```

Use `None` for unavailable optional outputs rather than failing silently.

## Package Structure

Target structure:

```text
advnt/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   └── advnt/
│       ├── __init__.py
│       ├── validation.py
│       ├── weights.py
│       ├── importances.py
│       ├── pseudo_labeling.py
│       ├── changepoint.py
│       ├── neutralization.py
│       ├── shap_diagnostics.py
│       └── utils.py
└── tests/
    ├── test_validation.py
    ├── test_weights.py
    ├── test_importances.py
    ├── test_pseudo_labeling.py
    └── test_changepoint.py
```

Keep the base class in `validation.py`. Put small pure helpers in separate modules only when they are independently testable.

## Implementation Priorities

### Phase 1: Core AV Estimator

Implement:

- `AdversarialValidator`
- CV scoring
- OOF probabilities
- final model fit
- model-based feature importances
- density-ratio sample weights
- pandas/numpy support
- unit tests

### Phase 2: Utility APIs

Implement:

```python
make_adversarial_dataset(X_train, X_test)
compute_density_ratio_weights(proba, eps=1e-6, clip=None, normalize=True)
extract_model_importances(model, feature_names=None)
select_safe_pseudo_labels(test_proba, threshold=0.1)
```

### Phase 3: Drift Modules

Implement:

- rolling-window adversarial validation for time series
- changepoint score curves
- feature neutralization via residualization
- SHAP diagnostics as optional dependency

### Phase 4: Optional Advanced Modules

Only after the tabular API is stable:

- PyTorch gradient reversal layer
- domain-adversarial neural network helpers
- representation-level domain invariance utilities

## Coding Style

Use simple, readable Python.

Prefer this:

```python
p = np.clip(proba, eps, 1 - eps)
weights = p / (1 - p)
```

Avoid unnecessarily abstract class hierarchies.

Avoid broad exception swallowing.

Keep public functions typed, but do not let typing make the code noisy.

Use numpy, pandas, and sklearn as the core dependency stack.

Optional dependencies should stay optional:

- `shap`
- `lightgbm`
- `xgboost`
- `torch`

Do not require optional dependencies for the base test suite.

## Testing Expectations

Tests should cover:

- sklearn-like constructor behavior
- `.fit(...)` returns `self`
- attributes exist after fit
- pandas feature names are preserved
- numpy inputs work
- fold scores have the expected length
- ROC AUC is high on intentionally shifted synthetic data
- ROC AUC is near random on matched synthetic data
- sample weights are finite and normalized when requested
- feature importances work for tree models
- coefficient importances work for linear models
- unsupported models return `None` importances

Use small synthetic datasets. Tests should run quickly.

Example synthetic drift setup:

```python
rng = np.random.default_rng(42)
X_train = pd.DataFrame({
    "stable": rng.normal(0, 1, 200),
    "shifted": rng.normal(0, 1, 200),
})
X_test = pd.DataFrame({
    "stable": rng.normal(0, 1, 200),
    "shifted": rng.normal(2, 1, 200),
})
```

The adversarial model should identify `shifted` as important.

## Documentation Expectations

README should include:

- what adversarial validation is
- quickstart example
- interpreting AV AUC
- sample-weighting example
- feature-importance example
- safe pseudo-labeling example
- warning that AV detects covariate shift, not target leakage by itself

Suggested quickstart:

```python
from advnt import AdversarialValidator

av = AdversarialValidator(random_state=42)
av.fit(X_train, X_test)

print(av.score_)
print(av.feature_importances_.head())

model.fit(X_train, y_train, sample_weight=av.sample_weights_)
```

## Design Principles

- The adversarial classifier is a reusable source of signal, not just a diagnostic score.
- The base estimator should expose artifacts downstream tools need.
- Keep defaults useful, but allow advanced users to pass their own model, CV splitter, and metric.
- Make tabular workflows excellent before expanding to neural workflows.
- Treat pandas support as first-class.
- Avoid magic behavior that surprises sklearn users.

## Do Not Do

- Do not add permutation importances to the base class.
- Do not make SHAP a required dependency.
- Do not make PyTorch a required dependency.
- Do not hide the adversarial probabilities from users.
- Do not drop shifted features automatically.
- Do not mutate `X_train` or `X_test`.
- Do not silently ignore CV/model incompatibilities.
- Do not build a CLI before the Python API is stable.

## Useful Interpretive Guidelines

Approximate AV AUC interpretation:

```text
0.50 - 0.55: little detectable train/test shift
0.55 - 0.70: mild to moderate shift
0.70 - 0.85: strong shift
0.85 - 1.00: severe shift or possible split artifact
```

These are heuristics, not universal rules. The package should expose the evidence, not pretend to make domain decisions automatically.

## Codex Task Guidance

When implementing a task:

1. Start from the smallest testable change.
2. Add or update tests in the same commit.
3. Keep the public API stable unless the task explicitly changes it.
4. Prefer pure helper functions for math-heavy pieces.
5. Keep estimator state explicit and inspectable.
6. Run the relevant tests before finishing.
7. Update README examples when behavior changes.

## First Milestone

A successful first milestone is:

```python
from advnt import AdversarialValidator

av = AdversarialValidator(random_state=42)
av.fit(X_train, X_test)

assert 0 <= av.score_ <= 1
assert av.oof_train_proba_.shape[0] == len(X_train)
assert av.test_proba_.shape[0] == len(X_test)
assert av.sample_weights_.shape[0] == len(X_train)
```

Once this works cleanly, build the higher-level ADVNT modules around it.
