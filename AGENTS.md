# AGENTS.md — Kaggle Playground Series S6E5: Predicting F1 Pit Stops

## Goal

Create a clean, reproducible Jupyter notebook for the Kaggle competition:

- Competition: `playground-series-s6e5`
- Title: `Predicting F1 Pit Stops`
- Task: binary classification
- Target: `PitNextLap`
- Metric: ROC AUC
- Required output: `submission.csv` with columns matching `sample_submission.csv`

The notebook should get from raw Kaggle input files to a valid submission, with a strong tabular baseline and enough diagnostics to iterate.

## Data location

Assume the notebook runs on Kaggle unless told otherwise.

```python
COMP_PATH = "/kaggle/input/competitions/playground-series-s6e5/"
TRAIN_PATH = COMP_PATH + "train.csv"
TEST_PATH = COMP_PATH + "test.csv"
SAMPLE_SUB_PATH = COMP_PATH + "sample_submission.csv"
```

Expected files:

```text
train.csv
test.csv
sample_submission.csv
```

Public notebooks show `train.csv`, `test.csv`, and `sample_submission.csv` being loaded from this competition path.

## Known schema notes

Use the actual CSV columns as source of truth. Based on public Kaggle notebook snippets, expect at least:

```text
id
Driver
Compound
Race
Year
PitStop
LapNumber
Stint
TyreLife
Position
LapTime (s)
Position_Change
PitNextLap   # train only target
```

Public snippets also show:

- `train` has about `439140` rows.
- `PitNextLap` is a float/binary target.
- Categorical/object columns include at least `Driver`, `Compound`, and likely `Race`.
- The task is to predict whether a driver will pit on the next lap.

Do not hard-code the full schema beyond `id` and `PitNextLap`. Print and validate the loaded columns.

## Notebook structure

Build the notebook with these sections:

1. Setup
2. Load data
3. Sanity checks
4. Light EDA
5. Feature engineering
6. Validation strategy
7. Baseline model
8. Model comparison / ensembling
9. Final training
10. Submission generation
11. Next experiments

## Setup cell

Use concise imports.

```python
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
```

Also try optional libraries if available:

```python
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
```

## Load data

```python
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

TARGET = "PitNextLap"
ID_COL = "id"

print(train.shape, test.shape, sample_sub.shape)
display(train.head())
display(test.head())
display(sample_sub.head())

assert TARGET in train.columns
assert ID_COL in train.columns
assert ID_COL in test.columns
```

## Sanity checks

Add checks for:

```python
print(train.info())
print(test.info())
print(train[TARGET].value_counts(dropna=False))
print(train[TARGET].mean())
print(train.isna().mean().sort_values(ascending=False).head(20))
print(test.isna().mean().sort_values(ascending=False).head(20))
```

Validate submission format:

```python
assert list(sample_sub.columns) == [ID_COL, TARGET]
assert len(sample_sub) == len(test)
```

## EDA requirements

Keep EDA useful, not bloated.

Include:

```python
# target prevalence
train[TARGET].mean()

# by obvious categorical/time features when present
for col in ["Compound", "Driver", "Race", "Year", "Stint", "LapNumber", "TyreLife", "Position"]:
    if col in train.columns:
        display(
            train.groupby(col)[TARGET]
                 .agg(["count", "mean"])
                 .sort_values("count", ascending=False)
                 .head(30)
        )
```

Plot only a few high-value charts:

```python
import matplotlib.pyplot as plt

if "LapNumber" in train.columns:
    train.groupby("LapNumber")[TARGET].mean().plot(figsize=(10, 4), title="Pit probability by lap")
    plt.show()

if "TyreLife" in train.columns:
    train.groupby("TyreLife")[TARGET].mean().plot(figsize=(10, 4), title="Pit probability by tyre life")
    plt.show()
```

## Feature engineering

Write feature engineering as one function applied to train and test together.

Rules:

- Do not use `PitNextLap` in features.
- Do not leak future test labels.
- Keep transformations deterministic.
- Preserve `id` only for submission, not modeling.

Start simple:

```python
def add_features(df):
    df = df.copy()

    if "LapNumber" in df.columns:
        df["lap_num_sq"] = df["LapNumber"] ** 2
        df["lap_num_log1p"] = np.log1p(df["LapNumber"])

    if "TyreLife" in df.columns:
        df["tyre_life_sq"] = df["TyreLife"] ** 2
        df["tyre_life_log1p"] = np.log1p(df["TyreLife"].clip(lower=0))

    if {"LapNumber", "TyreLife"}.issubset(df.columns):
        df["tyre_life_per_lap"] = df["TyreLife"] / (df["LapNumber"] + 1)
        df["laps_since_start_of_stint_est"] = df["TyreLife"]

    if {"Position", "Position_Change"}.issubset(df.columns):
        df["position_after_change_est"] = df["Position"] + df["Position_Change"]

    if {"LapTime (s)", "Driver"}.issubset(df.columns):
        # Computed later with train-only maps to avoid leakage if used.
        pass

    if "Compound" in df.columns:
        compound_order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
        df["compound_ord"] = df["Compound"].map(compound_order).fillna(-1)

    return df
```

Add train-derived aggregate encodings carefully inside each fold, not globally, for validation. For the first version, skip target encoding unless implementing it fold-safely.

Safe non-target aggregates can be fit on train and applied to test:

```python
def add_count_features(train_fe, test_fe, cols):
    all_df = pd.concat([train_fe[cols], test_fe[cols]], axis=0, ignore_index=True)
    for col in cols:
        if col in train_fe.columns:
            vc = all_df[col].value_counts(dropna=False)
            train_fe[f"{col}_count"] = train_fe[col].map(vc).astype("float32")
            test_fe[f"{col}_count"] = test_fe[col].map(vc).astype("float32")
    return train_fe, test_fe
```

## Validation strategy

Use ROC AUC.

Default:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Also test a group split by `Race` if `Race` exists, because random row splits may overestimate performance if rows from the same race are highly correlated.

```python
if "Race" in train.columns:
    # Use GroupKFold as a robustness check, not necessarily final leaderboard proxy.
    groups = train["Race"].astype(str)
```

Report both if time allows.

## Preprocessing

Use categorical handling compatible with tree models.

```python
feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in feature_cols if train_fe[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]
```

For sklearn models:

```python
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ), cat_cols),
    ],
    remainder="drop"
)
```

For CatBoost, pass categorical column names or indices and fill missing categories as strings.

## Baseline models

Implement at least these, depending on installed packages:

1. `HistGradientBoostingClassifier`
2. `XGBClassifier` if available
3. `CatBoostClassifier` if available
4. `LGBMClassifier` if available

Recommended first model:

```python
if HAS_XGB:
    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
```

CatBoost candidate:

```python
if HAS_CAT:
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
    )
```

LightGBM candidate:

```python
if HAS_LGBM:
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.025,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
```

## Cross-validation helper

Write a generic helper returning out-of-fold predictions, test predictions, and fold scores.

Pseudo-code:

```python
def run_cv_sklearn_model(model, X, y, X_test, splits, preprocess=None):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if preprocess is not None:
            clf = make_pipeline(preprocess, model)
        else:
            clf = clone(model)

        clf.fit(X_tr, y_tr)
        va_pred = clf.predict_proba(X_va)[:, 1]
        te_pred = clf.predict_proba(X_test)[:, 1]

        oof[va_idx] = va_pred
        test_pred += te_pred / len(splits)

        score = roc_auc_score(y_va, va_pred)
        scores.append(score)
        print(f"Fold {fold}: {score:.6f}")

    print(f"CV mean: {np.mean(scores):.6f} +/- {np.std(scores):.6f}")
    return oof, test_pred, scores
```

Remember to import:

```python
from sklearn.base import clone
```

## Submission

Use the model's averaged test probabilities.

```python
submission = sample_sub.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv("submission.csv", index=False)
display(submission.head())
print(submission.shape)
```

Check:

```python
assert list(submission.columns) == list(sample_sub.columns)
assert len(submission) == len(sample_sub)
assert submission[TARGET].between(0, 1).all()
```

## Experiments to include as TODOs

Add a final markdown cell with next experiments:

- Compare random `StratifiedKFold` vs `GroupKFold` by `Race`.
- Add fold-safe target encodings for `Driver`, `Race`, `Compound`, and interactions like `Driver_Compound`, `Race_Compound`.
- Add sequence/history features by sorting within `Race`, `Driver`, `Stint`, and `LapNumber`, if those columns exist.
- Add rolling or lag features only when they would be available at prediction time.
- Try probability calibration only if validation improves.
- Blend XGBoost, CatBoost, and LightGBM using simple weighted averages.
- Inspect high-probability predictions by lap/stint to catch obviously impossible patterns.
- If using the original public F1 strategy dataset, keep it optional and document whether it is allowed by competition rules.

## Important guardrails

- Do not optimize directly on the public leaderboard.
- Do not create features using `PitNextLap` outside fold-safe validation.
- Do not globally target-encode categories before CV.
- Do not assume row order is meaningful unless verified by columns like `Race`, `Driver`, `LapNumber`, and `Stint`.
- Always generate a valid `submission.csv`.
- Keep the notebook runnable top-to-bottom.
- Prefer simple, correct CV over clever leakage-prone features.

## Deliverable

Create a notebook named:

```text
s6e5_f1_pit_stop_baseline.ipynb
```

It should run end-to-end on Kaggle and write:

```text
submission.csv
```
