from __future__ import annotations

import warnings

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st

import shap

from sklearn.linear_model import LogisticRegression

from advnt import AdversarialValidator
from advnt.importances import compute_shap_values


warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(page_title="ADVNT Adversarial Validation", layout="wide")
st.title("ADVNT: Adversarial Validation Studio")
st.caption(
    "Upload train/test datasets, run adversarial validation, inspect AUC, feature importances, and SHAP force plots."
)


def _read_tabular_upload(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".parquet", ".pq")):
        return pd.read_parquet(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Parquet files.")


def _build_model(model_name: str, random_state: int):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, solver="lbfgs")

    if model_name == "LGBMClassifier":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LGBMClassifier selected, but lightgbm is not installed. Install lightgbm or choose another model."
            ) from exc

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model selection: {model_name}")


def _assert_same_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
    extra_in_test = [c for c in X_test.columns if c not in X_train.columns]

    if missing_in_test or extra_in_test:
        details = []
        if missing_in_test:
            details.append(f"Missing in test: {missing_in_test[:10]}")
        if extra_in_test:
            details.append(f"Extra in test: {extra_in_test[:10]}")
        raise ValueError("Train and test must contain the same columns. " + " | ".join(details))

    return X_test.loc[:, X_train.columns]


def _render_force_plot_html(force_plot) -> str:
    return f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"


with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Adversarial model",
        ["LGBMClassifier", "LogisticRegression"],
        index=0,
    )
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    top_n = st.slider("Top-N important features", min_value=3, max_value=50, value=10)
    shap_sample_idx = st.number_input("Row index for SHAP force plot", min_value=0, value=0, step=1)

col1, col2 = st.columns(2)

with col1:
    train_upload = st.file_uploader("Upload TRAIN data (CSV/Parquet)", type=["csv", "parquet", "pq"])
with col2:
    test_upload = st.file_uploader("Upload TEST data (CSV/Parquet)", type=["csv", "parquet", "pq"])

if train_upload is None or test_upload is None:
    st.info("Upload both train and test datasets to continue.")
    st.stop()

try:
    X_train = _read_tabular_upload(train_upload)
    X_test = _read_tabular_upload(test_upload)
    X_test = _assert_same_columns(X_train, X_test)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"Columns validated: {X_train.shape[1]} features in both datasets. Train rows={len(X_train):,}, Test rows={len(X_test):,}."
)

run_button = st.button("Run adversarial validation", type="primary")

if run_button:
    try:
        model = _build_model(model_name, int(random_state))
        validator = AdversarialValidator(model=model, random_state=int(random_state))
        validator.fit(X_train, X_test)
    except Exception as exc:
        st.error(f"Validation failed: {exc}")
        st.stop()

    st.subheader("Adversarial Validation Score")
    st.metric("Cross-validated AV AUC", f"{validator.score_:.4f}")
    st.write("Fold AUCs:", [round(s, 4) for s in validator.fold_scores_])

    st.subheader("Feature Importances")
    if validator.feature_importances_ is None:
        st.warning("The selected model does not expose model-based importances.")
    else:
        top_features = validator.feature_importances_.head(top_n)
        st.dataframe(top_features, use_container_width=True)
        st.bar_chart(
            top_features.set_index("feature")["importance"],
            use_container_width=True,
        )

    st.subheader("SHAP Diagnostics")
    st.caption("Force plots show a single train-domain and test-domain row explanation.")

    X_adv = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_adv = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

    try:
        shap_values = compute_shap_values(validator.model_, X_adv)
    except ImportError as exc:
        st.info(str(exc))
        st.stop()

    train_class_rows = np.where(y_adv == 0)[0]
    test_class_rows = np.where(y_adv == 1)[0]

    train_pick = train_class_rows[min(int(shap_sample_idx), len(train_class_rows) - 1)]
    test_pick = test_class_rows[min(int(shap_sample_idx), len(test_class_rows) - 1)]

    st.markdown("**Train-domain class (0) force plot**")
    train_force = shap.force_plot(
        shap_values.base_values[train_pick],
        shap_values.values[train_pick],
        X_adv.iloc[train_pick],
        matplotlib=False,
    )
    st.components.v1.html(_render_force_plot_html(train_force), height=320, scrolling=True)

    st.markdown("**Test-domain class (1) force plot**")
    test_force = shap.force_plot(
        shap_values.base_values[test_pick],
        shap_values.values[test_pick],
        X_adv.iloc[test_pick],
        matplotlib=False,
    )
    st.components.v1.html(_render_force_plot_html(test_force), height=320, scrolling=True)

    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_rank = (
        pd.DataFrame({"feature": X_adv.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    st.markdown("**Top features by mean absolute SHAP value**")
    st.dataframe(shap_rank.head(top_n), use_container_width=True)
