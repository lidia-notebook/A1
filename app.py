
# app.py
# Streamlit app for Hotel Cancellation Predictor

import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# 0) Page config
# -----------------------------
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

# -----------------------------
# 1) Recreate custom helper used during training
#    (needed so joblib can unpickle the pipeline)
# -----------------------------
def ensure_dataframe(X, feature_names):
    """
    This mirrors the function you used when saving the pipeline.
    It guarantees a DataFrame with exactly the training columns (order + names).
    """
    import pandas as pd
    import numpy as np

    if isinstance(X, pd.DataFrame):
        # add any missing columns
        missing = [c for c in feature_names if c not in X.columns]
        for c in missing:
            X[c] = np.nan
        # order to match training
        return X.loc[:, feature_names]
    else:
        return pd.DataFrame(X, columns=feature_names)


# -----------------------------
# 2) Load model + metadata (cached)
# -----------------------------
@st.cache_resource
def load_pipeline_and_meta():
    pipe = joblib.load("model_final.joblib")
    with open("model_meta.json", "r") as f:
        meta = json.load(f)
    # normalize to strings
    feature_names = [str(c) for c in meta.get("feature_names", [])]
    best_threshold = float(meta.get("threshold", 0.336))  # your chosen default
    return pipe, feature_names, best_threshold, meta

pipe, feature_names, default_threshold, meta = load_pipeline_and_meta()


# -----------------------------
# 3) Utilities
# -----------------------------
def make_compatible(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Force string column names, add missing as NaN, and reorder."""
    df = df.copy()
    df.columns = df.columns.astype(str)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan

    # Extra columns are okay; we drop them (pipeline expects training schema)
    df = df.loc[:, feature_names]
    return df


def predict_from_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return probabilities and class predictions (1 = will cancel)."""
    df_compat = make_compatible(df, feature_names)
    proba = pipe.predict_proba(df_compat)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["cancellation_probability"] = np.round(proba, 4)
    out["prediction"] = pred  # 1 = cancel, 0 = show
    return out


# -----------------------------
# 4) UI
# -----------------------------
st.title("🏨 Hotel Cancellation Predictor")
st.caption("Loads the trained XGBoost pipeline and predicts cancellation probabilities.")

with st.expander("ℹ️ Model info", expanded=False):
    st.write(
        {
            "sklearn_version": meta.get("sklearn_version", "unknown"),
            "num_features": len(feature_names),
            "default_threshold": default_threshold,
        }
    )
    st.write("First 10 expected features:", feature_names[:10])

# Threshold selection
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider(
    "Decision threshold (≥ predicts ‘cancel’)",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01,
)
st.sidebar.write(f"Using threshold: **{threshold:.3f}**")

# File uploader
uploaded = st.file_uploader("Upload a CSV to predict", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with the same columns used during training.")
else:
    # Read data
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head())

    # Schema diagnostics
    df_cols = set(df.columns.astype(str))
    expected = set(feature_names)
    missing = sorted(list(expected - df_cols))
    extra = sorted(list(df_cols - expected))

    colA, colB = st.columns(2)
    with colA:
        st.metric("Rows", len(df))
        st.metric("Expected features", len(feature_names))
    with colB:
        st.metric("Missing features auto-filled as NaN", len(missing))
        st.metric("Extra columns (ignored)", len(extra))

    if missing:
        with st.expander("Missing features (will be filled with NaN)", expanded=False):
            st.write(missing)
    if extra:
        with st.expander("Extra columns (will be dropped)", expanded=False):
            st.write(extra)

    # Predict
    with st.spinner("Scoring…"):
        results = predict_from_df(df, threshold)

    st.subheader("Results")
    st.write(
        "Prediction: **1 = will cancel**, **0 = will show**. "
        "You can sort by probability to prioritize risky bookings."
    )
    st.dataframe(results.head(50))

    # Simple summary
    cancels = int(results["prediction"].sum())
    shows = int(len(results) - cancels)
    st.write(f"**Predicted cancels:** {cancels}  |  **Predicted shows:** {shows}")

    # Download button
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results CSV",
        data=csv_bytes,
        file_name="cancellation_predictions.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Tip: keep **model_final.joblib** and **model_meta.json** in the same folder as this app. "
    "If you retrain/tune the model, re-export those two files and restart the app."
)
