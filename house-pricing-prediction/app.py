import streamlit as st
import pandas as pd
from pathlib import Path
import joblib


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "house_price_model.pkl")

st.title("🏠 Ames House Price Predictor")

st.markdown("""
Predict house prices using a **Ridge Regression** model trained on the
Ames Housing dataset.
""")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    ids = None
    if "Id" in df.columns:
        ids = df["Id"]
        df = df.drop(columns=["Id"])

    try:

        with st.spinner("Predicting..."):

            predictions = model.predict(df)

        result = df.copy()

        if ids is not None:
            result.insert(0, "Id", ids)

        result["PredictedPrice"] = predictions

        st.success("✅ Predictions completed!")

        st.dataframe(result.head())

        csv = result.to_csv(index=False)

        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(e)

col1, col2 = st.columns([3,1])


st.sidebar.title("About")

st.sidebar.info("""
Model: RidgeCV

Dataset:
Ames Housing

Competition:
Kaggle House Prices
""")

st.divider()

st.caption(
    "Built by Moez • Scikit-learn • Streamlit • Kaggle Ames Housing Dataset"
)