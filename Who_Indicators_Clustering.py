import streamlit as st
import pandas as pd
from src.utils import generate_cluster_trajectory_fig

st.title("Request Response Prediction")

with st.form("Request-Response"):
    model = st.selectbox(
        "Choose a model:",
        options=[
            "KMeans",
            "MiniBatchKMeans",
            "BisectingKMeans",
            "DBSCAN",
            "OPTICS",
            "AgglomerativeClustering",
            "MiniSom",
            "GaussianMixture",
            "AutoEncoder_KMeans",
        ],
    )

    st.subheader("Model Information")
    results_df = pd.read_csv("./results.csv")
    st.dataframe(results_df)

    countries = pd.read_csv(
        "./data/preprocessed/preprocessed.csv", usecols=lambda col: col == "Country"
    )["Country"]
    country = st.selectbox("Choose a country:", options=countries.unique())

    if st.form_submit_button():
        fig = generate_cluster_trajectory_fig(model, country)
        st.pyplot(fig)
