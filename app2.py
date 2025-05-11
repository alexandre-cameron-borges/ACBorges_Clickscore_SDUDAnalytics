import os
import streamlit as st
import pandas as pd
from models.predict import predict_cb, predict_ctr
import plotly.express as px

# Configuration
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# Inputs
gender_map = {"Male": 0, "Female": 1, "Unknown": 2}
age = st.slider("🎯 Âge cible", 18, 99, 30)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))
uploaded_file = st.file_uploader("📂 Importez votre CSV (colonnes: image, texte)", type="csv")
if not uploaded_file:
    st.info("Veuillez importer un fichier CSV.")
    st.stop()

# Data
df = pd.read_csv(uploaded_file)
if not {"image", "texte"}.issubset(df.columns):
    st.error("Les colonnes requises sont : image, texte")
    st.stop()
df = df.head(10)

# Prediction & Visualization
if st.button("🚀 Prédire"):
    age_norm = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    gender_id = gender_map[genre]
    records = []
    for _, row in df.iterrows():
        p_cb = predict_cb(row["texte"], age_norm, gender_id)
        p_ctr = predict_ctr(row["texte"])
        label = "❗ Clickbait" if p_cb >= 0.5 else "✅ Non-clickbait"
        records.append({
            "Texte": row["texte"],
            "P(clickbait)": p_cb,
            "CTR prédit": p_ctr,
            "Classification": label
        })
    res_df = pd.DataFrame(records)

    # Scatterplot
    fig = px.scatter(
        res_df,
        x="P(clickbait)",
        y="CTR prédit",
        color="Classification",
        hover_data=["Texte"],
        labels={"P(clickbait)": "Probabilité Clickbait", "CTR prédit": "CTR Prédit"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table with formatted percentages
    display_df = res_df.copy()
    display_df["P(clickbait)"] = display_df["P(clickbait)"].map("{:.1%}".format)
    display_df["CTR prédit"] = display_df["CTR prédit"].map("{:.1%}".format)
    st.table(display_df)
