import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

# 0) Token HF
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("⚠️ Vous devez définir HUGGINGFACE_TOKEN dans vos Secrets Streamlit.")
    st.stop()
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 1) Imports
from models.predict import predict_cb, predict_ctr

# 2) UI Setup
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# 3) Constantes & mapping
MEDIAN_AGE = 35.0
MAX_AGE    = 80.0
gender_map = {"Male":0, "Female":1, "Unknown":2}

# 4) Sélecteurs globaux
age   = st.slider("🎯 Âge cible", 18, 99, 30)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 5) CSV uploader
uploaded_file = st.file_uploader("📂 Importez votre CSV (colonnes: image, texte)", type="csv")
if not uploaded_file:
    st.info("Veuillez importer un fichier CSV.")
    st.stop()

df = pd.read_csv(uploaded_file)
if not {"image","texte"}.issubset(df.columns):
    st.error("Les colonnes requises sont : image, texte")
    st.stop()
df = df.head(10)

# 6) Batch prédiction
if st.button("🚀 Prédire"):
    age_norm  = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    gender_id = gender_map[genre]
    results = []
    for _, row in df.iterrows():
        p_cb  = predict_cb(row["texte"], age_norm, gender_id)
        p_ctr = predict_ctr(row["texte"])
        label = "❗ Clickbait" if p_cb >= 0.5 else "✅ Non-clickbait"
        results.append({
            "Texte": row["texte"],
            "P(clickbait)": f"{p_cb:.1%}",
            "Classification": label,
            "CTR prédit": f"{p_ctr:.1%}"
        })
    st.table(pd.DataFrame(results))

    # DataFrame résultat
    df_res = pd.DataFrame(results)
    df_res["color"] = df_res["Classification"].apply(
        lambda lab: "green" if "Clickbait" in lab else "red"
    )

    # === Création de fig et fig2 ===
    fig, ax = plt.subplots()
    ax.scatter(
        df_res["P(clickbait)"].str.rstrip("%").astype(float),
        df_res["CTR prédit"].str.rstrip("%").astype(float),
        c=df_res["color"]
    )
    ax.set_xlabel("P(clickbait) (%)")
    ax.set_ylabel("CTR prédit (%)")
    ax.set_title("Clickbait vs CTR pour chaque texte")

    counts = df_res["Classification"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[
            "green" if "Clickbait" in lab else "red"
            for lab in counts.index
        ]
    )
    ax2.set_title("Répartition Clickbait vs Non-clickbait")
    ax2.axis("equal")

    # === Affichage côte-à-côte ===
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Scatterplot")
        st.pyplot(fig)
    with col2:
        st.subheader("Pie Chart")
        st.pyplot(fig2)
