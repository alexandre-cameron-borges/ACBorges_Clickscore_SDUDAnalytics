import streamlit as st
import os
import pandas as pd

# 0) Récupère le token HF depuis les Secrets de Streamlit Cloud
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("⚠️ Vous devez définir HUGGINGFACE_TOKEN dans vos Secrets Streamlit.")
    st.stop()
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 1) Import des fonctions de prédiction
from models.predict import predict_cb, predict_ctr

# 2) Configuration de la page
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# 3) Constantes & mapping
MEDIAN_AGE = 35.0
MAX_AGE    = 80.0
gender_map = {"Male":0, "Female":1, "Unknown":2}

# 4) Sélecteurs globaux
age   = st.slider("🎯 Âge cible", 18, 99, 30, step=1)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 5) Uploader CSV (colonnes : image, texte)
uploaded_file = st.file_uploader("📂 Importez votre CSV (image, texte)", type="csv")
if not uploaded_file:
    st.info("Veuillez importer un fichier CSV.")
    st.stop()

df = pd.read_csv(uploaded_file)
required_cols = {"image", "texte"}
if not required_cols.issubset(df.columns):
    st.error(f"Les colonnes requises sont : {', '.join(required_cols)}")
    st.stop()

df = df.head(10)

# 6) Prédiction en batch
if st.button("🚀 Prédire"):
    age_norm  = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    gender_id = gender_map[genre]
    for _, row in df.iterrows():
        # Affichage image & texte
        try:
            st.image(row["image"], width=150)
        except:
            st.warning(f"Impossible d'afficher l'image : {row['image']}")
        st.write(row["texte"])
        # Prédictions
        p_cb  = predict_cb(row["texte"], age_norm, gender_id)
        p_ctr = predict_ctr(row["texte"])
        label = "❗ Clickbait" if p_cb >= 0.5 else "✅ Non-clickbait"
        st.metric("🔎 P(clickbait)", f"{p_cb:.1%}")
        st.write("**Classification :**", label)
        st.metric("📈 CTR prédit", f"{p_ctr:.1%}")
        st.markdown("---")
