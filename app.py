import streamlit as st
import os

# 0) Récupère le token depuis les Secrets de Streamlit Cloud
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("⚠️ Vous devez définir HUGGINGFACE_TOKEN dans vos Secrets Streamlit.")
    st.stop()

# 1) Injecte-le dans l’environnement pour predict.py
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 2) Maintenant que HUGGINGFACE_TOKEN est en place, on peut importer
from models.predict import predict_cb, predict_ctr

# 3) Config page
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# 4) Constantes & mapping
MEDIAN_AGE = 35.0
MAX_AGE    = 80.0
gender_map = {"Male":0, "Female":1, "Unknown":2}

# 5) Inputs
texte = st.text_area("📝 Texte publicitaire", height=150)
age   = st.slider("🎯 Âge cible", 18.0, 99.0, 30.0, step=1)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 6) Prédiction
if st.button("🚀 Prédire"):
    if not texte.strip():
        st.error("Le texte ne peut pas être vide.")
    else:
        age_norm = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
        p_cb     = predict_cb(texte, age_norm, gender_map[genre])
        p_ctr    = predict_ctr(texte)

        label_cb = "Clickbait ❗" if p_cb >= 0.5 else "Non-Clickbait ✅"
        st.metric("🔎 P(clickbait)", f"{p_cb:.1%}")
        st.write("**Classification :**", label_cb)
        st.metric("📈 CTR prédit",  f"{p_ctr:.1%}")

