import os
import streamlit as st
from models.predict import predict_cb, predict_ctr

# 0) Vérification du token
if not os.getenv("HUGGINGFACE_TOKEN"):
    st.error("⚠️ HUGGINGFACE_TOKEN non défini.")
    st.stop()

# 1) Config page
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")

# 2) Constantes âge & mapping genre
MEDIAN_AGE = 35.0
MAX_AGE    = 80.0
gender_map = {"Male":0, "Female":1, "Unknown":2}

# 3) Titre
st.title("💡 Détecteur de Clickbait & CTR")

# 4) Inputs utilisateur
texte = st.text_area("📝 Texte publicitaire", height=150)
age   = st.slider("🎯 Âge cible", 18.0, 99.0, 30.0)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 5) Prédiction
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
