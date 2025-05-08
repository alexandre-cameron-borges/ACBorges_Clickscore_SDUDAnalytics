import streamlit as st
from models.predict import predict_cb

# 1) Config page
st.set_page_config(page_title="Clickbait Detector", layout="centered")
st.title("Détection de Clickbait Publicitaire")

# 2) Constantes pour normalisation de l'âge (à ajuster)
MEDIAN_AGE = 35.0   # ta médiane réelle
MAX_AGE    = 80.0   # ton max réel

# 3) Mapping genre
gender_map = {"Male": 0, "Female": 1, "Unknown": 2}

# 4) Inputs utilisateur
texte = st.text_area("Texte publicitaire", height=150)
age   = st.slider("Âge cible", min_value=18.0, max_value=99.0, value=30.0)
genre = st.selectbox("Genre cible", list(gender_map.keys()))

# 5) Prédiction
if st.button("Prédire"):
    if not texte.strip():
        st.error("Le texte ne peut pas être vide.")
    else:
        age_norm = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
        proba = predict_cb(texte, age_norm, gender_map[genre])
        label = "Clickbait" if proba >= 0.5 else "Non-Clickbait"
        st.metric("Probabilité de Clickbait", f"{proba:.1%}")
        st.success(f"Résultat : {label}")
