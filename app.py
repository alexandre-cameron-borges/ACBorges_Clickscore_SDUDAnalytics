import os
import streamlit as st
from models.predict import predict_cb

# 1) Config page
st.set_page_config(page_title="Clickbait Detector", layout="centered")
st.title("Détection de Clickbait Publicitaire")

# 2) Constantes pour normalisation de l'âge (à ajuster)
MEDIAN_AGE = 35.0   # ta médiane réelle
MAX_AGE    = 80.0   # ton max réel

# 3) Charge le SEUIL calculé en amont (hardcode ou import d’un constante)
SEUIL = 0.57       # remplace par la valeur trouvée via ROC

# 4) Mapping genre
gender_map = {"Male": 0, "Female": 1, "Unknown": 2}

# 5) Inputs utilisateur
texte = st.text_area("Texte publicitaire", height=150)
age   = st.slider("Âge cible", min_value=18.0, max_value=99.0, value=30.0)
genre = st.selectbox("Genre cible", list(gender_map.keys()))

# 6) Bouton de prédiction
if st.button("Prédire"):
    # A) Filtre Texte vide ou trop court
    if not texte or len(texte.strip()) < 5:
        st.error("Le texte doit contenir au moins 5 caractères.")
    else:
        # B) Normalisation de l’âge
        age_norm = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
        # C) Appel du modèle
        proba = predict_cb(texte, age_norm, gender_map[genre])
        # D) Décision selon le SEUIL
        label = "Clickbait" if proba >= SEUIL else "Non-Clickbait"
        # E) Affichage
        st.metric("Probabilité de Clickbait", f"{proba:.1%}")
        if label == "Clickbait":
            st.error(f"Résultat : {label}")
        else:
            st.success(f"Résultat : {label}")

