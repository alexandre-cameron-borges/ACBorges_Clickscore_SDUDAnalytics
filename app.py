import streamlit as st
import torch
from models.predict import predict_cb
import pandas as pd

st.set_page_config(page_title="Clickbait Detector")
st.title("Détection de Clickbait Publicitaire")

# UI Inputs
texte  = st.text_area("Entrez votre texte publicitaire", height=150)
age    = st.slider("Âge cible", 18.0, 99.0, 30.0)
genre  = st.selectbox("Genre cible", ["Male","Female","Unknown"])
gender_map = {"Male":0,"Female":1,"Unknown":2}

if st.button("Prédire"):
    # Normalisation de l'âge (mêmes bornes que pour l'entraînement)
    med, mx =  df['age'].median(), df['age'].max()
    age_norm = (age - med) / (mx - med)
    # Appel à la fonction de prédiction
    proba = predict_cb(texte, age_norm, gender_map[genre])
    label = "Clickbait" if proba >= 0.5 else "Non-Clickbait"
    st.metric("Probabilité de clickbait", f"{proba:.1%}")
    st.success(f"Résultat : {label}")
