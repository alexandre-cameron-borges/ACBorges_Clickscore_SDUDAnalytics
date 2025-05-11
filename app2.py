import os
import streamlit as st

# ⚙️ Assurez-vous de définir le token *avant* tout import de predict
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

import pandas as pd
from models.predict import predict_cb, predict_ctr

st.title("Clickbait & CTR Predictor")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results = []
    for _, row in df.iterrows():
        texte = row["texte"]
        age_norm = row["age_norm"]
        gender_id = row["gender_id"]
        p_cb = predict_cb(texte, age_norm, gender_id)
        p_ctr = predict_ctr(texte)
        results.append({"texte": texte, "clickbait_prob": p_cb, "ctr_prob": p_ctr})
    st.dataframe(pd.DataFrame(results))
