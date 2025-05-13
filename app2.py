import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 0️⃣ Token HF
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("⚠️ Vous devez définir HUGGINGFACE_TOKEN dans vos Secrets Streamlit.")
    st.stop()
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 1️⃣ Imports prédiction
from models.predict import predict_cb, predict_tm, predict_ctr

# 2️⃣ UI Setup
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# 3️⃣ Constantes & mapping
MEDIAN_AGE = 35.0
MAX_AGE    = 80.0
gender_map = {"Male":0, "Female":1, "Unknown":2}

# 4️⃣ Sélecteurs
col_age, col_genre = st.columns([0.6, 0.4])
with col_age:
    age = st.slider("🎯 Âge cible", 18, 99, 30)
with col_genre:
    genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 5️⃣ Upload CSV
uploaded_file = st.file_uploader("📂 Importez votre CSV (colonnes: image, texte)", type="csv")
if not uploaded_file:
    st.info("Veuillez importer un fichier CSV.")
    st.stop()
df = pd.read_csv(uploaded_file)
if not {"image","texte"}.issubset(df.columns):
    st.error("Les colonnes requises sont : image, texte")
    st.stop()
df = df.head(10)

# 6️⃣ Prédiction + barre + animation plein-écran
if st.button("🚀 Prédire"):
    age_norm     = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    gender_id    = gender_map[genre]
    total        = len(df)
    progress_bar = st.progress(0, text=f"0/{total}")
    results      = []

    with st.spinner("🚀 Prédiction en cours…"):
        for i, row in enumerate(df.itertuples(), start=1):
            p_tm  = predict_tm(row.texte, age_norm, gender_id)
            p_ctr = predict_ctr(row.texte)
            label = {0:"❗ Nobait", 1:"Softbait", 2:"✅ Clickbait"}[p_tm]
            results.append({
                "Texte":          row.texte,
                "Classification": label,
                "CTR prédit":     f"{p_ctr:.2f}%"
            })
            progress_bar.progress(i/total, text=f"{i}/{total}")

    progress_bar.empty()
    st.success("✅ Prédiction terminée !")

    # 🎈 Animation plein-écran
    st.balloons()  # célèbre la fin de la prédiction :contentReference[oaicite:0]{index=0}

    # 7️⃣ Post-traitement & affichage
    df_res = pd.DataFrame(results)
    df_res["CTR_num"] = df_res["CTR prédit"].str.rstrip("%").astype(float)
    df_res = df_res.sort_values("CTR_num", ascending=False)

    st.subheader("🔽 Tableau trié par CTR prédit")
    st.table(df_res[["Texte","Classification","CTR prédit"]])

    color_map  = {"❗ Nobait":"red","Softbait":"orange","✅ Clickbait":"green"}
    df_res["color"] = df_res["Classification"].map(color_map)
    class_encode = {"❗ Nobait":0, "Softbait":1, "✅ Clickbait":2}
    x = df_res["Classification"].map(class_encode) + np.random.normal(0, 0.05, len(df_res))

    fig, ax = plt.subplots()
    ax.scatter(x, df_res["CTR_num"], c=df_res["color"])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Nobait","Softbait","Clickbait"])
    ax.set_ylabel("CTR prédit (%)")
    ax.set_title("CTR vs Classification")
    plt.tight_layout()

    counts = df_res["Classification"].value_counts().reindex(color_map.keys(), fill_value=0)
    fig2, ax2 = plt.subplots()
    ax2.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[color_map[l] for l in counts.index]
    )
    ax2.set_title("Répartition des classes")
    ax2.axis("equal")

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.pyplot(fig)
    with col2:
        st.pyplot(fig2)

