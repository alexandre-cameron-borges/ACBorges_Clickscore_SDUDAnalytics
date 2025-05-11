import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 0) Token HF
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("⚠️ Vous devez définir HUGGINGFACE_TOKEN dans vos Secrets Streamlit.")
    st.stop()
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 1) Imports
# Assuming models.predict is in a directory named 'models' relative to this script
# and contains predict_cb and predict_ctr functions.
# Make sure suggested_models_predict.py is saved as models/predict.py in your project.
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

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier CSV: {e}")
    st.stop()

if not {"image","texte"}.issubset(df.columns):
    st.error("Les colonnes requises dans le CSV sont : image, texte")
    st.stop()

df = df.head(10) # Limit to 10 texts as per original requirement

# 6) Batch prédiction
if st.button("🚀 Prédire"):
    if df.empty:
        st.warning("Le fichier CSV est vide ou ne contient pas de données après le filtrage initial.")
        st.stop()

    age_norm  = (age - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    gender_id = gender_map[genre]
    
    results_data = [] # To store data for both table and plot
    
    with st.spinner("Prédiction en cours..."):
        for index, row in df.iterrows():
            try:
                p_cb  = predict_cb(row["texte"], age_norm, gender_id)
                p_ctr = predict_ctr(row["texte"]) # Assuming this also returns a float
                
                # Ensure p_ctr is a float, handle potential errors if predict_ctr is a placeholder
                if not isinstance(p_ctr, (int, float)):
                    st.warning(f"CTR prediction for text 	'{row['texte'][:30]}...' did not return a number. Using 0.0 as default.")
                    p_ctr = 0.0

                label = "❗ Clickbait" if p_cb >= 0.5 else "✅ Non-clickbait"
                is_clickbait = (p_cb >= 0.5)
                
                results_data.append({
                    "Texte": row["texte"],
                    "p_cb_raw": p_cb,
                    "p_ctr_raw": p_ctr,
                    "is_clickbait": is_clickbait,
                    "P(clickbait)": f"{p_cb:.1%}",
                    "Classification": label,
                    "CTR prédit": f"{p_ctr:.1%}"
                })
            except RuntimeError as e:
                st.error(f"Erreur lors de la prédiction pour le texte: 	'{row['texte'][:50]}...'")
                st.error(f"Détail de l'erreur: {e}")
                st.error("Veuillez vérifier que le modèle est correctement chargé et que le fichier models/predict.py est à jour avec les chemins corrects.")
                st.stop()
            except Exception as e:
                st.error(f"Une erreur inattendue est survenue lors de la prédiction pour le texte: 	'{row['texte'][:50]}...'")
                st.error(f"Détail: {e}")
                st.stop()

    if results_data:
        results_df = pd.DataFrame(results_data)

        st.subheader("Visualisation: P(clickbait) vs. CTR Prédit")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate data for plotting for clarity
        clickbait_data = results_df[results_df['is_clickbait'] == True]
        non_clickbait_data = results_df[results_df['is_clickbait'] == False]
        
        ax.scatter(non_clickbait_data['p_cb_raw'], non_clickbait_data['p_ctr_raw'], color='red', label='Non-Clickbait', alpha=0.7)
        ax.scatter(clickbait_data['p_cb_raw'], clickbait_data['p_ctr_raw'], color='green', label='Clickbait', alpha=0.7)
        
        ax.set_xlabel("Probabilité de Clickbait (P(clickbait))")
        ax.set_ylabel("CTR Prédit")
        ax.set_title("P(clickbait) vs. CTR Prédit pour les 10 Textes")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Résultats des Prédictions")
        st.table(results_df[["Texte", "P(clickbait)", "Classification", "CTR prédit"]])
    else:
        st.info("Aucun résultat à afficher. La prédiction n'a pas pu être complétée pour les textes fournis.")

