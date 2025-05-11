import streamlit as st
import pandas as pd
import os

# It's good practice to define constants at the beginning or import them.
# These are needed for the age_norm calculation.
# Please ensure these values are appropriate for your model and data.
# Replace with actual median age and max age used for normalization in your model.
MEDIAN_AGE = 35.0  # Example: Actual median age from your dataset
MAX_AGE = 99.0     # Example: Actual maximum age from your dataset

# Ensure HUGGINGFACE_TOKEN is set in secrets
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("Le secret HUGGINGFACE_TOKEN n'est pas configuré.")
    st.stop()
os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]

# 1) Imports
# Make sure your custom prediction functions are available.
# For example, they might be in a 'models/predict.py' file.
try:
    from models.predict import predict_cb, predict_ctr
except ImportError:
    st.error("Erreur: Impossible d'importer les fonctions 'predict_cb' et 'predict_ctr' depuis 'models.predict'. Veuillez vérifier le fichier et son emplacement.")
    # Define dummy functions if actual models are not available for testing UI
    def predict_cb(text, age_norm, gender_id):
        st.warning("Fonction 'predict_cb' factice utilisée.")
        return 0.6 if 'clickbait' in text.lower() else 0.2
    def predict_ctr(text):
        st.warning("Fonction 'predict_ctr' factice utilisée.")
        return 0.05 if 'important' in text.lower() else 0.01
    # st.stop() # Uncomment if you want to stop execution if models can't be loaded

# 2) UI Setup
st.set_page_config(page_title="Clickbait & CTR Predictor", layout="centered")
st.title("💡 Détecteur de Clickbait & CTR Prédictif")

# Mapping for gender selection
gender_map = {"Male": 0, "Female": 1, "Unknown": 2}

# 4) Sélecteurs globaux
# The 'step=1' was removed from the age slider based on your provided diff.
age = st.slider("🎯 Âge cible", 18, 99, 30)
genre = st.selectbox("👤 Genre cible", list(gender_map.keys()))

# 5) CSV uploader
# The uploader description was changed based on your provided diff.
uploaded_file = st.file_uploader("📂 Importez votre CSV (colonnes: image, texte)", type="csv")

if not uploaded_file:
    st.info("Veuillez importer un fichier CSV pour commencer.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier CSV: {e}")
    st.stop()

# Validate required columns
if not {"image", "texte"}.issubset(df.columns):
    st.error("Le fichier CSV importé doit contenir les colonnes : 'image' et 'texte'.")
    st.stop()

# Limit processing to the first 10 rows as per the original script logic
df = df.head(10)

# 6) Batch prédiction
if st.button("🚀 Prédire"):
    # Normalize age
    # Ensure MEDIAN_AGE and MAX_AGE are correctly defined above or globally accessible
    if MAX_AGE == MEDIAN_AGE: # Avoid division by zero if ages are the same
        st.warning("MAX_AGE et MEDIAN_AGE sont égaux. La normalisation de l'âge peut produire des résultats inattendus.")
        age_norm = 0.0 # Default or handle as appropriate for your model
    else:
        age_norm = (float(age) - MEDIAN_AGE) / (MAX_AGE - MEDIAN_AGE)
    
    gender_id = gender_map[genre]
    
    table_results_data = []       # To store formatted data for the table display
    scatter_plot_raw_data = []  # To store raw numerical data for the scatter plot

    # Process each row of the DataFrame
    for index, row in df.iterrows():
        try:
            text_content = str(row["texte"]) # Ensure text is treated as a string

            # Call your prediction functions
            # These should return float values (probabilities)
            p_cb = predict_cb(text_content, age_norm, gender_id)
            p_ctr = predict_ctr(text_content)

            # Store raw probabilities for the scatter plot
            scatter_plot_raw_data.append({
                "P(clickbait)": p_cb,  # X-axis data point
                "CTR prédit": p_ctr      # Y-axis data point
            })

            # Determine classification label based on clickbait probability
            label = "❗ Clickbait" if p_cb >= 0.5 else "✅ Non-clickbait"

            # Store formatted data for the results table
            table_results_data.append({
                "Texte": text_content,
                "P(clickbait)": f"{p_cb:.1%}", # Format as percentage
                "Classification": label,
                "CTR prédit": f"{p_ctr:.1%}"  # Format as percentage
            })
        except Exception as e:
            st.error(f"Erreur lors de la prédiction pour la ligne {index} ('{row.get('texte', 'N/A')}'): {e}")
            # Optionally, add a placeholder to the table for rows that failed
            table_results_data.append({
                "Texte": str(row.get("texte", "N/A")),
                "P(clickbait)": "Erreur",
                "Classification": "Erreur",
                "CTR prédit": "Erreur"
            })
            # Do not add erroneous data to scatter_plot_raw_data

    # Display scatter plot if data points were successfully generated
    if scatter_plot_raw_data:
        df_scatter = pd.DataFrame(scatter_plot_raw_data)
        st.subheader("📊 Visualisation P(clickbait) vs. CTR Prédit")
        # Ensure the column names here match the keys used in scatter_plot_raw_data
        st.scatter_chart(df_scatter, x="P(clickbait)", y="CTR prédit")
    else:
        if table_results_data: # If there were table results but no scatter data (e.g. all predictions failed)
             st.info("Aucune donnée valide pour générer le graphique à nuage de points.")

    # Display table with prediction results
    if table_results_data:
        st.subheader("📋 Résultats des Prédictions")
        st.table(pd.DataFrame(table_results_data))
    else:
        # This case would typically only happen if the input df was empty or all rows failed before any processing.
        st.info("Aucun résultat à afficher dans le tableau.")
