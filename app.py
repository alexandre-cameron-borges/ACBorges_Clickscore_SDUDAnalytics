import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 1. Récupération du token Hugging Face (défini dans Streamlit Secrets sous HF_TOKEN)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Veuillez définir HF_TOKEN dans vos Secrets Streamlit.")
    st.stop()

@st.cache(allow_output_mutation=True)
def load_models():
    # Chargement du modèle de classification clickbait
    clickbait_clf = pipeline(
        "text-classification",
        model="alexandre-cameron-borges/clickbait-model",
        tokenizer="alexandre-cameron-borges/clickbait-model",
        use_auth_token=HF_TOKEN
    )
    # Chargement du modèle de régression CTR
    ctr_tok = AutoTokenizer.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=HF_TOKEN
    )
    ctr_model = AutoModelForSequenceClassification.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=HF_TOKEN
    )
    return clickbait_clf, ctr_tok, ctr_model

# --- Interface utilisateur ---
st.title("📝 Analyse Texte: Clickbait & CTR")
user_text = st.text_area("Entrez votre texte ici :", height=150)

if st.button("Évaluer"):
    clf, tok, mdl = load_models()

    # 1) Prédiction clickbait
    cb_out = clf(user_text)[0]
    st.markdown(f"**Clickbait ?** {cb_out['label']} (confiance : {cb_out['score']:.2f})")

    # 2) Prédiction CTR
    inputs = tok(user_text, return_tensors="pt", truncation=True, padding=True)
    ctr_logit = mdl(**inputs).logits.squeeze().item()
    st.markdown(f"**CTR prédit :** {ctr_logit:.3f}")
