import os
import streamlit as st
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 1. Authentification
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Définissez HF_TOKEN dans vos Secrets Streamlit.")
    st.stop()
login(token=HF_TOKEN)

@st.cache_resource
def load_models():
    # Clickbait
    clickbait_clf = pipeline(
        "text-classification",
        model="alexandre-cameron-borges/clickbait-model",
        tokenizer="alexandre-cameron-borges/clickbait-model",
        use_auth_token=HF_TOKEN
    )
    # CTR
    ctr_tok = AutoTokenizer.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=HF_TOKEN
    )
    ctr_model = AutoModelForSequenceClassification.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=HF_TOKEN
    )
    return clickbait_clf, ctr_tok, ctr_model

st.title("📝 Analyse Texte : Clickbait & CTR")
text = st.text_area("Entrez votre texte ici :", height=150)
if st.button("Évaluer"):
    clf, tok, mdl = load_models()
    cb = clf(text)[0]
    st.markdown(f"**Clickbait ?** {cb['label']} (confiance : {cb['score']:.2f})")
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    ctr = mdl(**inputs).logits.squeeze().item()
    st.markdown(f"**CTR prédit :** {ctr:.3f}")
