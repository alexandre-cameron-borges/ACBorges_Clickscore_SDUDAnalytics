import os
import streamlit as st
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Définissez HF_TOKEN dans vos Secrets Streamlit.")
    st.stop()
login(token=HF_TOKEN)

@st.cache_resource
def load_models():
    # 1) Clickbait : charger explicitement le modèle PyTorch (ou convertir TF→PT)
    cb_tok = AutoTokenizer.from_pretrained(
        "alexandre-cameron-borges/clickbait-model",
        use_auth_token=HF_TOKEN
    )
    cb_model = AutoModelForSequenceClassification.from_pretrained(
        "alexandre-cameron-borges/clickbait-model",
        use_auth_token=HF_TOKEN,
        # if needed: from_tf=True
    )
    clickbait_clf = pipeline(
        "text-classification",
        model=cb_model,
        tokenizer=cb_tok,
        framework="pt"
    )

    # 2) CTR : même principe pour la régression
    ctr_tok = AutoTokenizer.from_pretrained(
        "alexandre-cameron-borges/ctr-model", use_auth_token=HF_TOKEN
    )
    ctr_model = AutoModelForSequenceClassification.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=HF_TOKEN,
        # from_tf=True si votre Hub n’a que TF weights
    )
    return clickbait_clf, ctr_tok, ctr_model

