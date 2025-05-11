import requirements.txt 
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache(allow_output_mutation=True)
def load_models():
    token = os.getenv("HF_TOKEN")
    # Clickbait privé
    clf = pipeline(
        "text-classification",
        model="alexandre-cameron-borges/clickbait-model",
        tokenizer="alexandre-cameron-borges/clickbait-model",
        use_auth_token=token
    )
    # CTR privé
    tok = AutoTokenizer.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=token
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        "alexandre-cameron-borges/ctr-model",
        use_auth_token=token
    )
    return clf, tok, mdl
