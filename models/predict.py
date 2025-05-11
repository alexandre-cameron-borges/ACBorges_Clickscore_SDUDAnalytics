import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoConfig
import os

# --- 1. Device Configuration & Token ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The HUGGINGFACE_TOKEN should be set as an environment variable in the deployment environment.
# app2.py already attempts to set os.environ["HUGGINGFACE_TOKEN"] from st.secrets.
# The transformers library will try to use this token if available, especially for private models.
# Alternatively, ensure HUGGING_FACE_HUB_TOKEN environment variable is set.
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    print("Warning: HUGGINGFACE_TOKEN environment variable not found. Private model loading might fail.")

# --- 2. Model Definitions ---

# Clickbait Model: Assumes the HF model is the BERT part, and we add a custom head for age/gender.
class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, bert_model_name_or_path, n_genders, token=None):
        super().__init__()
        # Load the BERT model from Hugging Face
        # If the model on HF is a full classification model, this approach might need adjustment.
        # This assumes 'bert_model_name_or_path' provides the base BERT-like features.
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name_or_path, token=token)
        except Exception as e:
            print(f"Error loading BERT model from {bert_model_name_or_path}: {e}")
            print("Please ensure the Hugging Face model identifier is correct and the model is accessible (e.g., public or token provided for private models).")
            raise
            
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc = nn.Linear(1, 16)  # For normalized age
        
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, age, gender):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = bert_output.pooler_output
        
        age_features = F.relu(self.age_fc(age.unsqueeze(1)))
        gender_features = self.gender_emb(gender)
        
        combined_features = torch.cat([pooler_output, age_features, gender_features], dim=1)
        logits = self.head(combined_features)
        return logits.squeeze(-1)


# CTR Model: Assumes this is a standard sequence classification/regression model from Hugging Face.
# If it's for regression (predicting a CTR value), AutoModelForSequenceClassification might still work if output is single logit.
# Or it could be a custom model on HF. For now, let's assume it's a model that gives a single output suitable for CTR.
class CTRModel(nn.Module):
    def __init__(self, model_name_or_path, token=None):
        super().__init__()
        try:
            self.base_model = AutoModel.from_pretrained(model_name_or_path, token=token)
            # Add a regression head if the HF model is just the base encoder (e.g. BERT)
            # If the HF model is already a full regression model, this head might be redundant or need adjustment.
            self.config = AutoConfig.from_pretrained(model_name_or_path, token=token)
            self.regression_head = nn.Linear(self.config.hidden_size, 1)
        except Exception as e:
            print(f"Error loading CTR base model from {model_name_or_path}: {e}")
            raise

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output or last_hidden_state[:, 0] (CLS token) for sentence-level tasks
        pooler_output = outputs.pooler_output 
        ctr_logit = self.regression_head(pooler_output)
        return torch.sigmoid(ctr_logit).squeeze(-1) # Assuming CTR is a probability (0-1)

# --- 3. Tokenizer (Initialize once) ---
# Using a generic multilingual tokenizer. Adjust if your models require a specific one.
tokenizer_name = "bert-base-multilingual-cased"
try:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
except Exception as e:
    print(f"Error loading tokenizer {tokenizer_name}: {e}")
    # Fallback or specific model tokenizer might be needed if the generic one fails or is unsuitable.
    # For example, tokenizer = AutoTokenizer.from_pretrained(CLICKBAIT_MODEL_HF_ID, token=hf_token)
    raise

# --- 4. Load Trained Models from Hugging Face ---
CLICKBAIT_MODEL_HF_ID = "alexandre-cameron-borges/clickbait-model"
CTR_MODEL_HF_ID = "alexandre-cameron-borges/ctr-model"

# Number of unique gender IDs your clickbait model was trained with.
# This is for the nn.Embedding layer in ClickbaitModelWithCustomHead.
# If your gender_map in app2.py is {"Male":0, "Female":1, "Unknown":2}, then n_genders = 3.
N_GENDERS_FOR_CLICKBAIT_MODEL = 3

_cb_model = None
_ctr_model = None

try:
    print(f"Loading ClickbaitModelWithCustomHead using BERT part from: {CLICKBAIT_MODEL_HF_ID}")
    _cb_model = ClickbaitModelWithCustomHead(bert_model_name_or_path=CLICKBAIT_MODEL_HF_ID, 
                                             n_genders=N_GENDERS_FOR_CLICKBAIT_MODEL, 
                                             token=hf_token)
    _cb_model.to(device)
    _cb_model.eval()
    print(f"Clickbait model loaded successfully and moved to {device}.")
except Exception as e:
    print(f"ERROR: Could not load the Clickbait model from Hugging Face ({CLICKBAIT_MODEL_HF_ID}): {e}")
    _cb_model = None

try:
    print(f"Loading CTRModel from: {CTR_MODEL_HF_ID}")
    _ctr_model = CTRModel(model_name_or_path=CTR_MODEL_HF_ID, token=hf_token)
    _ctr_model.to(device)
    _ctr_model.eval()
    print(f"CTR model loaded successfully and moved to {device}.")
except Exception as e:
    print(f"ERROR: Could not load the CTR model from Hugging Face ({CTR_MODEL_HF_ID}): {e}")
    _ctr_model = None

# --- 5. Prediction Functions ---
def predict_cb(text: str, age_norm: float, gender_id: int):
    if _cb_model is None:
        raise RuntimeError(
            f"Clickbait model (_cb_model) from {CLICKBAIT_MODEL_HF_ID} is not loaded. "
            "Check Hugging Face model ID, token, and N_GENDERS. See console for loading errors."
        )
    _cb_model.eval()
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    age_tensor = torch.tensor([age_norm], dtype=torch.float).to(device)
    gender_tensor = torch.tensor([gender_id], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = _cb_model(input_ids, attention_mask, age_tensor, gender_tensor)
        probability = torch.sigmoid(logits).item()
    return probability

def predict_ctr(text: str):
    if _ctr_model is None:
        raise RuntimeError(
            f"CTR model (_ctr_model) from {CTR_MODEL_HF_ID} is not loaded. "
            "Check Hugging Face model ID and token. See console for loading errors."
        )
    _ctr_model.eval()
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        # The CTRModel is designed to output a sigmoid probability directly
        probability = _ctr_model(input_ids, attention_mask).item()
    return probability

if __name__ == '__main__':
    print("Testing predict.py script with Hugging Face model loading...")
    # Set HUGGINGFACE_TOKEN environment variable if your models are private for this test to work.
    # Example: export HUGGINGFACE_TOKEN='your_token_here'
    if os.environ.get("HUGGINGFACE_TOKEN"):
        if _cb_model:
            print(f"Clickbait model loaded. Running a test prediction.")
            cb_prob = predict_cb("This is an amazing title you won't believe!", 0.0, 0)
            print(f"Test Clickbait Prediction: {cb_prob:.4f}")
        else:
            print("Clickbait model not loaded. Cannot run clickbait test prediction.")
        
        if _ctr_model:
            print(f"CTR model loaded. Running a test prediction.")
            ctr_val = predict_ctr("An interesting article about technology.")
            print(f"Test CTR Prediction: {ctr_val:.4f}")
        else:
            print("CTR model not loaded. Cannot run CTR test prediction.")
    else:
        print("HUGGINGFACE_TOKEN not set. Skipping direct execution tests that require private model access.")

