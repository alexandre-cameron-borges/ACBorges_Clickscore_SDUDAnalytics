import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig
)

# 1. Device & Token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    print("Warning: HUGGINGFACE_TOKEN not found. Private model loading might fail.")
hf_kwargs = {"use_auth_token": hf_token} if hf_token else {}

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", **hf_kwargs
)

# 3. Modèles

class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, bert_model_name, n_genders):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name, **hf_kwargs)
        except Exception as e:
            print(f"Error loading BERT from {bert_model_name}: {e}")
            raise
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc = nn.Linear(1, 16)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(
            nn.Linear(hid, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, age, gender):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler = out.pooler_output
        age_feat = F.relu(self.age_fc(age.unsqueeze(1)))
        gen_feat = self.gender_emb(gender)
        cat = torch.cat([pooler, age_feat, gen_feat], dim=1)
        return self.head(cat).squeeze(-1)


class CTRModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        try:
            self.base = AutoModel.from_pretrained(model_name, **hf_kwargs)
            cfg = AutoConfig.from_pretrained(model_name, **hf_kwargs)
            self.reg_head = nn.Linear(cfg.hidden_size, 1)
        except Exception as e:
            print(f"Error loading CTR model {model_name}: {e}")
            raise

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.reg_head(out.pooler_output)).squeeze(-1)


# 4. Chargement
CLICKBAIT_ID = "alexandre-cameron-borges/clickbait-model"
CTR_ID       = "alexandre-cameron-borges/ctr-model"
N_GENDERS    = 3

_cb_model, _ctr_model = None, None
try:
    _cb_model = ClickbaitModelWithCustomHead(CLICKBAIT_ID, N_GENDERS).to(device).eval()
    print("Clickbait model ready.")
except: pass

try:
    _ctr_model = CTRModel(CTR_ID).to(device).eval()
    print("CTR model ready.")
except: pass


# 5. Prédiction
def predict_cb(text: str, age_norm: float, gender_id: int):
    if _cb_model is None:
        raise RuntimeError(f"Clickbait model not loaded (ID: {CLICKBAIT_ID}).")
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    in_ids = enc.input_ids.to(device)
    mask   = enc.attention_mask.to(device)
    age_t  = torch.tensor([age_norm], dtype=torch.float, device=device)
    gen_t  = torch.tensor([gender_id], dtype=torch.long, device=device)
    with torch.no_grad():
        return torch.sigmoid(_cb_model(in_ids, mask, age_t, gen_t)).item()

def predict_ctr(text: str):
    if _ctr_model is None:
        raise RuntimeError(f"CTR model not loaded (ID: {CTR_ID}).")
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    in_ids = enc.input_ids.to(device)
    mask   = enc.attention_mask.to(device)
    with torch.no_grad():
        return _ctr_model(in_ids, mask).item()
