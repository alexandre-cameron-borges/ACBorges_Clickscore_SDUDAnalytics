import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# 1️⃣ Config
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_token  = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN manquant : impossible de charger les modèles privés")
hf_kwargs = {"use_auth_token": hf_token}

# 2️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", **hf_kwargs)

# 3️⃣ Classes modèles (identiques)
class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, bert_id, n_genders):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(bert_id, **hf_kwargs)
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc     = nn.Linear(1, 16)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head      = nn.Sequential(
            nn.Linear(hid, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask, age, gender):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pool      = out.pooler_output
        age_feat  = F.relu(self.age_fc(age.unsqueeze(1)))
        gen_feat  = self.gender_emb(gender)
        return self.head(torch.cat([pool, age_feat, gen_feat], dim=1)).squeeze(-1)

class CTRModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base     = AutoModel.from_pretrained(model_id, **hf_kwargs)
        cfg         = AutoConfig.from_pretrained(model_id, **hf_kwargs)
        self.reg_head = nn.Linear(cfg.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.reg_head(out.pooler_output)).squeeze(-1)

# 4️⃣ Lazy loaders
CLICKBAIT_ID = "alexandre-cameron-borges/clickbait-model"
CTR_ID       = "alexandre-cameron-borges/ctr-model"
N_GENDERS    = 3

_cb_model = None
_ctr_model = None

def _load_cb_model():
    global _cb_model
    if _cb_model is None:
        try:
            _cb_model = ClickbaitModelWithCustomHead(CLICKBAIT_ID, N_GENDERS).to(device).eval()
        except Exception as e:
            raise RuntimeError(f"Échec chargement ClickbaitModel : {e}")
    return _cb_model

def _load_ctr_model():
    global _ctr_model
    if _ctr_model is None:
        try:
            _ctr_model = CTRModel(CTR_ID).to(device).eval()
        except Exception as e:
            raise RuntimeError(f"Échec chargement CTRModel : {e}")
    return _ctr_model

# 5️⃣ Fonctions de prédiction
def predict_cb(text: str, age_norm: float, gender_id: int):
    model = _load_cb_model()
    enc   = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids   = enc.input_ids.to(device)
    mask  = enc.attention_mask.to(device)
    age_t = torch.tensor([age_norm], dtype=torch.float, device=device)
    gen_t = torch.tensor([gender_id], dtype=torch.long, device=device)
    with torch.no_grad():
        return torch.sigmoid(model(ids, mask, age_t, gen_t)).item()

def predict_ctr(text: str):
    model = _load_ctr_model()
    enc   = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids   = enc.input_ids.to(device)
    mask  = enc.attention_mask.to(device)
    with torch.no_grad():
        return model(ids, mask).item()
