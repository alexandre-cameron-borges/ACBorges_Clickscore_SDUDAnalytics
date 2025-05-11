# models/predict.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download

# 1️⃣ Config
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN manquant : impossible de charger les modèles privés")
hf_kwargs = {"use_auth_token": hf_token}

# 2️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", **hf_kwargs
)

# 3️⃣ Architectures (chargent le backbone BERT public)
class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, backbone_id, n_genders):
        super().__init__()
        # Charger un BERT public comme backbone
        self.bert       = AutoModel.from_pretrained(backbone_id, **hf_kwargs)
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc     = nn.Linear(1, 16)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head      = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask, age, gender):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pool      = out.pooler_output
        age_feat  = F.relu(self.age_fc(age.unsqueeze(1)))
        gen_feat  = self.gender_emb(gender)
        return self.head(torch.cat([pool, age_feat, gen_feat], dim=1)).squeeze(-1)

class CTRModel(nn.Module):
    def __init__(self, backbone_id):
        super().__init__()
        self.base     = AutoModel.from_pretrained(backbone_id, **hf_kwargs)
        cfg         = AutoConfig.from_pretrained(backbone_id, **hf_kwargs)
        self.reg_head = nn.Linear(cfg.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.reg_head(out.pooler_output)).squeeze(-1)

# 4️⃣ Lazy-loading + téléchargement des .pt
CLICKBAIT_ID   = "alexandre-cameron-borges/clickbait-model"
CTR_ID         = "alexandre-cameron-borges/ctr-model"
BACKBONE_ID    = "bert-base-multilingual-cased"
N_GENDERS      = 3

_cb_model = None
_ctr_model = None

def _load_cb_model():
    global _cb_model
    if _cb_model is None:
        # 1. Télécharger votre checkpoint .pt
        ckpt = hf_hub_download(
            repo_id=CLICKBAIT_ID,
            filename="best_cb_model.pt",
            use_auth_token=hf_token
        )
        # 2. Instancier l’archi avec backbone public
        model = ClickbaitModelWithCustomHead(BACKBONE_ID, N_GENDERS).to(device)
        # 3. Charger les poids
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _cb_model = model
    return _cb_model

def _load_ctr_model():
    global _ctr_model
    if _ctr_model is None:
        ckpt = hf_hub_download(
            repo_id=CTR_ID,
            filename="best_ctr_model.pt",
            use_auth_token=hf_token
        )
        model = CTRModel(BACKBONE_ID).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _ctr_model = model
    return _ctr_model

# 5️⃣ Fonctions de prédiction
def predict_cb(text: str, age_norm: float, gender_id: int):
    model = _load_cb_model()
    enc   = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    )
    ids   = enc.input_ids.to(device)
    mask  = enc.attention_mask.to(device)
    age_t = torch.tensor([age_norm], dtype=torch.float, device=device)
    gen_t = torch.tensor([gender_id], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids, mask, age_t, gen_t)
        return torch.sigmoid(logits).item()

def predict_ctr(text: str):
    model = _load_ctr_model()
    enc   = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    )
    ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
    with torch.no_grad():
        return model(ids, mask).item()
