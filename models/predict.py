import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

# Toujours CPU
DEVICE = torch.device("cpu")

# Tokenizer partagé
BERT_NAME = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

# === Modèle Clickbait ===
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders=3):
        super().__init__()
        # Chargement immédiat CPU en float32
        self.bert       = BertModel.from_pretrained(BERT_NAME, torch_dtype=torch.float32)
        self.age_fc     = nn.Linear(1, 16)
        self.gender_emb = nn.Embedding(n_genders, 8)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, age, gender):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        a   = F.relu(self.age_fc(age.unsqueeze(1)))
        g   = self.gender_emb(gender)
        x   = torch.cat([cls, a, g], dim=1)
        return self.head(x).squeeze(-1)

_cb_model = None
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    global _cb_model
    if _cb_model is None:
        # Télécharger et charger les poids
        path = hf_hub_download(
            repo_id="alexandre-cameron-borges/clickbait-model",
            filename="best_cb_model.pt",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        _cb_model = ClickbaitModel(n_genders=3)
        _cb_model.load_state_dict(torch.load(path, map_location="cpu"))
        _cb_model.eval()
        # Warm-up pour éliminer tout meta-tensor
        dummy_ids  = torch.zeros(1,128, dtype=torch.long)
        dummy_mask = torch.zeros_like(dummy_ids)
        dummy_age  = torch.tensor([0.0])
        dummy_gen  = torch.tensor([0])
        with torch.no_grad():
            _cb_model(dummy_ids, dummy_mask, dummy_age, dummy_gen)

    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    age_t     = torch.tensor([age_norm], dtype=torch.float32)
    gen_t     = torch.tensor([gender_id], dtype=torch.long)
    with torch.no_grad():
        logit = _cb_model(ids, mask, age_t, gen_t)
    return torch.sigmoid(logit).item()

# === Modèle CTR ===
class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME, torch_dtype=torch.float32)
        hid = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(cls).squeeze(-1)

_ctr_model = None
def predict_ctr(text: str) -> float:
    global _ctr_model
    if _ctr_model is None:
        path = hf_hub_download(
            repo_id="alexandre-cameron-borges/ctr-model",
            filename="best_ctr_model.pt",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        _ctr_model = CTRModel()
        _ctr_model.load_state_dict(torch.load(path, map_location="cpu"))
        _ctr_model.eval()
        # Warm-up
        dummy_ids  = torch.zeros(1,128, dtype=torch.long)
        dummy_mask = torch.zeros_like(dummy_ids)
        with torch.no_grad():
            _ctr_model(dummy_ids, dummy_mask)

    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    with torch.no_grad():
        val = _ctr_model(ids, mask)
    return float(val.item())
