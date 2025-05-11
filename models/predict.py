# 1) → Désactiver TorchDynamo & passer en mode eager
import torch
try:
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.disable()
except ImportError:
    pass

import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cpu")
BERT_NAME = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

class ClickbaitModel(nn.Module):
    def __init__(self, n_genders=3):
        super().__init__()
        self.bert       = BertModel.from_pretrained(BERT_NAME, torch_dtype=torch.float32)
        self.age_fc     = nn.Linear(1,16)
        self.gender_emb = nn.Embedding(n_genders,8)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(
            nn.Linear(hid,64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64,1)
        )
    def forward(self, input_ids, attention_mask, age, gender):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        a   = F.relu(self.age_fc(age.unsqueeze(1)))
        g   = self.gender_emb(gender)
        x   = torch.cat([out, a, g], dim=1)
        return self.head(x).squeeze(-1)

class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME, torch_dtype=torch.float32)
        hid = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid,64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64,1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(out).squeeze(-1)

# Chargement des poids (CPU par défaut)
_cb_model = ClickbaitModel(n_genders=3)
path_cb   = hf_hub_download("alexandre-cameron-borges/clickbait-model", "best_cb_model.pt",
                            token=os.getenv("HUGGINGFACE_TOKEN"))
_cb_model.load_state_dict(torch.load(path_cb, map_location="cpu"))
_cb_model.eval()

_ctr_model = CTRModel()
path_ctr   = hf_hub_download("alexandre-cameron-borges/ctr-model", "best_ctr_model.pt",
                             token=os.getenv("HUGGINGFACE_TOKEN"))
_ctr_model.load_state_dict(torch.load(path_ctr, map_location="cpu"))
_ctr_model.eval()

def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    age_t  = torch.tensor([age_norm], dtype=torch.float32)
    gen_t  = torch.tensor([gender_id], dtype=torch.long)
    with torch.no_grad():
        logit = _cb_model(ids, mask, age_t, gen_t)
    return torch.sigmoid(logit).item()

def predict_ctr(text: str) -> float:
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    with torch.no_grad():
        val = _ctr_model(ids, mask)
    return float(val.item())
