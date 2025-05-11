import os
import torch
import torch.nn as nn, torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HUGGINGFACE_TOKEN introuvable dans l'environnement.")

# -- Tokenizer et architecture BERT de base (pas ton repo) --
BERT_BASE = "bert-base-multilingual-cased"
tokenizer_cb  = BertTokenizer.from_pretrained(BERT_BASE)
tokenizer_ctr = BertTokenizer.from_pretrained(BERT_BASE)

# -- Clickbait Model checkpoint repo --
CB_REPO = "alexandre-cameron-borges/clickbait-model"
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders=3):
        super().__init__()
        # on charge BERT de base
        self.bert       = BertModel.from_pretrained(BERT_BASE)
        self.age_fc     = nn.Linear(1,16)
        self.gender_emb = nn.Embedding(n_genders,8)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(
            nn.Linear(hid,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,1)
        )
    def forward(self, input_ids, attention_mask, age, gender):
        out = self.bert(input_ids, attention_mask).pooler_output
        a   = F.relu(self.age_fc(age.unsqueeze(1)))
        g   = self.gender_emb(gender)
        x   = torch.cat([out, a, g], dim=1)
        return self.head(x).squeeze(-1)

_cb_model = None
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    global _cb_model
    if _cb_model is None:
        # DL du checkpoint
        path = hf_hub_download(CB_REPO, "best_cb_model.pt", token=HF_TOKEN)
        _cb_model = ClickbaitModel(n_genders=3)
        _cb_model.load_state_dict(torch.load(path, map_location="cpu"))
        _cb_model.eval()
    enc = tokenizer_cb(text, padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    logits = _cb_model(enc.input_ids, enc.attention_mask,
                       torch.tensor(age_norm), torch.tensor(gender_id))
    return torch.sigmoid(logits).item()


# -- CTR Model checkpoint repo --
CTR_REPO = "alexandre-cameron-borges/ctr-model"
class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_BASE)
        hid = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask).pooler_output
        return self.head(out).squeeze(-1)

_ctr_model = None
def predict_ctr(text: str) -> float:
    global _ctr_model
    if _ctr_model is None:
        path = hf_hub_download(CTR_REPO, "best_ctr_model.pt", token=HF_TOKEN)
        _ctr_model = CTRModel()
        _ctr_model.load_state_dict(torch.load(path, map_location="cpu"))
        _ctr_model.eval()
    enc = tokenizer_ctr(text, padding="max_length", truncation=True,
                        max_length=128, return_tensors="pt")
    val = _ctr_model(enc.input_ids, enc.attention_mask)
    return float(val.item())

