import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

# 0) Device global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Token HF
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HUGGINGFACE_TOKEN introuvable dans l'environnement.")

# 2) Tokenizers BERT
BERT_BASE     = "bert-base-multilingual-cased"
tokenizer_cb  = BertTokenizer.from_pretrained(BERT_BASE)
tokenizer_ctr = BertTokenizer.from_pretrained(BERT_BASE)

# 3) Modèle Clickbait
CB_REPO = "alexandre-cameron-borges/clickbait-model"
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders=3):
        super().__init__()
        self.bert       = BertModel.from_pretrained(BERT_BASE)
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        a   = F.relu(self.age_fc(age.unsqueeze(1)))
        g   = self.gender_emb(gender)
        x   = torch.cat([out, a, g], dim=1)
        return self.head(x).squeeze(-1)

_cb_model = None
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    global _cb_model
    if _cb_model is None:
        path = hf_hub_download(CB_REPO, "best_cb_model.pt", token=HF_TOKEN)
        _cb_model = ClickbaitModel(n_genders=3).to(DEVICE)
        _cb_model.load_state_dict(torch.load(path, map_location=DEVICE))
        _cb_model.eval()
    enc = tokenizer_cb(text, padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    input_ids      = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)
    age_tensor     = torch.tensor([age_norm], dtype=torch.float, device=DEVICE)
    gender_tensor  = torch.tensor([gender_id], dtype=torch.long, device=DEVICE)
    logits = _cb_model(input_ids, attention_mask, age_tensor, gender_tensor)
    return torch.sigmoid(logits).item()

# 4) Modèle CTR
CTR_REPO = "alexandre-cameron-borges/ctr-model"
class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_BASE)
        hid = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(out).squeeze(-1)

_ctr_model = None
def predict_ctr(text: str) -> float:
    global _ctr_model
    if _ctr_model is None:
        path = hf_hub_download(CTR_REPO, "best_ctr_model.pt", token=HF_TOKEN)
        _ctr_model = CTRModel().to(DEVICE)
        _ctr_model.load_state_dict(torch.load(path, map_location=DEVICE))
        _ctr_model.eval()
    enc = tokenizer_ctr(text, padding="max_length", truncation=True,
                        max_length=128, return_tensors="pt")
    input_ids      = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)
    val = _ctr_model(input_ids, attention_mask)
    return float(val.item())
