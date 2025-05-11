import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

# ——— 1. Auth & Device ———
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— 2. Download private checkpoints ———
HF_CB_REPO  = "alexandre-cameron-borges/clickbait-model"
HF_CTR_REPO = "alexandre-cameron-borges/ctr-model"
MODEL_PATH_CB  = hf_hub_download(repo_id=HF_CB_REPO,
                                 filename="pytorch_model.bin",
                                 use_auth_token=HF_TOKEN)
MODEL_PATH_CTR = hf_hub_download(repo_id=HF_CTR_REPO,
                                 filename="pytorch_model.bin",
                                 use_auth_token=HF_TOKEN)

# ——— 3. Clickbait Model Definition ———
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders, bert_name="bert-base-multilingual-cased"):
        super().__init__()
        self.bert       = BertModel.from_pretrained(bert_name)
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc     = nn.Linear(1, 16)
        hid_size        = self.bert.config.hidden_size + 8 + 16
        self.head       = nn.Sequential(
            nn.Linear(hid_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, age, gender):
        pooler_out   = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask).pooler_output
        age_feat     = F.relu(self.age_fc(age.unsqueeze(1)))
        gender_feat  = self.gender_emb(gender)
        combined     = torch.cat([pooler_out, age_feat, gender_feat], dim=1)
        return self.head(combined).squeeze(-1)

# ——— 4. Load & Prepare ———
N_GENDERS = 3  # must match training
_cb_model = ClickbaitModel(n_genders=N_GENDERS)
_cb_model.load_state_dict(torch.load(MODEL_PATH_CB, map_location=device))
_cb_model.to(device).eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# ——— 5. Prediction Functions ———
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    enc = tokenizer(text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    age  = torch.tensor([age_norm], dtype=torch.float, device=device)
    gen  = torch.tensor([gender_id], dtype=torch.long, device=device)
