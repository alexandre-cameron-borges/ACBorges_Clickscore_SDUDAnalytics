import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download

# 1️⃣ Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN manquant : impossible de charger les modèles privés")
hf_kwargs = {"use_auth_token": hf_token}

# 2️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", **hf_kwargs
)

# 3️⃣ Architectures multi-tâche
class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, backbone_id, n_genders, n_tm_classes=3):
        super().__init__()
        # Charger directement sur GPU/CPU avec device_map et low_cpu_mem_usage
        self.bert = AutoModel.from_pretrained(
            backbone_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            **hf_kwargs
        )
        # déplacer la tête sur le même device
        hidden = self.bert.config.hidden_size
        self.gender_emb = nn.Embedding(n_genders, 8).to(self.bert.device)
        self.age_fc     = nn.Linear(1, 16).to(self.bert.device)
        hid = hidden + 16 + 8

        self.cb_head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        ).to(self.bert.device)

        self.tm_head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_tm_classes)
        ).to(self.bert.device)

    def forward(self, input_ids, attention_mask, age, gender):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device
