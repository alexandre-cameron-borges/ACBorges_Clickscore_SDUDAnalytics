import os, traceback
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# 1️⃣ Device & HF token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
hf_kwargs = {"use_auth_token": hf_token} if hf_token else {}

# 2️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", **hf_kwargs
)

# 3️⃣ Modèles
class ClickbaitModelWithCustomHead(nn.Module):
    def __init__(self, bert_id, n_genders):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_id, **hf_kwargs)
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc = nn.Linear(1, 16)
        hid = self.bert.config.hidden_size + 16 + 8
        self.head = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, input_ids, attention_mask, age, gender):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pool = out.pooler_output
        age_feat = F.relu(self.age_fc(age.unsqueeze(1)))
        gen_feat = self.gender_emb(gender)
        cat = torch.cat([pool, age_feat, gen_feat], dim=1)
        return self.head(cat).squeeze(-1)

class CTRModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_id, **hf_kwargs)
        cfg = AutoConfig.from_pretrained(model_id, **hf_kwargs)
        self.reg_head = nn.Linear(cfg.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(self.reg_head(out.pooler_output)).squeeze(-1)

# 4️⃣ Chargement
CLICKBAIT_ID = "alexandre-cameron-borges/clickbait-model"
CTR_ID       = "alexandre-cameron-borges/ctr-model"
N_GENDERS    = 3

_cb_model, _ctr_model = None, None
try:
    _cb_model = ClickbaitModelWithCustomHead(CLICKBAIT_ID, N_GENDERS).to(device).eval()
    print("✅ Clickbait model loaded.")
except Exception as e:
    traceback.print_exc(); print(f"❌ Clickbait load error: {e}")

try:
    _ctr_model = CTRModel(CTR_ID).to(device).eval()
    print("✅ CTR model loaded.")
except Exception as e:
    traceback.print_exc(); print(f"❌ CTR load error: {e}")

# 5️⃣ Fonctions de prédiction
def predict_cb(text: str, age_norm: float, gender_id: int):
    if _cb_model is None:
        raise RuntimeError(f"Clickbait model not loaded (ID: {CLICKBAIT_ID}).")
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    in_ids = enc.input_ids.to(device); mask = enc.attention
