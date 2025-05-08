import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

# 1) Architecture identique à celle utilisée à l'entraînement
class ClickbaitModel(torch.nn.Module):
    def __init__(self, n_genders: int):
        super().__init__()
        self.bert       = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.age_fc     = torch.nn.Linear(1, 16)
        self.gender_emb = torch.nn.Embedding(n_genders, 8)
        hid_size = self.bert.config.hidden_size + 16 + 8
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hid_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, age, gender):
        # Extraction BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        # Passage âge & genre
        a   = F.relu(self.age_fc(age.unsqueeze(1)))
        g   = self.gender_emb(gender)
        # Concaténation et head
        x   = torch.cat([out, a, g], dim=1)
        return self.head(x).squeeze(-1)

# 2) Tokenizer (on garde celui de Hugging Face)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# 3) Chargement lazy du modèle depuis HF Hub
_model = None
def _load_model():
    global _model
    if _model is None:
        # Récupère ton token défini en env var
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        # Télécharge le checkpoint .pt
        local_path = hf_hub_download(
            repo_id="alexandre-cameron-borges/clickbait-model",
            filename="best_cb_model.pt",
            token=hf_token
        )
        # Instancie la classe et charge les poids
        _model = ClickbaitModel(n_genders=3)
        state_dict = torch.load(local_path, map_location="cpu")
        _model.load_state_dict(state_dict)
        _model.eval()
    return _model

# 4) Fonction exposée pour l’inférence
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    model = _load_model()
    enc   = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    logits = model(
        enc.input_ids,
        enc.attention_mask,
        torch.tensor([age_norm]),
        torch.tensor([gender_id])
    )
    # Renvoie une probabilité [0,1]
    return torch.sigmoid(logits).item()
