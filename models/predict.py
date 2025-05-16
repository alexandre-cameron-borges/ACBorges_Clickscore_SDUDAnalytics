import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download

# 1️⃣ Config
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise RuntimeError("HUGGINGFACE_TOKEN manquant : impossible de charger les modèles privés")
hf_kwargs = {"use_auth_token": hf_token}

# Device (CPU ou CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", **hf_kwargs
)

# 3️⃣ Architectures
class ClickbaitModelWithCustomHead(nn.Module):
    # … inchangé …

class CTRModel(nn.Module):
    # … inchangé …

# 4️⃣ Lazy-loading + poids
CLICKBAIT_ID = "alexandre-cameron-borges/clickbait-model"
CTR_ID       = "alexandre-cameron-borges/ctr-model"
BACKBONE_ID  = "bert-base-multilingual-cased"
N_GENDERS    = 3

_cb_model = None
_ctr_model = None

def _load_cb_model():
    global _cb_model
    if _cb_model is None:
        ckpt = hf_hub_download(
            repo_id=CLICKBAIT_ID,
            filename="best_cb_model.pt",
            use_auth_token=hf_token
        )
        model = ClickbaitModelWithCustomHead(BACKBONE_ID, N_GENDERS, n_tm_classes=3)
        # → déplacement sur GPU **uniquement** si CUDA
        if device.type == "cuda":
            try:
                model = model.to(device)
            except NotImplementedError:
                pass

        state = torch.load(ckpt, map_location=device)
        model_dict      = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
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
        model = CTRModel(BACKBONE_ID)
        # → déplacement sur GPU **uniquement** si CUDA
        if device.type == "cuda":
            try:
                model = model.to(device)
            except NotImplementedError:
                pass

        state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"⚠️ Missing keys in CTR model: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys in CTR model: {unexpected}")
        model.eval()
        _ctr_model = model
    return _ctr_model

# 5️⃣ Fonctions de prédiction
def predict_cb(text: str, age_norm: float, gender_id: int) -> float:
    # inchangé …

def predict_ctr(text: str) -> float:
    # inchangé …
