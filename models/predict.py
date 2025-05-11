import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# Forcer l'utilisation du CPU dès le départ
device = torch.device("cpu")
# (Optionnel) S'assurer que PyTorch n'essaie pas d'utiliser d'éventuelles optimisations GPU/Flash sur CPU
torch.backends.cuda.matmul.allow_tf32 = False  # désactive le TF32 (précaution pour GPU, sans effet sur CPU)
# Pas besoin de désactiver explicitement scaled_dot_product_attention sur CPU, 
# PyTorch utilisera automatiquement la version standard si GPU non utilisé.

# Charger le tokenizer BERT correspondant aux modèles entraînés
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Définition du modèle Clickbait (texte + âge + genre) exactement comme lors de l'entraînement
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders: int):
        super().__init__()
        # Charger le modèle Bert en mémoire CPU, en float32, sans device_map ni lazy loading
        self.bert = BertModel.from_pretrained(
            "bert-base-multilingual-cased",
            torch_dtype=torch.float32   # forcer les poids en float32
            # Pas de device_map (None par défaut) et low_cpu_mem_usage=False par défaut pour chargement complet
        )
        # Couches supplémentaires pour l'âge et le genre
        self.age_fc     = nn.Linear(1, 16)
        self.gender_emb = nn.Embedding(n_genders, 8)
        # Couche de sortie (tête) prenant en compte la concaténation [BERT, age, gender]
        hidden_size = self.bert.config.hidden_size  # dimension de sortie de BERT (poule-output)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 16 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        # Tout le calcul est fait sur CPU, il faut donc que tous les tenseurs soient sur CPU
        # (Ce sera le cas si on n'utilise que CPU depuis le début)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Récupérer le vecteur [CLS] (pooler_output) du modèle BERT
        cls_output = bert_output.pooler_output  # shape: (batch_size, hidden_size)
        # Passer l'âge dans la couche linéaire + ReLU
        age_feat = F.relu(self.age_fc(age.unsqueeze(1)))     # shape: (batch_size, 16)
        # Passer le genre dans l'embedding
        gender_feat = self.gender_emb(gender)                # shape: (batch_size, 8)
        # Concaténer les caractéristiques texte (BERT), âge et genre
        combined = torch.cat([cls_output, age_feat, gender_feat], dim=1)  # shape: (batch_size, hidden_size+24)
        # Calculer le logit final (valeur réelle) de sortie
        logit = self.head(combined).squeeze(-1)  # shape: (batch_size,) après squeeze
        return logit

# Définition du modèle CTR (texte seul) comme lors de l'entraînement
class CTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Charger Bert en float32 sur CPU
        self.bert = BertModel.from_pretrained(
            "bert-base-multilingual-cased",
            torch_dtype=torch.float32
        )
        hidden_size = self.bert.config.hidden_size
        # Tête de régression du CTR
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output  = bert_output.pooler_output          # shape: (batch_size, hidden_size)
        prediction  = self.head(cls_output).squeeze(-1)  # shape: (batch_size,)
        return prediction

# Initialisation des instances de modèle 
# (On charge d'abord les poids entraînés avant d'instancier pour connaître n_genders)
# Chemins ou téléchargements des poids des modèles entraînés (.pt)
cb_weights_path = "best_cb_model.pt"   # Assurez-vous que ce fichier existe dans le répertoire courant
ctr_weights_path = "best_ctr_model.pt"

# Si les modèles sont hébergés sur Hugging Face Hub (privés), utiliser HfApi ou hf_hub_download avec un token:
# from huggingface_hub import hf_hub_download
# cb_weights_path = hf_hub_download(repo_id="alexandre-cameron-borges/clickbait-model", filename="best_cb_model.pt")
# ctr_weights_path = hf_hub_download(repo_id="alexandre-cameron-borges/ctr-model", filename="best_ctr_model.pt")

# Charger les dictionnaires d'état (poids) sur CPU
cb_state_dict = torch.load(cb_weights_path, map_location="cpu")
ctr_state_dict = torch.load(ctr_weights_path, map_location="cpu")

# Déterminer le nombre de genres utilisé pendant l'entraînement à partir des poids de l'embedding
n_genders = cb_state_dict["gender_emb.weight"].size(0)  # taille 0 du poids de l'embedding = nombre de catégories de genre

# Créer les modèles avec les architectures appropriées
model_cb = ClickbaitModel(n_genders=n_genders)
model_ctr = CTRModel()

# Charger les poids entraînés dans les modèles
model_cb.load_state_dict(cb_state_dict)
model_ctr.load_state_dict(ctr_state_dict)

# Mettre les modèles en mode évaluation et s'assurer qu'ils sont sur CPU
model_cb.eval()
model_ctr.eval()
model_cb.to(device)
model_ctr.to(device)

# Fonctions de prédiction

def predict_cb(text: str, age: float = None, gender: int = None) -> float:
    """
    Prédit la probabilité qu'un titre soit du clickbait (retourne une valeur entre 0 et 1).
    - text : le titre à évaluer.
    - age : l'âge associé (doit être **normalisé** comme dans l'entraînement, sinon None pour utiliser la valeur médiane par défaut).
    - gender : l'identifiant de genre (0/1 pour M/F, 2 pour inconnu) ou None pour inconnu par défaut.
    """
    # Préparer les tenseurs d'entrée pour BERT
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Valeurs par défaut pour age et gender si non fournis
    if age is None:
        # Si aucun âge fourni, on utilise l'âge médian normalisé (qui vaut 0.0 après normalisation par construction)
        age_tensor = torch.tensor([0.0], dtype=torch.float32)
    else:
        # Si un âge brut est fourni, il faut le normaliser comme pendant l'entraînement :
        #   age_norm = (age_brut - median_age) / (max_age - median_age).
        # **Important**: Assurez-vous d'utiliser la même median_age et max_age que celles du jeu de données d'entraînement.
        median_age = 0.0  # remplacer par la valeur numérique utilisée pendant l'entraînement
        max_age = 1.0     # remplacer par la valeur numérique utilisée pendant l'entraînement
        age_norm = (age - median_age) / (max_age - median_age) if max_age != median_age else 0.0
        age_tensor = torch.tensor([age_norm], dtype=torch.float32)
    if gender is None:
        # Utiliser la catégorie 'inconnu' par défaut si aucun genre fourni
        gender_id = torch.tensor([2], dtype=torch.long)
    else:
        gender_id = torch.tensor([gender], dtype=torch.long)

    # Inférence (pas de gradient)
    with torch.no_grad():
        logit = model_cb(input_ids, attention_mask, age_tensor, gender_id)
        # Appliquer une sigmoïde pour obtenir une probabilité entre 0 et 1
        prob = torch.sigmoid(logit).item()
    return prob

def predict_ctr(text: str) -> float:
    """
    Prédit le CTR (taux de clics) estimé pour un titre donné.  
    Retourne une valeur entre 0.0 et 1.0 correspondant à la fraction de clics attendue.
    """
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Inférence sans gradient
    with torch.no_grad():
        pred = model_ctr(input_ids, attention_mask)
        ctr_value = pred.item()
    # S'assurer que la sortie est bornée entre 0 et 1 (par précaution, car le modèle n'a pas de contrainte explicite)
    if ctr_value < 0.0:
        ctr_value = 0.0
    if ctr_value > 1.0:
        ctr_value = 1.0
    return ctr_value

# Les fonctions predict_cb et predict_ctr sont maintenant prêtes à être utilisées.
# Exemple d'utilisation sur 10 titres (appel consécutif sans rechargement du modèle) :
# for title in titles_list:
#     print(predict_cb(title), predict_ctr(title))
