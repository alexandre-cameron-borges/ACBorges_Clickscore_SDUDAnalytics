import math
import torch
import torch.nn.functional as F

# --- Patch scaled_dot_product_attention to avoid CPU 'meta' tensor issues ---
if hasattr(F, "scaled_dot_product_attention"):
    def safe_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        """Custom scaled dot-product attention (CPU-safe)."""
        # Calculate raw attention scores (Q*K^T) scaled by sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # For boolean mask: False means mask out that position
                attn_scores.masked_fill_(~attn_mask, float('-inf'))
            else:
                # For additive mask (float): add directly (mask contains -inf or large negative for masked positions)
                attn_scores += attn_mask
        # Apply causal mask if needed (prevent attending to future positions)
        if is_causal:
            L, S = attn_scores.size(-2), attn_scores.size(-1)
            causal_mask = torch.triu(torch.ones((L, S), dtype=torch.bool, device=attn_scores.device), diagonal=1)
            attn_scores.masked_fill_(causal_mask, float('-inf'))
        # Softmax to get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Apply dropout if specified (usually 0.0 during inference)
        if dropout_p > 0.0:
            attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)
        # Compute attention output = probs * V
        return torch.matmul(attn_probs, value)
    F.scaled_dot_product_attention = safe_scaled_dot_product_attention

# --- Force all operations on CPU ---
device = torch.device("cpu")

# Load tokenizer and define model architectures
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

class ClickbaitModel(torch.nn.Module):
    def __init__(self, n_genders: int):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        # Additional feature encoders
        self.age_fc = torch.nn.Linear(1, 16)
        self.gender_emb = torch.nn.Embedding(n_genders, 8)
        hidden_size = self.bert.config.hidden_size  # BERT's hidden layer size
        # Combined input size = BERT output + age_feat (16) + gender_feat (8)
        combined_size = hidden_size + 16 + 8
        self.head = torch.nn.Sequential(
            torch.nn.Linear(combined_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),   # use same dropout as training (disabled during eval)
            torch.nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask, age, gender):
        # BERT forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        # Process additional features
        age_feat = torch.relu(self.age_fc(age.unsqueeze(1)))       # [batch_size, 16]
        gender_feat = self.gender_emb(gender)                      # [batch_size, 8]
        # Concatenate BERT output with additional features
        combined = torch.cat([pooled_output, age_feat, gender_feat], dim=1)
        logits = self.head(combined).squeeze(-1)  # [batch_size] (single output per example)
        return logits  # (raw score for clickbait; apply sigmoid for probability)

class CTRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        hidden_size = self.bert.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),  # same dropout as used in training
            torch.nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        score = self.head(pooled_output).squeeze(-1)  # [batch_size] (predicted CTR value)
        return score

# Initialize models and load pre-trained weights (on CPU)
clickbait_model = ClickbaitModel(n_genders=3).to(device)  # n_genders=3 as in training dataset
clickbait_model.load_state_dict(torch.load("models/best_cb_model.pt", map_location=device))
clickbait_model.eval()  # set to evaluation mode (disables dropout)

ctr_model = CTRModel().to(device)
ctr_model.load_state_dict(torch.load("models/best_ctr_model.pt", map_location=device))
ctr_model.eval()

# --- Prediction functions ---

def predict_cb(text: str, age: float, gender: int) -> str:
    """
    Predicts whether the given text is clickbait or not.
    Returns the label "clickbait" or "no-clickbait".
    """
    # Tokenize the input text for BERT
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    age_tensor = torch.tensor(age, dtype=torch.float32, device=device)
    gender_tensor = torch.tensor(gender, dtype=torch.long, device=device)
    # Run inference
    with torch.no_grad():
        logit = clickbait_model(input_ids, attention_mask, age_tensor, gender_tensor)
        prob = torch.sigmoid(logit)  # probability of being "clickbait"
    # Convert probability to human-readable label
    return "clickbait" if prob.item() >= 0.5 else "no-clickbait"

def predict_ctr(text: str) -> float:
    """
    Predicts the click-through rate (CTR) for the given text.
    Returns the predicted CTR value.
    """
    # Tokenize the input text for BERT
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    # Run inference
    with torch.no_grad():
        pred = ctr_model(input_ids, attention_mask)
    # Return the predicted CTR score (as a float)
    return pred.item()
