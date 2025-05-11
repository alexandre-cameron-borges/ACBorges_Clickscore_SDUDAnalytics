import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# --- 1. Device Configuration ---
# This will automatically use CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Model Definition (ensure this matches your trained model's architecture) ---
class ClickbaitModel(nn.Module):
    def __init__(self, n_genders, bert_model_name="bert-base-multilingual-cased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Ensure the embedding layer matches the number of unique gender IDs used during training
        self.gender_emb = nn.Embedding(n_genders, 8)
        self.age_fc = nn.Linear(1, 16) # For normalized age
        
        # Calculate hidden size for the head
        # BERT pooler output + age features + gender features
        hid = self.bert.config.hidden_size + 16 + 8 
        self.head = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Adjust dropout if different in your trained model
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, age, gender):
        # The input tensors (input_ids, attention_mask, age, gender) 
        # should be moved to the model's device *before* calling this forward method.
        # This is handled in the predict_cb function.

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = bert_output.pooler_output
        
        age_features = F.relu(self.age_fc(age.unsqueeze(1))) # age should be [batch_size, 1]
        gender_features = self.gender_emb(gender) # gender should be [batch_size]
        
        combined_features = torch.cat([pooler_output, age_features, gender_features], dim=1)
        
        logits = self.head(combined_features)
        return logits.squeeze(-1) # Remove last dimension if it's 1

# --- 3. Tokenizer (Initialize once) ---
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# --- 4. Load Trained Clickbait Model ---
# IMPORTANT: Replace with the actual path to your saved clickbait model file (.pth).
# This path must be accessible from your Streamlit application's environment.
# Example: "models/clickbait_model_final.pth" or an absolute path.
MODEL_PATH_CB = "./clickbait_model.pth"  # <<< ### YOU MUST CHANGE THIS ###

# IMPORTANT: Replace with the number of unique gender IDs your model was trained with.
# This is for the nn.Embedding layer in ClickbaitModel.
# If your gender_map in app2.py is {"Male":0, "Female":1, "Unknown":2}, then n_genders = 3.
N_GENDERS_USED_IN_TRAINING_CB = 3  # <<< ### VERIFY AND CHANGE THIS IF NEEDED ###

_cb_model = None
try:
    print(f"Attempting to load ClickbaitModel with n_genders={N_GENDERS_USED_IN_TRAINING_CB}")
    _cb_model = ClickbaitModel(n_genders=N_GENDERS_USED_IN_TRAINING_CB)
    print(f"Loading state_dict from: {MODEL_PATH_CB} to device: {device}")
    # Load state dict, mapping to the configured device.
    _cb_model.load_state_dict(torch.load(MODEL_PATH_CB, map_location=device))
    _cb_model.to(device)  # Explicitly move model to the configured device
    _cb_model.eval()      # Set the model to evaluation mode
    print(f"Clickbait model loaded successfully and moved to {device}.")
except FileNotFoundError:
    print(f"ERROR: Clickbait model file not found at {MODEL_PATH_CB}. Please check the path.")
    # In a Streamlit app, you might use st.error() here.
    _cb_model = None # Ensure model is None if loading fails
except Exception as e:
    print(f"ERROR: Could not load the Clickbait model: {e}")
    _cb_model = None # Ensure model is None if loading fails

# --- 5. Prediction Functions ---
def predict_cb(text: str, age_norm: float, gender_id: int):
    if _cb_model is None:
        # This error will be raised if the model didn't load correctly.
        raise RuntimeError(
            "Clickbait model (_cb_model) is not loaded. "
            "Please check the MODEL_PATH_CB and N_GENDERS_USED_IN_TRAINING_CB in models/predict.py, "
            "and ensure the model file is accessible. Check console for loading errors."
        )

    _cb_model.eval()  # Ensure model is in evaluation mode for inference

    # 1. Tokenize input text
    # The tokenizer should handle padding and truncation as per your training.
    enc = tokenizer(
        text,
        padding="max_length",        # Or False, if you handle batching differently
        truncation=True,
        max_length=128,              # Ensure this matches training
        return_tensors="pt"          # Return PyTorch tensors
    )

    # 2. Prepare inputs and move all to the correct device
    # .unsqueeze(0) adds a batch dimension (batch_size=1)
    input_ids = enc.input_ids.to(device) # Already [1, seq_len] from tokenizer if text is single string
    attention_mask = enc.attention_mask.to(device) # Already [1, seq_len]
    
    # Ensure age_tensor and gender_tensor are correctly shaped (batch_size, ...)
    age_tensor = torch.tensor([age_norm], dtype=torch.float).to(device) # Shape: [1]
    gender_tensor = torch.tensor([gender_id], dtype=torch.long).to(device) # Shape: [1]

    # 3. Make prediction
    with torch.no_grad():  # Disable gradient calculations during inference
        logits = _cb_model(input_ids, attention_mask, age_tensor, gender_tensor)
        probability = torch.sigmoid(logits).item()  # Apply sigmoid and get scalar probability

    return probability

# --- Placeholder for predict_ctr --- 
# If your predict_ctr function also uses a PyTorch model, it will need similar 
# model loading, device handling, and input tensor preparation as predict_cb.

# Example structure for predict_ctr if it uses a model:
# MODEL_PATH_CTR = "./ctr_model.pth"  # <<< ### YOU MUST CHANGE THIS ###
# N_FEATURES_CTR = ... # Or other params for CTRModel
# _ctr_model = None
# try:
#     # _ctr_model = YourCTRModel(...) # Define or import YourCTRModel
#     # _ctr_model.load_state_dict(torch.load(MODEL_PATH_CTR, map_location=device))
#     # _ctr_model.to(device)
#     # _ctr_model.eval()
#     # print(f"CTR model loaded successfully on {device}.")
#     pass # Replace with actual CTR model loading
# except Exception as e:
#     print(f"Error loading the CTR model: {e}")
#     _ctr_model = None

def predict_ctr(text: str):
    print("predict_ctr function is currently a placeholder.")
    # if _ctr_model is None:
    #     raise RuntimeError("CTR model (_ctr_model) is not loaded.")
    #
    # # Process text, create tensors, move to device
    # with torch.no_grad():
    #     # ctr_prediction = _ctr_model(...)
    #     ctr_prediction = 0.05 # Placeholder
    # return ctr_prediction
    return 0.05  # Return a dummy value for now

if __name__ == '__main__':
    # This block is for testing the script directly.
    # You'll need to have a model file at the path specified in MODEL_PATH_CB.
    print("Testing predict.py script...")
    if _cb_model is None:
        print("Clickbait model not loaded. Cannot run test.")
    else:
        print(f"Clickbait model loaded on {device}. Running a test prediction.")
        sample_text = "This is an amazing title you won't believe!"
        sample_age_norm = 0.0 # (35 - 35) / (80 - 35)
        sample_gender_id = 0 # Male
        try:
            cb_prob = predict_cb(sample_text, sample_age_norm, sample_gender_id)
            print(f"Test Clickbait Prediction for '{sample_text}': {cb_prob:.4f}")
        except Exception as e:
            print(f"Error during test prediction: {e}")

    # Test predict_ctr (currently a placeholder)
    ctr_val = predict_ctr("some text for ctr")
    print(f"Test CTR Prediction: {ctr_val}")
