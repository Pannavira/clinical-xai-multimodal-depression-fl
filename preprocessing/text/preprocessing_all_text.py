import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Load environment variables dynamically from project root .env file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
env_path = os.path.join(WORKSPACE_ROOT, ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from transformers import AutoTokenizer, AutoModel

# Configuration
E_DAIC_DIR = os.path.join(WORKSPACE_ROOT, "data", "E-DAIC")
LABELS_PATH = os.path.join(WORKSPACE_ROOT, "data", "detailed_lables.csv")
OUTPUT_PATH = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "text", "text_embeddings_all.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
MODEL_NAME = "mental/mental-bert-base-uncased"
BATCH_SAVE_INTERVAL = 10  # Save progress incrementally every 10 subjects

# 1. Load Hugging Face model & tokenizer
print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)

# Set up device acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
encoder_model = encoder_model.to(device)
encoder_model.eval()

# Load ground truth labels
if os.path.exists(LABELS_PATH):
    print(f"Loading labels from: {LABELS_PATH}")
    df_labels = pd.read_csv(LABELS_PATH)
else:
    print(f"Warning: Labels file not found at {LABELS_PATH}. Labels will be default to 0.")
    df_labels = None

def get_depression_label(participant_id):
    if df_labels is None:
        return 0
    try:
        # Extract digits from participant_id (e.g. "P300" -> 300)
        p_num = int("".join(filter(str.isdigit, participant_id)))
        row = df_labels[df_labels['Participant'] == p_num]
        if not row.empty:
            return int(row.iloc[0]['Depression_label'])
    except Exception as e:
        print(f"Error fetching label for {participant_id}: {e}")
    return 0

def load_and_aggregate_raw_transcript(file_path):
    raw_df = pd.read_csv(file_path)
    combined_text = " ".join(raw_df['Text'].astype(str).tolist())
    
    # Advanced normalization (lowercasing, single spacing)
    clean_text = combined_text.lower().strip()
    clean_text = " ".join(clean_text.split())
    
    return combined_text, clean_text

def process_tokenization_and_embedding(text, max_seq_len=512):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = encoder_model(**inputs)
        
    # CLS embedding at index 0
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return cls_embedding.tolist()

# Find all participant folders (e.g. 300_P)
print(f"Scanning directories in {E_DAIC_DIR}...")
participant_folders = []
if os.path.exists(E_DAIC_DIR):
    for name in os.listdir(E_DAIC_DIR):
        dir_path = os.path.join(E_DAIC_DIR, name)
        if os.path.isdir(dir_path) and name.endswith("_P"):
            participant_folders.append((name, dir_path))
            
# Sort by participant number numerically
participant_folders.sort(key=lambda x: int("".join(filter(str.isdigit, x[0]))))
print(f"Found {len(participant_folders)} participant folders.")

# Apply test limit if environment variable is set
test_limit = os.environ.get("TEST_LIMIT")
if test_limit:
    try:
        limit = int(test_limit)
        participant_folders = participant_folders[:limit]
        print(f"Testing mode enabled: limiting processing to the first {limit} participants.")
    except ValueError:
        pass

# Process subjects
processed_data = []

# Loop through all participants
for idx, (folder_name, folder_path) in enumerate(tqdm(participant_folders, desc="Processing transcripts")):
    # E.g. "300_P" -> "P300"
    p_digits = "".join(filter(str.isdigit, folder_name))
    participant_id = f"P{p_digits}"
    
    transcript_file = f"{p_digits}_Transcript.csv"
    transcript_path = os.path.join(folder_path, transcript_file)
    
    record = {
        "participant_id": participant_id,
        "raw_text": "",
        "clean_text": "",
        "depression_label": get_depression_label(participant_id),
        "status": "failed",
        "text_embedding": None
    }
    
    if not os.path.exists(transcript_path):
        print(f"\nWarning: Transcript not found at {transcript_path}")
        processed_data.append(record)
        continue
        
    try:
        raw_txt, clean_txt = load_and_aggregate_raw_transcript(transcript_path)
        record["raw_text"] = raw_txt
        record["clean_text"] = clean_txt
        
        # Check if clean text is empty
        if not clean_txt:
            record["status"] = "empty"
            processed_data.append(record)
            continue
            
        # Extract embeddings
        embedding = process_tokenization_and_embedding(clean_txt)
        record["text_embedding"] = embedding
        record["status"] = "success"
        
    except Exception as e:
        print(f"\nError processing {participant_id}: {e}")
        record["status"] = f"error: {str(e)}"
        
    processed_data.append(record)
    
    # Periodically save intermediate results
    if (idx + 1) % BATCH_SAVE_INTERVAL == 0 or (idx + 1) == len(participant_folders):
        df_temp = pd.DataFrame(processed_data)
        df_temp.to_csv(OUTPUT_PATH, index=False)
        
        # Save Excel version at the very end
        if (idx + 1) == len(participant_folders):
            excel_path = OUTPUT_PATH.replace(".csv", ".xlsx")
            df_temp.to_excel(excel_path, index=False)
            print(f"Excel output saved to {excel_path}")

print(f"\nBatch processing complete! Output saved to {OUTPUT_PATH}")