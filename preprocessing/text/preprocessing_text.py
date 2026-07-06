import os

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

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
# =====================================================================
# LANGKAH 1: Simulasi Agregasi & Pembersihan (Dari Berkas Mentah Anda)
# =====================================================================
def load_and_aggregate_raw_transcript(file_path):
    raw_df = pd.read_csv(file_path)
    
    # CRITIQUE REMEDY: Idealnya lakukan penyaringan pembicara di sini jika memungkinkan.
    # Untuk contoh ini, kita asumsikan penggabungan seluruh teks transkrip.
    combined_text = " ".join(raw_df['Text'].astype(str).tolist())
    
    # Normalisasi tingkat lanjut (lowercasing, pembersihan spasi ganda)
    clean_text = combined_text.lower().strip()
    clean_text = " ".join(clean_text.split())
    
    return combined_text, clean_text

def get_depression_label(participant_id, labels_file_path=None):
    if labels_file_path is None:
        labels_file_path = os.path.join(WORKSPACE_ROOT, "data", "detailed_lables.csv")
    try:
        # Ekstrak angka saja (misal "P300" -> 300)
        p_num = int("".join(filter(str.isdigit, participant_id)))
        df_labels = pd.read_csv(labels_file_path)
        row = df_labels[df_labels['Participant'] == p_num]
        if not row.empty:
            return int(row.iloc[0]['Depression_label'])
        else:
            print(f"Warning: Partisipan {participant_id} tidak ditemukan. Default ke 0.")
            return 0
    except Exception as e:
        print(f"Error membaca label depresi untuk {participant_id}: {e}. Default ke 0.")
        return 0

# Misalkan kita bangun baris untuk clean_text.csv
participant_id = "P300"
raw_txt, clean_txt = load_and_aggregate_raw_transcript(os.path.join(WORKSPACE_ROOT, "data", "E-DAIC", "300_P", "300_Transcript.csv"))
data_koordinasi = {
    "participant_id": [participant_id],
    "raw_text": [raw_txt],
    "clean_text": [clean_txt],
    "depression_label": [get_depression_label(participant_id)],
    "status": ["success"]
}
df_clean = pd.DataFrame(data_koordinasi)

# =====================================================================
# LANGKAH 2: Tokenisasi (Langkah 2.4) & Ekstraksi Embedding (Langkah 2.5)
# =====================================================================
# Menggunakan MentalBERT untuk domain kesehatan mental klinis
MODEL_NAME = "mental/mental-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)

# Atur akselerasi perangkat (GPU jika tersedia)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_model = encoder_model.to(device)
encoder_model.eval()

def process_tokenization_and_embedding(text, max_seq_len=512):
    # 1. Tokenisasi dengan padding dan truncation terstandarisasi
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    
    # Pindahkan tensor input ke device target
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. Ekstraksi Vektor Fitur Konten Teks tanpa kalkulasi gradien
    with torch.no_grad():
        outputs = encoder_model(**inputs)
        
    # Mengambil representasi token [CLS] pada indeks ke-0 sebagai ringkasan semantik kalimat
    # Ukuran output dimensi: [1, 768]
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    return cls_embedding.tolist()

# Eksekusi fungsi pada dataframe koordinasi
extracted_embeddings = []
for idx, row in df_clean.iterrows():
    if row['status'] == 'success':
        vector = process_tokenization_and_embedding(row['clean_text'])
        extracted_embeddings.append(vector)
    else:
        extracted_embeddings.append(None)

# Tambahkan matriks representasi ke dalam DataFrame baru untuk Kesiapan Late Fusion
df_clean['text_embedding'] = extracted_embeddings

# Ekspor hasil ke bentuk akhir pembentukan representasi teks
output_path = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "text", "text_embeddings.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_clean.to_csv(output_path, index=False)
print("Proses Tokenisasi dan Ekstraksi Vektor Embedding Selesai Beroperasi.")