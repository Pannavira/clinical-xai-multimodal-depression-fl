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

# 1. Konfigurasi Parameter dan Inisialisasi Model
MODEL_NAME = "mental/mental-bert-base-uncased"
MAX_LENGTH = 512  # Diubah dari 256 ke 512 untuk menangkap transkrip E-DAIC yang lebih panjang secara optimal
PADDING = "max_length"
TRUNCATION = True

print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Pindahkan ke GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
model.eval()

# 2. Setup Direktori Output
output_base_dir = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "text")
output_dir = os.path.join(output_base_dir, "text_embeddings")
os.makedirs(output_dir, exist_ok=True)

# 3. Pemuatan Dataset (E-DAIC / Preprocessed CSV / Simulasi Fallback)
E_DAIC_DIR = os.path.join(WORKSPACE_ROOT, "data", "E-DAIC")
PREPROCESSED_CSV = os.path.join(output_base_dir, "text_embeddings_all.csv")

data_input = None

# Opsi A: Coba muat dari hasil prapemrosesan teragregasi
if os.path.exists(PREPROCESSED_CSV):
    print(f"[INFO] Memuat data dari hasil prapemrosesan sebelumnya: '{PREPROCESSED_CSV}'")
    df_pre = pd.read_csv(PREPROCESSED_CSV)
    # Gunakan kolom 'clean_text' jika tersedia, jika tidak coba 'text'
    text_col = "clean_text" if "clean_text" in df_pre.columns else ("text" if "text" in df_pre.columns else "")
    if text_col:
        data_input = pd.DataFrame({
            'participant_id': df_pre['participant_id'],
            'text': df_pre[text_col].fillna("")
        })
        print(f"[INFO] Berhasil memuat {len(data_input)} baris data partisipan.")
    else:
        print("[WARNING] Kolom teks tidak ditemukan di text_embeddings_all.csv.")

# Opsi B: Coba memindai direktori E-DAIC asli jika Opsi A gagal
if data_input is None and os.path.exists(E_DAIC_DIR):
    print(f"[INFO] Memindai direktori E-DAIC asli di '{E_DAIC_DIR}'...")
    participant_folders = []
    for name in os.listdir(E_DAIC_DIR):
        dir_path = os.path.join(E_DAIC_DIR, name)
        if os.path.isdir(dir_path) and name.endswith("_P"):
            participant_folders.append((name, dir_path))
            
    if participant_folders:
        # Urutkan secara numerik
        participant_folders.sort(key=lambda x: int("".join(filter(str.isdigit, x[0]))) if any(c.isdigit() for c in x[0]) else 0)
        print(f"[INFO] Ditemukan {len(participant_folders)} folder partisipan E-DAIC.")
        
        records = []
        for name, dir_path in participant_folders:
            p_digits = "".join(filter(str.isdigit, name))
            participant_id = f"P{p_digits}"
            transcript_file = f"{p_digits}_Transcript.csv"
            transcript_path = os.path.join(dir_path, transcript_file)
            
            clean_txt = ""
            if os.path.exists(transcript_path):
                try:
                    raw_df = pd.read_csv(transcript_path)
                    if 'Text' in raw_df.columns:
                        combined_text = " ".join(raw_df['Text'].astype(str).tolist())
                        clean_txt = " ".join(combined_text.lower().strip().split())
                except Exception as e:
                    print(f"[WARNING] Gagal membaca transcript {participant_id}: {e}")
            
            records.append({
                'participant_id': participant_id,
                'text': clean_txt
            })
        data_input = pd.DataFrame(records)
        print(f"[INFO] Berhasil memproses {len(data_input)} partisipan dari E-DAIC.")

# Opsi C: Simulasi fallback jika data asli tidak dapat diakses sama sekali
if data_input is None:
    print("[WARNING] Dataset asli tidak ditemukan. Menggunakan data simulasi fallback...")
    data_input = pd.DataFrame({
        'participant_id': ['P001', 'P002', 'P003'],
        'text': [
            "I have been feeling really down lately and cannot sleep well.",
            "Today is fine, just doing my usual routine at the university.",
            ""  # Simulasi data kosong/gagal
        ]
    })

index_records = []
embeddings_list = []
embedding_dim = model.config.hidden_size

print("Memulai ekstraksi embedding...")
for idx, row in tqdm(data_input.iterrows(), total=len(data_input)):
    pid = row['participant_id']
    text = row['text']
    
    # Validasi input kosong (Koreksi bug: len(text).strip() -> len(text.strip()))
    if not isinstance(text, str) or len(text.strip()) == 0:
        index_records.append({
            "participant_id": pid,
            "embedding_file": "None",
            "embedding_dim": 0,
            "model_used": MODEL_NAME,
            "status": "failed_empty_text"
        })
        embeddings_list.append(np.zeros(embedding_dim))
        continue
        
    try:
        # Tokenisasi
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=MAX_LENGTH, 
            padding=PADDING, 
            truncation=TRUNCATION
        )
        
        # Pindahkan input ke device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Ekstraksi embedding tanpa menghitung gradien
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Strategi: Mengambil Mean Pooling dari Last Hidden State
            # (Lebih representatif untuk klasifikasi kalimat daripada sekadar token [CLS])
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            
            # Kalikan dengan mask untuk mengabaikan padding token
            masked_embeddings = token_embeddings * attention_mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            
            mean_pooled = sum_embeddings / sum_mask
            embedding_np = mean_pooled.cpu().numpy().flatten()
            
        # Simpan ke format .npy per partisipan
        file_path = os.path.join(output_dir, f"{pid}.npy")
        np.save(file_path, embedding_np)
        
        # Catat sukses ke indeks dan simpan array embedding
        rel_embedding_file = f"text_embeddings/{pid}.npy"
        index_records.append({
            "participant_id": pid,
            "embedding_file": rel_embedding_file,
            "embedding_dim": embedding_np.shape[0],
            "model_used": MODEL_NAME,
            "status": "success"
        })
        embeddings_list.append(embedding_np)
        
    except Exception as e:
        index_records.append({
            "participant_id": pid,
            "embedding_file": "None",
            "embedding_dim": 0,
            "model_used": MODEL_NAME,
            "status": f"failed_{str(e)}"
        })
        embeddings_list.append(np.zeros(embedding_dim))

# 4. Export berkas indeks pelacak dan stacked numpy array
df_index = pd.DataFrame(index_records)
index_csv_path = os.path.join(output_base_dir, "text_embedding_index.csv")
df_index.to_csv(index_csv_path, index=False)

# Simpan juga versi Excel (.xlsx) untuk kemudahan inspeksi manual
index_excel_path = os.path.join(output_base_dir, "text_embedding_index.xlsx")
df_index.to_excel(index_excel_path, index=False)

# Buat berkas npy gabungan (stacked array)
stacked_embeddings = np.vstack(embeddings_list)
stacked_npy_path = os.path.join(output_base_dir, "text_embeddings.npy")
np.save(stacked_npy_path, stacked_embeddings)

print(f"\n[SUCCESS] Berkas indeks berhasil disimpan di: '{index_csv_path}' dan '{index_excel_path}'")
print(f"[SUCCESS] Berkas stacked embeddings berhasil disimpan di: '{stacked_npy_path}' dengan shape {stacked_embeddings.shape}")

# Log Ringkasan Dokumentasi
print("\n=== RINGKASAN EKSTRAKSI EMBEDDING ===")
print(f"Nama Model          : {MODEL_NAME}")
print(f"Dimensi Embedding   : {embedding_dim}")
print(f"Parameter Tokenizer : max_length={MAX_LENGTH}, padding={PADDING}, truncation={TRUNCATION}")
print(f"Total Sukses        : {len(df_index[df_index['status'] == 'success'])}")
print(f"Total Gagal         : {len(df_index[df_index['status'] != 'success'])}")
if len(df_index[df_index['status'] == 'success']) > 0:
    # Cari indeks pertama yang sukses untuk ditampilkan sebagai contoh
    first_success_idx = df_index[df_index['status'] == 'success'].index[0]
    sample_pid = df_index.loc[first_success_idx, "participant_id"]
    sample_vector = embeddings_list[first_success_idx]
    print(f"Contoh hasil ({sample_pid}) (10 dimensi pertama): {sample_vector[:10].tolist()}")
print("=====================================")