import os
import pandas as pd

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

from transformers import AutoTokenizer

def run_text_tokenization(input_csv="text_embeddings_all.csv", output_csv="tokenized_text.csv", max_length=512):
    """
    Menjalankan proses tokenisasi menggunakan tokenizer Hugging Face (BERT)
    dan mengekspor metrik kontrol kualitas ke dalam format CSV target.
    """
    # 1. Validasi keberadaan berkas input
    # Coba cari di folder output jika di root tidak ada
    WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    output_dir = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "text")
    if not os.path.exists(input_csv):
        alt_input = os.path.join(output_dir, input_csv)
        if os.path.exists(alt_input):
            input_csv = alt_input
            print(f"[INFO] Menggunakan berkas input dari folder output: '{input_csv}'")
        else:
            print(f"[ERROR] Berkas '{input_csv}' tidak ditemukan.")
            print("Membuat data simulasi untuk pengetesan...")
            # Data simulasi berdasarkan sampel Transcript 300 yang Anda berikan
            df_input = pd.DataFrame({
                "participant_id": ["P300", "P301"],
                "clean_text": [
                    "so i'm going to interview in spanish okay good atlanta georgia my parents are from here",
                    "i like reading books i enjoy cooking exercise is great"
                ],
                "depression_label": [1, 0]
            })
            df_input.to_csv(input_csv, index=False)

    if os.path.exists(input_csv):
        df_input = pd.read_csv(input_csv)

    # Sesuaikan lokasi output ke folder 'output' jika ada
    if os.path.exists(output_dir) and not os.path.dirname(output_csv):
        output_csv = os.path.join(output_dir, output_csv)

    # 2. Inisialisasi Tokenizer (Menggunakan BERT-base-uncased sebagai baseline)
    # Anda dapat menggantinya ke "mental/mental-bert-base-uncased" jika diperlukan
    model_checkpoint = "mental/mental-bert-base-uncased"
    print(f"[INFO] Mengunduh/memuat tokenizer: {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_rows = []

    print("[INFO] Memulai pemrosesan tokenisasi per baris...")
    for idx, row in df_input.iterrows():
        pid = row["participant_id"]
        text = str(row["clean_text"]) if pd.notna(row["clean_text"]) else ""
        
        try:
            # Hitung jumlah asli token dari teks tanpa batasan panjang (max_length)
            raw_tokens = tokenizer.tokenize(text)
            token_count = len(raw_tokens)
            
            # Evaluasi metrik kontrol kualitas (Truncation & Padding)
            truncated = 1 if token_count > max_length else 0
            padding = 1 if token_count < max_length else 0
            
            # Masukkan metrik ke dalam list sesuai format tabel target
            tokenized_rows.append({
                "participant_id": pid,
                "token_count": token_count,
                "truncated": truncated,
                "padding": padding,
                "status": "success"
            })
            
        except Exception as e:
            # Penanganan jika terjadi error pemrosesan teks
            tokenized_rows.append({
                "participant_id": pid,
                "token_count": 0,
                "truncated": 0,
                "padding": 0,
                "status": f"failed: {str(e)}"
            })

    # 3. Konstruksi DataFrame baru dan Ekspor ke CSV & Excel
    df_output = pd.DataFrame(tokenized_rows)
    df_output.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Berkas luaran berhasil disimpan di: '{output_csv}'")
    
    # Ekspor ke Excel (.xlsx) juga
    excel_output = output_csv.replace(".csv", ".xlsx")
    df_output.to_excel(excel_output, index=False)
    print(f"[SUCCESS] Berkas luaran Excel berhasil disimpan di: '{excel_output}'")

if __name__ == "__main__":
    # Jalankan fungsi utama dengan panjang maksimum sekuens BERT (512 token)
    run_text_tokenization(
        input_csv="text_embeddings_all.csv", 
        output_csv="tokenized_text.csv", 
        max_length=512
    )