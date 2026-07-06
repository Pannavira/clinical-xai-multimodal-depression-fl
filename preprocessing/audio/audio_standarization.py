import os
import glob
import time
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import noisereduce as nr
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET_SR = 16000  # 16 kHz standar untuk ekstraksi fitur klinis
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
E_DAIC_DIR = os.path.join(WORKSPACE_ROOT, "data", "E-DAIC")

# Output directories inside preprocessing_output/audio
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "audio")
STANDARDIZED_DIR = os.path.join(OUTPUT_DIR, "standardized_audio")
DENOISED_DIR = os.path.join(OUTPUT_DIR, "denoised_audio")
OUTPUT_DOC_DIR = os.path.join(OUTPUT_DIR, "documentation")

for folder in [STANDARDIZED_DIR, DENOISED_DIR, OUTPUT_DOC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Inisialisasi list untuk logging
standardization_records = []
denoising_records = []

# Scan all participant directories (e.g. 300_P) in E-DAIC
audio_files = []
if os.path.exists(E_DAIC_DIR):
    for name in os.listdir(E_DAIC_DIR):
        dir_path = os.path.join(E_DAIC_DIR, name)
        if os.path.isdir(dir_path) and name.endswith("_P"):
            p_digits = "".join(filter(str.isdigit, name))
            wav_file = f"{p_digits}_AUDIO.wav"
            wav_path = os.path.join(dir_path, wav_file)
            if os.path.exists(wav_path):
                audio_files.append((p_digits, wav_path))
            else:
                # Fallback: scan any wav files in directory
                fallback_wavs = glob.glob(os.path.join(dir_path, "*.wav"))
                if fallback_wavs:
                    audio_files.append((p_digits, fallback_wavs[0]))

# Sort participant folders numerically
audio_files.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)

print(f"Ditemukan {len(audio_files)} file audio untuk diproses.\n")

# ==========================================
# PIPELINE PROCESSING
# ==========================================
for idx, (p_digits, file_path) in enumerate(audio_files):
    participant_id = f"P{p_digits}"  # e.g., P300, aligning with text embeddings standard
    
    print(f"[{idx+1}/{len(audio_files)}] Memproses: {participant_id} (Path: {file_path})")
    start_time = time.time()
    
    try:
        # ------------------------------------------
        # LANGKAH 2.6 — STANDARISASI FILE AUDIO
        # ------------------------------------------
        # Ambil metadata original tanpa me-resample penuh terlebih dahulu
        orig_info = sf.info(file_path)
        orig_sr = orig_info.samplerate
        orig_channels = orig_info.channels
        
        # Load audio dan paksa menjadi Mono (mono=True) serta Resample ke 16kHz
        y_raw, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        duration_sec = librosa.get_duration(y=y_raw, sr=sr)
        
        # Simpan hasil standarisasi
        std_file_path = os.path.join(STANDARDIZED_DIR, f"{participant_id}.wav")
        sf.write(std_file_path, y_raw, sr, subtype='PCM_16')
        
        # Catat log standarisasi
        standardization_records.append({
            "participant_id": participant_id,
            "original_filename": os.path.basename(file_path),
            "original_format": "wav",
            "new_format": "wav",
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sr,
            "status": "success"
        })
        
        # ------------------------------------------
        # LANGKAH 2.7 — DENOISING & NORMALISASI
        # ------------------------------------------
        # 1. Noise Reduction ringan (prop_decrease=0.75 agar karakteristik vokal tidak hilang, stationary=True untuk performa cepat)
        y_denoised = nr.reduce_noise(y=y_raw, sr=sr, prop_decrease=0.75, n_fft=1024, stationary=True)
        
        # 2. Amplitudo Normalization (Peak normalization ke -1.0 hingga 1.0 dB)
        y_normalized = librosa.util.normalize(y_denoised)
        
        # 3. Trim Silence (Potong bagian hening di awal/akhir jika melebihi ambang batas)
        y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=30)
        
        # Simpan hasil akhir denoised
        denoised_file_path = os.path.join(DENOISED_DIR, f"{participant_id}.wav")
        sf.write(denoised_file_path, y_trimmed, sr, subtype='PCM_16')
        
        # Catat log denoising
        denoising_records.append({
            "Document": "audio_preprocessing_log.xlsx",
            "Participant_ID": participant_id,
            "Original_Filename": os.path.basename(file_path),
            "Status": "success",
            "Remark": "Denoised, Normalized, and Trimmed"
        })
        
        # 4. Eksport Visualisasi Waveform (Hanya untuk sampel pertama sebagai dokumentasi)
        if idx == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(np.linspace(0, duration_sec, len(y_raw)), y_raw, color='gray')
            plt.title(f"Waveform Sebelum Denoising ({participant_id})")
            plt.ylabel("Amplitudo")
            
            duration_denoised = librosa.get_duration(y=y_trimmed, sr=sr)
            plt.subplot(2, 1, 2)
            plt.plot(np.linspace(0, duration_denoised, len(y_trimmed)), y_trimmed, color='blue')
            plt.title(f"Waveform Sesudah Denoising & Normalisasi ({participant_id})")
            plt.xlabel("Waktu (detik)")
            plt.ylabel("Amplitudo")
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DOC_DIR, "waveform_before_after.png"), dpi=300)
            plt.close()
            
        print(f"    Selesai dalam {time.time() - start_time:.2f} detik")
            
    except Exception as e:
        print(f"Gagal memproses {participant_id}: {str(e)}")
        standardization_records.append({
            "participant_id": participant_id,
            "original_filename": os.path.basename(file_path),
            "original_format": "wav",
            "new_format": "FAILED",
            "duration_sec": 0,
            "sample_rate": TARGET_SR,
            "status": f"failed: {str(e)}"
        })
        denoising_records.append({
            "Document": "audio_preprocessing_log.xlsx",
            "Participant_ID": participant_id,
            "Original_Filename": os.path.basename(file_path),
            "Status": "failed",
            "Remark": str(e)
        })

# ==========================================
# EXPORT LOGS TO EXCEL
# ==========================================
if standardization_records:
    df_std = pd.DataFrame(standardization_records)
    df_std.to_excel(os.path.join(OUTPUT_DOC_DIR, "audio_standardization_log.xlsx"), index=False)

if denoising_records:
    df_denoise = pd.DataFrame(denoising_records)
    df_denoise.to_excel(os.path.join(OUTPUT_DOC_DIR, "audio_preprocessing_log.xlsx"), index=False)

print("\n=== PROSES PRAPEMROSESAN AUDIO SELESAI ===")
print(f"Log dan grafik telah disimpan di folder: {OUTPUT_DOC_DIR}")