import os
import time
import numpy as np
import pandas as pd
import soundfile as sf

# ==========================================
# CONFIGURATION & DIRECTORY SETUP
# ==========================================
SEGMENT_LENGTH_SEC = 10    # Durasi target tiap segmen (detik)
RMS_THRESHOLD = 0.001      # Ambang batas amplitudo RMS untuk mendeteksi suara aktif
MIN_DURATION_SEC = 2.0     # Durasi minimal segmen agar dianggap valid untuk dianalisis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "audio")
DENOISED_DIR = os.path.join(OUTPUT_DIR, "denoised_audio")
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented_audio")
OUTPUT_DOC_DIR = os.path.join(OUTPUT_DIR, "documentation")

# Buat direktori output jika belum ada
for folder in [SEGMENTED_DIR, OUTPUT_DOC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Scan file audio hasil denoising
audio_files = []
if os.path.exists(DENOISED_DIR):
    for f in os.listdir(DENOISED_DIR):
        if f.endswith(".wav"):
            p_id = os.path.splitext(f)[0]  # Mengambil ID partisipan (misal P300)
            file_path = os.path.join(DENOISED_DIR, f)
            audio_files.append((p_id, file_path))

# Urutkan file berdasarkan nomor ID partisipan secara numerik
def extract_digits(p_id):
    digits = "".join(filter(str.isdigit, p_id))
    return int(digits) if digits.isdigit() else 0

audio_files.sort(key=lambda x: extract_digits(x[0]))

print(f"Ditemukan {len(audio_files)} file audio denoised untuk proses segmentasi.\n")

segmentation_records = []

# ==========================================
# PIPELINE PROSES SEGMENTASI
# ==========================================
for idx, (participant_id, file_path) in enumerate(audio_files):
    print(f"[{idx+1}/{len(audio_files)}] Memproses: {participant_id} (Path: {file_path})")
    start_time = time.time()
    
    try:
        # Load audio dengan soundfile (sangat cepat untuk format wav yang sudah terstandar)
        y, sr = sf.read(file_path)
        
        # Pastikan data berupa 1D (Mono)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        duration_sec = len(y) / sr
        seg_length_samples = int(SEGMENT_LENGTH_SEC * sr)
        
        # Hitung total segmen secara teoretis
        total_segments = int(np.ceil(len(y) / seg_length_samples))
        valid_segments_count = 0
        
        # Buat subfolder khusus per partisipan agar terorganisir dengan rapi
        participant_seg_dir = os.path.join(SEGMENTED_DIR, participant_id)
        os.makedirs(participant_seg_dir, exist_ok=True)
        
        for seg_idx in range(total_segments):
            start_sample = seg_idx * seg_length_samples
            end_sample = min(len(y), (seg_idx + 1) * seg_length_samples)
            
            y_seg = y[start_sample:end_sample]
            seg_duration = len(y_seg) / sr
            
            # 1. Validasi Durasi Minimal (untuk segmen sisa di akhir rekaman)
            if seg_duration < MIN_DURATION_SEC:
                continue
                
            # 2. Validasi Keheningan dengan RMS Energy
            rms_energy = np.sqrt(np.mean(y_seg ** 2))
            
            if rms_energy >= RMS_THRESHOLD:
                valid_segments_count += 1
                seg_filename = f"{participant_id}_seg{valid_segments_count:03d}.wav"
                seg_filepath = os.path.join(participant_seg_dir, seg_filename)
                
                # Simpan segmen valid sebagai 16-bit PCM WAV
                sf.write(seg_filepath, y_seg, sr, subtype='PCM_16')
                
        # Hapus subfolder jika tidak ada satu pun segmen yang valid (untuk kebersihan folder)
        if valid_segments_count == 0:
            try:
                os.rmdir(participant_seg_dir)
            except Exception:
                pass
                
        # Catat ke log records
        segmentation_records.append({
            "participant_id": participant_id,
            "duration_sec": int(round(duration_sec)),
            "segment_length": f"{SEGMENT_LENGTH_SEC} sec",
            "total_segments": total_segments,
            "valid_segments": valid_segments_count,
            "status": "success"
        })
        
        print(f"    Selesai: {total_segments} total segmen, {valid_segments_count} segmen valid disimpan. ({time.time() - start_time:.2f} detik)")
        
    except Exception as e:
        print(f"    Gagal memproses {participant_id}: {str(e)}")
        segmentation_records.append({
            "participant_id": participant_id,
            "duration_sec": 0,
            "segment_length": f"{SEGMENT_LENGTH_SEC} sec",
            "total_segments": 0,
            "valid_segments": 0,
            "status": f"failed: {str(e)}"
        })

# ==========================================
# EXPORT LOGS KE EXCEL
# ==========================================
if segmentation_records:
    df_seg = pd.DataFrame(segmentation_records)
    log_file_path = os.path.join(OUTPUT_DOC_DIR, "audio_segmentation_log.xlsx")
    df_seg.to_excel(log_file_path, index=False)
    print(f"\n=== PROSES SEGMENTASI AUDIO SELESAI ===")
    print(f"Log segmentasi telah berhasil disimpan di: {log_file_path}")
else:
    print("\nTidak ada berkas yang berhasil diproses.")
