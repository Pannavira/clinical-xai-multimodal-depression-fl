import os
import time
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET_SR = 16000  # 16 kHz standar untuk ekstraksi fitur klinis
N_FFT = 1024
HOP_LENGTH = 512
RMS_THRESHOLD = 0.005  # Ambang batas RMS untuk mendeteksi keheningan (pause)
YIN_HOP = 1024         # Hop length lebih besar khusus untuk YIN agar cepat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "audio")
DENOISED_DIR = os.path.join(OUTPUT_DIR, "denoised_audio")
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented_audio")
OUTPUT_DOC_DIR = os.path.join(OUTPUT_DIR, "documentation")

# Subfolder output untuk fitur
MFCC_OUT_DIR = os.path.join(OUTPUT_DIR, "mfcc")
SPECTROGRAM_OUT_DIR = os.path.join(OUTPUT_DIR, "spectrogram")
SPECTROGRAM_FEATURES_DIR = os.path.join(OUTPUT_DIR, "spectrogram_features")

# Buat direktori output jika belum ada
for folder in [OUTPUT_DIR, DENOISED_DIR, SEGMENTED_DIR, OUTPUT_DOC_DIR, MFCC_OUT_DIR, SPECTROGRAM_OUT_DIR, SPECTROGRAM_FEATURES_DIR]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS FOR ACOUSTIC FEATURES
# ==========================================
def estimate_speech_rate(y, sr, voiced_duration):
    """
    Estimasi speech rate (suku kata per detik bicara) menggunakan deteksi peak 
    pada selubung amplitudo (loudness envelope) sinyal.
    """
    try:
        # Window size 100ms untuk menghaluskan amplop
        window_len = int(0.100 * sr)
        envelope = np.convolve(np.abs(y), np.ones(window_len)/window_len, mode='same')
        
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)
            
        # Jeda minimal antar suku kata kira-kira 150ms
        min_dist = int(0.150 * sr)
        peaks, _ = scipy.signal.find_peaks(envelope, height=0.05, distance=min_dist)
        syllable_count = len(peaks)
        
        speech_rate = syllable_count / voiced_duration if voiced_duration > 0 else 0.0
        return speech_rate, syllable_count
    except Exception:
        return 0, 0.0

def extract_features_for_audio(y, sr, participant_id):
    """
    Mengekstrak 128 fitur akustik terintegrasi dari sinyal audio.
    """
    # 1. Hitung frame-level RMS energy dan Pitch/F0
    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
    frame_duration = HOP_LENGTH / sr
    total_frames = len(rms_frames)
    
    # Estimasi Pitch/F0 menggunakan YIN dengan hop length lebih tinggi agar cepat
    try:
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, hop_length=YIN_HOP)
        rms_yin = librosa.feature.rms(y=y, frame_length=2048, hop_length=YIN_HOP)[0]
        min_len = min(len(f0), len(rms_yin))
        f0 = f0[:min_len]
        rms_yin = rms_yin[:min_len]
        
        # Hilangkan F0 pada frame sunyi untuk kalkulasi statistik vokal aktif
        voiced_mask = (rms_yin >= RMS_THRESHOLD) & (f0 > 65) & (f0 < 500)
        voiced_f0 = f0[voiced_mask]
        
        if len(voiced_f0) > 0:
            pitch_mean = np.mean(voiced_f0)
            pitch_std = np.std(voiced_f0)
            pitch_min = np.min(voiced_f0)
            pitch_max = np.max(voiced_f0)
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = 0.0
    except Exception:
        voiced_mask = np.zeros(1, dtype=bool)
        voiced_f0 = np.array([])
        pitch_mean = pitch_std = pitch_min = pitch_max = 0.0

    # 2. Fitur Jeda dan Dinamika Bicara (Speech & Pause Dynamics)
    is_pause = rms_frames < RMS_THRESHOLD
    total_pause_frames = np.sum(is_pause)
    total_pause_duration = total_pause_frames * frame_duration
    voicing_fraction = np.sum(~is_pause) / total_frames if total_frames > 0 else 0.0
    
    # Hitung jumlah segmen jeda (diam)
    pause_segments = 0
    in_pause = False
    for frame in is_pause:
        if frame and not in_pause:
            pause_segments += 1
            in_pause = True
        elif not frame:
            in_pause = False
            
    avg_pause_duration = total_pause_duration / pause_segments if pause_segments > 0 else 0.0
    voiced_duration = np.sum(~is_pause) * frame_duration
    
    # Kecepatan Bicara (Speech rate)
    speech_rate, syllable_count = estimate_speech_rate(y, sr, voiced_duration)
    
    # Jitter & Shimmer (Gangguan kestabilan suara / mikro-fluktuasi)
    try:
        if len(voiced_f0) > 1:
            periods = 1.0 / voiced_f0
            diffs = np.abs(np.diff(periods))
            jitter = np.mean(diffs) / np.mean(periods)
        else:
            jitter = 0.0
    except Exception:
        jitter = 0.0
        
    try:
        if len(voiced_f0) > 1:
            # Recompute voiced mask on standard rms frames (512 hop) for better resolution
            rms_f512 = rms_frames
            # Downsample voiced mask from YIN or align lengths
            # Simple approximation: RMS energy variance of the voiced frames
            voiced_rms = rms_f512[rms_f512 >= RMS_THRESHOLD]
            if len(voiced_rms) > 1:
                diffs = np.abs(np.diff(voiced_rms))
                shimmer = np.mean(diffs) / np.mean(voiced_rms)
            else:
                shimmer = 0.0
        else:
            shimmer = 0.0
    except Exception:
        shimmer = 0.0

    # ----------------------------------------------------
    # BENTUK VEKTOR FITUR (TOTAL 128 DIMENSI)
    # ----------------------------------------------------
    feature_vector = []
    
    # 1. MFCC (16 koefisien * 6 statistik = 96 fitur)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    feature_vector.extend(np.mean(mfcc, axis=1))       # 16
    feature_vector.extend(np.std(mfcc, axis=1))        # 16
    feature_vector.extend(np.mean(delta_mfcc, axis=1))  # 16
    feature_vector.extend(np.std(delta_mfcc, axis=1))   # 16
    feature_vector.extend(np.mean(delta2_mfcc, axis=1)) # 16
    feature_vector.extend(np.std(delta2_mfcc, axis=1))  # 16
    
    # 2. Spectral Features (16 fitur)
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(cent))
    feature_vector.append(np.std(cent))
    
    # Spectral Bandwidth
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(band))
    feature_vector.append(np.std(band))
    
    # Spectral Roll-off
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)[0]
    feature_vector.append(np.mean(roll))
    feature_vector.append(np.std(roll))
    
    # Spectral Contrast (n_bands=4 menghasilkan 5 sub-band)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=4)
    feature_vector.extend(np.mean(contrast, axis=1))   # 5
    feature_vector.extend(np.std(contrast, axis=1))    # 5
    
    # 3. Temporal, Energy & Dynamics (16 fitur)
    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(zcr))
    feature_vector.append(np.std(zcr))
    
    # RMS Energy
    feature_vector.append(np.mean(rms_frames))
    feature_vector.append(np.std(rms_frames))
    feature_vector.append(np.max(rms_frames))
    
    # Pitch/F0
    feature_vector.append(pitch_mean)
    feature_vector.append(pitch_std)
    feature_vector.append(pitch_min)
    feature_vector.append(pitch_max)
    
    # Jeda dan Dinamika (Speech & Pause Dynamics)
    feature_vector.append(voicing_fraction)
    feature_vector.append(total_pause_duration)
    feature_vector.append(float(pause_segments))
    feature_vector.append(avg_pause_duration)
    feature_vector.append(speech_rate)
    feature_vector.append(jitter)
    feature_vector.append(shimmer)
    
    # Pastikan ukuran persis 128
    feature_vector = np.array(feature_vector, dtype=np.float32)
    assert len(feature_vector) == 128, f"Eror: Dimensi fitur {len(feature_vector)} tidak sesuai dengan 128!"
    
    # Kembalikan vektor fitur dan komponen untuk tabular CSV terpisah
    return feature_vector, {
        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_std": np.std(mfcc, axis=1),
        "delta_mean": np.mean(delta_mfcc, axis=1),
        "delta_std": np.std(delta_mfcc, axis=1),
        "delta2_mean": np.mean(delta2_mfcc, axis=1),
        "delta2_std": np.std(delta2_mfcc, axis=1),
        "pitch_energy": {
            "zcr_mean": np.mean(zcr), "zcr_std": np.std(zcr),
            "rms_mean": np.mean(rms_frames), "rms_std": np.std(rms_frames), "rms_max": np.max(rms_frames),
            "pitch_mean": pitch_mean, "pitch_std": pitch_std, "pitch_min": pitch_min, "pitch_max": pitch_max,
            "spectral_centroid_mean": np.mean(cent), "spectral_centroid_std": np.std(cent),
            "spectral_bandwidth_mean": np.mean(band), "spectral_bandwidth_std": np.std(band),
            "spectral_rolloff_mean": np.mean(roll), "spectral_rolloff_std": np.std(roll),
            **{f"spectral_contrast_band_{i+1}_mean": val for i, val in enumerate(np.mean(contrast, axis=1))},
            **{f"spectral_contrast_band_{i+1}_std": val for i, val in enumerate(np.std(contrast, axis=1))},
            "voicing_fraction": voicing_fraction, "total_pause_duration": total_pause_duration,
            "num_pauses": pause_segments, "avg_pause_duration": avg_pause_duration,
            "speech_rate": speech_rate, "jitter": jitter, "shimmer": shimmer
        }
    }

def process_single_participant(participant_id, file_path, is_first):
    """
    Memproses ekstraksi fitur untuk satu partisipan tunggal (dirancang untuk multiprocessing).
    """
    try:
        # Load audio (mono=True)
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        duration_sec = len(y) / sr
        
        # Ekstrak fitur
        feature_vector, raw_components = extract_features_for_audio(y, sr, participant_id)
        
        # Simpan fitur .npy per partisipan
        npy_filename = f"{participant_id}.npy"
        npy_filepath = os.path.join(MFCC_OUT_DIR, npy_filename)
        np.save(npy_filepath, feature_vector)
        
        # Plot Mel-Spectrogram
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {participant_id}')
        plt.tight_layout()
        
        # Simpan ke folder spectrogram dan spectrogram_features
        png_filename = f"{participant_id}.png"
        spec_path1 = os.path.join(SPECTROGRAM_OUT_DIR, png_filename)
        spec_path2 = os.path.join(SPECTROGRAM_FEATURES_DIR, png_filename)
        plt.savefig(spec_path1, dpi=100)
        plt.savefig(spec_path2, dpi=100)
        plt.close()
        
        # Siapkan data baris untuk mfcc_features.csv (96 kolom MFCC)
        mfcc_row = {"participant_id": participant_id}
        for i in range(16):
            mfcc_row[f"mfcc_{i+1}_mean"] = raw_components["mfcc_mean"][i]
            mfcc_row[f"mfcc_{i+1}_std"] = raw_components["mfcc_std"][i]
            mfcc_row[f"mfcc_{i+1}_delta_mean"] = raw_components["delta_mean"][i]
            mfcc_row[f"mfcc_{i+1}_delta_std"] = raw_components["delta_std"][i]
            mfcc_row[f"mfcc_{i+1}_delta2_mean"] = raw_components["delta2_mean"][i]
            mfcc_row[f"mfcc_{i+1}_delta2_std"] = raw_components["delta2_std"][i]
            
        # Siapkan data baris untuk pitch_energy_features.csv (32 kolom)
        pe_row = {"participant_id": participant_id}
        pe_row.update(raw_components["pitch_energy"])
        
        # Simpan visualisasi waveform & spectrogram jika is_first == True
        if is_first:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            # Waveform
            times = np.linspace(0, duration_sec, len(y))
            ax1.plot(times, y, color='#1f77b4', alpha=0.8)
            ax1.set_title(f"Waveform Sinyal Audio Denoised ({participant_id})", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Amplitudo", fontsize=10)
            ax1.set_xlabel("Waktu (detik)", fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Spectrogram
            img = librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
            ax2.set_title(f"Mel Spectrogram ({participant_id})", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Frekuensi Mel", fontsize=10)
            ax2.set_xlabel("Waktu (detik)", fontsize=10)
            fig.colorbar(img, ax=ax2, format='%+2.0f dB')
            
            plt.tight_layout()
            viz_path = os.path.join(OUTPUT_DOC_DIR, "audio_feature_examples.png")
            plt.savefig(viz_path, dpi=300)
            plt.close()
            
        return {
            "status": "success",
            "participant_id": participant_id,
            "npy_filename": npy_filename,
            "png_filename": png_filename,
            "feature_vector": feature_vector,
            "mfcc_row": mfcc_row,
            "pe_row": pe_row
        }
    except Exception as e:
        return {
            "status": "failed",
            "participant_id": participant_id,
            "error_message": str(e)
        }

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Scan file audio hasil denoising
    audio_files = []
    if os.path.exists(DENOISED_DIR):
        for f in os.listdir(DENOISED_DIR):
            if f.endswith(".wav"):
                p_id = os.path.splitext(f)[0]
                file_path = os.path.join(DENOISED_DIR, f)
                audio_files.append((p_id, file_path))

    # Urutkan file secara numerik berdasarkan ID partisipan
    def extract_digits(p_id):
        digits = "".join(filter(str.isdigit, p_id))
        return int(digits) if digits.isdigit() else 0

    audio_files.sort(key=lambda x: extract_digits(x[0]))
    print(f"Ditemukan {len(audio_files)} file audio denoised untuk ekstraksi fitur.")
    print("Memulai ekstraksi fitur akustik secara paralel (Multiprocessing)...")
    
    # Gunakan ProcessPoolExecutor untuk paralelisasi
    # Batasi ke 10 workers agar sistem tetap responsif
    max_workers = min(10, os.cpu_count() or 1)
    print(f"Menjalankan dengan {max_workers} worker processes.\n")
    
    index_records = []
    stacked_features = []
    failed_log_records = []
    
    mfcc_csv_data = []
    pitch_energy_csv_data = []
    
    # Kirim task ke pool
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, (participant_id, file_path) in enumerate(audio_files):
            # Kirim flag is_first untuk partisipan pertama agar membuat plot dokumentasi
            is_first = (idx == 0)
            futures[executor.submit(process_single_participant, participant_id, file_path, is_first)] = participant_id
            
        # Kumpulkan hasil saat selesai
        completed = 0
        total = len(audio_files)
        for future in as_completed(futures):
            participant_id = futures[future]
            completed += 1
            
            try:
                res = future.result()
                if res["status"] == "success":
                    # Sukses
                    pid = res["participant_id"]
                    npy_fn = res["npy_filename"]
                    png_fn = res["png_filename"]
                    
                    index_records.append({
                        "participant_id": pid,
                        "mfcc_file": f"mfcc/{npy_fn}",
                        "spectrogram_file": f"spectrogram/{png_fn}",
                        "feature_dim": 128,
                        "status": "success"
                    })
                    stacked_features.append((pid, res["feature_vector"]))
                    mfcc_csv_data.append(res["mfcc_row"])
                    pitch_energy_csv_data.append(res["pe_row"])
                    print(f"[{completed}/{total}] Berhasil memproses {pid}")
                else:
                    # Gagal di fungsi
                    err_msg = res["error_message"]
                    print(f"[{completed}/{total}] Gagal memproses {participant_id}: {err_msg}")
                    failed_log_records.append({
                        "participant_id": participant_id,
                        "error_message": err_msg,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    index_records.append({
                        "participant_id": participant_id,
                        "mfcc_file": "None",
                        "spectrogram_file": "None",
                        "feature_dim": 128,
                        "status": f"failed: {err_msg}"
                    })
                    # Simpan nol sementara
                    stacked_features.append((participant_id, np.zeros(128, dtype=np.float32)))
            except Exception as exc:
                # Gagal di executor
                print(f"[{completed}/{total}] Gagal memproses {participant_id} (Executor error): {str(exc)}")
                failed_log_records.append({
                    "participant_id": participant_id,
                    "error_message": f"Executor error: {str(exc)}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                index_records.append({
                    "participant_id": participant_id,
                    "mfcc_file": "None",
                    "spectrogram_file": "None",
                    "feature_dim": 128,
                    "status": f"failed: {str(exc)}"
                })
                stacked_features.append((participant_id, np.zeros(128, dtype=np.float32)))

    # Urutkan kembali hasil berdasarkan participant_id numerik untuk memastikan alignment array stacked npy
    # Karena as_completed mengembalikan hasil secara tidak berurutan
    print("\nMengurutkan dan menyelaraskan seluruh data fitur...")
    
    # Sort index records
    index_records.sort(key=lambda x: extract_digits(x["participant_id"]))
    
    # Sort stacked features
    stacked_features.sort(key=lambda x: extract_digits(x[0]))
    stacked_array = np.vstack([x[1] for x in stacked_features])
    
    # Sort CSV data
    mfcc_csv_data.sort(key=lambda x: extract_digits(x["participant_id"]))
    pitch_energy_csv_data.sort(key=lambda x: extract_digits(x["participant_id"]))

    # ==========================================
    # SAVE ALL OUTPUTS AND EXPORTS
    # ==========================================
    print("Menyimpan berkas indeks dan visualisasi...")

    # 1. Simpan audio_feature_index.csv & .xlsx
    df_index = pd.DataFrame(index_records)
    index_csv_path = os.path.join(OUTPUT_DIR, "audio_feature_index.csv")
    index_excel_path = os.path.join(OUTPUT_DOC_DIR, "audio_feature_index.xlsx")
    df_index.to_csv(index_csv_path, index=False)
    df_index.to_excel(index_excel_path, index=False)

    # 2. Simpan stacked numpy array audio_features.npy
    stacked_npy_path = os.path.join(OUTPUT_DIR, "audio_features.npy")
    np.save(stacked_npy_path, stacked_array)

    # 3. Simpan mfcc_features.csv
    df_mfcc = pd.DataFrame(mfcc_csv_data)
    mfcc_csv_path = os.path.join(OUTPUT_DIR, "mfcc_features.csv")
    df_mfcc.to_csv(mfcc_csv_path, index=False)

    # 4. Simpan pitch_energy_features.csv
    df_pe = pd.DataFrame(pitch_energy_csv_data)
    pe_csv_path = os.path.join(OUTPUT_DIR, "pitch_energy_features.csv")
    df_pe.to_csv(pe_csv_path, index=False)

    # 5. Tulis log kegagalan jika ada
    failed_log_path = os.path.join(OUTPUT_DOC_DIR, "audio_feature_extraction_failed.log")
    with open(failed_log_path, "w", encoding="utf-8") as f_log:
        if failed_log_records:
            f_log.write("=== LOG FILES GAGAL PROSES EKSTRAKSI FITUR ===\n\n")
            for rec in failed_log_records:
                f_log.write(f"[{rec['timestamp']}] Partisipan: {rec['participant_id']}\n")
                f_log.write(f"Eror: {rec['error_message']}\n")
                f_log.write("-" * 50 + "\n")
            print(f"[WARNING] Ada beberapa error. Log kegagalan disimpan di: {failed_log_path}")
        else:
            f_log.write("Semua proses ekstraksi fitur berhasil diselesaikan tanpa kegagalan.\n")
            print("[SUCCESS] Semua file diproses dengan sukses tanpa error.")

    # ==========================================
    # GENERATE EMPIRICAL STATS AND DOCUMENTATION
    # ==========================================
    print("\nMenghitung statistik durasi dan segmen...")
    segment_durations = []
    valid_segment_counts = []
    
    if os.path.exists(SEGMENTED_DIR):
        for pid_folder in os.listdir(SEGMENTED_DIR):
            folder_path = os.path.join(SEGMENTED_DIR, pid_folder)
            if os.path.isdir(folder_path):
                seg_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
                valid_segment_counts.append(len(seg_files))
                for sf_name in seg_files:
                    sf_path = os.path.join(folder_path, sf_name)
                    try:
                        info = sf.info(sf_path)
                        segment_durations.append(info.duration)
                    except Exception:
                        pass

    # Load segment log to retrieve total segment count
    total_segments_log = 0
    valid_segments_log = 0
    segment_log_path = os.path.join(OUTPUT_DOC_DIR, "audio_segmentation_log.xlsx")
    if os.path.exists(segment_log_path):
        df_seg_log = pd.read_excel(segment_log_path)
        total_segments_log = int(df_seg_log["total_segments"].sum())
        valid_segments_log = int(df_seg_log["valid_segments"].sum())

    # Tulis Laporan Lanjutan
    report_path = os.path.join(OUTPUT_DOC_DIR, "audio_feature_extraction_report.md")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("# Laporan Hasil Ekstraksi Fitur Audio (Langkah 2.9)\n\n")
        f_rep.write("## 1. Jenis Fitur Audio & Konfigurasi\n")
        f_rep.write("- **Sampling Rate**: 16.000 Hz (mono, PCM 16-bit).\n")
        f_rep.write("- **Jumlah Fitur per Partisipan**: 128 dimensi.\n")
        f_rep.write("- **Konfigurasi Ekstraksi MFCC**:\n")
        f_rep.write("  - Jumlah koefisien: 16 koefisien.\n")
        f_rep.write("  - Ukuran Frame FFT (n_fft): 1024 sampel (64 ms).\n")
        f_rep.write("  - Panjang Langkah (hop_length): 512 sampel (32 ms).\n")
        f_rep.write("  - Derivatif: First Derivative (Delta) & Second Derivative (Delta-Delta).\n")
        f_rep.write("- **Rincian Alokasi Fitur (128 dimensi)**:\n")
        f_rep.write("  - Fitur MFCC (96): Rerata (16) & Simpangan Baku (16) untuk MFCC, Delta-MFCC, dan Delta-Delta-MFCC.\n")
        f_rep.write("  - Fitur Spektral (16): Rerata & Simpangan Baku untuk Spectral Centroid, Bandwidth, Roll-off, serta Spectral Contrast (5 sub-band).\n")
        f_rep.write("  - Fitur Temporal & Energi (16): ZCR (Rerata, Std), RMS Energy (Rerata, Std, Max), Pitch/F0 (Rerata, Std, Min, Max), dan Dinamika Bicara (Voicing Fraction, Pause Duration, Pause Count, Avg Pause Duration, Speech Rate, Jitter, Shimmer).\n\n")
        
        f_rep.write("## 2. Alasan Pemilihan Fitur\n")
        f_rep.write("- **MFCC**: Sangat krusial untuk menganalisis timbre suara dan karakteristik vokal spektral.\n")
        f_rep.write("- **Pitch (F0) & Energy (RMS)**: Representasi tinggi-rendah dan intensitas suara yang sering berfluktuasi secara abnormal pada pasien depresi (prosodi datar).\n")
        f_rep.write("- **Pause Dynamics & Speech Rate**: Karakteristik melambatnya aktivitas motorik-vokal (speech rate menurun, durasi jeda memanjang) merupakan biomarker klinis utama depresi.\n")
        f_rep.write("- **Spectral Centroid, Bandwidth & Contrast**: Mengidentifikasi kejelasan artikulasi (vocal clarity) dan kecerahan spektrum suara.\n")
        f_rep.write("- **Jitter & Shimmer**: Menangkap ketidakstabilan mikro pada pita suara (instability/breathiness) yang berkorelasi dengan kelelahan vokal.\n\n")
        
        f_rep.write("## 3. Statistik Durasi & Segmen Audio\n")
        if segment_durations:
            f_rep.write(f"- **Total Partisipan**: {len(audio_files)} orang.\n")
            f_rep.write(f"- **Total Segmen Teoretis**: {total_segments_log} segmen.\n")
            f_rep.write(f"- **Total Segmen Valid (Setelah Filter Sunyi)**: {len(segment_durations)} segmen.\n")
            f_rep.write(f"- **Rerata Durasi Segmen**: {np.mean(segment_durations):.2f} detik.\n")
            f_rep.write(f"- **Simpangan Baku Durasi Segmen**: {np.std(segment_durations):.2f} detik.\n")
            f_rep.write(f"- **Durasi Minimum Segmen**: {np.min(segment_durations):.2f} detik.\n")
            f_rep.write(f"- **Durasi Maksimum Segmen**: {np.max(segment_durations):.2f} detik.\n")
            f_rep.write(f"- **Total Durasi Hasil Segmentasi Aktif**: {np.sum(segment_durations)/60:.2f} menit.\n")
        else:
            f_rep.write("- Data segmen tidak ditemukan atau gagal dipindai.\n")
        f_rep.write("\n")
        
        f_rep.write("## 4. Hasil Proses & Log Status\n")
        success_count = len(df_index[df_index['status'] == 'success'])
        fail_count = len(df_index[df_index['status'] != 'success'])
        f_rep.write(f"- **Total Sukses**: {success_count} partisipan.\n")
        f_rep.write(f"- **Total Gagal**: {fail_count} partisipan.\n")
        if fail_count > 0:
            f_rep.write(f"- **Rincian Kegagalan**: Lihat berkas `audio_feature_extraction_failed.log`.\n")
        else:
            f_rep.write("- **Rincian Kegagalan**: Tidak ada kegagalan terdeteksi.\n")

    print(f"\n[SUCCESS] Laporan hasil ekstraksi fitur disimpan di: {report_path}")
    print(f"Total durasi eksekusi paralel: {time.time() - start_time:.2f} detik.")
    print(f"=== PIPELINE EKSTRAKSI FITUR SELESAI ===")
