import os
import time
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
RMS_THRESHOLD = 0.005
YIN_HOP = 1024

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "preprocessing_output", "audio")
DENOISED_DIR = os.path.join(OUTPUT_DIR, "denoised_audio")
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented_audio")
OUTPUT_DOC_DIR = os.path.join(OUTPUT_DIR, "documentation")

MFCC_OUT_DIR = os.path.join(OUTPUT_DIR, "mfcc")
SPECTROGRAM_OUT_DIR = os.path.join(OUTPUT_DIR, "spectrogram")
SPECTROGRAM_FEATURES_DIR = os.path.join(OUTPUT_DIR, "spectrogram_features")

# Buat direktori output jika belum ada
for folder in [OUTPUT_DIR, DENOISED_DIR, SEGMENTED_DIR, OUTPUT_DOC_DIR, MFCC_OUT_DIR, SPECTROGRAM_OUT_DIR, SPECTROGRAM_FEATURES_DIR]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# REUSE FEATURE EXTRACTION LOGIC
# ==========================================
def estimate_speech_rate(y, sr, voiced_duration):
    try:
        window_len = int(0.100 * sr)
        envelope = np.convolve(np.abs(y), np.ones(window_len)/window_len, mode='same')
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)
        min_dist = int(0.150 * sr)
        peaks, _ = scipy.signal.find_peaks(envelope, height=0.05, distance=min_dist)
        syllable_count = len(peaks)
        speech_rate = syllable_count / voiced_duration if voiced_duration > 0 else 0.0
        return speech_rate, syllable_count
    except Exception:
        return 0, 0.0

def extract_features_for_audio(y, sr, participant_id):
    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
    frame_duration = HOP_LENGTH / sr
    total_frames = len(rms_frames)
    
    try:
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, hop_length=YIN_HOP)
        rms_yin = librosa.feature.rms(y=y, frame_length=2048, hop_length=YIN_HOP)[0]
        min_len = min(len(f0), len(rms_yin))
        f0 = f0[:min_len]
        rms_yin = rms_yin[:min_len]
        
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

    is_pause = rms_frames < RMS_THRESHOLD
    total_pause_frames = np.sum(is_pause)
    total_pause_duration = total_pause_frames * frame_duration
    voicing_fraction = np.sum(~is_pause) / total_frames if total_frames > 0 else 0.0
    
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
    speech_rate, syllable_count = estimate_speech_rate(y, sr, voiced_duration)
    
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
            rms_f512 = rms_frames
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

    feature_vector = []
    
    # 1. MFCC (96)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    feature_vector.extend(np.mean(mfcc, axis=1))
    feature_vector.extend(np.std(mfcc, axis=1))
    feature_vector.extend(np.mean(delta_mfcc, axis=1))
    feature_vector.extend(np.std(delta_mfcc, axis=1))
    feature_vector.extend(np.mean(delta2_mfcc, axis=1))
    feature_vector.extend(np.std(delta2_mfcc, axis=1))
    
    # 2. Spectral (16)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(cent))
    feature_vector.append(np.std(cent))
    
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(band))
    feature_vector.append(np.std(band))
    
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)[0]
    feature_vector.append(np.mean(roll))
    feature_vector.append(np.std(roll))
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=4)
    feature_vector.extend(np.mean(contrast, axis=1))
    feature_vector.extend(np.std(contrast, axis=1))
    
    # 3. Temporal, Energy & Dynamics (16)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
    feature_vector.append(np.mean(zcr))
    feature_vector.append(np.std(zcr))
    
    feature_vector.append(np.mean(rms_frames))
    feature_vector.append(np.std(rms_frames))
    feature_vector.append(np.max(rms_frames))
    
    feature_vector.append(pitch_mean)
    feature_vector.append(pitch_std)
    feature_vector.append(pitch_min)
    feature_vector.append(pitch_max)
    
    feature_vector.append(voicing_fraction)
    feature_vector.append(total_pause_duration)
    feature_vector.append(float(pause_segments))
    feature_vector.append(avg_pause_duration)
    feature_vector.append(speech_rate)
    feature_vector.append(jitter)
    feature_vector.append(shimmer)
    
    feature_vector = np.array(feature_vector, dtype=np.float32)
    
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

# ==========================================
# HEALING PIPELINE
# ==========================================
def heal():
    print("=== MEMULAI PROSES PENYEMBUHAN (HEALING) FITUR AUDIO ===")
    
    # 1. Load existing files
    index_csv_path = os.path.join(OUTPUT_DIR, "audio_feature_index.csv")
    mfcc_csv_path = os.path.join(OUTPUT_DIR, "mfcc_features.csv")
    pe_csv_path = os.path.join(OUTPUT_DIR, "pitch_energy_features.csv")
    
    if not os.path.exists(index_csv_path):
        print("[ERROR] Index audio_feature_index.csv tidak ditemukan. Silakan jalankan script utama terlebih dahulu.")
        return
        
    df_index = pd.read_csv(index_csv_path)
    df_mfcc = pd.read_csv(mfcc_csv_path) if os.path.exists(mfcc_csv_path) else pd.DataFrame(columns=["participant_id"])
    df_pe = pd.read_csv(pe_csv_path) if os.path.exists(pe_csv_path) else pd.DataFrame(columns=["participant_id"])
    
    # Cari yang gagal
    failed_rows = df_index[df_index["status"] != "success"]
    failed_pids = failed_rows["participant_id"].tolist()
    
    print(f"Ditemukan {len(failed_pids)} partisipan yang gagal diproses: {failed_pids}")
    
    if not failed_pids:
        print("[SUCCESS] Semua partisipan sudah berstatus 'success'. Tidak ada yang perlu disembuhkan.")
        return
        
    failed_log_records = []
    
    # Proses secara SEQUENTIAL untuk menghemat RAM
    for idx, pid in enumerate(failed_pids):
        print(f"[{idx+1}/{len(failed_pids)}] Menyembuhkan: {pid}...")
        file_path = os.path.join(DENOISED_DIR, f"{pid}.wav")
        
        if not os.path.exists(file_path):
            print(f"  [ERROR] File audio tidak ditemukan: {file_path}")
            continue
            
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            
            # Ekstrak fitur
            feature_vector, raw_components = extract_features_for_audio(y, sr, pid)
            
            # Simpan .npy
            npy_filename = f"{pid}.npy"
            npy_filepath = os.path.join(MFCC_OUT_DIR, npy_filename)
            np.save(npy_filepath, feature_vector)
            
            # Plot spectrogram
            plt.figure(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {pid}')
            plt.tight_layout()
            
            png_filename = f"{pid}.png"
            spec_path1 = os.path.join(SPECTROGRAM_OUT_DIR, png_filename)
            spec_path2 = os.path.join(SPECTROGRAM_FEATURES_DIR, png_filename)
            plt.savefig(spec_path1, dpi=100)
            plt.savefig(spec_path2, dpi=100)
            plt.close()
            
            # Update index dataframe
            df_index.loc[df_index["participant_id"] == pid, "mfcc_file"] = f"mfcc/{npy_filename}"
            df_index.loc[df_index["participant_id"] == pid, "spectrogram_file"] = f"spectrogram/{png_filename}"
            df_index.loc[df_index["participant_id"] == pid, "feature_dim"] = 128
            df_index.loc[df_index["participant_id"] == pid, "status"] = "success"
            
            # Siapkan row data untuk mfcc
            mfcc_row = {"participant_id": pid}
            for i in range(16):
                mfcc_row[f"mfcc_{i+1}_mean"] = raw_components["mfcc_mean"][i]
                mfcc_row[f"mfcc_{i+1}_std"] = raw_components["mfcc_std"][i]
                mfcc_row[f"mfcc_{i+1}_delta_mean"] = raw_components["delta_mean"][i]
                mfcc_row[f"mfcc_{i+1}_delta_std"] = raw_components["delta_std"][i]
                mfcc_row[f"mfcc_{i+1}_delta2_mean"] = raw_components["delta2_mean"][i]
                mfcc_row[f"mfcc_{i+1}_delta2_std"] = raw_components["delta2_std"][i]
            
            # Hapus baris lama jika ada, lalu tambah yang baru
            df_mfcc = df_mfcc[df_mfcc["participant_id"] != pid]
            df_mfcc = pd.concat([df_mfcc, pd.DataFrame([mfcc_row])], ignore_index=True)
            
            # Siapkan row data untuk pitch/energy
            pe_row = {"participant_id": pid}
            pe_row.update(raw_components["pitch_energy"])
            
            df_pe = df_pe[df_pe["participant_id"] != pid]
            df_pe = pd.concat([df_pe, pd.DataFrame([pe_row])], ignore_index=True)
            
            print(f"  [SUCCESS] Berhasil menyembuhkan {pid}")
            
        except Exception as e:
            print(f"  [FAILED] Gagal menyembuhkan {pid}: {str(e)}")
            failed_log_records.append({
                "participant_id": pid,
                "error_message": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            df_index.loc[df_index["participant_id"] == pid, "status"] = f"failed_heal: {str(e)}"
            
    # ==========================================
    # RESORT AND RE-STACK
    # ==========================================
    def extract_digits(p_id):
        digits = "".join(filter(str.isdigit, p_id))
        return int(digits) if digits.isdigit() else 0
        
    print("\nMenyelaraskan data dan menyusun ulang stacked numpy array...")
    
    # Urutkan berdasarkan PID numerik
    df_index["sort_key"] = df_index["participant_id"].apply(extract_digits)
    df_index = df_index.sort_values("sort_key").drop(columns=["sort_key"])
    
    df_mfcc["sort_key"] = df_mfcc["participant_id"].apply(extract_digits)
    df_mfcc = df_mfcc.sort_values("sort_key").drop(columns=["sort_key"])
    
    df_pe["sort_key"] = df_pe["participant_id"].apply(extract_digits)
    df_pe = df_pe.sort_values("sort_key").drop(columns=["sort_key"])
    
    # Re-stack audio_features.npy
    stacked_features = []
    for pid in df_index["participant_id"]:
        npy_path = os.path.join(MFCC_OUT_DIR, f"{pid}.npy")
        if os.path.exists(npy_path):
            vec = np.load(npy_path)
        else:
            vec = np.zeros(128, dtype=np.float32)
        stacked_features.append(vec)
        
    stacked_array = np.vstack(stacked_features)
    stacked_npy_path = os.path.join(OUTPUT_DIR, "audio_features.npy")
    np.save(stacked_npy_path, stacked_array)
    
    # Save CSVs
    df_index.to_csv(index_csv_path, index=False)
    df_index.to_excel(os.path.join(OUTPUT_DOC_DIR, "audio_feature_index.xlsx"), index=False)
    
    df_mfcc.to_csv(mfcc_csv_path, index=False)
    df_pe.to_csv(pe_csv_path, index=False)
    
    # Update log failed
    failed_log_path = os.path.join(OUTPUT_DOC_DIR, "audio_feature_extraction_failed.log")
    with open(failed_log_path, "w", encoding="utf-8") as f_log:
        still_failed = df_index[df_index["status"] != "success"]
        if not still_failed.empty:
            f_log.write("=== LOG FILES GAGAL PROSES EKSTRAKSI FITUR (SETELAH HEALING) ===\n\n")
            for _, row in still_failed.iterrows():
                f_log.write(f"Partisipan: {row['participant_id']}\n")
                f_log.write(f"Status: {row['status']}\n")
                f_log.write("-" * 50 + "\n")
            print(f"[WARNING] Masih ada {len(still_failed)} kegagalan. Log disimpan di: {failed_log_path}")
        else:
            f_log.write("Semua proses ekstraksi fitur berhasil diselesaikan tanpa kegagalan.\n")
            print("[SUCCESS] Semua file berhasil diproses 100% tanpa error.")

    # ==========================================
    # RECALCULATE EMPIRICAL STATS AND REPORT
    # ==========================================
    print("Memperbarui statistik durasi dan segmen...")
    segment_durations = []
    if os.path.exists(SEGMENTED_DIR):
        for pid_folder in os.listdir(SEGMENTED_DIR):
            folder_path = os.path.join(SEGMENTED_DIR, pid_folder)
            if os.path.isdir(folder_path):
                for sf_name in os.listdir(folder_path):
                    if sf_name.endswith(".wav"):
                        try:
                            info = sf.info(os.path.join(folder_path, sf_name))
                            segment_durations.append(info.duration)
                        except Exception:
                            pass

    total_segments_log = 0
    segment_log_path = os.path.join(OUTPUT_DOC_DIR, "audio_segmentation_log.xlsx")
    if os.path.exists(segment_log_path):
        df_seg_log = pd.read_excel(segment_log_path)
        total_segments_log = int(df_seg_log["total_segments"].sum())

    # Rewrite Markdown Report
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
            f_rep.write(f"- **Total Partisipan**: {len(df_index)} orang.\n")
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
        f_rep.write(f"- **Total Sukses**: {success_count} / {len(df_index)} partisipan.\n")
        f_rep.write(f"- **Total Gagal**: {fail_count} / {len(df_index)} partisipan.\n")
        if fail_count > 0:
            f_rep.write(f"- **Rincian Kegagalan**: Lihat berkas `audio_feature_extraction_failed.log`.\n")
        else:
            f_rep.write("- **Rincian Kegagalan**: Tidak ada kegagalan terdeteksi (100% Sukses).\n")

    print(f"\n[SUCCESS] Laporan diperbarui di: {report_path}")
    print("=== PROSES PENYEMBUHAN SELESAI DENGAN SUKSES ===")

if __name__ == "__main__":
    heal()
