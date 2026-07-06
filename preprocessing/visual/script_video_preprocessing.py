"""
script_video_preprocessing.py
==============================
Pipeline prapemrosesan modalitas VISUAL untuk dataset E-DAIC.

Input  : File CSV OpenFace2 per partisipan
         ({participant_id}_OpenFace2_1_0_Pose_gaze_AUs.csv)
         yang sudah diekstrak oleh OpenFace 2.x dari rekaman video sesi DAIC-WOZ.
         TIDAK diperlukan video mentah — script ini bekerja murni dari file CSV.

Output : (semua di --output_dir / visual)
  - pose_features.csv                   : Statistik agregat fitur pose per partisipan
  - gaze_features.csv                   : Statistik agregat fitur gaze per partisipan
  - expression_features.csv             : Statistik AU intensity + AU presence rate per partisipan
  - visual_features.csv                 : Gabungan ketiga CSV di atas + kolom quality summary;
                                          siap di-merge dengan fitur audio/teks via participant_id
  - visual_features_sequence.npy        : Array 3-D (n_participants, seq_len, 31 fitur)
                                          untuk model temporal (LSTM / Transformer)
  - visual_features_sequence_index.csv  : Peta row_index <-> participant_id untuk array .npy
  - visual_feature_scaler_params.json   : Parameter scaler (mean/std atau min/max per kolom)
  - video_preprocessing_log.xlsx        : Log status per partisipan (kualitas, error, dll.)

Referensi Desain:
  - Ringayati et al. (2023) menggunakan agregat statistik OpenFace AUs sebagai input SVM/RF
    untuk deteksi depresi pada DAIC-WOZ.
  - AVEC 2017 baseline menggunakan sequence-level features dengan panjang tetap 300 frame.
  - Scaler di-fit HANYA pada subset training untuk menghindari data leakage
    (Kapoor & Etchells, 2023, "Leakage and the reproducibility crisis in ML-based science").

Struktur Kode:
  load_openface_csv()           — membaca dan memvalidasi 53 kolom
  filter_quality()              — filter per-frame berdasarkan success & confidence
  aggregate_statistics()        — agregasi temporal: statistik kontinu + activation rate biner
  build_fixed_length_sequence() — resampling ke panjang tetap untuk model sekuensial
  normalize_features()          — normalisasi tanpa data leakage via fit_mask
  save_outputs()                — menyimpan semua file output
  main()                        — entry point dengan argparse

Cara menjalankan (contoh):
  python script_video_preprocessing.py \\
      --input_dir "../../data/E-DAIC" \\
      --output_dir "../../preprocessing_output" \\
      --confidence_threshold 0.7 \\
      --min_valid_ratio 0.6 \\
      --sequence_length 300 \\
      --scaler_method standard
"""

import os
import sys
import glob
import json
import logging
import argparse
import traceback
import warnings
import shutil
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# =============================================================================
# KONSTANTA SKEMA KOLOM
# Dideklarasikan secara eksplisit (bukan di-derive dari file) agar script bisa
# mendeteksi mismatch kolom lebih awal sebelum menyentuh data.
# =============================================================================

METADATA_COLS: List[str] = ["frame", "timestamp", "confidence", "success"]

POSE_COLS: List[str] = [
    "pose_Tx", "pose_Ty", "pose_Tz",   # Translasi kepala (mm)
    "pose_Rx", "pose_Ry", "pose_Rz",   # Rotasi kepala (radian)
]

GAZE_COLS: List[str] = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",   # Vektor gaze mata kiri
    "gaze_1_x", "gaze_1_y", "gaze_1_z",   # Vektor gaze mata kanan
    "gaze_angle_x", "gaze_angle_y",         # Sudut gaze agregat
]

AU_INTENSITY_COLS: List[str] = [
    # Intensitas AU kontinu (skala 0-5). Digunakan untuk menangkap
    # derajat ekspresi wajah - bukan hanya ada/tidaknya.
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

AU_PRESENCE_COLS: List[str] = [
    # Kehadiran AU biner (0/1). AU28 HANYA punya versi _c (tidak ada AU28_r).
    # Agregasi menggunakan activation_rate (proporsi frame bernilai 1),
    # BUKAN mean nilai kontinu, karena skalanya berbeda.
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

# Kolom yang digunakan di sequence 3-D (au_presence TIDAK disertakan karena
# informasi biner sudah tercakup di activation_rate versi agregat)
SEQUENCE_FEATURE_COLS: List[str] = POSE_COLS + GAZE_COLS + AU_INTENSITY_COLS
# => 6 pose + 8 gaze + 17 au_intensity = 31 fitur per frame

# Total kolom wajib yang harus ada di setiap CSV
ALL_REQUIRED_COLS: List[str] = (
    METADATA_COLS + POSE_COLS + GAZE_COLS + AU_INTENSITY_COLS + AU_PRESENCE_COLS
)  # 4 + 6 + 8 + 17 + 18 = 53 kolom


# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Menyiapkan logger dua-jalur: ke console (INFO) dan ke file teks (DEBUG).

    File log teks disimpan di output_dir untuk keperluan debugging.
    Log terstruktur per-partisipan dikumpulkan ke DataFrame dan disimpan
    sebagai video_preprocessing_log.xlsx oleh fungsi save_outputs().

    Parameters
    ----------
    output_dir : str
        Direktori tempat file log teks akan disimpan.

    Returns
    -------
    logging.Logger
        Logger bernama "visual_preprocessing".
    """
    logger = logging.getLogger("visual_preprocessing")
    logger.setLevel(logging.DEBUG)

    # Handler console - tampilkan INFO ke atas
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Handler file - simpan semua level termasuk DEBUG
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "video_preprocessing_run.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    return logger


# =============================================================================
# 1. LOAD & VALIDATE
# =============================================================================

def load_openface_csv(
    file_path: str,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Membaca file CSV OpenFace2 dan memvalidasi kehadiran 53 kolom yang
    ditetapkan oleh skema E-DAIC.

    Desain:
    - Validasi dilakukan SEBELUM operasi apapun sehingga error terdeteksi dini.
    - Kolom dibaca apa adanya; tidak ada rename otomatis agar konsisten dengan
      file asli E-DAIC yang sudah diverifikasi langsung dari data.
    - strip() pada nama kolom mengantisipasi spasi tidak kasat mata di header.
    - Jika kolom tidak lengkap, mengembalikan (None, pesan_error) sehingga
      pipeline bisa mencatat warning dan melanjutkan ke partisipan berikutnya
      tanpa crash seluruh pipeline.

    Parameters
    ----------
    file_path : str
        Path absolut ke file CSV OpenFace2.
    logger : logging.Logger
        Logger untuk mencatat peringatan/error.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], str]
        (DataFrame, "") jika berhasil, atau (None, pesan_error) jika gagal.
    """
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        # Hapus spasi dari nama kolom (bisa terjadi pada output OpenFace versi lama)
        df.columns = [c.strip() for c in df.columns]

        missing_cols = [c for c in ALL_REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            msg = (
                f"Kolom berikut tidak ditemukan di file: {missing_cols}. "
                f"Total kolom file: {len(df.columns)}"
            )
            logger.warning(f"[VALIDASI GAGAL] {os.path.basename(file_path)}: {msg}")
            return None, msg

        logger.debug(
            f"Berhasil membaca {file_path}: {len(df)} baris, {len(df.columns)} kolom."
        )
        return df, ""

    except Exception as e:
        msg = f"Gagal membaca CSV: {str(e)}"
        logger.error(f"[LOAD ERROR] {os.path.basename(file_path)}: {msg}")
        return None, msg


# =============================================================================
# 2. QUALITY FILTERING
# =============================================================================

def filter_quality(
    df: pd.DataFrame,
    participant_id: str,
    confidence_threshold: float = 0.7,
    min_valid_ratio: float = 0.6,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Memfilter frame berkualitas rendah berdasarkan kolom `success` dan `confidence`.

    Rasional filter:
    - success == 0  : OpenFace gagal mendeteksi wajah pada frame tersebut.
                      Data AU/pose dari frame ini tidak reliable.
    - confidence < threshold : Walaupun deteksi berhasil, landmark akurasi rendah
                      meningkatkan noise pada estimasi AU. Threshold 0.7 adalah
                      nilai umum yang digunakan di literatur (Baltrusaitis et al., 2018).

    Keputusan "low_quality" (valid_frame_ratio < min_valid_ratio) HANYA dicatat
    sebagai flag di log - partisipan TETAP diproses. Keputusan exclude final
    dilakukan secara manual di tahap Quality Control terpisah (bukan otomatis
    di script ini), sesuai praktik standar dalam penelitian klinis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah hasil load_openface_csv().
    participant_id : str
        ID partisipan untuk keperluan logging.
    confidence_threshold : float
        Ambang batas minimum nilai kolom `confidence`. Default 0.7.
    min_valid_ratio : float
        Jika rasio frame valid di bawah nilai ini, tandai sebagai 'low_quality'.
        Default 0.6.
    logger : logging.Logger, optional
        Logger untuk mencatat peringatan.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (df_valid, quality_stats) — DataFrame frame valid dan dict statistik kualitas.
    """
    n_total = len(df)

    # Mask frame valid: success == 1 DAN confidence >= threshold
    valid_mask = (df["success"] == 1) & (df["confidence"] >= confidence_threshold)
    df_valid = df[valid_mask].copy()

    n_valid = len(df_valid)
    valid_ratio = n_valid / n_total if n_total > 0 else 0.0
    mean_conf = df_valid["confidence"].mean() if n_valid > 0 else 0.0
    # session_duration diambil dari timestamp maksimum SELURUH data (termasuk frame invalid),
    # karena durasi sesi tidak tergantung pada kualitas deteksi
    session_duration = df["timestamp"].max() if n_total > 0 else 0.0

    quality_flag = "ok"
    if valid_ratio < min_valid_ratio:
        quality_flag = "low_quality"
        if logger:
            logger.warning(
                f"[LOW QUALITY] Partisipan {participant_id}: valid_frame_ratio="
                f"{valid_ratio:.3f} < min_valid_ratio={min_valid_ratio}. "
                "Partisipan tetap diproses; keputusan exclude dilakukan manual."
            )

    quality_stats: Dict[str, Any] = {
        "n_total_frames":       n_total,
        "n_valid_frames":       n_valid,
        "valid_frame_ratio":    round(valid_ratio, 4),
        "mean_confidence":      round(float(mean_conf), 4),
        "session_duration_sec": round(float(session_duration), 2),
        "quality_flag":         quality_flag,
    }

    if logger:
        logger.info(
            f"  [{participant_id}] Quality filter: {n_valid}/{n_total} frame valid "
            f"({valid_ratio:.1%}), flag={quality_flag}"
        )

    return df_valid, quality_stats


# =============================================================================
# 3 & 4. FEATURE GROUPING + AGREGASI TEMPORAL
# =============================================================================

def _aggregate_continuous(
    df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, float]:
    """
    Menghitung 5 statistik deskriptif (mean, std, median, min, max) untuk
    setiap kolom kontinu.

    Nama kolom output: "{feature}_{stat}", mis. "pose_Tx_mean".

    Lima statistik ini dipilih karena:
    - mean & median : menangkap tendensi sentral (median lebih robust terhadap outlier)
    - std           : menangkap variabilitas temporal (dinamika gerakan/ekspresi)
    - min & max     : menangkap rentang dan peak aktivitas
    Bersama-sama keempatnya merepresentasikan distribusi marginal tiap fitur
    tanpa memerlukan asumsi distribusi apapun.

    Parameters
    ----------
    df : pd.DataFrame
        Frame data yang sudah difilter quality.
    cols : List[str]
        Daftar nama kolom yang akan diagregasi.

    Returns
    -------
    Dict[str, float]
        Dict berisi {"{feature}_{stat}": nilai}.
    """
    result: Dict[str, float] = {}
    for col in cols:
        series = df[col].dropna()
        n = len(series)
        result[f"{col}_mean"]   = float(series.mean())   if n > 0 else np.nan
        result[f"{col}_std"]    = float(series.std())    if n > 1 else 0.0
        result[f"{col}_median"] = float(series.median()) if n > 0 else np.nan
        result[f"{col}_min"]    = float(series.min())    if n > 0 else np.nan
        result[f"{col}_max"]    = float(series.max())    if n > 0 else np.nan
    return result


def _aggregate_binary(
    df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, float]:
    """
    Menghitung activation_rate (proporsi frame bernilai 1) untuk kolom biner AU presence.

    Alasan tidak dicampur dengan agregasi kontinu:
    - Kolom AU*_c berskala 0/1 sehingga mean-nya bermakna sebagai
      proporsi kemunculan (activation rate), berbeda dari AU*_r yang
      mean-nya merepresentasikan intensitas rata-rata.
    - Menjaga pemisahan ini mencegah interpretasi yang ambigu dan
      memudahkan analisis feature importance.

    Nama kolom output: "{feature}_rate", mis. "AU01_c_rate".

    Parameters
    ----------
    df : pd.DataFrame
        Frame data yang sudah difilter quality.
    cols : List[str]
        Daftar nama kolom biner AU presence.

    Returns
    -------
    Dict[str, float]
        Dict berisi {"{feature}_rate": proporsi 0 - 1}.
    """
    result: Dict[str, float] = {}
    for col in cols:
        series = df[col].dropna()
        result[f"{col}_rate"] = float(series.mean()) if len(series) > 0 else np.nan
    return result


def aggregate_statistics(
    df_valid: pd.DataFrame,
    participant_id: str,
    quality_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Mengagregasi fitur temporal per partisipan dari frame yang lolos quality filter.

    Tiga kelompok fitur dikembalikan secara terpisah agar bisa disimpan ke
    file CSV yang berbeda (pose_features.csv, gaze_features.csv,
    expression_features.csv) sebelum digabung ke visual_features.csv.

    Desain split tiga grup ini mengikuti konvensi multimodal deep learning di mana
    masing-masing sub-modalitas (head pose, eye gaze, facial expression)
    sering diproses secara terpisah oleh stream/encoder yang berbeda.
    Quality summary disertakan di setiap grup agar setiap file CSV berdiri sendiri
    (self-contained) tanpa memerlukan join tambahan.

    Parameters
    ----------
    df_valid : pd.DataFrame
        Frame yang sudah difilter quality (output filter_quality()).
    participant_id : str
        ID partisipan (digunakan sebagai key pada merge antar modalitas).
    quality_stats : Dict[str, Any]
        Statistik kualitas dari filter_quality() untuk disematkan ke setiap baris.

    Returns
    -------
    Tuple[Dict, Dict, Dict]
        (pose_row, gaze_row, expression_row) — satu dict per grup fitur,
        masing-masing sudah menyertakan participant_id dan quality_stats.
    """
    base: Dict[str, Any] = {"participant_id": participant_id, **quality_stats}

    pose_row: Dict[str, Any] = {
        **base,
        **_aggregate_continuous(df_valid, POSE_COLS),
    }
    gaze_row: Dict[str, Any] = {
        **base,
        **_aggregate_continuous(df_valid, GAZE_COLS),
    }
    expression_row: Dict[str, Any] = {
        **base,
        **_aggregate_continuous(df_valid, AU_INTENSITY_COLS),
        **_aggregate_binary(df_valid, AU_PRESENCE_COLS),
    }

    return pose_row, gaze_row, expression_row


# =============================================================================
# 6. SEQUENCE BUILDING (untuk model temporal LSTM / Transformer)
# =============================================================================

def build_fixed_length_sequence(
    df_valid: pd.DataFrame,
    sequence_length: int = 300,
) -> np.ndarray:
    """
    Membangun representasi sekuensial dengan panjang tetap dari frame valid.

    Strategi resampling:
    - Jika jumlah frame > sequence_length : pilih frame secara merata
      (linspace-based indexing) - ekuivalen dengan temporal downsampling.
      Ini lebih baik dari hard-crop karena mempertahankan distribusi
      temporal secara keseluruhan (tidak bias ke awal/akhir sesi).
    - Jika jumlah frame < sequence_length : padding dengan nol di akhir
      (zero-padding post). Padding dilakukan sebelum normalisasi agar tidak
      memengaruhi statistik scaler; nilai 0 aman karena setelah normalisasi
      z-score, nilai 0 akan ter-transform ke -mean/std (bukan 0 absolut) -
      ini perlu dipertimbangkan saat masking di model Transformer.
    - Jika jumlah frame == 0 : kembalikan array nol penuh.

    Fitur yang digunakan: 31 kolom (6 pose + 8 gaze + 17 au_intensity).
    AU presence biner TIDAK disertakan di sequence karena activation_rate
    sudah merepresentasikan informasi biner tersebut di versi agregat.

    Parameters
    ----------
    df_valid : pd.DataFrame
        Frame valid (output filter_quality()).
    sequence_length : int
        Panjang target sequence (jumlah frame). Default 300.
        Sesuai baseline AVEC 2017.

    Returns
    -------
    np.ndarray
        Array shape (sequence_length, n_features) dtype float32.
    """
    n_features = len(SEQUENCE_FEATURE_COLS)
    n_frames = len(df_valid)

    if n_frames == 0:
        return np.zeros((sequence_length, n_features), dtype=np.float32)

    arr = df_valid[SEQUENCE_FEATURE_COLS].values.astype(np.float32)

    if n_frames >= sequence_length:
        # Temporal downsampling: ambil sequence_length frame secara merata
        indices = np.linspace(0, n_frames - 1, sequence_length, dtype=int)
        sequence = arr[indices]
    else:
        # Zero-padding di akhir (post-padding)
        pad_length = sequence_length - n_frames
        sequence = np.vstack([
            arr,
            np.zeros((pad_length, n_features), dtype=np.float32),
        ])

    return sequence  # shape: (sequence_length, n_features)


# =============================================================================
# 7. NORMALISASI (tanpa data leakage)
# =============================================================================

def normalize_features(
    df: pd.DataFrame,
    method: str = "standard",
    fit_mask: Optional[pd.Series] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Menormalisasi kolom numerik DataFrame menggunakan statistik yang di-fit
    HANYA dari baris training (fit_mask == True).

    Desain anti-data-leakage:
    - Scaler HANYA melihat baris training (fit_mask) untuk menghitung
      mean/std atau min/max.
    - Statistik tersebut kemudian diterapkan ke SEMUA baris (transform),
      termasuk dev dan test.
    - Ini mencegah informasi dari dev/test "bocor" ke pipeline training
      (Kapoor & Etchells, 2023, "Leakage and the reproducibility crisis
       in machine learning-based science").

    Kolom yang dikecualikan dari normalisasi: participant_id, kolom quality
    summary, dan kolom kategoris (quality_flag, status, error_message).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame visual_features.csv dengan satu baris per partisipan.
    method : str
        'standard' untuk StandardScaler (z-score, direkomendasikan untuk
        neural network karena asumsi distribusi normal lebih terpenuhi),
        'minmax' untuk MinMaxScaler (cocok untuk tree-based model).
    fit_mask : pd.Series, optional
        Boolean mask (index sesuai df) yang menandai baris training.
        Jika None, gunakan SEMUA baris (WARNING: potensi leakage!).
    exclude_cols : List[str], optional
        Kolom tambahan yang tidak dinormalisasi.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (df_normalized, scaler_params) - DataFrame ternormalisasi dan
        dict parameter scaler per kolom untuk disimpan ke JSON.
    """
    df_out = df.copy()

    # Kolom yang dikecualikan dari normalisasi secara default
    _exclude = set(exclude_cols or [])
    _exclude.update([
        "participant_id", "quality_flag", "status", "error_message",
        "n_total_frames", "n_valid_frames", "valid_frame_ratio",
        "mean_confidence", "session_duration_sec",
    ])

    numeric_cols = [
        c for c in df.columns
        if c not in _exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        warnings.warn("Tidak ada kolom numerik untuk dinormalisasi.")
        return df_out, {}

    if fit_mask is None:
        warnings.warn(
            "fit_mask tidak disediakan. Scaler akan di-fit dari SEMUA data. "
            "Ini berpotensi menyebabkan data leakage jika data belum dibagi split."
        )
        fit_mask = pd.Series(True, index=df.index)

    X_all   = df_out[numeric_cols].values
    X_train = df_out.loc[fit_mask, numeric_cols].values

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"method harus 'standard' atau 'minmax', bukan '{method}'.")

    scaler.fit(X_train)
    df_out[numeric_cols] = scaler.transform(X_all)

    # Kemas parameter scaler ke dict untuk disimpan ke JSON
    scaler_params: Dict[str, Any] = {"method": method, "columns": {}}
    for i, col in enumerate(numeric_cols):
        if method == "standard":
            scaler_params["columns"][col] = {
                "mean": float(scaler.mean_[i]),
                "std":  float(scaler.scale_[i]),
            }
        else:  # minmax
            scaler_params["columns"][col] = {
                "min":   float(scaler.data_min_[i]),
                "max":   float(scaler.data_max_[i]),
                "scale": float(scaler.scale_[i]),
            }

    return df_out, scaler_params


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

def save_outputs(
    pose_rows: List[Dict],
    gaze_rows: List[Dict],
    expression_rows: List[Dict],
    sequence_arrays: List[np.ndarray],
    sequence_ids: List[str],
    log_records: List[Dict],
    scaler_params: Dict,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Menyimpan semua output pipeline ke output_dir.

    Struktur output:
      {output_dir}/
      |- pose_features.csv
      |- gaze_features.csv
      |- expression_features.csv
      |- visual_features.csv
      |- visual_features_sequence.npy
      |- visual_features_sequence_index.csv
      |- visual_feature_scaler_params.json
      |- video_preprocessing_log.xlsx
      |- script_video_preprocessing.py   (salinan script ini)
      `- video_preprocessing_run.log     (log teks detail)

    Parameters
    ----------
    pose_rows, gaze_rows, expression_rows : List[Dict]
        Satu dict per partisipan dari aggregate_statistics().
    sequence_arrays : List[np.ndarray]
        Daftar array (sequence_length, n_features) per partisipan.
    sequence_ids : List[str]
        participant_id dalam urutan yang sama dengan sequence_arrays.
    log_records : List[Dict]
        Satu dict log per partisipan untuk disimpan ke Excel.
    scaler_params : Dict
        Output normalize_features(); dict kosong jika normalisasi di-skip.
    output_dir : str
        Direktori tujuan (dibuat otomatis jika belum ada).
    logger : logging.Logger
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- CSV: per-grup fitur -----------------------------------------------
    df_pose       = pd.DataFrame(pose_rows)       if pose_rows       else pd.DataFrame()
    df_gaze       = pd.DataFrame(gaze_rows)       if gaze_rows       else pd.DataFrame()
    df_expression = pd.DataFrame(expression_rows) if expression_rows else pd.DataFrame()

    df_pose.to_csv(os.path.join(output_dir, "pose_features.csv"), index=False)
    df_gaze.to_csv(os.path.join(output_dir, "gaze_features.csv"), index=False)
    df_expression.to_csv(os.path.join(output_dir, "expression_features.csv"), index=False)
    logger.info("Tersimpan: pose_features.csv, gaze_features.csv, expression_features.csv")

    # -- CSV: visual_features (gabungan semua grup + quality summary) ------
    quality_cols = [
        "participant_id", "n_total_frames", "n_valid_frames",
        "valid_frame_ratio", "mean_confidence", "session_duration_sec", "quality_flag",
    ]

    if not df_pose.empty and not df_gaze.empty and not df_expression.empty:
        # Kolom fitur murni (tanpa quality summary) untuk gaze dan expression
        # agar tidak ada duplikasi saat merge
        gaze_feat_cols       = [c for c in df_gaze.columns       if c not in quality_cols]
        expression_feat_cols = [c for c in df_expression.columns if c not in quality_cols]

        df_visual = (
            df_pose
            .merge(
                df_gaze[["participant_id"] + gaze_feat_cols],
                on="participant_id", how="outer",
            )
            .merge(
                df_expression[["participant_id"] + expression_feat_cols],
                on="participant_id", how="outer",
            )
        )
        df_visual.to_csv(os.path.join(output_dir, "visual_features.csv"), index=False)
        logger.info(
            f"Tersimpan: visual_features.csv — {len(df_visual)} partisipan, "
            f"{len(df_visual.columns)} kolom total."
        )
    else:
        logger.warning("Salah satu grup fitur kosong; visual_features.csv tidak dibuat.")

    # -- NPY + index CSV: sequence -----------------------------------------
    if sequence_arrays:
        seq_3d = np.stack(sequence_arrays, axis=0)  # (n_participants, seq_len, n_feat)
        npy_path = os.path.join(output_dir, "visual_features_sequence.npy")
        np.save(npy_path, seq_3d)
        logger.info(
            f"Tersimpan: visual_features_sequence.npy — shape {seq_3d.shape} "
            f"(n_participants={seq_3d.shape[0]}, seq_len={seq_3d.shape[1]}, "
            f"n_features={seq_3d.shape[2]})"
        )

        df_seq_idx = pd.DataFrame({
            "row_index":      range(len(sequence_ids)),
            "participant_id": sequence_ids,
        })
        df_seq_idx.to_csv(
            os.path.join(output_dir, "visual_features_sequence_index.csv"),
            index=False,
        )
        logger.info("Tersimpan: visual_features_sequence_index.csv")
    else:
        logger.warning(
            "sequence_arrays kosong. File .npy dan index CSV tidak dibuat."
        )

    # -- JSON: scaler params -----------------------------------------------
    if scaler_params:
        scaler_path = os.path.join(output_dir, "visual_feature_scaler_params.json")
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=2, ensure_ascii=False)
        logger.info("Tersimpan: visual_feature_scaler_params.json")
    else:
        logger.info(
            "scaler_params kosong (normalisasi di-skip). "
            "visual_feature_scaler_params.json TIDAK dibuat."
        )

    # -- XLSX: log terstruktur per partisipan ------------------------------
    if log_records:
        df_log = pd.DataFrame(log_records)
        log_cols_order = [
            "participant_id", "n_total_frames", "n_valid_frames",
            "valid_frame_ratio", "mean_confidence", "session_duration_sec",
            "quality_flag", "status", "error_message",
        ]
        # Tambahkan kolom tambahan yang tidak termasuk dalam urutan di atas
        extra_cols = [c for c in df_log.columns if c not in log_cols_order]
        df_log = df_log.reindex(columns=log_cols_order + extra_cols)

        log_xlsx = os.path.join(output_dir, "video_preprocessing_log.xlsx")
        df_log.to_excel(log_xlsx, index=False)
        logger.info(f"Tersimpan: video_preprocessing_log.xlsx — {len(df_log)} entri")


# =============================================================================
# MAIN - Entry Point
# =============================================================================

def main() -> None:
    """
    Entry point utama pipeline prapemrosesan visual E-DAIC.

    Alur eksekusi:
    1. Parse argumen CLI via argparse
    2. Setup logger (console + file)
    3. Scan file CSV OpenFace2 di input_dir (subfolder {id}_P/ atau flat)
    4. Per partisipan: load -> validate 53 kolom -> quality filter ->
       aggregate statistics -> build fixed-length sequence
    5. Normalisasi (jika file split AVEC 2017 tersedia di input_dir)
    6. Save semua output ke output_dir/Visual_Preprocessing/Output/

    Tentang struktur folder E-DAIC yang didukung:
    Script ini mencari file CSV di dua lokasi secara berurutan:
    (a) Subfolder {id}_P/features/ (struktur aktual E-DAIC):
        Contoh: E-DAIC/300_P/features/300_OpenFace2.1.0_Pose_gaze_AUs.csv
        Juga dicek langsung di {id}_P/ jika subfolder features/ tidak ada.
    (b) Langsung di input_dir (flat structure) sebagai fallback.
    Pola glob menangkap kedua format nama: OpenFace2.1.0 (titik) dan
    OpenFace2_1_0 (underscore).
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    DEFAULT_INPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "E-DAIC")
    DEFAULT_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "preprocessing_output")

    # -- Argparse ----------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline prapemrosesan modalitas visual E-DAIC "
            "dari file CSV OpenFace2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Direktori root E-DAIC yang berisi subfolder {id}_P/ "
            "atau file CSV OpenFace2 langsung."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Direktori tujuan. Output disimpan di "
            "{output_dir}/visual/."
        ),
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help=(
            "Ambang batas minimum nilai kolom `confidence` OpenFace2. "
            "Frame dengan confidence < threshold akan dibuang."
        ),
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.6,
        help=(
            "Jika valid_frame_ratio partisipan < nilai ini, tandai sebagai "
            "'low_quality' di log (partisipan TETAP diproses)."
        ),
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=300,
        help=(
            "Panjang tetap sequence (jumlah frame) untuk output .npy. "
            "Sesuai dengan baseline AVEC 2017 (300 frame ~ 10 detik @ 30 fps)."
        ),
    )
    parser.add_argument(
        "--scaler_method",
        type=str,
        default="standard",
        choices=["standard", "minmax"],
        help=(
            "'standard' (z-score, direkomendasikan untuk neural network) atau "
            "'minmax' (skala 0-1, cocok untuk tree-based model)."
        ),
    )

    args = parser.parse_args()

    # -- Tentukan direktori output final -----------------------------------
    final_output_dir = os.path.join(
        args.output_dir, "visual"
    )
    os.makedirs(final_output_dir, exist_ok=True)

    # -- Setup logging -----------------------------------------------------
    logger = setup_logging(final_output_dir)
    logger.info("=" * 70)
    logger.info("PIPELINE PRAPEMROSESAN VISUAL E-DAIC - MULAI")
    logger.info(f"  input_dir           : {args.input_dir}")
    logger.info(f"  output_dir (final)  : {final_output_dir}")
    logger.info(f"  confidence_threshold: {args.confidence_threshold}")
    logger.info(f"  min_valid_ratio     : {args.min_valid_ratio}")
    logger.info(f"  sequence_length     : {args.sequence_length}")
    logger.info(f"  scaler_method       : {args.scaler_method}")
    logger.info("=" * 70)

    # -- Scan file CSV OpenFace2 -------------------------------------------
    csv_files: List[Tuple[str, str]] = []  # [(participant_id, file_path), ...]

    if not os.path.exists(args.input_dir):
        logger.error(f"input_dir tidak ditemukan: {args.input_dir}")
        sys.exit(1)

    # Struktur aktual E-DAIC:
    #   {input_dir}/{id}_P/features/{id}_OpenFace2.1.0_Pose_gaze_AUs.csv
    # Pola glob "*OpenFace*Pose_gaze_AUs.csv" menangkap kedua format nama:
    #   - OpenFace2.1.0  (titik, format dataset asli)
    #   - OpenFace2_1_0  (underscore, format alternatif)
    OPENFACE_GLOB = "*OpenFace*Pose_gaze_AUs.csv"

    # (a) Cari di dalam subfolder {id}_P/features/ (struktur E-DAIC asli)
    for entry in os.listdir(args.input_dir):
        entry_path = os.path.join(args.input_dir, entry)
        if os.path.isdir(entry_path) and entry.endswith("_P"):
            # Cek subfolder features/ terlebih dahulu, lalu fallback ke entry_path
            features_subdir = os.path.join(entry_path, "features")
            search_dirs = [features_subdir, entry_path]  # prioritas: features/ dulu
            found_for_this_participant = False
            for search_dir in search_dirs:
                if not os.path.isdir(search_dir):
                    continue
                matches = glob.glob(os.path.join(search_dir, OPENFACE_GLOB))
                for m in matches:
                    fname = os.path.basename(m)
                    # participant_id = prefix sebelum "_OpenFace"
                    pid = fname.split("_OpenFace")[0]
                    csv_files.append((pid, m))
                    found_for_this_participant = True
                # Jika sudah ditemukan di search_dir ini, jangan cek fallback
                if found_for_this_participant:
                    break

    # (b) Fallback: cari langsung di input_dir (flat structure)
    if not csv_files:
        for m in glob.glob(os.path.join(args.input_dir, OPENFACE_GLOB)):
            fname = os.path.basename(m)
            pid = fname.split("_OpenFace")[0]
            csv_files.append((pid, m))

    # Hapus duplikat dan urutkan numerik
    seen: set = set()
    csv_files_unique: List[Tuple[str, str]] = []
    for pid, fpath in csv_files:
        if fpath not in seen:
            seen.add(fpath)
            csv_files_unique.append((pid, fpath))

    def _sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        pid = item[0]
        return (0, pid) if not pid.isdigit() else (1, pid.zfill(10))

    csv_files = sorted(csv_files_unique, key=_sort_key)

    logger.info(f"Ditemukan {len(csv_files)} file CSV OpenFace2 untuk diproses.")

    if not csv_files:
        logger.error(
            "Tidak ada file CSV ditemukan. Periksa --input_dir dan pastikan "
            "file bernama '*_OpenFace2_1_0_Pose_gaze_AUs.csv' tersedia."
        )
        sys.exit(1)

    # -- Proses per partisipan --------------------------------------------
    pose_rows:       List[Dict] = []
    gaze_rows:       List[Dict] = []
    expression_rows: List[Dict] = []
    sequence_arrays: List[np.ndarray] = []
    sequence_ids:    List[str] = []
    log_records:     List[Dict] = []

    for idx, (participant_id, file_path) in enumerate(csv_files, start=1):
        logger.info(
            f"[{idx:3d}/{len(csv_files)}] Memproses: {participant_id} | "
            f"{os.path.basename(file_path)}"
        )

        # Inisialisasi log record
        log_rec: Dict[str, Any] = {
            "participant_id":       participant_id,
            "n_total_frames":       0,
            "n_valid_frames":       0,
            "valid_frame_ratio":    0.0,
            "mean_confidence":      0.0,
            "session_duration_sec": 0.0,
            "quality_flag":         "unknown",
            "status":               "failed",
            "error_message":        "",
        }

        try:
            # LANGKAH 1: Load & Validate
            df_raw, load_err = load_openface_csv(file_path, logger)
            if df_raw is None:
                log_rec["error_message"] = load_err
                log_records.append(log_rec)
                logger.warning(
                    f"  Skip {participant_id}: validasi kolom gagal."
                )
                continue

            # LANGKAH 2: Quality Filtering
            df_valid, quality_stats = filter_quality(
                df_raw,
                participant_id,
                args.confidence_threshold,
                args.min_valid_ratio,
                logger,
            )
            log_rec.update(quality_stats)

            # LANGKAH 3 & 4: Agregasi Statistik
            pose_row, gaze_row, expression_row = aggregate_statistics(
                df_valid, participant_id, quality_stats
            )
            pose_rows.append(pose_row)
            gaze_rows.append(gaze_row)
            expression_rows.append(expression_row)

            # LANGKAH 6: Build Fixed-Length Sequence
            seq = build_fixed_length_sequence(df_valid, args.sequence_length)
            sequence_arrays.append(seq)
            sequence_ids.append(participant_id)

            log_rec["status"]        = "success"
            log_rec["error_message"] = ""
            logger.info(
                f"  [{participant_id}] OK | "
                f"{quality_stats['n_valid_frames']}/{quality_stats['n_total_frames']} "
                f"frame valid | flag={quality_stats['quality_flag']}"
            )

        except Exception as exc:
            log_rec["status"]        = "failed"
            log_rec["error_message"] = str(exc)
            logger.error(
                f"  [ERROR] {participant_id}: {exc}\n{traceback.format_exc()}"
            )

        finally:
            log_records.append(log_rec)

    # Ringkasan
    n_success = sum(1 for r in log_records if r["status"] == "success")
    n_fail    = sum(1 for r in log_records if r["status"] == "failed")
    n_lowq    = sum(
        1 for r in log_records
        if r.get("quality_flag") == "low_quality" and r["status"] == "success"
    )
    logger.info(
        f"\nRingkasan: {n_success} sukses "
        f"({n_lowq} low_quality tetap diproses), "
        f"{n_fail} gagal, dari total {len(csv_files)} partisipan."
    )

    if not pose_rows:
        logger.error(
            "Tidak ada partisipan yang berhasil diproses. "
            "Pipeline dihentikan; log tetap disimpan."
        )
        save_outputs([], [], [], [], [], log_records, {}, final_output_dir, logger)
        sys.exit(1)

    # -- LANGKAH 7: Normalisasi -------------------------------------------
    scaler_params: Dict = {}

    # Periksa ketersediaan file split E-DAIC
    split_train_path = os.path.join(
        args.input_dir, "train_split_Depression_AVEC2017.csv"
    )
    train_split_found = os.path.exists(split_train_path)

    if train_split_found:
        logger.info(f"File split training ditemukan: {split_train_path}")
        df_train_split = pd.read_csv(split_train_path)
        train_ids = set(
            df_train_split["Participant_ID"].astype(str).str.strip().tolist()
        )

        # Buat DataFrame gabungan untuk normalisasi
        # Kolom quality summary yang tidak dinormalisasi sudah ditangani
        # oleh fungsi normalize_features() melalui _exclude
        df_all_poses = pd.DataFrame(pose_rows)
        all_pids = df_all_poses["participant_id"].astype(str)
        fit_mask = all_pids.isin(train_ids).values
        fit_mask_series = pd.Series(fit_mask, index=df_all_poses.index)

        logger.info(
            f"Fit mask normalisasi: {fit_mask.sum()} partisipan training "
            f"dari {len(df_all_poses)} total."
        )

        # Normalisasi masing-masing DataFrame
        df_pose_norm, scaler_params = normalize_features(
            df_all_poses, method=args.scaler_method, fit_mask=fit_mask_series
        )
        df_gaze_norm, _ = normalize_features(
            pd.DataFrame(gaze_rows), method=args.scaler_method, fit_mask=fit_mask_series
        )
        df_expr_norm, _ = normalize_features(
            pd.DataFrame(expression_rows), method=args.scaler_method, fit_mask=fit_mask_series
        )

        pose_rows       = df_pose_norm.to_dict(orient="records")
        gaze_rows       = df_gaze_norm.to_dict(orient="records")
        expression_rows = df_expr_norm.to_dict(orient="records")

        logger.info(
            "Normalisasi selesai. Fitur ternormalisasi menggunakan "
            "statistik training set."
        )
    else:
        logger.warning(
            "File 'train_split_Depression_AVEC2017.csv' TIDAK ditemukan di "
            f"input_dir ({args.input_dir}). Langkah normalisasi di-skip. "
            "Data mentah (unnormalized) akan disimpan. Normalisasi dapat "
            "dilakukan ulang setelah file split tersedia."
        )

    # -- LANGKAH 5 & 8: Save Outputs --------------------------------------
    save_outputs(
        pose_rows=pose_rows,
        gaze_rows=gaze_rows,
        expression_rows=expression_rows,
        sequence_arrays=sequence_arrays,
        sequence_ids=sequence_ids,
        log_records=log_records,
        scaler_params=scaler_params,
        output_dir=final_output_dir,
        logger=logger,
    )

    # Salin script ini ke output_dir untuk dokumentasi reprodusibilitas
    script_src  = os.path.abspath(__file__)
    script_dest = os.path.join(final_output_dir, "script_video_preprocessing.py")
    if os.path.abspath(script_src) != os.path.abspath(script_dest):
        try:
            shutil.copy2(script_src, script_dest)
            logger.info(f"Script disalin ke output: {script_dest}")
        except Exception as e:
            logger.warning(f"Gagal menyalin script: {e}")

    logger.info("=" * 70)
    logger.info("PIPELINE PRAPEMROSESAN VISUAL E-DAIC - SELESAI")
    logger.info(f"Semua output tersimpan di: {final_output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
