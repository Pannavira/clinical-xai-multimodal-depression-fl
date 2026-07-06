# Integrasi Explainable AI Klinis pada Multimodal Deep Learning untuk Deteksi Depresi Berbasis Federated Learning (MODELING)

Proyek ini bertujuan untuk membangun sistem deteksi depresi berbasis **Multimodal Deep Learning** (mengintegrasikan modalitas Teks, Audio, dan Visual) menggunakan dataset **E-DAIC** (Extended Distress Analysis Interview Corpus) dalam kerangka kerja **Federated Learning** (Pembelajaran Terfederasi) serta dilengkapi dengan **Explainable AI (XAI)** klinis untuk memberikan interpretasi keputusan model yang dapat dipertanggungjawabkan dalam konteks medis.

---

## Daftar Isi

- [Overview Proyek](#overview-proyek)
- [Struktur Direktori Utama](#struktur-direktori-utama)
- [Instalasi & Persiapan Lingkungan](#instalasi--persiapan-lingkungan)
  - [1. Kloning Repositori](#1-kloning-repositori)
  - [2. Membuat Virtual Environment](#2-membuat-virtual-environment)
  - [3. Aktivasi Virtual Environment](#3-aktivasi-virtual-environment)
  - [4. Install Library / Dependensi](#4-install-library--dependensi)
  - [5. Konfigurasi Token Hugging Face](#5-konfigurasi-token-hugging-face)
- [Pipeline Prapemrosesan (Preprocessing)](#pipeline-prapemrosesan-preprocessing)
  - [1. Prapemrosesan Teks (Text Preprocessing)](#1-prapemrosesan-teks-text-preprocessing)
  - [2. Prapemrosesan Audio (Audio Preprocessing)](#2-prapemrosesan-audio-audio-preprocessing)
  - [3. Prapemrosesan Visual (Visual Preprocessing)](#3-prapemrosesan-visual-visual-preprocessing)

---

## Overview Proyek

Sistem ini dirancang untuk mendeteksi depresi melalui analisis wawancara klinis virtual (Avatar Ellie pada dataset DAIC-WOZ/E-DAIC). Untuk melindungi privasi pasien/partisipan, proyek ini dirancang agar dapat dilatih secara terdistribusi menggunakan **Federated Learning**. Model akhir dilengkapi dengan metode **Explainable AI (XAI)** untuk menjelaskan fitur akustik, linguistik, dan visual mana yang paling berkontribusi terhadap diagnosis depresi.

### Fitur Utama Modalitas:

- **Teks (Linguistik):** Transkrip wawancara dibersihkan, ditokenisasi, dan diekstraksi representasi semantiknya menggunakan model bahasa klinis khusus **MentalBERT** (`mental/mental-bert-base-uncased`).
- **Audio (Akustik):** Sinyal audio distandarisasi (16 kHz, mono), dibersihkan dari noise menggunakan spectral gating, disegmentasikan berdasarkan keaktifan suara (RMS threshold), dan diekstraksi fiturnya (MFCC, Spectrogram, F0/pitch, speech rate, voiced duration).
- **Visual (Ekspresi & Pose):** Mengekstrak fitur gerak pose wajah, arah tatapan mata (_gaze_), serta intensitas dan kemunculan gerakan otot wajah (_Action Units_ - AU) menggunakan data OpenFace 2.x, kemudian diselaraskan menjadi sekuens temporal dengan panjang tetap untuk pemodelan deret waktu.

---

## Struktur Direktori Utama

```text
clinical-xai-multimodal-depression-fl/
│
├── data/                                 # Dataset mentah & Split Data
│   ├── E-DAIC/                           # Berisi subfolder partisipan (misal: 300_P, 301_P, dll.)
│   ├── detailed_lables.csv               # Label depresi (ground-truth) partisipan
│   ├── train_split_Depression_AVEC2017.csv
│   └── test_split_Depression_AVEC2017.csv
│
├── preprocessing/                        # Kode sumber prapemrosesan data
│   ├── text/                             # Pemrosesan transkrip & MentalBERT embeddings
│   ├── audio/                            # Standardisasi, denoising, segmentasi, & ekstraksi fitur audio
│   └── visual/                           # Pemrosesan fitur pose, gaze, & Action Units dari OpenFace
│
├── preprocessing_output/                 # Output hasil prapemrosesan (dibuat otomatis)
│   ├── text/
│   ├── audio/
│   └── visual/
│
├── modeling/                             # Pemodelan Federated Learning (dalam pengembangan)
├── xai/                                  # Metode Explainable AI klinis (dalam pengembangan)
│
├── .env                                  # Berkas konfigurasi rahasia (token HF, dll.)
├── .env.example                          # Template konfigurasi berkas .env
├── requirements.txt                      # Daftar pustaka Python yang diperlukan
└── README.md                             # Dokumentasi proyek (berkas ini)
```

---

## Instalasi & Persiapan Lingkungan

Ikuti langkah-langkah di bawah ini untuk memasang proyek ini di komputer lokal Anda.

### 1. Kloning Repositori

Clone proyek ini dari GitHub terlebih dahulu, lalu masuk ke direktori proyek:

```bash
git clone https://github.com/Pannavira/clinical-xai-multimodal-depression-fl.git
cd clinical-xai-multimodal-depression-fl
```

### 2. Membuat Virtual Environment

Buat lingkungan virtual Python (`.venv`) agar pustaka proyek tidak bentrok dengan pustaka sistem global Anda.

- **Windows:**
  ```powershell
  python -m venv .venv
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  ```

### 3. Aktivasi Virtual Environment

Aktifkan lingkungan virtual yang telah dibuat sebelum menginstal pustaka apa pun.

- **Windows (Command Prompt / PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux / macOS:**
  `bash
source .venv/bin/activate
`
  Setelah aktif, indikator `(.venv)` akan muncul di sebelah kiri baris perintah terminal Anda.

### 4. Install Library / Dependensi

Pasang semua pustaka yang tertera di [requirements.txt](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> Jika Anda menggunakan GPU Nvidia dan ingin memanfaatkan akselerasi CUDA untuk model MentalBERT, pastikan versi `torch` yang terinstal sudah mendukung CUDA. Kunjungi [pytorch.org](https://pytorch.org/) untuk detail perintah instalasi PyTorch CUDA yang sesuai dengan sistem Anda.

### 5. Konfigurasi Token Hugging Face

Model **MentalBERT** (`mental/mental-bert-base-uncased`) merupakan model gated. Anda memerlukan Hugging Face Access Token dengan hak akses **Read** untuk mengunduhnya.

1.  Dapatkan token Anda di: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  Salin berkas `.env.example` menjadi `.env`:
    ```bash
    cp .env.example .env
    ```
    _(Pada Windows PowerShell, gunakan perintah: `copy .env.example .env`)_
3.  Buka berkas `.env` dan masukkan token Anda pada variabel `HF_TOKEN`:
    ```env
    HF_TOKEN=hf_masukkan_token_huggingface_anda_di_sini
    ```

---

## Pipeline Prapemrosesan (Preprocessing)

Fase prapemrosesan membagi tugas pengolahan data ke dalam 3 modalitas utama secara terstruktur. Seluruh script prapemrosesan terletak di folder [preprocessing/](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing).

### 1. Prapemrosesan Teks (Text Preprocessing)

Folder: [preprocessing/text/](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/text)

Sub-pipeline teks bertanggung jawab untuk membersihkan transkrip wawancara mentah, mencocokkan ID partisipan dengan label tingkat keparahan depresi (`detailed_lables.csv`), dan menghasilkan representasi vektor kata (_embedding_) menggunakan MentalBERT.

- **[preprocessing_all_text.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/text/preprocessing_all_text.py)**:
  Script utama untuk memproses transkrip seluruh partisipan di folder `data/E-DAIC`. Teks digabungkan, dinormalisasi ke huruf kecil, dibersihkan dari spasi ganda, ditokenisasi, lalu dikonversi menjadi embedding CLS 768-dimensi. Output disimpan ke:
  - `preprocessing_output/text/text_embeddings_all.csv`
  - `preprocessing_output/text/text_embeddings_all.xlsx`
- **[text_embeddings.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/text/text_embeddings.py)**:
  Melakukan ekstraksi embedding tingkat lanjut menggunakan tokenisasi terstandar (panjang sekuens maks 512 token) dengan representasi CLS MentalBERT.
- **[script_text_tokenization.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/text/script_text_tokenization.py)**:
  Menjalankan proses uji coba tokenisasi untuk verifikasi kualitas token, menghitung jumlah token riil, rasio pemotongan kalimat (_truncation rate_), dan memvalidasi kebenaran input teks.
- **[preprocessing_text.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/text/preprocessing_text.py)**:
  Script simulasi prapemrosesan teks tunggal untuk pengujian cepat alur integrasi tanpa memproses seluruh dataset.

**Cara Menjalankan:**

```bash
# Memproses transkrip semua subjek
python preprocessing/text/preprocessing_all_text.py

# Ekstraksi embedding tingkat lanjut
python preprocessing/text/text_embeddings.py
```

---

### 2. Prapemrosesan Audio (Audio Preprocessing)

Folder: [preprocessing/audio/](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/audio)

Sub-pipeline audio mempersiapkan data audio wawancara untuk ekstraksi fitur akustik klinis dengan meminimalkan noise lingkungan dan memisahkan rekaman suara panjang menjadi klip suara aktif berdurasi pendek.

- **[audio_standarization.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/audio/audio_standarization.py)**:
  Membaca audio mentah `{p_num}_AUDIO.wav`, melakukan resampling paksa ke **16.000 Hz (mono)**, dan mereduksi noise latar belakang menggunakan algoritma spectral gating dari library `noisereduce`. Hasil disimpan di:
  - `preprocessing_output/audio/standardized_audio/`
  - `preprocessing_output/audio/denoised_audio/`
- **[audio_segmentation.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/audio/audio_segmentation.py)**:
  Memotong audio bersih (_denoised_) menjadi segmen-segmen pendek berdurasi **10 detik**. Bagian audio yang hening disaring menggunakan detektor energi RMS (Ambang Batas RMS < 0.001) agar hanya menyisakan klip suara aktif. Hasil disimpan di:
  - `preprocessing_output/audio/segmented_audio/{participant_id}/`
- **[audio_feature_extraction.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/audio/audio_feature_extraction.py)**:
  Melakukan ekstraksi fitur akustik komprehensif pada tingkat frame dan tingkat segmen. Fitur yang diekstraksi meliputi:
  - **MFCC** (13 koefisien + delta + delta-delta)
  - **Log-Mel Spectrogram**
  - **F0 / Pitch** (Frekuensi fundamental menggunakan algoritma YIN, statistik min, max, mean, std)
  - **Metrik Kecepatan Bicara** (_Speech Rate_, hitungan suku kata, durasi bersuara/_voiced duration_)
- **[heal_audio_features.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/audio/heal_audio_features.py)**:
  Utilitas untuk memulihkan atau mengekstrak ulang fitur audio jika terjadi kegagalan/error di tengah jalan, memastikan integritas seluruh subjek terjaga.

**Cara Menjalankan:**

```bash
# 1. Standardisasi dan Noise Reduction
python preprocessing/audio/audio_standarization.py

# 2. Segmentasi Audio Suara Aktif
python preprocessing/audio/audio_segmentation.py

# 3. Ekstraksi Fitur Akustik (MFCC, Spectrogram, F0)
python preprocessing/audio/audio_feature_extraction.py
```

---

### 3. Prapemrosesan Visual (Visual Preprocessing)

Folder: [preprocessing/visual/](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/visual)

Sub-pipeline visual berfokus pada data deskriptor wajah yang diekstrak oleh OpenFace 2.x. Script ini memproses file CSV OpenFace untuk menghasilkan agregat statistik fitur ekspresi wajah dan poses temporal.

- **[script_video_preprocessing.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/preprocessing/visual/script_video_preprocessing.py)**:
  Membaca file CSV OpenFace (`{participant_id}_OpenFace2_1_0_Pose_gaze_AUs.csv`) yang berisi koordinat pose kepala, arah pandangan mata (_gaze_), serta intensitas/keberadaan _Action Units_ (AU) ekspresi wajah.
  - **Filter Kualitas:** Menyaring frame dengan tingkat kepercayaan OpenFace di bawah ambang batas (default: >0.70).
  - **Agregasi Temporal:** Menghitung statistik deskriptif (mean, std, persentil, range) untuk sinyal kontinu (gaze/pose/intensitas AU) dan activation rate untuk fitur biner (AU presence).
  - **Penyelarasan Panjang Tetap:** Melakukan resampling deret waktu menjadi panjang sekuens tetap **300 frame** untuk model temporal (LSTM/Transformer).
  - **Pencegahan Kebocoran Data (Data Leakage):** Parameter normalisasi (scaler) di-fit **hanya** menggunakan data set training (sesuai train split), lalu diterapkan pada data test/validation.
  - **Output Berkas:**
    - `visual_features.csv` (gabungan fitur statistik teragregasi)
    - `visual_features_sequence.npy` (array NumPy 3D siap latih untuk LSTM/Transformer)
    - `visual_feature_scaler_params.json` (parameter penskalaan tersimpan)

**Cara Menjalankan:**

```bash
python preprocessing/visual/script_video_preprocessing.py \
    --input_dir "data/E-DAIC" \
    --output_dir "preprocessing_output" \
    --confidence_threshold 0.7 \
    --sequence_length 300 \
    --scaler_method standard
```

_(Catatan: Dataset E-DAIC tidak menyediakan file video raw sehingga, Anda tidak memerlukan file video mentah `.mp4`/`.avi` untuk menjalankan script ini. Script bekerja secara efisien dengan mengolah file CSV OpenFace yang sudah disediakan di folder fitur masing-masing subjek)._
