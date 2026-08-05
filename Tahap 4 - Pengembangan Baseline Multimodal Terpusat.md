# Tahap 4 - Pengembangan Baseline Multimodal Terpusat

## 1\. Tujuan Tahap Ini

Tujuan Tahap 4 adalah membangun model pembanding awal berbasis **multimodal deep learning secara terpusat**. Model ini disebut baseline karena akan menjadi titik acuan untuk menilai apakah model Federated Learning pada tahap berikutnya masih memiliki performa yang kompetitif.

Pada tahap ini, seluruh data training digunakan secara terpusat dalam satu lingkungan pelatihan. Pendekatan ini **bukan model final penelitian**, tetapi dipakai sebagai pembanding ilmiah.

Tahap ini mencakup tujuh pekerjaan utama:

- Menyiapkan dataset centralized dari hasil partisi.
- Membangun baseline unimodal untuk teks, audio, dan visual.
- Membangun baseline bimodal.
- Membangun baseline multimodal penuh.
- Menentukan strategi fusi multimodal.
- Melakukan training dan hyperparameter tuning.
- Mengevaluasi performa baseline sebagai benchmark untuk tahap Federated Learning.

## 2\. Input dan Output Tahap 4

### 2.1 Input dari Tahap 3

Tahap 4 menggunakan hasil Tahap 3, terutama global split dan indeks fitur multimodal.

| **Input**                               | **Fungsi**                     |
| --------------------------------------- | ------------------------------ |
| global_train_index.csv                  | data training centralized      |
| global_validation_index.csv             | data validasi centralized      |
| global_test_index.csv                   | data uji akhir                 |
| multimodal_feature_index.csv            | indeks fitur teks-audio-visual |
| final_text_features/                    | fitur atau embedding teks      |
| final_audio_features/                   | fitur audio                    |
| final_visual_features/                  | fitur visual                   |
| label_distribution_report.xlsx          | distribusi label               |
| federated_partition_quality_report.docx | bukti validitas split          |
| data_dictionary.xlsx                    | definisi variabel              |

### 2.2 Output Akhir Tahap 4

Pada akhir tahap ini, Bapak/Ibu harus memiliki:

| **Output**              | **Nama File/Dokumen yang Disarankan**                                     |
| ----------------------- | ------------------------------------------------------------------------- |
| Protokol baseline model | centralized_baseline_protocol.docx                                        |
| Konfigurasi eksperimen  | centralized_baseline_config.yaml                                          |
| Model text-only         | text_only_baseline.pt                                                     |
| Model audio-only        | audio_only_baseline.pt                                                    |
| Model visual-only       | visual_only_baseline.pt                                                   |
| Model bimodal           | text_audio_baseline.pt, text_visual_baseline.pt, audio_visual_baseline.pt |
| Model multimodal penuh  | centralized_multimodal_baseline.pt                                        |
| Training log            | centralized_training_log.csv                                              |
| Hyperparameter log      | hyperparameter_tuning_log.xlsx                                            |
| Tabel metrik performa   | centralized_baseline_metrics.xlsx                                         |
| Confusion matrix        | confusion_matrix_baseline.png                                             |
| ROC curve               | roc_curve_baseline.png                                                    |
| PR curve                | pr_curve_baseline.png                                                     |
| Laporan baseline        | centralized_baseline_report.docx                                          |

## 3\. Struktur Folder Tahap 4

Gunakan struktur folder berikut:

Tahap_6_Pengembangan_Baseline_Multimodal_Terpusat/  
│  
├── 01_Input_From_Partition/  
│ ├── global_train_index.csv  
│ ├── global_validation_index.csv  
│ ├── global_test_index.csv  
│ ├── multimodal_feature_index.csv  
│ ├── final_text_features/  
│ ├── final_audio_features/  
│ ├── final_visual_features/  
│ └── data_dictionary.xlsx  
│  
├── 02_Protocol_And_Config/  
│ ├── centralized_baseline_protocol.docx  
│ ├── centralized_baseline_config.yaml  
│ ├── model_architecture_plan.docx  
│ └── random_seed_record.txt  
│  
├── 03_Unimodal_Baseline/  
│ ├── Text_Only/  
│ ├── Audio_Only/  
│ └── Visual_Only/  
│  
├── 04_Bimodal_Baseline/  
│ ├── Text_Audio/  
│ ├── Text_Visual/  
│ └── Audio_Visual/  
│  
├── 05_Multimodal_Baseline/  
│ ├── Late_Fusion/  
│ ├── Cross_Attention_Optional/  
│ ├── model_checkpoints/  
│ └── training_logs/  
│  
├── 06_Hyperparameter_Tuning/  
│ ├── tuning_config.yaml  
│ ├── hyperparameter_tuning_log.xlsx  
│ └── best_model_selection.xlsx  
│  
├── 07_Evaluation/  
│ ├── centralized_baseline_metrics.xlsx  
│ ├── confusion_matrix/  
│ ├── roc_curve/  
│ ├── pr_curve/  
│ ├── error_analysis.xlsx  
│ └── evaluation_summary.docx  
│  
├── 08_Scripts/  
│ ├── dataset_loader.py  
│ ├── text_encoder.py  
│ ├── audio_encoder.py  
│ ├── visual_encoder.py  
│ ├── multimodal_fusion.py  
│ ├── train_centralized.py  
│ ├── evaluate_model.py  
│ └── plot_metrics.py  
│  
└── 09_Final_Output/  
├── centralized_multimodal_baseline.pt  
├── centralized_baseline_report.docx  
├── baseline_summary_for_report.docx  
└── README_baseline_experiment.md

## 4\. Langkah Detail yang Harus Dikerjakan

### Langkah 4.1 - Menyiapkan Protokol Baseline Terpusat

Sebelum membuat model, buat dokumen protokol agar eksperimen baseline dapat direplikasi.

**Dokumen yang dibuat**

centralized_baseline_protocol.docx

**Isi dokumen**

| **Bagian**             | **Isi**                                         |
| ---------------------- | ----------------------------------------------- |
| Tujuan baseline        | menjelaskan fungsi baseline sebagai benchmark   |
| Jenis model            | unimodal, bimodal, multimodal                   |
| Input data             | fitur teks, audio, visual                       |
| Strategi training      | centralized training                            |
| Strategi fusi          | late fusion sebagai eksperimen utama            |
| Cross-attention        | opsional jika waktu dan komputasi cukup         |
| Split data             | train, validation, global test                  |
| Metrik evaluasi        | accuracy, precision, recall, F1-score, AUC      |
| Pencegahan leakage     | global test tidak dipakai saat tuning           |
| Kriteria model terbaik | F1-score/AUC tertinggi pada validation set      |
| Output                 | model checkpoint, tabel metrik, grafik evaluasi |
| Penanggung jawab       | ketua, anggota, mahasiswa terkait               |

**Prinsip penting**

Global test set hanya digunakan **sekali untuk evaluasi akhir**, bukan untuk tuning. Tuning dilakukan dengan validation set agar hasil evaluasi tidak bias.

### Langkah 4.2 - Menyiapkan Konfigurasi Eksperimen

Buat file konfigurasi agar eksperimen mudah diulang.

**File yang dibuat**

centralized_baseline_config.yaml

**Contoh isi**

project_name: FED-MIND  
experiment_stage: centralized_multimodal_baseline  
<br/>data:  
train_index: global_train_index.csv  
validation_index: global_validation_index.csv  
test_index: global_test_index.csv  
id_column: participant_id  
label_column: depression_label  
text_feature_column: text_feature_path  
audio_feature_column: audio_feature_path  
visual_feature_column: visual_feature_path  
<br/>model:  
text_encoder: bert_embedding_mlp  
audio_encoder: cnn_or_mlp  
visual_encoder: cnn_or_mlp  
fusion_strategy: late_fusion  
classifier: mlp  
dropout: 0.3  
<br/>training:  
batch_size: 16  
epochs: 50  
learning_rate: 0.0001  
optimizer: adam  
loss_function: binary_cross_entropy  
early_stopping: true  
patience: 10  
random_seed: 42  
<br/>evaluation:  
metrics:  
\- accuracy  
\- precision  
\- recall  
\- f1_score  
\- auc_roc  
threshold: 0.5

**Yang harus didokumentasikan**

- versi Python;
- versi library;
- jenis GPU/CPU;
- batch size;
- epoch;
- learning rate;
- optimizer;
- loss function;
- random seed;
- strategi early stopping;
- strategi pemilihan model terbaik.

### Langkah 4.3 - Memverifikasi Data Input Baseline

**Tujuan**

Memastikan data training, validation, dan test sudah siap digunakan oleh model.

**Langkah teknis**

- Buka global_train_index.csv, global_validation_index.csv, dan global_test_index.csv.
- Pastikan tidak ada overlap participant_id.
- Pastikan setiap data memiliki fitur teks, audio, visual, dan label.
- Pastikan path fitur dapat dibaca.
- Periksa distribusi label pada masing-masing split.
- Periksa dimensi fitur setiap modalitas.
- Tandai data yang bermasalah sebelum training.

**Dokumen yang dibuat**

centralized_input_verification_log.xlsx

**Format tabel**

| **participant_id** | **split**  | **text_valid** | **audio_valid** | **visual_valid** | **label_valid** | **status**       |
| ------------------ | ---------- | -------------- | --------------- | ---------------- | --------------- | ---------------- |
| P001               | train      | 1              | 1               | 1                | 1               | valid            |
| P002               | validation | 1              | 1               | 1                | 1               | valid            |
| P003               | test       | 1              | 0               | 1                | 1               | excluded/warning |

## 5\. Membangun Baseline Unimodal

Baseline unimodal penting untuk mengetahui kontribusi masing-masing modalitas. Dengan eksperimen ini, Bapak/Ibu dapat menjawab pertanyaan: **modalitas mana yang paling informatif untuk deteksi depresi?**

### Langkah 4.4 - Text-Only Baseline

**Tujuan**

Mengukur performa model hanya menggunakan fitur teks.

**Input**

| **Input**         | **Keterangan** |
| ----------------- | -------------- |
| text_feature_path | embedding teks |
| depression_label  | label depresi  |

**Arsitektur sederhana**

Text Embedding → Dense/MLP → Dropout → Classifier → Output

Jika embedding sudah berasal dari BERT, model tidak perlu terlalu kompleks. Gunakan MLP sebagai classifier awal.

**Output**

Text_Only/  
├── text_only_baseline.pt  
├── text_only_training_log.csv  
├── text_only_metrics.xlsx  
├── text_only_confusion_matrix.png  
└── text_only_report.docx

**Yang harus dicatat**

- dimensi embedding teks;
- panjang token maksimum jika relevan;
- model embedding yang digunakan;
- F1-score;
- recall;
- AUC;
- kesalahan prediksi.

### Langkah 4.5 - Audio-Only Baseline

**Tujuan**

Mengukur kemampuan fitur audio untuk mendeteksi depresi.

**Input**

| **Input**          | **Keterangan**                   |
| ------------------ | -------------------------------- |
| audio_feature_path | MFCC, pitch, energy, spectrogram |
| depression_label   | label depresi                    |

**Arsitektur pilihan**

| **Jenis Input Audio**           | **Model yang Cocok**        |
| ------------------------------- | --------------------------- |
| fitur tabular MFCC/pitch/energy | MLP                         |
| spectrogram                     | CNN                         |
| sequence audio feature          | LSTM/GRU/Transformer ringan |

Untuk baseline awal, gunakan MLP untuk fitur numerik atau CNN sederhana untuk spectrogram.

**Output**

Audio_Only/  
├── audio_only_baseline.pt  
├── audio_only_training_log.csv  
├── audio_only_metrics.xlsx  
├── audio_only_confusion_matrix.png  
└── audio_only_report.docx

**Yang harus dicatat**

- jenis fitur audio;
- dimensi fitur;
- durasi/segmentasi yang digunakan;
- performa model;
- apakah audio lebih kuat/lemah dibanding teks.

### Langkah 4.6 - Visual-Only Baseline

**Tujuan**

Mengukur kontribusi fitur visual seperti ekspresi wajah, landmark, gaze, dan head movement.

**Input**

| **Input**           | **Keterangan**             |
| ------------------- | -------------------------- |
| visual_feature_path | landmark, expression, gaze |
| depression_label    | label depresi              |

**Arsitektur pilihan**

| **Jenis Input Visual**      | **Model yang Cocok**  |
| --------------------------- | --------------------- |
| fitur landmark/gaze tabular | MLP                   |
| frame wajah/image           | CNN                   |
| sequence frame              | LSTM/GRU/Temporal CNN |

Untuk tahap awal, gunakan fitur visual teragregasi dan MLP agar stabil.

**Output**

Visual_Only/  
├── visual_only_baseline.pt  
├── visual_only_training_log.csv  
├── visual_only_metrics.xlsx  
├── visual_only_confusion_matrix.png  
└── visual_only_report.docx

## 6\. Membangun Baseline Bimodal

Baseline bimodal digunakan untuk melihat apakah kombinasi dua modalitas lebih baik daripada satu modalitas.

### Langkah 4.7 - Text + Audio Baseline

**Tujuan**

Menguji apakah informasi verbal dari teks dan informasi vokal dari audio saling melengkapi.

**Arsitektur**

Text Encoder → Text Representation  
Audio Encoder → Audio Representation  
Text + Audio Representation → Fusion Layer → Classifier

**Output**

Text_Audio/  
├── text_audio_baseline.pt  
├── text_audio_training_log.csv  
├── text_audio_metrics.xlsx  
└── text_audio_report.docx

### Langkah 4.8 - Text + Visual Baseline

**Tujuan**

Menguji hubungan antara isi verbal dan ekspresi non-verbal.

**Output**

Text_Visual/  
├── text_visual_baseline.pt  
├── text_visual_training_log.csv  
├── text_visual_metrics.xlsx  
└── text_visual_report.docx

### Langkah 4.9 - Audio + Visual Baseline

**Tujuan**

Menguji kekuatan sinyal non-verbal tanpa teks.

**Output**

Audio_Visual/  
├── audio_visual_baseline.pt  
├── audio_visual_training_log.csv  
├── audio_visual_metrics.xlsx  
└── audio_visual_report.docx

## 7\. Membangun Baseline Multimodal Penuh

### Langkah 4.10 - Menentukan Strategi Fusi Multimodal

Ada beberapa strategi fusi yang dapat digunakan.

| **Strategi Fusi** | **Keterangan**                                          | **Prioritas**        |
| ----------------- | ------------------------------------------------------- | -------------------- |
| Early fusion      | fitur digabung di awal                                  | opsional             |
| Late fusion       | representasi tiap modalitas digabung sebelum classifier | utama                |
| Decision fusion   | prediksi tiap model digabung                            | pembanding sederhana |
| Attention fusion  | bobot modalitas dipelajari model                        | tambahan             |
| Cross-attention   | relasi lintas modalitas dimodelkan eksplisit            | opsional lanjutan    |

Untuk hibah ini, gunakan **late fusion sebagai model utama** karena lebih stabil, mudah dijelaskan, dan sesuai dengan kebutuhan baseline sebelum Federated Learning.

### Langkah 4.11 - Membangun Arsitektur Late Fusion

**Arsitektur umum**

Text Feature → Text Encoder → Text Representation  
Audio Feature → Audio Encoder → Audio Representation  
Visual Feature → Visual Encoder → Visual Representation  
<br/>Text Representation + Audio Representation + Visual Representation  
→ Fusion Layer  
→ Dense Layer  
→ Dropout  
→ Classifier  
→ Depression Probability

**Komponen model**

| **Komponen**   | **Fungsi**                                         |
| -------------- | -------------------------------------------------- |
| Text encoder   | mengubah embedding teks menjadi representasi laten |
| Audio encoder  | mengubah fitur audio menjadi representasi laten    |
| Visual encoder | mengubah fitur visual menjadi representasi laten   |
| Fusion layer   | menggabungkan representasi tiga modalitas          |
| Classifier     | menghasilkan prediksi depresi/non-depresi          |

**Output**

Late_Fusion/  
├── centralized_multimodal_late_fusion.pt  
├── multimodal_training_log.csv  
├── multimodal_metrics.xlsx  
├── multimodal_confusion_matrix.png  
├── multimodal_roc_curve.png  
├── multimodal_pr_curve.png  
└── multimodal_late_fusion_report.docx

### Langkah 4.12 - Cross-Attention sebagai Eksperimen Opsional

Cross-attention dapat digunakan jika waktu dan komputasi cukup. Tujuannya untuk mempelajari hubungan lintas modalitas, misalnya hubungan antara pola teks negatif dan ekspresi wajah datar.

**Gunakan jika**

- baseline late fusion sudah stabil;
- data cukup;
- komputasi mencukupi;
- anggota tim sudah menguasai implementasi attention.

**Jangan diprioritaskan jika**

- dataset kecil;
- model late fusion belum stabil;
- training sering overfitting;
- waktu hibah terbatas.

**Output opsional**

Cross_Attention_Optional/  
├── centralized_multimodal_cross_attention.pt  
├── cross_attention_training_log.csv  
├── cross_attention_metrics.xlsx  
└── cross_attention_report.docx

## 8\. Training dan Hyperparameter Tuning

### Langkah 4.13 - Menentukan Hyperparameter Awal

Gunakan konfigurasi awal berikut:

| **Parameter**  | **Nilai Awal yang Disarankan** |
| -------------- | ------------------------------ |
| Epoch          | 50                             |
| Batch size     | 16 atau 32                     |
| Learning rate  | 0.001 atau 0.0001              |
| Optimizer      | Adam                           |
| Loss function  | Binary Cross-Entropy           |
| Dropout        | 0.2-0.5                        |
| Early stopping | ya                             |
| Patience       | 5-10 epoch                     |
| Random seed    | 42                             |
| Threshold awal | 0.5                            |

### Langkah 4.14 - Melakukan Hyperparameter Tuning

**Parameter yang diuji**

| **Parameter**    | **Kandidat Nilai**    |
| ---------------- | --------------------- |
| Learning rate    | 0.001, 0.0005, 0.0001 |
| Batch size       | 8, 16, 32             |
| Dropout          | 0.2, 0.3, 0.5         |
| Hidden dimension | 64, 128, 256          |
| Fusion dimension | 128, 256              |
| Optimizer        | Adam, AdamW           |
| Weight decay     | 0, 1e-5, 1e-4         |

**Strategi tuning**

Tidak perlu terlalu banyak kombinasi. Gunakan pendekatan bertahap:

- tuning model unimodal;
- pilih encoder yang stabil;
- tuning model multimodal late fusion;
- pilih konfigurasi terbaik berdasarkan validation F1-score atau AUC;
- evaluasi final pada global test set.

**Dokumen yang dibuat**

hyperparameter_tuning_log.xlsx

**Format tabel**

| **Experiment ID** | **Model**              | **LR** | **Batch** | **Dropout** | **Hidden Dim** | **Val F1** | **Val AUC** | **Catatan** |
| ----------------- | ---------------------- | ------ | --------- | ----------- | -------------- | ---------- | ----------- | ----------- |
| EXP001            | Text-only              | 0.001  | 16        | 0.3         | 128            | ...        | ...         | stabil      |
| EXP002            | Multimodal late fusion | 0.0001 | 16        | 0.5         | 256            | ...        | ...         | terbaik     |

### Langkah 4.15 - Menjalankan Training Baseline

**Langkah teknis**

- Load data training.
- Load fitur teks, audio, visual.
- Load label.
- Jalankan model unimodal.
- Jalankan model bimodal.
- Jalankan model multimodal.
- Simpan checkpoint terbaik berdasarkan validation performance.
- Catat loss dan metrik setiap epoch.
- Evaluasi model terbaik pada global test set.
- Simpan hasil akhir.

**Output training**

training_logs/  
├── text_only_training_log.csv  
├── audio_only_training_log.csv  
├── visual_only_training_log.csv  
├── text_audio_training_log.csv  
├── text_visual_training_log.csv  
├── audio_visual_training_log.csv  
└── multimodal_training_log.csv

**Format training log**

| **Epoch** | **Train Loss** | **Val Loss** | **Val Accuracy** | **Val Recall** | **Val F1** | **Val AUC** |
| --------- | -------------- | ------------ | ---------------- | -------------- | ---------- | ----------- |
| 1         | ...            | ...          | ...              | ...            | ...        | ...         |
| 2         | ...            | ...          | ...              | ...            | ...        | ...         |
| 3         | ...            | ...          | ...              | ...            | ...        | ...         |

**9\. Evaluasi Model Baseline**

### Langkah 4.16 - Menghitung Metrik Evaluasi

**Metrik wajib**

| **Metrik**       | **Fungsi**                                  |
| ---------------- | ------------------------------------------- |
| Accuracy         | akurasi umum                                |
| Precision        | seberapa tepat prediksi depresi             |
| Recall           | seberapa baik model menangkap kasus depresi |
| F1-score         | keseimbangan precision dan recall           |
| AUC-ROC          | kemampuan membedakan depresi/non-depresi    |
| Confusion matrix | analisis kesalahan klasifikasi              |

Dalam konteks deteksi depresi, **recall dan F1-score harus diberi perhatian khusus**, karena false negative berarti individu dengan potensi depresi tidak terdeteksi.

### Langkah 4.17 - Membandingkan Semua Baseline

**Tabel utama**

centralized_baseline_metrics.xlsx

| **Model**       | **Modalitas**         | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC** |
| --------------- | --------------------- | ------------ | ------------- | ---------- | ------------ | ------- |
| Text-only       | Teks                  | ...          | ...           | ...        | ...          | ...     |
| Audio-only      | Audio                 | ...          | ...           | ...        | ...          | ...     |
| Visual-only     | Visual                | ...          | ...           | ...        | ...          | ...     |
| Text + Audio    | Teks + Audio          | ...          | ...           | ...        | ...          | ...     |
| Text + Visual   | Teks + Visual         | ...          | ...           | ...        | ...          | ...     |
| Audio + Visual  | Audio + Visual        | ...          | ...           | ...        | ...          | ...     |
| Full Multimodal | Teks + Audio + Visual | ...          | ...           | ...        | ...          | ...     |

### Langkah 4.18 - Membuat Confusion Matrix

Buat confusion matrix untuk minimal empat model:

- text-only;
- audio-only;
- visual-only;
- full multimodal.

**Format analisis**

| **Komponen**   | **Makna**                      |
| -------------- | ------------------------------ |
| True Positive  | depresi terdeteksi benar       |
| True Negative  | non-depresi terdeteksi benar   |
| False Positive | non-depresi diprediksi depresi |
| False Negative | depresi tidak terdeteksi       |

**Output**

confusion_matrix/  
├── cm_text_only.png  
├── cm_audio_only.png  
├── cm_visual_only.png  
└── cm_multimodal.png

### Langkah 4.19 - Membuat ROC Curve dan PR Curve

**Output**

roc_curve/  
├── roc_text_only.png  
├── roc_audio_only.png  
├── roc_visual_only.png  
└── roc_multimodal.png  
<br/>pr_curve/  
├── pr_text_only.png  
├── pr_audio_only.png  
├── pr_visual_only.png  
└── pr_multimodal.png

**Catatan**

PR curve berguna jika dataset tidak seimbang. Jika distribusi depresi dan non-depresi imbalanced, PR curve sebaiknya dilaporkan selain ROC curve.

### Langkah 4.20 - Error Analysis

Error analysis penting agar laporan tidak hanya berisi angka performa.

**Yang dianalisis**

| **Jenis Error** | **Pertanyaan Analisis**                                   |
| --------------- | --------------------------------------------------------- |
| False Negative  | kasus depresi apa yang gagal terdeteksi?                  |
| False Positive  | kasus non-depresi apa yang salah diklasifikasi?           |
| Modalitas lemah | apakah error terjadi pada audio/video berkualitas rendah? |
| Teks pendek     | apakah model gagal pada transkrip pendek?                 |
| Visual noisy    | apakah face detection rendah memengaruhi prediksi?        |
| Label ambiguity | apakah skor depresi dekat cut-off membuat model bingung?  |

**Dokumen**

error_analysis.xlsx

**Format tabel**

| **participant_id** | **True Label** | **Predicted Label** | **Probability** | **Error Type** | **Kemungkinan Penyebab** |
| ------------------ | -------------- | ------------------- | --------------- | -------------- | ------------------------ |
| P001               | 1              | 0                   | 0.42            | False Negative | teks pendek/audio noisy  |
| P002               | 0              | 1                   | 0.71            | False Positive | ekspresi visual ambigu   |

**10\. Analisis Kontribusi Modalitas**

### Langkah 4.21 - Membuat Ablation Study

Ablation study menunjukkan kontribusi setiap modalitas.

**Eksperimen ablation**

| **Eksperimen**  | **Tujuan**                |
| --------------- | ------------------------- |
| Full multimodal | performa lengkap          |
| Tanpa teks      | melihat pentingnya teks   |
| Tanpa audio     | melihat pentingnya audio  |
| Tanpa visual    | melihat pentingnya visual |
| Text-only       | kontribusi teks           |
| Audio-only      | kontribusi audio          |
| Visual-only     | kontribusi visual         |

**Tabel ablation**

| **Model**       | **Teks** | **Audio** | **Visual** | **F1-score** | **AUC** | **Interpretasi**        |
| --------------- | -------- | --------- | ---------- | ------------ | ------- | ----------------------- |
| Full multimodal | Ya       | Ya        | Ya         | ...          | ...     | performa utama          |
| Without text    | Tidak    | Ya        | Ya         | ...          | ...     | dampak hilangnya teks   |
| Without audio   | Ya       | Tidak     | Ya         | ...          | ...     | dampak hilangnya audio  |
| Without visual  | Ya       | Ya        | Tidak      | ...          | ...     | dampak hilangnya visual |

**Dokumen**

modality_ablation_study.xlsx  
modality_contribution_analysis.docx

**11\. Pemilihan Model Baseline Terbaik**

### Langkah 4.22 - Menentukan Model Terbaik

Model terbaik dipilih berdasarkan validation set, lalu diuji pada global test set.

**Kriteria pemilihan**

| **Kriteria**                   | **Prioritas**                        |
| ------------------------------ | ------------------------------------ |
| F1-score tinggi                | utama                                |
| Recall tinggi                  | sangat penting untuk deteksi depresi |
| AUC tinggi                     | penting                              |
| Val loss stabil                | penting                              |
| Tidak overfitting              | penting                              |
| Arsitektur tidak terlalu berat | penting untuk tahap FL               |

**Dokumen**

best_baseline_model_selection.xlsx

**Format tabel**

| **Model**                   | **Val F1** | **Val AUC** | **Test F1** | **Test AUC** | **Overfitting** | **Dipilih?** |
| --------------------------- | ---------- | ----------- | ----------- | ------------ | --------------- | ------------ |
| Text-only                   | ...        | ...         | ...         | ...          | rendah          | tidak        |
| Full multimodal late fusion | ...        | ...         | ...         | ...          | rendah          | ya           |

## 12\. Dokumentasi Wajib untuk Laporan Hibah

### 12.1 Dokumentasi Teknis Utama

| **No** | **Dokumen**                             | **Fungsi**                   |
| ------ | --------------------------------------- | ---------------------------- |
| 1      | centralized_baseline_protocol.docx      | rancangan baseline           |
| 2      | centralized_baseline_config.yaml        | konfigurasi eksperimen       |
| 3      | model_architecture_plan.docx            | desain encoder dan fusi      |
| 4      | centralized_input_verification_log.xlsx | validasi input               |
| 5      | training_logs/                          | bukti proses training        |
| 6      | hyperparameter_tuning_log.xlsx          | bukti tuning                 |
| 7      | centralized_baseline_metrics.xlsx       | tabel performa               |
| 8      | modality_ablation_study.xlsx            | kontribusi modalitas         |
| 9      | error_analysis.xlsx                     | analisis kesalahan           |
| 10     | centralized_baseline_report.docx        | laporan akhir tahap baseline |

### 12.2 Dokumentasi Model

| **Dokumen/File**                   | **Isi**              |
| ---------------------------------- | -------------------- |
| text_only_baseline.pt              | model teks           |
| audio_only_baseline.pt             | model audio          |
| visual_only_baseline.pt            | model visual         |
| text_audio_baseline.pt             | model teks+audio     |
| text_visual_baseline.pt            | model teks+visual    |
| audio_visual_baseline.pt           | model audio+visual   |
| centralized_multimodal_baseline.pt | model baseline utama |
| model_checkpoint_log.xlsx          | catatan checkpoint   |

### 12.3 Dokumentasi Evaluasi

| **Dokumen**                   | **Isi**                      |
| ----------------------------- | ---------------------------- |
| confusion_matrix_baseline.png | confusion matrix model utama |
| roc_curve_baseline.png        | ROC curve model utama        |
| pr_curve_baseline.png         | precision-recall curve       |
| classification_report.xlsx    | precision, recall, F1        |
| threshold_analysis.xlsx       | analisis threshold prediksi  |
| evaluation_summary.docx       | ringkasan evaluasi           |

### 12.4 Dokumentasi untuk Artikel Ilmiah

| **Dokumen**                        | **Digunakan untuk Bagian Artikel** |
| ---------------------------------- | ---------------------------------- |
| centralized_baseline_protocol.docx | Methods                            |
| model_architecture_plan.docx       | Model Architecture                 |
| centralized_baseline_metrics.xlsx  | Results                            |
| modality_ablation_study.xlsx       | Ablation Study                     |
| error_analysis.xlsx                | Discussion                         |
| roc_curve_baseline.png             | Results Figure                     |
| confusion_matrix_baseline.png      | Results Figure                     |
| centralized_baseline_report.docx   | Experimental Setup                 |

## 13\. Template Tabel yang Perlu Disiapkan

**Tabel 1. Konfigurasi Model Baseline**

| **Komponen**    | **Konfigurasi**      |
| --------------- | -------------------- |
| Text encoder    | BERT embedding + MLP |
| Audio encoder   | MLP/CNN              |
| Visual encoder  | MLP/CNN              |
| Fusion strategy | Late fusion          |
| Classifier      | MLP                  |
| Loss function   | Binary Cross-Entropy |
| Optimizer       | Adam                 |
| Epoch           | ...                  |
| Batch size      | ...                  |
| Learning rate   | ...                  |
| Dropout         | ...                  |

**Tabel 2. Hasil Unimodal Baseline**

| **Model**   | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC** |
| ----------- | ------------ | ------------- | ---------- | ------------ | ------- |
| Text-only   | ...          | ...           | ...        | ...          | ...     |
| Audio-only  | ...          | ...           | ...        | ...          | ...     |
| Visual-only | ...          | ...           | ...        | ...          | ...     |

**Tabel 3. Hasil Bimodal Baseline**

| **Model**      | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC** |
| -------------- | ------------ | ------------- | ---------- | ------------ | ------- |
| Text + Audio   | ...          | ...           | ...        | ...          | ...     |
| Text + Visual  | ...          | ...           | ...        | ...          | ...     |
| Audio + Visual | ...          | ...           | ...        | ...          | ...     |

**Tabel 4. Hasil Multimodal Baseline**

| **Model**       | **Strategi Fusi** | **Accuracy** | **Precision** | **Recall** | **F1-score** | **AUC** |
| --------------- | ----------------- | ------------ | ------------- | ---------- | ------------ | ------- |
| Full multimodal | Late fusion       | ...          | ...           | ...        | ...          | ...     |
| Full multimodal | Cross-attention   | ...          | ...           | ...        | ...          | ...     |

**Tabel 5. Ablation Study**

| **Eksperimen**  | **Teks** | **Audio** | **Visual** | **F1-score** | **AUC** | **Dampak**     |
| --------------- | -------- | --------- | ---------- | ------------ | ------- | -------------- |
| Full multimodal | Ya       | Ya        | Ya         | ...          | ...     | baseline utama |
| Without text    | Tidak    | Ya        | Ya         | ...          | ...     | ...            |
| Without audio   | Ya       | Tidak     | Ya         | ...          | ...     | ...            |
| Without visual  | Ya       | Ya        | Tidak      | ...          | ...     | ...            |

**Tabel 6. Hyperparameter Tuning**

| **Experiment ID** | **Model**              | **LR** | **Batch** | **Dropout** | **Hidden Dim** | **Val F1** | **Val AUC** | **Status** |
| ----------------- | ---------------------- | ------ | --------- | ----------- | -------------- | ---------- | ----------- | ---------- |
| EXP001            | Text-only              | ...    | ...       | ...         | ...            | ...        | ...         | selesai    |
| EXP002            | Audio-only             | ...    | ...       | ...         | ...            | ...        | ...         | selesai    |
| EXP003            | Multimodal late fusion | ...    | ...       | ...         | ...            | ...        | ...         | terbaik    |

**Tabel 7. Error Analysis**

| **Error Type**   | **Jumlah** | **Persentase** | **Kemungkinan Penyebab** |
| ---------------- | ---------- | -------------- | ------------------------ |
| False Positive   | ...        | ...%           | fitur non-verbal ambigu  |
| False Negative   | ...        | ...%           | teks pendek/audio noisy  |
| Correct Positive | ...        | ...%           | indikator depresi kuat   |
| Correct Negative | ...        | ...%           | sinyal depresi rendah    |

## 14\. Narasi Siap Pakai untuk Laporan Hibah

Berikut narasi yang dapat digunakan dalam laporan kemajuan atau laporan akhir:

Tahap pengembangan baseline multimodal terpusat dilakukan untuk memperoleh model pembanding sebelum sistem dikembangkan ke dalam skema Federated Learning. Dataset multimodal hasil pra-pemrosesan dan partisi digunakan dengan tetap menjaga pemisahan data train, validation, dan global test. Pada tahap ini, model dibangun secara bertahap mulai dari baseline unimodal berbasis teks, audio, dan visual, kemudian dilanjutkan dengan kombinasi bimodal, dan akhirnya model multimodal penuh.

Pada model multimodal penuh, masing-masing modalitas diproses melalui encoder khusus. Fitur teks diproses menggunakan representasi embedding bahasa, fitur audio diproses melalui encoder akustik, sedangkan fitur visual diproses melalui encoder visual berbasis fitur ekspresi, landmark, dan gaze. Representasi dari ketiga modalitas kemudian digabungkan menggunakan strategi late fusion sebelum masuk ke classifier akhir. Model dilatih secara centralized sebagai benchmark performa, kemudian dievaluasi menggunakan metrik accuracy, precision, recall, F1-score, dan AUC. Hasil baseline ini digunakan sebagai acuan untuk menilai performa model Federated Learning pada tahap berikutnya.

## 15\. Checklist Pekerjaan Tahap 4

| **No** | **Aktivitas**                                         | **Status** |
| ------ | ----------------------------------------------------- | ---------- |
| 1      | Menyiapkan centralized_baseline_protocol.docx         | ☐          |
| 2      | Membuat centralized_baseline_config.yaml              | ☐          |
| 3      | Memverifikasi input train/validation/test             | ☐          |
| 4      | Membuat centralized_input_verification_log.xlsx       | ☐          |
| 5      | Membangun text-only baseline                          | ☐          |
| 6      | Membangun audio-only baseline                         | ☐          |
| 7      | Membangun visual-only baseline                        | ☐          |
| 8      | Membangun text+audio baseline                         | ☐          |
| 9      | Membangun text+visual baseline                        | ☐          |
| 10     | Membangun audio+visual baseline                       | ☐          |
| 11     | Menentukan strategi late fusion                       | ☐          |
| 12     | Membangun full multimodal baseline                    | ☐          |
| 13     | Melakukan hyperparameter tuning                       | ☐          |
| 14     | Menyimpan model checkpoint terbaik                    | ☐          |
| 15     | Menghitung accuracy, precision, recall, F1-score, AUC | ☐          |
| 16     | Membuat confusion matrix                              | ☐          |
| 17     | Membuat ROC curve dan PR curve                        | ☐          |
| 18     | Melakukan ablation study                              | ☐          |
| 19     | Melakukan error analysis                              | ☐          |
| 20     | Memilih model baseline terbaik                        | ☐          |
| 21     | Membuat centralized_baseline_metrics.xlsx             | ☐          |
| 22     | Membuat centralized_baseline_report.docx              | ☐          |
| 23     | Membuat README_baseline_experiment.md                 | ☐          |

## 16\. Kesimpulan Praktis Tahap 4

Pada Tahap 4, pekerjaan utama adalah membangun **model baseline multimodal terpusat** sebagai pembanding ilmiah sebelum masuk ke Federated Learning. Baseline ini harus menunjukkan performa dari model:

- text-only;
- audio-only;
- visual-only;
- bimodal;
- full multimodal.

Dokumen paling penting yang wajib tersedia adalah:

- centralized_baseline_protocol.docx
- centralized_baseline_config.yaml
- centralized_input_verification_log.xlsx
- text_only_metrics.xlsx
- audio_only_metrics.xlsx
- visual_only_metrics.xlsx
- centralized_baseline_metrics.xlsx
- hyperparameter_tuning_log.xlsx
- modality_ablation_study.xlsx
- error_analysis.xlsx
- confusion_matrix_baseline.png
- roc_curve_baseline.png
- pr_curve_baseline.png
- centralized_multimodal_baseline.pt
- centralized_baseline_report.docx

Jika seluruh dokumen ini lengkap, maka Tahap 4 sudah kuat untuk dilaporkan sebagai bukti bahwa penelitian telah memiliki **benchmark performa multimodal** yang valid, terukur, dan siap dibandingkan dengan Tahap 7, yaitu **desain arsitektur Federated Multimodal Learning**.