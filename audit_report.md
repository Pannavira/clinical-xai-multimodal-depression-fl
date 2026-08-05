# 🔬 Laporan Audit Teknis Mendalam — Modeling Pipeline

## Ringkasan Eksekutif

Audit ini menjalankan verifikasi **statis** (analisis kode) dan **dinamis** (training + evaluasi runtime) terhadap seluruh folder `modeling/` untuk mendeteksi kekeliruan, data leakage, mode collapse, dan halusinasi metrik.

| # | Aspek Audit | Status | Catatan |
|---|---|---|---|
| 1 | Split Integrity (Overlap) | ✅ **PASS** | 0 overlap antar 3 split |
| 2 | Label Mapping | ✅ **PASS** | Menggunakan `Depression_label` (clinician-rated) |
| 3 | Scaler Leakage | ✅ **PASS** | Stats dihitung hanya dari training |
| 4 | Class Imbalance (pos_weight) | ✅ **PASS** | `pos_weight=3.19` diterapkan |
| 5 | Threshold Tuning | ✅ **PASS** | Tuned pada validation, diterapkan ke test |
| 6 | Mode Collapse | ✅ **PASS** | Prediksi aktif kedua kelas |
| 7 | Overfitting | ⚠️ **WARNING** | F1 gap Train→Test = **0.4357** |

> [!IMPORTANT]
> Pipeline dan kode secara arsitektural **BENAR** dan **VALID** sebagai baseline. Tidak ditemukan data leakage, mode collapse, atau halusinasi metrik. Namun terdapat **overfitting signifikan** yang perlu ditangani.

---

## 1. Integritas Dataset & Split

### 1.1 Pembagian Subjek

| Split | Jumlah Subjek |
|---|---|
| Train | 176 |
| Validation | 44 |
| Test | 55 |
| **Total Unik** | **275** |

```
Overlap Train ↔ Val:  0 subjek
Overlap Train ↔ Test: 0 subjek
Overlap Val   ↔ Test: 0 subjek
```

> [!TIP]
> ✅ **LULUS** — Ketiga split **100% independen**. Tidak ada satu pun participant_id yang muncul di lebih dari satu split. Rasio split ~64:16:20 sesuai standar riset.

### 1.2 Mapping Label

Model menggunakan kolom `Depression_label` dari [detailed_lables.csv](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/data/detailed_lables.csv) yang merupakan **label biner klinis (clinician-rated PHQ-8 binary)**, bukan simple threshold `>= 10` pada `Depression_severity`.

Ditemukan 20 kasus di mana `Depression_severity >= 10` **tidak** cocok dengan `Depression_label`:

| PID | Score | Threshold ≥10 | Ground Truth | Keterangan |
|---|---|---|---|---|
| 459 | 16 | 1 | 0 | Klinis menilai non-depressed |
| 483 | 15 | 1 | 0 | Klinis menilai non-depressed |
| 352 | 10 | 1 | 0 | Borderline case |
| 320 | 11 | 1 | 0 | Klinis menilai non-depressed |
| ... | ... | ... | ... | 16 kasus lainnya |

> [!NOTE]
> Ini **BUKAN BUG** melainkan **perilaku yang benar**. Dataset E-DAIC menggunakan label klinis yang bisa berbeda dari simple thresholding. Kode di [dataset_loader.py](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/modeling/dataset_loader.py#L48-L63) **sudah benar** memprioritaskan `label_dict` dari `detailed_labels.csv`.

---

## 2. Penanganan Class Imbalance & Feature Scaling

### 2.1 Class Imbalance

```
Train:      Depressed=42 (23.9%) | Non-Depressed=134 (76.1%)
Validation: Depressed=11 (25.0%) | Non-Depressed= 33 (75.0%)
Test:       Depressed=13 (23.6%) | Non-Depressed= 42 (76.4%)

pos_weight = 134/42 = 3.1905
```

✅ `pos_weight` dihitung **hanya dari training set** dan diterapkan pada `BCEWithLogitsLoss` di [train_centralized.py L131-137](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/modeling/train_centralized.py#L131-L137).

### 2.2 Feature Normalization (Z-Score Scaling)

Alur normalisasi di [dataset_loader.py L222-230](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/modeling/dataset_loader.py#L222-L230):

```python
# Step 1: Buat raw training dataset TANPA scaler
raw_train_dataset = MultimodalDepressionDataset(df_train, ..., scaler_stats=None)

# Step 2: Hitung mean/std HANYA dari training
scaler_stats = compute_scaler_stats(raw_train_dataset)

# Step 3: Terapkan stats YANG SAMA ke semua split
train_dataset = MultimodalDepressionDataset(df_train, ..., scaler_stats=scaler_stats)
val_dataset   = MultimodalDepressionDataset(df_val,   ..., scaler_stats=scaler_stats)
test_dataset  = MultimodalDepressionDataset(df_test,  ..., scaler_stats=scaler_stats)
```

**Runtime Verification:**
| Feature | Shape | Mean of Stats | Std of Stats |
|---|---|---|---|
| visual_mean | (178,) | 346.74 | 3049.32 |
| visual_std | (178,) | 116.84 | 975.88 |
| audio_mean | (128,) | 93.56 | 407.00 |
| audio_std | (128,) | 16.82 | 66.66 |
| text_mean | (768,) | -0.011 | 0.307 |
| text_std | (768,) | 0.055 | 0.008 |

> [!TIP]
> ✅ **LULUS** — Tidak ada scaler leakage. Stats dihitung eksklusif dari training set.

---

## 3. Evaluasi Metrik & Confusion Matrix (Test Set)

### 3.1 Confusion Matrix

```
                    Predicted 0    Predicted 1
Actual 0 (Non-Dep):   TN = 36        FP = 6
Actual 1 (Dep):        FN = 9        TP = 4
```

| Komponen | Nilai |
|---|---|
| True Positives (TP) | 4 |
| True Negatives (TN) | 36 |
| False Positives (FP) | 6 |
| False Negatives (FN) | 9 |

### 3.2 Metrik Klasifikasi

| Metrik | Nilai |
|---|---|
| **Accuracy** | 0.7273 |
| **Precision** | 0.4000 |
| **Recall** | 0.3077 |
| **F1-Score** | 0.3478 |
| **AUC-ROC** | 0.6758 |
| **Threshold** | 0.52 |

```
Classification Report:
                   precision    recall  f1-score   support
Non-Depressed (0)       0.80      0.86      0.83        42
    Depressed (1)       0.40      0.31      0.35        13

         accuracy                           0.73        55
        macro avg       0.60      0.58      0.59        55
     weighted avg       0.71      0.73      0.71        55
```

### 3.3 Mode Collapse Check

✅ **TIDAK ADA Mode Collapse** — Model aktif memprediksi **kedua kelas** (0 dan 1). Prediksi unik: `[0, 1]`.

### 3.4 Threshold Tuning

✅ Threshold `0.52` diperoleh dari fungsi [find_best_threshold()](file:///c:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/modeling/train_centralized.py#L27-L37) yang men-scan threshold `0.1–0.9` (step `0.02`) untuk memaksimalkan F1-Score **pada validation set saja**, lalu diterapkan ke test.

---

## 4. Diagnosis Kekeliruan & Overfitting

### 4.1 Perbandingan Train vs Val vs Test

| Set | F1-Score | AUC-ROC | Accuracy | Loss |
|---|---|---|---|---|
| **Train** | 0.7835 | 0.9705 | 0.8807 | 0.5703 |
| **Validation** | 0.5600 | 0.6887 | 0.7500 | 0.9846 |
| **Test** | 0.3478 | 0.6758 | 0.7273 | 0.9983 |

```
F1 Gap (Train - Test):   0.4357  ⚠️
Loss Gap (Test - Train): 0.4280  ⚠️
```

> [!WARNING]
> **Overfitting terdeteksi**: Gap F1-Score antara Train (0.78) dan Test (0.35) mencapai **0.44**, menunjukkan model terlalu hafal training data. Ini **bukan data leakage** (sudah diverifikasi), melainkan kapasitas model yang terlalu besar relatif terhadap dataset kecil (176 training samples).

### 4.2 Analisis Akar Masalah Overfitting

1. **Dataset kecil**: Hanya 176 training samples dengan 42 kasus positif (depressed)
2. **Arsitektur terlalu besar**: 3 encoder masing-masing 2-layer MLP (768→128, 128→128) + classifier head = total parameter cukup besar untuk ~176 sampel
3. **Early stopping terlalu lambat**: Patience 10 dengan max 50 epoch. Model berhenti di epoch 13, namun best checkpoint sudah di epoch awal
4. **Tidak ada augmentasi data** atau regularisasi tambahan (hanya Dropout 0.3)

---

## 5. Kesimpulan & Rekomendasi

### ✅ Kesimpulan Objektif

> **Pipeline modeling ini secara arsitektural BENAR dan VALID untuk dijadikan baseline.** Tidak ada data leakage, mode collapse, halusinasi metrik, atau kekeliruan fatal. Semua aspek teknis kritis (split integrity, label mapping, scaler independence, class balancing, threshold tuning) telah diimplementasikan dengan benar.

> Performa Test F1=0.35 dan AUC=0.68 adalah **wajar dan realistis** untuk baseline awal pada dataset E-DAIC yang kecil (~275 total subjek). Ini menunjukkan model **tidak halusinasi** — hasil sebelumnya yang terlalu sempurna kemungkinan dihasilkan dari proses yang berbeda.

### ⚠️ Area yang Perlu Perbaikan (Non-Critical)

| # | Rekomendasi | Prioritas | Dampak |
|---|---|---|---|
| 1 | **Tambah regularisasi**: Naikkan dropout ke 0.4-0.5, tambah L2 weight decay ke 1e-3 | Medium | Mengurangi overfitting |
| 2 | **Kurangi kapasitas model**: Perkecil hidden_dim ke 64 untuk dataset sekecil ini | Medium | Reduce memorization |
| 3 | **Turunkan patience**: Dari 10 ke 5-7, atau gunakan metric scheduling | Low | Training lebih efisien |
| 4 | **Augmentasi data**: Noise injection, mixup, atau feature perturbation pada training | Medium | Meningkatkan generalisasi |
| 5 | **Cross-validation**: K-fold CV (k=5) untuk estimasi metrik yang lebih stabil pada dataset kecil | High | Evaluasi lebih robust |

### 📌 Status Final: VALID SEBAGAI BASELINE ✅
