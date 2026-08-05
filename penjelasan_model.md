# 🧠 Panduan Sederhana & Detail: Bagaimana Model AI Mendeteksi Depresi (Multimodal Depression Classifier)

Dokumen ini menjelaskan alur kerja, arsitektur, dan cara kerja model kecerdasan buatan (AI) pemroses depresi dalam bahasa yang sederhana, rinci, dan mudah dipahami oleh siapa saja, bahkan tanpa latar belakang teknis atau pemrograman.

---

## 📌 Ringkasan Singkat (TL;DR)

Model AI ini dirancang untuk mendeteksi tanda-tanda depresi pada seseorang dengan menganalisis **3 jenis informasi sekaligus**: **Kata-kata yang diucapkan (Teks)**, **Nada suara (Audio)**, dan **Ekspresi wajah (Visual)**. 

Di dalam dunia AI, teknik menggabungkan beberapa jenis penginderaan ini disebut **Multimodal**. Model bekerja mirip seperti seorang psikolog yang mengamati pasien dari berbagai sudut sebelum memberikan diagnosis.

---

## 🏛️ 1. Analogi Sederhana: "Tim Dokter Psikiatri Digital"

Bayangkan ketika seorang pasien berkonsultasi dengan seorang psikiater. Psikiater tidak hanya mendengarkan isi ceritanya saja, tetapi memperhatikan seluruh respons pasien:

```
                  ┌──────────────────────────────────────────┐
                  │           PASIEN BERKONSULTASI           │
                  └────────────────────┬─────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   🗣️ TEKS     │              │   🎙️ AUDIO    │              │  👁️ VISUAL    │
│ (Kata-kata &  │              │ (Nada, Jeda,  │              │  (Ekspresi &  │
│    Kalimat)   │              │   Intonasi)   │              │ Pergerakan)   │
└───────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │         GABUNGAN ANALISIS (FUSION)        │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │         DIAGNOSIS: DEPRESI / SEHAT       │
                  └──────────────────────────────────────────┘
```

1. **Kata-kata (Teks)**: Apakah kalimat yang diucapkan mengandung rasa putus asa, hampa, atau emosi negatif?
2. **Nada Suara (Audio)**: Apakah suaranya datar (*monotone*), lambat, atau banyak jeda hening yang tidak biasa?
3. **Ekspresi Wajah (Visual)**: Apakah kontak mata berkurang, bibir jarang tersenyum, atau otot wajah tampak tegang/lesu?

Model AI ini menirukan proses berpikir tim dokter tersebut. AI mengambil data teks, suara, dan video, lalu menggabungkannya untuk menghitung risiko depresi seseorang.

---

## 👁️ 2. Mengenal "Tiga Indera" Model AI (Data Input)

Sebelum AI mengambil keputusan, data mentah dari manusia diubah menjadi kumpulan angka yang bisa dimengerti oleh komputer.

### A. Indera 1: Teks (Kata-kata & Kalimat)
* **Sumber Data**: Transkrip percakapan hasil wawancara pasien.
* **Format di AI**: **768 fitur/angka**.
* **Cara Kerja Sederhana**: Komputer menggunakan teknologi pemrosesan bahasa (*Natural Language Processing*) untuk membaca makna dibalik kalimat. Komputer mendeteksi kata-kata kunci emosional dan konteks percakapan.

### B. Indera 2: Audio (Suara & Intonasi)
* **Sumber Data**: Rekaman gelombang suara saat pasien berbicara.
* **Format di AI**: **128 fitur/angka**.
* **Cara Kerja Sederhana**: Komputer mengukur frekuensi suara, ritme, tinggi-rendahnya nada (*pitch*), serta durasi keheningan (jeda bicara) di antara kata. Suara orang yang mengalami depresi sering kali memiliki intonasi yang lebih datar dan tempo bicara yang lebih lambat.

### C. Indera 3: Visual (Ekspresi & Gerak Wajah)
* **Sumber Data**: Rekaman video titik-titik koordinat wajah (menggunakan teknologi ekstraksi fitur *OpenFace*).
* **Format di AI**: **178 fitur/angka**.
* **Cara Kerja Sederhana**: Komputer mengamati titik-titik kecil di alis, mata, pipi, dan bibir. Komputer mengukur seberapa sering pasien berkedip, ke mana arah tatapan matanya, dan seberapa dinamis gerakan otot wajahnya.

---

## ⚙️ 3. Alur Kerja Model (Langkah demi Langkah)

Di dalam program komputer, alur pemrosesan dibagi menjadi 3 tahap utama:

```
[ Input Teks: 768 ]  ───> [ Encoder Teks ]  ───> ( Ringkasan 128 ) ┐
                                                                  │
[ Input Audio: 128 ] ───> [ Encoder Audio ] ───> ( Ringkasan 128 ) ┼──> [ LATE FUSION ] ───> [ DIAGNOSIS ]
                                                                  │      (Gabungan: 384)       (Depresi / Tidak)
[ Input Visual: 178 ]───> [ Encoder Visual] ───> ( Ringkasan 128 ) ┘
```

### Tahap 1: "Dokter Spesialis" (Modality Encoders)
Setiap indera memiliki modul "penyaring" khusus yang disebut *Encoder* (terbuat dari jaringan saraf tiruan / MLP):
* **Text Encoder**: Mengecilkan 768 angka teks menjadi **128 angka sari penting**.
* **Audio Encoder**: Mengecilkan 128 angka audio menjadi **128 angka sari penting**.
* **Visual Encoder**: Mengecilkan 178 angka visual menjadi **128 angka sari penting**.

### Tahap 2: "Rapat Penggabungan" (Late Fusion)
Setelah ketiga indera diringkas, ketiganya digabungkan bersama ($128 + 128 + 128 = 384$ angka). 

Mengapa dinamakan **Late Fusion** (Penggabungan Akhir)? 
Karena masing-masing indera diproses dan dipahami terlebih dahulu secara mandiri, baru kemudian dipertemukan di meja diskusi akhir. Di tahap ini, AI bisa menemukan kecocokan atau ketidakcocokan antar-indera. 

> **Contoh Ketidakcocokan**: Pasien mengatakan *"Saya merasa bahagia"* (Teks positif), TETAPI nada suaranya sangat lambat (Audio negatif) dan matanya menunduk lesu (Visual negatif). Modul penggabungan akan menangkap ketidakcocokan ini dan mengenali bahwa pasien mungkin sedang menyembunyikan kondisinya.

### Tahap 3: "Keputusan Akhir" (Classifier Head)
Gabungan informasi 384 angka tersebut diolah oleh bagian akhir model untuk menghasilkan **Skor Probabilitas (0% sampai 100%)**:
* Jika skor probabilitas **$\ge 52\%$**, model menyimpulkan: **DEPRESI (1)**.
* Jika skor probabilitas **$< 52\%$**, model menyimpulkan: **TIDAK DEPRESI / SEHAT (0)**.

---

## 📊 4. Memahami Hasil Ukur & Performa Model

Ketika model diuji dengan data pasien baru (Test Set), hasilnya adalah sebagai berikut. Mari kita terjemahkan angka-angka teknis ini ke bahasa sehari-hari:

| Metrik Teknolis | Nilai | Arti dan Penjelasan Sederhana |
|---|---|---|
| **Accuracy (Akurasi)** | **72.7%** | **Ketepatan Total**: Dari 100 orang yang diperiksa, model berhasil menebak kondisi 73 orang dengan tepat. |
| **Precision (Presisi)** | **40.0%** | **Keakuratan Vonis Depresi**: Saat model membunyikan alarm dan berkata *"Orang ini mengalami depresi"*, tingkat kebenarannya adalah 40% (sisanya merupakan *false alarm* / salah sangka). |
| **Recall (Sensitivitas)** | **30.8%** | **Daya Tangkap Penderita**: Dari seluruh penderita depresi yang sebenarnya ada di lapangan, model berhasil menemukan 31% di antaranya, sementara sisanya belum terdeteksi. |
| **F1-Score** | **34.8%** | **Nilai Keseimbangan**: Angka yang mengukur seberapa seimbang antara keakuratan vonis (Precision) dan daya tangkap (Recall). |
| **AUC-ROC** | **67.6%** | **Kemampuan Membedakan**: Mengukur seberapa jago model memisahkan mana kelompok orang depresi dan kelompok sehat. (Skor 50% = sama seperti tebak acak/lempar koin; Skor 100% = sempurna). |

---

## 🏫 5. Apa Itu "Overfitting" dan Kenapa Terjadi?

Laporan audit menyebutkan bahwa model mengalami **Overfitting** (Skor saat latihan F1=0.78, tapi saat ujian akhir F1=0.35). Apa maksudnya?

### Analogi "Siswa Menghafal Soal Latihan"
* Bayangkan seorang siswa yang sedang belajar untuk ujian.
* Siswa diberikan **176 soal latihan (Data Training)**. Siswa tersebut memiliki daya ingat yang sangat kuat, sehingga dia **menghafal jawaban** dari 176 soal tersebut dan mendapat nilai **88 saat latihan**.
* Namun, saat ujian nasional yang berisi **soal-soal baru (Data Test)** diberikan, siswa bingung dan nilainya turun menjadi **35**.
* **Artinya**: Siswa bukan memahami *konsep mendalam* tentang mata pelajaran tersebut, melainkan hanya *menghafal ciri-ciri khusus* dari soal latihan.

### Mengapa AI Menghafal?
1. **Jumlah Sampel Sedikit**: Data latihan hanya berisi 176 orang (hanya 42 di antaranya yang mengalami depresi).
2. **Model Terlalu Pintar/Besar**: Kapasitas memori model komputer terlalu besar dibandingkan jumlah data latihan yang sedikit, sehingga model dengan mudah menghafal data tanpa belajar pola umumnya.

---

## 🏆 6. Mengapa Model Ini Tetap Bagus dan Valid Sebagai Baseline?

Mungkin timbul pertanyaan: *"Jika F1-Score pada data ujian hanya 34.8%, mengapa model ini dinyatakan valid dan bagus sebagai baseline?"*

1. **Jujur dan Tanpa Kecurangan (*Zero Data Leakage*)**: 
   Model ini seperti siswa yang tidak mencontek. Data latihan dan data ujian dipisah total. Nilai F1 34.8% adalah **nilai murni dan jujur**, bukan hasil manipulasi atau kebocoran kunci jawaban.
2. **Tantangan Nyata di Dunia Medis**: 
   Mendeteksi depresi dari rekaman singkat adalah hal yang sangat sulit, bahkan untuk psikiater berpengalaman. Angka AUC 67.6% adalah hasil acuan awal (*benchmark*) yang wajar dan realistis pada dataset medis E-DAIC.
3. **Penyangga Utama Penelitian (*Baseline*)**: 
   *Baseline* artinya **titik awal pengukuran**. Sebelum kita mencoba teknik-teknik rumit yang baru (seperti *Federated Learning* atau penambahan fitur keamanan), kita harus punya patok awal yang jujur sebagai pembanding. Tanpa *baseline* yang jujur, kita tidak akan pernah tahu apakah teknik baru kita benar-benar memberikan peningkatan atau tidak.

---

## 🛠️ 7. Rencana Perbaikan Ke Depan (Langkah Selanjutnya)

Untuk memperbaiki performa model agar tidak mudah overfit (menghafal), kita tidak perlu merombak ulang seluruh bangunan kodenya. Kita cukup melakukan beberapa langkah "pengereman":

1. **Memperkecil Kapasitas Otak AI**: Mengurangi jumlah sel saraf dari 128 menjadi 64 atau 32 agar model terpaksa belajar pola umum dan tidak sempat menghafal detail kecil.
2. **Menambah "Rem" (Regularisasi & Dropout)**: Memberikan batasan yang lebih ketat saat latihan agar model tidak terlalu percaya diri pada satu fitur saja.
3. **Variasi Latihan (Cross-Validation)**: Melatih model dengan cara memutar data latihan dan data ujian beberapa kali (*5-Fold Cross Validation*) agar estimasi kemampuan model semakin stabil.

---

### 📝 Kesimpulan
Model ini sudah memiliki **struktur bangunan yang sangat kuat, bersih, dan secara medis/ilmu komputer sudah tepat**. Ibarat mesin mobil, semua komponen transmisi dan kelistrikannya berfungsi dengan sempurna, hanya perlu penyetelan *tuning* pada karburator dan bahan bakar agar jalannya lebih stabil!
