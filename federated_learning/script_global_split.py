import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.5 — Membuat Global Test Set dan Split Dataset")
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='preprocessing_output/processed_multimodal_dataset.csv',
        help='Jalur file dataset multimodal (CSV)'
    )
    parser.add_argument(
        '--labels', 
        type=str, 
        default='data/detailed_lables.csv',
        help='Jalur file label detail (CSV) untuk mengambil Depression_label biner'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='federated_learning_output',
        help='Direktori penyimpanan hasil split dan statistik'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Proporsi data untuk global test set (0.0 - 1.0)'
    )
    parser.add_argument(
        '--val_size', 
        type=float, 
        default=0.2,
        help='Proporsi data untuk validation pool relatif terhadap sisa data client (0.0 - 1.0)'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state seed untuk replikasi hasil split'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Langkah 4.5 — Pembuatan Global Test Set ===")
    
    # 1. Ambil processed_multimodal_dataset.csv
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset tidak ditemukan di {dataset_path}")
        return
        
    print(f"Membaca dataset dari: {dataset_path}")
    df_dataset = pd.read_csv(dataset_path)
    print(f"Total baris awal dataset: {len(df_dataset)}")
    
    # 2. Pastikan semua data memiliki participant_id, fitur teks, audio, visual, dan label
    required_cols = [
        'participant_id', 
        'text_feature_path', 
        'audio_feature_path', 
        'visual_feature_path', 
        'depression_label'
    ]
    
    # Periksa ketersediaan kolom
    for col in required_cols:
        if col not in df_dataset.columns:
            print(f"Error: Kolom wajib '{col}' tidak ditemukan di dataset!")
            return
            
    # Hapus baris dengan nilai null pada kolom wajib
    df_clean = df_dataset.dropna(subset=required_cols).copy()
    
    # Hapus baris dengan string kosong
    for col in ['text_feature_path', 'audio_feature_path', 'visual_feature_path']:
        df_clean = df_clean[df_clean[col].astype(str).str.strip() != '']
        
    print(f"Total baris setelah pembersihan (tidak null/kosong): {len(df_clean)}")
    if len(df_clean) < len(df_dataset):
        print(f"Peringatan: {len(df_dataset) - len(df_clean)} baris tidak valid dibuang.")
        
    # Standardisasi tipe data ID
    df_clean['participant_id'] = df_clean['participant_id'].astype(int)
    
    # Ambil binary label untuk stratifikasi dari detailed_lables.csv
    labels_path = args.labels
    if os.path.exists(labels_path):
        print(f"Membaca label detail dari: {labels_path}")
        df_labels = pd.read_csv(labels_path)
        df_labels['Participant'] = df_labels['Participant'].astype(int)
        
        # Merge untuk mendapatkan Depression_label biner
        df_merged = pd.merge(
            df_clean, 
            df_labels[['Participant', 'Depression_label']], 
            left_on='participant_id', 
            right_on='Participant', 
            how='left'
        )
        
        # Fallback jika ada participant yang tidak memiliki label biner di detailed_labels
        if df_merged['Depression_label'].isnull().any():
            print("Peringatan: Beberapa participant tidak ditemukan di detailed_lables.csv. Fallback ke threshold >= 10.")
            df_merged['binary_label'] = df_merged['Depression_label'].fillna(
                (df_merged['depression_label'] >= 10).astype(int)
            ).astype(int)
        else:
            df_merged['binary_label'] = df_merged['Depression_label'].astype(int)
            
        df_clean['binary_label'] = df_merged['binary_label'].values
    else:
        print(f"Peringatan: File label detail '{labels_path}' tidak ditemukan.")
        print("Menggunakan threshold default 'depression_label >= 10' untuk mendeteksi depresi biner.")
        df_clean['binary_label'] = (df_clean['depression_label'] >= 10).astype(int)
        
    # 3. Pisahkan 20% data sebagai global_test_index.csv (menggunakan stratifikasi label biner)
    print(f"Melakukan split global test set ({args.test_size * 100}%) dengan stratifikasi...")
    
    df_client_pool, df_global_test = train_test_split(
        df_clean,
        test_size=args.test_size,
        stratify=df_clean['binary_label'],
        random_state=args.random_state
    )
    
    # 5. Sisa 80% data digunakan untuk pembagian klien (split menjadi train pool dan validation pool)
    # Kami menggunakan stratifikasi berdasarkan label biner juga
    print(f"Melakukan split sisa data klien menjadi Train pool dan Validation pool ({args.val_size * 100}% dari data klien)...")
    
    df_train_pool, df_val_pool = train_test_split(
        df_client_pool,
        test_size=args.val_size,
        stratify=df_client_pool['binary_label'],
        random_state=args.random_state
    )
    
    # Hapus kolom pembantu 'binary_label' sebelum menyimpan indeks
    df_train_save = df_train_pool.drop(columns=['binary_label'], errors='ignore')
    df_val_save = df_val_pool.drop(columns=['binary_label'], errors='ignore')
    df_test_save = df_global_test.drop(columns=['binary_label'], errors='ignore')
    
    # Pastikan direktori output ada
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tentukan path file output
    train_path = os.path.join(args.output_dir, 'global_train_index.csv')
    val_path = os.path.join(args.output_dir, 'global_validation_index.csv')
    test_path = os.path.join(args.output_dir, 'global_test_index.csv')
    xlsx_path = os.path.join(args.output_dir, 'global_split_distribution.xlsx')
    
    # 6. Simpan hasil split ke CSV
    print("Menyimpan berkas indeks split data...")
    df_train_save.to_csv(train_path, index=False)
    df_val_save.to_csv(val_path, index=False)
    df_test_save.to_csv(test_path, index=False)
    
    print(f"- Saved: {train_path} ({len(df_train_save)} baris)")
    print(f"- Saved: {val_path} ({len(df_val_save)} baris)")
    print(f"- Saved: {test_path} ({len(df_test_save)} baris)")
    
    # Hitung statistik distribusi untuk Excel
    total_all = len(df_clean)
    
    def get_stats(df_split):
        total = len(df_split)
        depresi = int(df_split['binary_label'].sum())
        non_depresi = total - depresi
        pct = (total / total_all) * 100
        return depresi, non_depresi, total, f"{pct:.1f}%"
        
    stats_train = get_stats(df_train_pool)
    stats_val = get_stats(df_val_pool)
    stats_test = get_stats(df_global_test)
    
    # Buat DataFrame distribusi split
    dist_data = {
        'Split': ['Train pool', 'Validation pool', 'Global test'],
        'Depresi': [stats_train[0], stats_val[0], stats_test[0]],
        'Non-depresi': [stats_train[1], stats_val[1], stats_test[1]],
        'Total': [stats_train[2], stats_val[2], stats_test[2]],
        'Persentase': [stats_train[3], stats_val[3], stats_test[3]]
    }
    
    df_dist = pd.DataFrame(dist_data)
    
    # Cetak tabel statistik di terminal
    print("\n--- Ringkasan Distribusi Split Data ---")
    print(df_dist.to_string(index=False))
    print("---------------------------------------")
    
    # Simpan hasil split_distribution ke Excel
    print(f"Menyimpan statistik distribusi ke: {xlsx_path}")
    try:
        df_dist.to_excel(xlsx_path, index=False, sheet_name='Distribution')
        print("Penyimpanan Excel berhasil.")
    except Exception as e:
        print(f"Error saat menulis ke Excel: {e}")
        
    # Verifikasi integritas split (mencegah overlapping participant)
    train_pids = set(df_train_pool['participant_id'])
    val_pids = set(df_val_pool['participant_id'])
    test_pids = set(df_global_test['participant_id'])
    
    intersect_train_val = train_pids.intersection(val_pids)
    intersect_train_test = train_pids.intersection(test_pids)
    intersect_val_test = val_pids.intersection(test_pids)
    
    print("\n=== Validasi Integritas Split ===")
    if len(intersect_train_val) == 0 and len(intersect_train_test) == 0 and len(intersect_val_test) == 0:
        print("OK: Tidak ada tumpang tindih participant_id antara set Train, Validation, dan Test.")
    else:
        print("ERROR: Ditemukan kebocoran data (overlapping participant IDs)!")
        if len(intersect_train_val) > 0:
            print(f"  Overlap Train-Val: {intersect_train_val}")
        if len(intersect_train_test) > 0:
            print(f"  Overlap Train-Test: {intersect_train_test}")
        if len(intersect_val_test) > 0:
            print(f"  Overlap Val-Test: {intersect_val_test}")
            
    print(f"Total partisipan unik di seluruh split: {len(train_pids) + len(val_pids) + len(test_pids)} dari {total_all}")

if __name__ == '__main__':
    main()
