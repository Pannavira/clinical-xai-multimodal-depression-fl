import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.7 — Pembuatan Skenario IID (Partisi Klien dan Validasi Lokal)")
    parser.add_argument(
        '--input_train', 
        type=str, 
        default='federated_learning_output/global_train_index.csv',
        help='Jalur file index train global (CSV)'
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
        default='federated_learning_output/IID_3Clients',
        help='Direktori penyimpanan hasil split klien IID'
    )
    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=3,
        help='Jumlah klien federated'
    )
    parser.add_argument(
        '--val_size', 
        type=float, 
        default=0.15,
        help='Proporsi data klien untuk validasi lokal (0.0 - 1.0)'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state seed untuk replikasi hasil split'
    )
    return parser.parse_args()

def partition_stratified(df, num_clients, random_state):
    """
    Membagi dataframe ke sejumlah klien secara IID terstratifikasi
    berdasarkan kolom 'binary_label'.
    """
    groups = df.groupby('binary_label')
    client_lists = [[] for _ in range(num_clients)]
    
    for label, group in groups:
        # Acak kelompok sampel
        group_shuffled = group.sample(frac=1.0, random_state=random_state)
        # Ambil daftar indeks
        indices = group_shuffled.index.tolist()
        # Bagi indeks sama rata
        chunks = np.array_split(indices, num_clients)
        for i in range(num_clients):
            client_lists[i].append(df.loc[chunks[i]])
            
    client_dfs = []
    for i in range(num_clients):
        # Gabungkan dan acak ulang agar baris tercampur
        df_client = pd.concat(client_lists[i]).sample(frac=1.0, random_state=random_state)
        client_dfs.append(df_client)
        
    return client_dfs

def main():
    args = parse_args()
    
    print("=== Langkah 4.7 — Pembuatan Skenario IID ===")
    print(f"Jalur Input Train Global: {args.input_train}")
    print(f"Jalur Label Detail: {args.labels}")
    print(f"Direktori Output: {args.output_dir}")
    print(f"Jumlah Klien: {args.num_clients}")
    print(f"Ukuran Validasi Lokal: {args.val_size * 100}%")
    print(f"Random State Seed: {args.random_state}")
    
    # 1. Pastikan input train global ada
    if not os.path.exists(args.input_train):
        print(f"Error: File input '{args.input_train}' tidak ditemukan!")
        return
        
    # Muat global train pool
    df_global_train = pd.read_csv(args.input_train)
    total_train_samples = len(df_global_train)
    print(f"Berhasil memuat global train pool. Total sampel: {total_train_samples}")
    
    # 2. Pemuatan dan pemetaan label depresi biner
    if os.path.exists(args.labels):
        print(f"Membaca label detail dari: {args.labels}")
        df_labels = pd.read_csv(args.labels)
        df_labels['Participant'] = df_labels['Participant'].astype(int)
        
        label_dict = dict(zip(df_labels['Participant'], df_labels['Depression_label']))
        
        # Tambahkan kolom biner untuk stratifikasi
        df_global_train['binary_label'] = df_global_train['participant_id'].map(label_dict).fillna(
            (df_global_train['depression_label'] >= 10).astype(int)
        ).astype(int)
    else:
        print(f"Peringatan: File label detail '{args.labels}' tidak ditemukan. Menggunakan fallback 'depression_label >= 10'.")
        df_global_train['binary_label'] = (df_global_train['depression_label'] >= 10).astype(int)
        
    print("\n--- Distribusi Depresi Global Train Pool ---")
    val_counts = df_global_train['binary_label'].value_counts()
    for val, count in val_counts.items():
        label_str = "Depresi (1)" if val == 1 else "Non-depresi (0)"
        print(f"- {label_str}: {count} sampel ({(count / total_train_samples) * 100:.1f}%)")
        
    # 3. Bagi data ke sejumlah klien secara seimbang (IID)
    client_dfs = partition_stratified(df_global_train, args.num_clients, args.random_state)
    
    # Buat direktori output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List untuk menyimpan split train & validation per klien
    client_train_dfs = []
    client_val_dfs = []
    
    # 4 & 5. Lakukan pembagian local train & local validation untuk masing-masing klien
    print("\n=== Melakukan Pembagian Latih & Validasi Lokal ===")
    for i in range(args.num_clients):
        c_idx = i + 1
        df_client_all = client_dfs[i]
        
        # Split menjadi train (1 - val_size) dan validation (val_size) secara terstratifikasi
        df_c_train, df_c_val = train_test_split(
            df_client_all,
            test_size=args.val_size,
            stratify=df_client_all['binary_label'],
            random_state=args.random_state
        )
        
        client_train_dfs.append(df_c_train)
        client_val_dfs.append(df_c_val)
        
        # Buat salinan untuk disimpan (tanpa kolom helper binary_label)
        df_train_save = df_c_train.drop(columns=['binary_label'], errors='ignore')
        df_val_save = df_c_val.drop(columns=['binary_label'], errors='ignore')
        
        train_path = os.path.join(args.output_dir, f'client_{c_idx}_train.csv')
        val_path = os.path.join(args.output_dir, f'client_{c_idx}_val.csv')
        
        df_train_save.to_csv(train_path, index=False)
        df_val_save.to_csv(val_path, index=False)
        
        print(f"Klien {c_idx}:")
        print(f"  - Total sampel: {len(df_client_all)}")
        print(f"  - Simpan Latih    : {train_path} ({len(df_train_save)} sampel)")
        print(f"  - Simpan Validasi : {val_path} ({len(df_val_save)} sampel)")

    # 6. Buat Laporan Distribusi IID (iid_distribution_report.xlsx)
    report_path = os.path.join(args.output_dir, 'iid_distribution_report.xlsx')
    print(f"\nMenyusun laporan distribusi ke: {report_path}")
    
    summary_data = []
    modality_data = []
    
    for i in range(args.num_clients):
        c_idx = i + 1
        tr_df = client_train_dfs[i]
        vl_df = client_val_dfs[i]
        all_df = client_dfs[i]
        
        # Hitung statistik
        tr_total = len(tr_df)
        tr_dep = int(tr_df['binary_label'].sum())
        tr_non = tr_total - tr_dep
        tr_pct = (tr_dep / tr_total * 100) if tr_total > 0 else 0.0
        
        vl_total = len(vl_df)
        vl_dep = int(vl_df['binary_label'].sum())
        vl_non = vl_total - vl_dep
        vl_pct = (vl_dep / vl_total * 100) if vl_total > 0 else 0.0
        
        all_total = len(all_df)
        all_dep = int(all_df['binary_label'].sum())
        all_non = all_total - all_dep
        all_pct = (all_dep / all_total * 100) if all_total > 0 else 0.0
        
        summary_data.append({
            'Client': f'Client {c_idx}',
            'Train Depresi': tr_dep,
            'Train Non-depresi': tr_non,
            'Train Total': tr_total,
            'Train Persentase Depresi': f"{tr_pct:.1f}%",
            'Val Depresi': vl_dep,
            'Val Non-depresi': vl_non,
            'Val Total': vl_total,
            'Val Persentase Depresi': f"{vl_pct:.1f}%",
            'Client Total': all_total,
            'Client Persentase Depresi': f"{all_pct:.1f}%"
        })
        
        # Hitung modalitas (Teks, Audio, Visual) yang terisi
        # (Seluruh baris memiliki data lengkap 3 modalitas di global_train)
        has_text = len(all_df[all_df['text_feature_path'].notna() & (all_df['text_feature_path'].str.strip() != '')])
        has_audio = len(all_df[all_df['audio_feature_path'].notna() & (all_df['audio_feature_path'].str.strip() != '')])
        has_visual = len(all_df[all_df['visual_feature_path'].notna() & (all_df['visual_feature_path'].str.strip() != '')])
        
        modality_data.append({
            'Client': f'Client {c_idx}',
            'Sampel Teks': has_text,
            'Persentase Teks': f"{(has_text/all_total)*100:.1f}%",
            'Sampel Audio': has_audio,
            'Persentase Audio': f"{(has_audio/all_total)*100:.1f}%",
            'Sampel Visual': has_visual,
            'Persentase Visual': f"{(has_visual/all_total)*100:.1f}%",
            'Total': all_total
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_modality = pd.DataFrame(modality_data)
    
    # Tampilkan summary di terminal
    print("\n--- Ringkasan Distribusi IID ---")
    print(df_summary[['Client', 'Client Total', 'Client Persentase Depresi']])
    print("--------------------------------")
    
    # Tulis Excel menggunakan openpyxl
    try:
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, index=False, sheet_name='Summary')
            df_modality.to_excel(writer, index=False, sheet_name='Modality Composition')
        print("Laporan Excel 'iid_distribution_report.xlsx' berhasil disimpan.")
    except Exception as e:
        print(f"Error saat menulis file laporan Excel: {e}")
        
    # 7. Buat Log Partisi Detail (iid_partition_log.xlsx)
    log_path = os.path.join(args.output_dir, 'iid_partition_log.xlsx')
    print(f"Menyusun log partisi detail ke: {log_path}")
    
    log_rows = []
    for i in range(args.num_clients):
        c_idx = i + 1
        
        # Train data log
        for _, row in client_train_dfs[i].iterrows():
            log_rows.append({
                'participant_id': int(row['participant_id']),
                'depression_label': int(row['depression_label']),
                'binary_label': int(row['binary_label']),
                'client_assignment': f'Client {c_idx}',
                'local_split': 'Train',
                'text_feature_path': row['text_feature_path'],
                'audio_feature_path': row['audio_feature_path'],
                'visual_feature_path': row['visual_feature_path']
            })
            
        # Val data log
        for _, row in client_val_dfs[i].iterrows():
            log_rows.append({
                'participant_id': int(row['participant_id']),
                'depression_label': int(row['depression_label']),
                'binary_label': int(row['binary_label']),
                'client_assignment': f'Client {c_idx}',
                'local_split': 'Validation',
                'text_feature_path': row['text_feature_path'],
                'audio_feature_path': row['audio_feature_path'],
                'visual_feature_path': row['visual_feature_path']
            })
            
    df_log = pd.DataFrame(log_rows)
    df_log = df_log.sort_values(by=['client_assignment', 'local_split', 'participant_id'])
    
    try:
        with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
            df_log.to_excel(writer, index=False, sheet_name='Partition Log')
        print("Log Excel 'iid_partition_log.xlsx' berhasil disimpan.")
    except Exception as e:
        print(f"Error saat menulis file log Excel: {e}")
        
    # === Validasi Integritas Data ===
    print("\n=== Validasi Integritas Data ===")
    
    # 1. Cek overlap participant antar klien
    overlap_clients = False
    client_pids = [set(df['participant_id']) for df in client_dfs]
    for i in range(args.num_clients):
        for j in range(i + 1, args.num_clients):
            intersect = client_pids[i].intersection(client_pids[j])
            if len(intersect) > 0:
                print(f"ERROR: Tumpang tindih partisipan terdeteksi antara Client {i+1} dan Client {j+1}: {intersect}")
                overlap_clients = True
    if not overlap_clients:
        print("OK: Tidak ada tumpang tindih partisipan antar klien.")
        
    # 2. Cek overlap train-val pada klien yang sama
    overlap_train_val = False
    for i in range(args.num_clients):
        tr_pids = set(client_train_dfs[i]['participant_id'])
        vl_pids = set(client_val_dfs[i]['participant_id'])
        intersect = tr_pids.intersection(vl_pids)
        if len(intersect) > 0:
            print(f"ERROR: Tumpang tindih train-val terdeteksi pada Client {i+1}: {intersect}")
            overlap_train_val = True
    if not overlap_train_val:
        print("OK: Tidak ada tumpang tindih train-val pada klien yang sama.")
        
    # 3. Cek jumlah total baris cocok dengan global train pool
    total_partitioned = sum(len(df) for df in client_train_dfs) + sum(len(df) for df in client_val_dfs)
    if total_partitioned == total_train_samples:
        print(f"OK: Jumlah total sampel terpartisi ({total_partitioned}) cocok dengan dataset awal ({total_train_samples}).")
    else:
        print(f"ERROR: Ketidakcocokan jumlah total sampel! Terpartisi={total_partitioned}, Awal={total_train_samples}")

if __name__ == '__main__':
    main()
