import os
import argparse
import pandas as pd
import numpy as np
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.6 — Membuat Data Latih dan Validasi Klien untuk Federated Learning")
    parser.add_argument(
        '--config', 
        type=str, 
        default='preprocessing/federated_partition_config.yaml',
        help='Jalur berkas konfigurasi YAML'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='federated_learning_output',
        help='Direktori penyimpanan global train dan validation index'
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
        help='Direktori penyimpanan hasil split klien'
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

def partition_dirichlet(df, num_clients, alpha, random_state):
    """
    Membagi dataframe ke sejumlah klien secara Non-IID dengan alokasi Dirichlet
    berdasarkan kolom 'binary_label'.
    """
    np.random.seed(random_state)
    groups = df.groupby('binary_label')
    client_indices = [[] for _ in range(num_clients)]
    
    # Simpan indeks asli agar loc berfungsi dengan benar
    df_with_reset_idx = df.copy()
    
    for label, group in groups:
        indices = group.index.tolist()
        np.random.shuffle(indices)
        
        # Dirichlet alokasi proporsi
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Hitung jumlah sampel per klien
        counts = np.round(proportions * len(indices)).astype(int)
        
        # Koreksi pembulatan jika total jumlah tidak sesuai jumlah grup asli
        diff = len(indices) - sum(counts)
        if diff != 0:
            idx = np.argmax(counts)
            counts[idx] += diff
            
        # Distribusikan indeks
        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(indices[start:end])
            start = end
            
    client_dfs = []
    for i in range(num_clients):
        df_client = df_with_reset_idx.loc[client_indices[i]].sample(frac=1.0, random_state=random_state)
        client_dfs.append(df_client)
        
    return client_dfs

def main():
    args = parse_args()
    
    print("=== Langkah 4.6 — Pemisahan Data Latih dan Validasi Klien ===")
    
    # Muat konfigurasi dari YAML jika tersedia
    num_clients = 3
    distribution_method = 'iid'
    non_iid_alpha = 0.5
    random_state = 42
    
    if os.path.exists(args.config):
        print(f"Membaca konfigurasi dari: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if 'federated_clients' in config:
            client_cfg = config['federated_clients']
            num_clients = client_cfg.get('num_clients', num_clients)
            distribution_method = client_cfg.get('distribution_method', distribution_method).lower()
            non_iid_alpha = client_cfg.get('non_iid_alpha', non_iid_alpha)
            random_state = client_cfg.get('random_state', random_state)
            
    print(f"Konfigurasi FL:")
    print(f"- Jumlah Klien: {num_clients}")
    print(f"- Metode Distribusi: {distribution_method.upper()}")
    if distribution_method == 'non_iid':
        print(f"- Dirichlet Alpha: {non_iid_alpha}")
    print(f"- Random State Seed: {random_state}")
    
    # Jalur masukan
    train_pool_path = os.path.join(args.input_dir, 'global_train_index.csv')
    val_pool_path = os.path.join(args.input_dir, 'global_validation_index.csv')
    labels_path = args.labels
    
    if not os.path.exists(train_pool_path) or not os.path.exists(val_pool_path):
        print(f"Error: Berkas global train/val index tidak ditemukan di {args.input_dir}!")
        print("Pastikan Langkah 4.5 sudah dijalankan.")
        return
        
    # Muat data pool
    df_train = pd.read_csv(train_pool_path)
    df_val = pd.read_csv(val_pool_path)
    
    print(f"Membaca train pool ({len(df_train)} baris) dan validation pool ({len(df_val)} baris)...")
    
    # Ambil label biner untuk stratifikasi / partisi
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        df_labels['Participant'] = df_labels['Participant'].astype(int)
        
        label_dict = dict(zip(df_labels['Participant'], df_labels['Depression_label']))
        
        df_train['binary_label'] = df_train['participant_id'].map(label_dict).fillna(
            (df_train['depression_label'] >= 10).astype(int)
        ).astype(int)
        
        df_val['binary_label'] = df_val['participant_id'].map(label_dict).fillna(
            (df_val['depression_label'] >= 10).astype(int)
        ).astype(int)
    else:
        print("Peringatan: File label detail tidak ditemukan. Menggunakan threshold skor >= 10.")
        df_train['binary_label'] = (df_train['depression_label'] >= 10).astype(int)
        df_val['binary_label'] = (df_val['depression_label'] >= 10).astype(int)
        
    # Lakukan partisi data train dan validation ke masing-masing klien
    if distribution_method == 'iid':
        client_train_dfs = partition_stratified(df_train, num_clients, random_state)
        client_val_dfs = partition_stratified(df_val, num_clients, random_state)
    elif distribution_method == 'non_iid':
        client_train_dfs = partition_dirichlet(df_train, num_clients, non_iid_alpha, random_state)
        # Untuk validation set, kita juga membagi secara non-IID mengikuti distribusi train
        client_val_dfs = partition_dirichlet(df_val, num_clients, non_iid_alpha, random_state)
    else:
        print(f"Error: Metode distribusi '{distribution_method}' tidak dikenali. Pilih 'iid' atau 'non_iid'.")
        return
        
    # Pastikan direktori output ada
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Simpan hasil partisi dan tampilkan statistik
    print("\n=== Menyimpan Berkas Indeks Klien ===")
    for i in range(num_clients):
        client_idx = i + 1
        
        # Hapus kolom pembantu binary_label sebelum diekspor
        df_train_save = client_train_dfs[i].drop(columns=['binary_label'], errors='ignore')
        df_val_save = client_val_dfs[i].drop(columns=['binary_label'], errors='ignore')
        
        train_out_path = os.path.join(args.output_dir, f'client_{client_idx}_train.csv')
        val_out_path = os.path.join(args.output_dir, f'client_{client_idx}_val.csv')
        
        df_train_save.to_csv(train_out_path, index=False)
        df_val_save.to_csv(val_out_path, index=False)
        
        print(f"Klien {client_idx}:")
        print(f"- Saved Latih: {train_out_path} ({len(df_train_save)} baris)")
        print(f"- Saved Validasi: {val_out_path} ({len(df_val_save)} baris)")
        
    # Tampilkan Ringkasan Statistik Distribusi Klien
    stats_rows = []
    
    # 1. Total Pool Latih
    tr_total = len(df_train)
    tr_dep = df_train['binary_label'].sum()
    tr_non = tr_total - tr_dep
    stats_rows.append({
        'Klien/Pool': 'Global Train Pool',
        'Set': 'Train',
        'Depresi': tr_dep,
        'Non-depresi': tr_non,
        'Total': tr_total,
        'Rasio Depresi': f"{(tr_dep/tr_total)*100:.1f}%"
    })
    
    # 2. Total Pool Validasi
    vl_total = len(df_val)
    vl_dep = df_val['binary_label'].sum()
    vl_non = vl_total - vl_dep
    stats_rows.append({
        'Klien/Pool': 'Global Val Pool',
        'Set': 'Validation',
        'Depresi': vl_dep,
        'Non-depresi': vl_non,
        'Total': vl_total,
        'Rasio Depresi': f"{(vl_dep/vl_total)*100:.1f}%"
    })
    
    # 3. Klien detail
    for i in range(num_clients):
        c_idx = i + 1
        
        # Train stats
        tr_c_total = len(client_train_dfs[i])
        tr_c_dep = client_train_dfs[i]['binary_label'].sum()
        tr_c_non = tr_c_total - tr_c_dep
        stats_rows.append({
            'Klien/Pool': f'Client {c_idx}',
            'Set': 'Train',
            'Depresi': tr_c_dep,
            'Non-depresi': tr_c_non,
            'Total': tr_c_total,
            'Rasio Depresi': f"{(tr_c_dep/tr_c_total)*100:.1f}%" if tr_c_total > 0 else "0.0%"
        })
        
        # Val stats
        vl_c_total = len(client_val_dfs[i])
        vl_c_dep = client_val_dfs[i]['binary_label'].sum()
        vl_c_non = vl_c_total - vl_c_dep
        stats_rows.append({
            'Klien/Pool': f'Client {c_idx}',
            'Set': 'Validation',
            'Depresi': vl_c_dep,
            'Non-depresi': vl_c_non,
            'Total': vl_c_total,
            'Rasio Depresi': f"{(vl_c_dep/vl_c_total)*100:.1f}%" if vl_c_total > 0 else "0.0%"
        })
        
    df_stats = pd.DataFrame(stats_rows)
    print("\n--- Ringkasan Distribusi Data Klien ---")
    print(df_stats.to_string(index=False))
    print("----------------------------------------")
    
    # Ekspor / Update Excel file dengan sheet baru
    xlsx_path = os.path.join(args.output_dir, 'global_split_distribution.xlsx')
    print(f"Memperbarui berkas Excel dengan data klien di: {xlsx_path}")
    try:
        # Jika file sudah ada, gunakan ExcelWriter mode overlay/append
        if os.path.exists(xlsx_path):
            with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_stats.to_excel(writer, index=False, sheet_name='Client Distribution')
        else:
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df_stats.to_excel(writer, index=False, sheet_name='Client Distribution')
        print("Excel berhasil diperbarui.")
    except Exception as e:
        print(f"Peringatan: Gagal menulis sheet ke Excel: {e}")
        
    # Verifikasi Integritas Split Klien
    print("\n=== Validasi Integritas Split Klien ===")
    
    # 1. Pastikan tidak ada overlap participant antar klien pada set training
    train_pids_list = [set(df['participant_id']) for df in client_train_dfs]
    overlap_found = False
    
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            intersect = train_pids_list[i].intersection(train_pids_list[j])
            if len(intersect) > 0:
                print(f"ERROR: Tumpang tindih participant_id terdeteksi antara Client {i+1} Latih dan Client {j+1} Latih: {intersect}")
                overlap_found = True
                
    # 2. Pastikan tidak ada overlap participant antar set train & val milik klien yang sama
    for i in range(num_clients):
        tr_set = set(client_train_dfs[i]['participant_id'])
        vl_set = set(client_val_dfs[i]['participant_id'])
        intersect = tr_set.intersection(vl_set)
        if len(intersect) > 0:
            print(f"ERROR: Tumpang tindih participant_id terdeteksi di Client {i+1} antara Train dan Val: {intersect}")
            overlap_found = True
            
    # 3. Pastikan jumlah total baris train dan val sama dengan total pool
    total_train_partitioned = sum(len(df) for df in client_train_dfs)
    total_val_partitioned = sum(len(df) for df in client_val_dfs)
    
    if total_train_partitioned == len(df_train) and total_val_partitioned == len(df_val):
        print(f"Sukses: Jumlah baris data terpartisi cocok dengan pool asal (Train: {total_train_partitioned}/{len(df_train)}, Val: {total_val_partitioned}/{len(df_val)}).")
    else:
        print(f"ERROR: Ketidakcocokan jumlah baris!")
        print(f"  Train: Partitioned={total_train_partitioned}, Pool={len(df_train)}")
        print(f"  Val: Partitioned={total_val_partitioned}, Pool={len(df_val)}")
        overlap_found = True
        
    if not overlap_found:
        print("OK: Seluruh validasi integritas lolos tanpa masalah kebocoran data.")
        
if __name__ == '__main__':
    main()
