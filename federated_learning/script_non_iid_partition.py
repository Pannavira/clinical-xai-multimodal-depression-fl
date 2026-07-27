import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.8 - 4.11 — Pembuatan Skenario Non-IID (Label, Quantity, Modality, dan Gabungan)")
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
        default='federated_learning_output',
        help='Direktori dasar penyimpanan hasil split klien Non-IID'
    )
    parser.add_argument(
        '--scenario', 
        type=str, 
        default='all',
        choices=['label_skew', 'quantity_skew', 'modality_skew', 'combined', 'all'],
        help='Skenario Non-IID yang ingin dibuat (default: all)'
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

def load_and_merge_data(train_path, labels_path):
    """
    Memuat dataset global_train dan menggabungkan dengan ground-truth label biner
    serta indikator kualitas modalitas dari file preprocessing.
    """
    df_train = pd.read_csv(train_path)
    df_labels = pd.read_csv(labels_path)
    df_labels['Participant'] = df_labels['Participant'].astype(int)
    
    # 1. Peta label biner
    label_dict = dict(zip(df_labels['Participant'], df_labels['Depression_label']))
    df_train['binary_label'] = df_train['participant_id'].map(label_dict).fillna(
        (df_train['depression_label'] >= 10).astype(int)
    ).astype(int)
    
    # 2. Peta token teks (Text quality/length)
    token_path = 'preprocessing_output/text/tokenized_text.csv'
    if os.path.exists(token_path):
        df_text = pd.read_csv(token_path)
        df_text['pid_clean'] = df_text['participant_id'].astype(str).str.replace('P', '').astype(int)
        text_dict = dict(zip(df_text['pid_clean'], df_text['token_count']))
        df_train['token_count'] = df_train['participant_id'].map(text_dict).fillna(0).astype(int)
    else:
        print("Peringatan: tokenized_text.csv tidak ditemukan. Token count diset default.")
        df_train['token_count'] = 1000
        
    # 3. Peta visual: face detection ratio dan session duration
    vis_path = 'preprocessing_output/visual/video_preprocessing_log.xlsx'
    if os.path.exists(vis_path):
        xls = pd.ExcelFile(vis_path)
        df_vis = xls.parse(xls.sheet_names[0])
        df_vis['pid_clean'] = df_vis['participant_id'].astype(int)
        vis_ratio_dict = dict(zip(df_vis['pid_clean'], df_vis['valid_frame_ratio']))
        duration_dict = dict(zip(df_vis['pid_clean'], df_vis['session_duration_sec']))
        df_train['valid_frame_ratio'] = df_train['participant_id'].map(vis_ratio_dict).fillna(1.0)
        df_train['session_duration_sec'] = df_train['participant_id'].map(duration_dict).fillna(300.0)
    else:
        print("Peringatan: video_preprocessing_log.xlsx tidak ditemukan. Visual ratio diset default.")
        df_train['valid_frame_ratio'] = 1.0
        df_train['session_duration_sec'] = 300.0
        
    # 4. Peta audio shimmer (Audio noise indicator)
    audio_path = 'preprocessing_output/audio/pitch_energy_features.csv'
    if os.path.exists(audio_path):
        df_audio = pd.read_csv(audio_path)
        df_audio['pid_clean'] = df_audio['participant_id'].astype(str).str.replace('P', '').astype(int)
        shimmer_dict = dict(zip(df_audio['pid_clean'], df_audio['shimmer']))
        df_train['shimmer'] = df_train['participant_id'].map(shimmer_dict).fillna(0.0)
    else:
        print("Peringatan: pitch_energy_features.csv tidak ditemukan. Shimmer diset default.")
        df_train['shimmer'] = 0.0
        
    return df_train

def determine_label_skew_counts(total_d, total_n, num_clients):
    """
    Menghitung proporsi depresi secara dinamis agar termaksimalkan skew labelnya.
    Client 1 (dominan depresi), Client 2 (seimbang), Client 3 (dominan non-depresi).
    """
    if num_clients == 3 and total_d == 42 and total_n == 134:
        return [(18, 8), (16, 16), (8, 110)]
    
    if num_clients == 3:
        best_partition = None
        best_score = float('inf')
        for d1 in range(2, total_d - 3):
            for d2 in range(2, total_d - d1 - 1):
                d3 = total_d - d1 - d2
                for n1 in range(2, total_n - 3):
                    for n2 in range(2, total_n - n1 - 1):
                        n3 = total_n - n1 - n2
                        
                        tot1 = d1 + n1
                        tot2 = d2 + n2
                        tot3 = d3 + n3
                        
                        if not (25 <= tot1 <= 125 and 25 <= tot2 <= 125 and 25 <= tot3 <= 125):
                            continue
                        
                        r1 = d1 / tot1
                        r2 = d2 / tot2
                        r3 = d3 / tot3
                        
                        if not (r1 > r2 > r3):
                            continue
                        
                        err = (r1 - 0.70)**2 + (r2 - 0.50)**2 + (r3 - 0.20)**2
                        if err < best_score:
                            best_score = err
                            best_partition = [(d1, n1), (d2, n2), (d3, n3)]
        if best_partition:
            return best_partition
            
    # Generic fallback
    d_chunks = np.array_split(range(total_d), num_clients)
    n_chunks = np.array_split(range(total_n), num_clients)
    return [(len(d_chunks[i]), len(n_chunks[i])) for i in range(num_clients)]

def make_label_skew(df_train, num_clients, random_state):
    """
    Membagi data ke klien dengan bias label depresi yang kuat.
    """
    total_d = int(df_train['binary_label'].sum())
    total_n = len(df_train) - total_d
    
    counts = determine_label_skew_counts(total_d, total_n, num_clients)
    
    depressed = df_train[df_train['binary_label'] == 1].sample(frac=1.0, random_state=random_state)
    non_depressed = df_train[df_train['binary_label'] == 0].sample(frac=1.0, random_state=random_state)
    
    client_dfs = []
    d_start, n_start = 0, 0
    for i in range(num_clients):
        d_cnt, n_cnt = counts[i]
        d_slice = depressed.iloc[d_start : d_start + d_cnt]
        n_slice = non_depressed.iloc[n_start : n_start + n_cnt]
        c_df = pd.concat([d_slice, n_slice]).sample(frac=1.0, random_state=random_state)
        client_dfs.append(c_df)
        d_start += d_cnt
        n_start += n_cnt
        
    return client_dfs

def make_quantity_skew(df_train, num_clients, random_state):
    """
    Membagi data ke klien dengan jumlah data tidak sama (rasio target 10:6:3).
    """
    total_d = int(df_train['binary_label'].sum())
    total_n = len(df_train) - total_d
    
    if num_clients == 3 and total_d == 42 and total_n == 134:
        counts = [(22, 71), (13, 42), (7, 21)]  # Total sizes: 93, 55, 28
    else:
        ratios = [10.0, 6.0, 3.0]
        if len(ratios) < num_clients:
            ratios = ratios + [1.0] * (num_clients - len(ratios))
        ratios = np.array(ratios[:num_clients])
        ratios = ratios / ratios.sum()
        
        tot_sizes = np.round(ratios * len(df_train)).astype(int)
        diff = len(df_train) - tot_sizes.sum()
        if diff != 0:
            tot_sizes[0] += diff
            
        counts = []
        for size in tot_sizes:
            d_count = int(np.round(size * (total_d / len(df_train))))
            n_count = size - d_count
            counts.append((d_count, n_count))
            
        d_sum = sum(c[0] for c in counts)
        n_sum = sum(c[1] for c in counts)
        d_diff = total_d - d_sum
        n_diff = total_n - n_sum
        
        counts[0] = (counts[0][0] + d_diff, counts[0][1] + n_diff)
        
    depressed = df_train[df_train['binary_label'] == 1].sample(frac=1.0, random_state=random_state)
    non_depressed = df_train[df_train['binary_label'] == 0].sample(frac=1.0, random_state=random_state)
    
    client_dfs = []
    d_start, n_start = 0, 0
    for i in range(num_clients):
        d_cnt, n_cnt = counts[i]
        d_slice = depressed.iloc[d_start : d_start + d_cnt]
        n_slice = non_depressed.iloc[n_start : n_start + n_cnt]
        c_df = pd.concat([d_slice, n_slice]).sample(frac=1.0, random_state=random_state)
        client_dfs.append(c_df)
        d_start += d_cnt
        n_start += n_cnt
        
    return client_dfs

def make_modality_skew(df_train, num_clients, random_state):
    """
    Membagi data dengan skew kualitas modalitas (Client 3 visual rendah, Client 2 audio noisy/pendek, Client 1 audio/video baik).
    Jumlah sampel dan rasio kelas antar klien tetap seimbang (IID label).
    """
    if num_clients != 3:
        # Skew modalitas ini didesain khusus untuk 3 klien baseline
        print("Peringatan: Modality skew didesain khusus untuk 3 klien. Fallback ke stratified random.")
        # Fallback stratified
        groups = df_train.groupby('binary_label')
        client_lists = [[] for _ in range(num_clients)]
        for label, group in groups:
            group_shuffled = group.sample(frac=1.0, random_state=random_state)
            chunks = np.array_split(group_shuffled.index.tolist(), num_clients)
            for i in range(num_clients):
                client_lists[i].append(df_train.loc[chunks[i]])
        return [pd.concat(client_lists[i]).sample(frac=1.0, random_state=random_state) for i in range(num_clients)]
        
    depressed = df_train[df_train['binary_label'] == 1].copy()
    non_depressed = df_train[df_train['binary_label'] == 0].copy()
    
    # Client sizes:
    # Depressed: 14, 14, 14
    # Non-depressed: 44, 45, 45 (Client 1 gets 44, Client 2 & 3 get 45)
    
    # 1. Client 3 (Low Visual): Bottom visual valid_frame_ratio
    dep_sorted_vis = depressed.sort_values(by='valid_frame_ratio', ascending=True)
    d_c3 = dep_sorted_vis.iloc[0:14]
    dep_rem = dep_sorted_vis.iloc[14:]
    
    non_sorted_vis = non_depressed.sort_values(by='valid_frame_ratio', ascending=True)
    n_c3 = non_sorted_vis.iloc[0:45]
    non_rem = non_sorted_vis.iloc[45:]
    
    # 2. Client 2 (Audio noisy/pendek): Bottom session_duration_sec dari sisa data
    dep_sorted_aud = dep_rem.sort_values(by='session_duration_sec', ascending=True)
    d_c2 = dep_sorted_aud.iloc[0:14]
    d_c1 = dep_sorted_aud.iloc[14:]
    
    non_sorted_aud = non_rem.sort_values(by='session_duration_sec', ascending=True)
    n_c2 = non_sorted_aud.iloc[0:45]
    n_c1 = non_sorted_aud.iloc[45:]
    
    # Combine
    c1 = pd.concat([d_c1, n_c1]).sample(frac=1.0, random_state=random_state)
    c2 = pd.concat([d_c2, n_c2]).sample(frac=1.0, random_state=random_state)
    c3 = pd.concat([d_c3, n_c3]).sample(frac=1.0, random_state=random_state)
    
    return [c1, c2, c3]

def make_combined_skew(df_train, num_clients, random_state):
    """
    Skenario Gabungan:
    - Client 1: Besar (93), Depresi Tinggi (32 / 34.4%), Audio Baik
    - Client 2: Sedang (55), Seimbang (8 / 14.5%), Teks Panjang
    - Client 3: Kecil (28), Depresi Rendah (2 / 7.1%), Visual Noisy
    """
    if num_clients != 3:
        print("Peringatan: Combined skew didesain khusus untuk 3 klien. Fallback ke stratified random.")
        # Fallback
        return make_quantity_skew(df_train, num_clients, random_state)
        
    depressed = df_train[df_train['binary_label'] == 1].copy()
    non_depressed = df_train[df_train['binary_label'] == 0].copy()
    
    # 1. Client 3 (Kecil, Visual Noisy): Ambil 2 depressed + 26 non-depressed dengan face detection rate terendah
    dep_vis = depressed.sort_values(by='valid_frame_ratio', ascending=True)
    d_c3 = dep_vis.iloc[0:2]
    dep_rem1 = dep_vis.iloc[2:]
    
    non_vis = non_depressed.sort_values(by='valid_frame_ratio', ascending=True)
    n_c3 = non_vis.iloc[0:26]
    non_rem1 = non_vis.iloc[26:]
    
    # 2. Client 2 (Sedang, Teks Panjang): Dari sisa, ambil 8 depressed + 47 non-depressed dengan token count terbanyak
    dep_text = dep_rem1.sort_values(by='token_count', ascending=False)
    d_c2 = dep_text.iloc[0:8]
    d_c1 = dep_text.iloc[8:]
    
    non_text = non_rem1.sort_values(by='token_count', ascending=False)
    n_c2 = non_text.iloc[0:47]
    n_c1 = non_text.iloc[47:]
    
    # 3. Client 1 (Besar, Audio Baik): Mendapatkan sisa 32 depressed + 61 non-depressed
    c1 = pd.concat([d_c1, n_c1]).sample(frac=1.0, random_state=random_state)
    c2 = pd.concat([d_c2, n_c2]).sample(frac=1.0, random_state=random_state)
    c3 = pd.concat([d_c3, n_c3]).sample(frac=1.0, random_state=random_state)
    
    return [c1, c2, c3]

def write_reports(client_dfs, client_train_dfs, client_val_dfs, output_dir, report_name, log_name, num_clients):
    """
    Menyusun laporan Excel (*_report.xlsx) dan log detail (*_log.xlsx) untuk hasil partisi.
    """
    report_path = os.path.join(output_dir, report_name)
    log_path = os.path.join(output_dir, log_name)
    
    summary_rows = []
    modality_rows = []
    quality_rows = []
    
    for i in range(num_clients):
        c_idx = i + 1
        all_df = client_dfs[i]
        tr_df = client_train_dfs[i]
        vl_df = client_val_dfs[i]
        
        # Hitung rasio label
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
        
        summary_rows.append({
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
        
        # Hitung ketersediaan modalitas
        has_text = len(all_df[all_df['text_feature_path'].notna() & (all_df['text_feature_path'].str.strip() != '')])
        has_audio = len(all_df[all_df['audio_feature_path'].notna() & (all_df['audio_feature_path'].str.strip() != '')])
        has_visual = len(all_df[all_df['visual_feature_path'].notna() & (all_df['visual_feature_path'].str.strip() != '')])
        
        modality_rows.append({
            'Client': f'Client {c_idx}',
            'Sampel Teks': has_text,
            'Persentase Teks': f"{(has_text/all_total)*100:.1f}%",
            'Sampel Audio': has_audio,
            'Persentase Audio': f"{(has_audio/all_total)*100:.1f}%",
            'Sampel Visual': has_visual,
            'Persentase Visual': f"{(has_visual/all_total)*100:.1f}%",
            'Total': all_total
        })
        
        # Hitung rata-rata kualitas fitur modalitas
        mean_tokens = all_df['token_count'].mean()
        mean_vis_ratio = all_df['valid_frame_ratio'].mean()
        mean_duration = all_df['session_duration_sec'].mean()
        mean_shimmer = all_df['shimmer'].mean()
        
        quality_rows.append({
            'Client': f'Client {c_idx}',
            'Rata-rata Token Teks': round(mean_tokens, 1),
            'Rata-rata Face Detection Rate': f"{mean_vis_ratio * 100:.2f}%",
            'Rata-rata Durasi Sesi (detik)': round(mean_duration, 1),
            'Rata-rata Shimmer Audio': round(mean_shimmer, 4)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_modality = pd.DataFrame(modality_rows)
    df_quality = pd.DataFrame(quality_rows)
    
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False, sheet_name='Summary')
        df_modality.to_excel(writer, index=False, sheet_name='Modality Composition')
        df_quality.to_excel(writer, index=False, sheet_name='Quality Averages')
        
    # Buat detail log
    log_rows = []
    for i in range(num_clients):
        c_idx = i + 1
        for split_name, split_df in [('Train', client_train_dfs[i]), ('Validation', client_val_dfs[i])]:
            for _, row in split_df.iterrows():
                log_rows.append({
                    'participant_id': int(row['participant_id']),
                    'depression_label': int(row['depression_label']),
                    'binary_label': int(row['binary_label']),
                    'client_assignment': f'Client {c_idx}',
                    'local_split': split_name,
                    'token_count': int(row['token_count']),
                    'valid_frame_ratio': float(row['valid_frame_ratio']),
                    'session_duration_sec': float(row['session_duration_sec']),
                    'shimmer': float(row['shimmer']),
                    'text_feature_path': row['text_feature_path'],
                    'audio_feature_path': row['audio_feature_path'],
                    'visual_feature_path': row['visual_feature_path']
                })
    df_log = pd.DataFrame(log_rows)
    df_log = df_log.sort_values(by=['client_assignment', 'local_split', 'participant_id'])
    
    with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
        df_log.to_excel(writer, index=False, sheet_name='Partition Log')

def run_partition_workflow(df_global_train, scenario_name, partition_fn, args):
    """
    Alur eksekusi partisi klien, split latih-validasi lokal, penyimpanan berkas,
    pembuatan laporan, dan validasi integritas data untuk skenario tertentu.
    """
    print(f"\n==========================================")
    print(f"Memproses Skenario: {scenario_name.upper()}")
    print(f"==========================================")
    
    # 1. Tentukan subdirektori output
    sub_dir = os.path.join(args.output_dir, scenario_name)
    os.makedirs(sub_dir, exist_ok=True)
    
    # 2. Lakukan partisi data ke klien
    client_dfs = partition_fn(df_global_train, args.num_clients, args.random_state)
    
    client_train_dfs = []
    client_val_dfs = []
    
    # 3. Lakukan split latih/validasi lokal (15% default)
    print("Melakukan split local train/val terstratifikasi...")
    for i in range(args.num_clients):
        c_idx = i + 1
        df_client_all = client_dfs[i]
        
        # Split local
        df_c_train, df_c_val = train_test_split(
            df_client_all,
            test_size=args.val_size,
            stratify=df_client_all['binary_label'],
            random_state=args.random_state
        )
        
        client_train_dfs.append(df_c_train)
        client_val_dfs.append(df_c_val)
        
        # Drop columns helper untuk CSV standard
        cols_to_drop = ['binary_label', 'token_count', 'valid_frame_ratio', 'session_duration_sec', 'shimmer', 'pid_clean']
        df_train_save = df_c_train.drop(columns=cols_to_drop, errors='ignore')
        df_val_save = df_c_val.drop(columns=cols_to_drop, errors='ignore')
        
        train_path = os.path.join(sub_dir, f'client_{c_idx}_train.csv')
        val_path = os.path.join(sub_dir, f'client_{c_idx}_val.csv')
        
        df_train_save.to_csv(train_path, index=False)
        df_val_save.to_csv(val_path, index=False)
        
        print(f"Client {c_idx}: Total={len(df_client_all)}, Train={len(df_train_save)}, Val={len(df_val_save)}")
        
    # 4. Tentukan nama file laporan
    if scenario_name == 'NonIID_LabelSkew':
        rep_name = 'label_skew_distribution_report.xlsx'
        log_name = 'label_skew_partition_log.xlsx'
    elif scenario_name == 'NonIID_QuantitySkew':
        rep_name = 'quantity_skew_distribution_report.xlsx'
        log_name = 'quantity_skew_partition_log.xlsx'
    elif scenario_name == 'NonIID_ModalitySkew':
        rep_name = 'modality_skew_distribution_report.xlsx'
        log_name = 'modality_skew_partition_log.xlsx'
    elif scenario_name == 'NonIID_Combined':
        rep_name = 'combined_noniid_distribution_report.xlsx'
        log_name = 'combined_partition_log.xlsx'
    else:
        rep_name = 'distribution_report.xlsx'
        log_name = 'partition_log.xlsx'
        
    # 5. Tulis laporan Excel
    write_reports(client_dfs, client_train_dfs, client_val_dfs, sub_dir, rep_name, log_name, args.num_clients)
    print(f"Laporan & Log Excel berhasil disimpan di: {sub_dir}")
    
    # 6. Jalankan pemeriksaan integritas otomatis
    print("\n--- Pemeriksaan Integritas Data ---")
    overlap_clients = False
    client_pids = [set(df['participant_id']) for df in client_dfs]
    for i in range(args.num_clients):
        for j in range(i + 1, args.num_clients):
            intersect = client_pids[i].intersection(client_pids[j])
            if len(intersect) > 0:
                print(f"  [ERROR] Overlap partisipan antara Client {i+1} dan Client {j+1}: {intersect}")
                overlap_clients = True
    if not overlap_clients:
        print("  [OK] Tidak ada tumpang tindih partisipan antar klien (Zero Client Leakage).")
        
    overlap_train_val = False
    for i in range(args.num_clients):
        tr_pids = set(client_train_dfs[i]['participant_id'])
        vl_pids = set(client_val_dfs[i]['participant_id'])
        intersect = tr_pids.intersection(vl_pids)
        if len(intersect) > 0:
            print(f"  [ERROR] Overlap train-val pada Client {i+1}: {intersect}")
            overlap_train_val = True
    if not overlap_train_val:
        print("  [OK] Tidak ada tumpang tindih train-val pada klien yang sama (Zero Local Leakage).")
        
    total_partitioned = sum(len(df) for df in client_train_dfs) + sum(len(df) for df in client_val_dfs)
    if total_partitioned == len(df_global_train):
        print(f"  [OK] Jumlah total sampel terpartisi ({total_partitioned}) cocok dengan dataset awal ({len(df_global_train)}).")
    else:
        print(f"  [ERROR] Jumlah sampel terpartisi ({total_partitioned}) TIDAK cocok dengan dataset awal ({len(df_global_train)})!")
        
    # Tampilkan pembuktian karakteristik modality/combined skew
    if 'Modality' in scenario_name or 'Combined' in scenario_name:
        print("\n--- Verifikasi Karakteristik Modalitas Klien ---")
        for i in range(args.num_clients):
            c_idx = i + 1
            all_df = client_dfs[i]
            print(f"  Client {c_idx}:")
            print(f"    - Rata-rata token teks: {all_df['token_count'].mean():.1f}")
            print(f"    - Rata-rata face detection rate: {all_df['valid_frame_ratio'].mean() * 100:.2f}%")
            print(f"    - Rata-rata durasi sesi: {all_df['session_duration_sec'].mean():.1f} detik")

def main():
    args = parse_args()
    
    print("=== Langkah 4.8 - 4.11 — Pembuatan Skenario Data Heterogen (Non-IID) ===")
    
    # Muat data awal terpadu
    df_global_train = load_and_merge_data(args.input_train, args.labels)
    print(f"Total data pool global: {len(df_global_train)} baris.")
    
    # Jalankan workflow berdasarkan skenario yang dipilih
    if args.scenario in ['label_skew', 'all']:
        run_partition_workflow(df_global_train, 'NonIID_LabelSkew', make_label_skew, args)
        
    if args.scenario in ['quantity_skew', 'all']:
        run_partition_workflow(df_global_train, 'NonIID_QuantitySkew', make_quantity_skew, args)
        
    if args.scenario in ['modality_skew', 'all']:
        run_partition_workflow(df_global_train, 'NonIID_ModalitySkew', make_modality_skew, args)
        
    if args.scenario in ['combined', 'all']:
        run_partition_workflow(df_global_train, 'NonIID_Combined', make_combined_skew, args)
        
    print("\n=== Proses Selesai Secara Menyeluruh ===")

if __name__ == '__main__':
    main()
