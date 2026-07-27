# script_federated_partition.py
# Skrip orkestrator utama untuk menjalankan seluruh alur partisi data
# dan menghasilkan laporan konsolidasi terpadu.

import os
import sys
import subprocess
import pandas as pd
import numpy as np

def run_script(script_path, args=[]):
    """
    Menjalankan script Python sebagai subprocess.
    """
    print(f"\n>>> Menjalankan: {script_path} {' '.join(args)}")
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {script_path} gagal dijalankan dengan kode return {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"Sukses: {script_path} selesai dijalankan.")

def load_metadata_and_metrics():
    """
    Memuat data ground truth label dan indikator kualitas modalitas.
    """
    labels_path = 'data/detailed_lables.csv'
    token_path = 'preprocessing_output/text/tokenized_text.csv'
    vis_path = 'preprocessing_output/visual/video_preprocessing_log.xlsx'
    audio_path = 'preprocessing_output/audio/pitch_energy_features.csv'
    
    # 1. Peta label biner
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        df_labels['Participant'] = df_labels['Participant'].astype(int)
        label_dict = dict(zip(df_labels['Participant'], df_labels['Depression_label']))
    else:
        print(f"Peringatan: {labels_path} tidak ditemukan. Menggunakan fallback >= 10.")
        label_dict = {}
        
    # 2. Peta token teks
    if os.path.exists(token_path):
        df_text = pd.read_csv(token_path)
        df_text['pid_clean'] = df_text['participant_id'].astype(str).str.replace('P', '').astype(int)
        text_dict = dict(zip(df_text['pid_clean'], df_text['token_count']))
    else:
        text_dict = {}
        
    # 3. Peta visual
    if os.path.exists(vis_path):
        xls = pd.ExcelFile(vis_path)
        df_vis = xls.parse(xls.sheet_names[0])
        df_vis['pid_clean'] = df_vis['participant_id'].astype(int)
        vis_ratio_dict = dict(zip(df_vis['pid_clean'], df_vis['valid_frame_ratio']))
        duration_dict = dict(zip(df_vis['pid_clean'], df_vis['session_duration_sec']))
    else:
        vis_ratio_dict = {}
        duration_dict = {}
        
    # 4. Peta audio
    if os.path.exists(audio_path):
        df_audio = pd.read_csv(audio_path)
        df_audio['pid_clean'] = df_audio['participant_id'].astype(str).str.replace('P', '').astype(int)
        shimmer_dict = dict(zip(df_audio['pid_clean'], df_audio['shimmer']))
    else:
        shimmer_dict = {}
        
    return label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict

def get_binary_label(pid, score, label_dict):
    """
    Mengembalikan label biner (1 untuk Depresi, 0 untuk Non-depresi).
    """
    if pid in label_dict:
        return int(label_dict[pid])
    return 1 if score >= 10 else 0

def process_client_file(file_path, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict):
    """
    Membaca berkas CSV klien dan menghitung statistik deskriptif dan metrik kualitas.
    """
    df = pd.read_csv(file_path)
    df['participant_id'] = df['participant_id'].astype(int)
    
    # Tambahkan kolom metrik
    df['binary_label'] = df.apply(lambda r: get_binary_label(r['participant_id'], r['depression_label'], label_dict), axis=1)
    df['token_count'] = df['participant_id'].map(text_dict).fillna(1000).astype(int)
    df['valid_frame_ratio'] = df['participant_id'].map(vis_ratio_dict).fillna(1.0)
    df['session_duration_sec'] = df['participant_id'].map(duration_dict).fillna(300.0)
    df['shimmer'] = df['participant_id'].map(shimmer_dict).fillna(0.0)
    
    return df

def generate_consolidated_reports():
    print("\n==========================================")
    print("MENGONSOLIDASIKAN LAPORAN AKHIR (EXCEL)")
    print("==========================================")
    
    output_dir = 'federated_learning_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata dan kamus metrik
    label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict = load_metadata_and_metrics()
    
    # Definisikan skenario dan foldernya
    scenarios = {
        'IID_3Clients': 'IID_3Clients',
        'NonIID_LabelSkew': 'NonIID_LabelSkew',
        'NonIID_QuantitySkew': 'NonIID_QuantitySkew',
        'NonIID_ModalitySkew': 'NonIID_ModalitySkew',
        'NonIID_Combined': 'NonIID_Combined'
    }
    
    # ----------------------------------------------------
    # 1. KONSOLIDASI: client_data_summary.xlsx
    # ----------------------------------------------------
    summary_rows = []
    scenario_details = {sc: [] for sc in scenarios.keys()}
    
    # Muat global test & global validation pool untuk data pembanding
    global_test_path = os.path.join(output_dir, 'global_test_index.csv')
    global_train_path = os.path.join(output_dir, 'global_train_index.csv')
    global_val_path = os.path.join(output_dir, 'global_validation_index.csv')
    
    test_pids = set()
    train_pool_pids = set()
    val_pool_pids = set()
    
    if os.path.exists(global_test_path):
        df_test = process_client_file(global_test_path, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
        test_pids = set(df_test['participant_id'])
        t_total = len(df_test)
        t_dep = int(df_test['binary_label'].sum())
        t_non = t_total - t_dep
        summary_rows.append({
            'Skenario': 'Global Split',
            'Klien/Set': 'Global Test Set',
            'Latih': 0,
            'Validasi': t_total,
            'Total': t_total,
            'Depresi': t_dep,
            'Non-depresi': t_non,
            'Rasio Depresi': f"{(t_dep/t_total)*100:.1f}%" if t_total > 0 else "0.0%"
        })
        
    if os.path.exists(global_train_path):
        df_train_pool = process_client_file(global_train_path, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
        train_pool_pids = set(df_train_pool['participant_id'])
        
    if os.path.exists(global_val_path):
        df_val_pool = process_client_file(global_val_path, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
        val_pool_pids = set(df_val_pool['participant_id'])
        
    # Proses klien untuk masing-masing skenario
    for sc_name, sc_folder in scenarios.items():
        sc_path = os.path.join(output_dir, sc_folder)
        if not os.path.exists(sc_path):
            print(f"Peringatan: Folder skenario '{sc_path}' tidak ditemukan. Dilewati.")
            continue
            
        for i in range(3):
            c_idx = i + 1
            train_file = os.path.join(sc_path, f'client_{c_idx}_train.csv')
            val_file = os.path.join(sc_path, f'client_{c_idx}_val.csv')
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                df_c_tr = process_client_file(train_file, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
                df_c_vl = process_client_file(val_file, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
                
                tr_len = len(df_c_tr)
                vl_len = len(df_c_vl)
                tot_len = tr_len + vl_len
                
                dep_c = int(df_c_tr['binary_label'].sum() + df_c_vl['binary_label'].sum())
                non_c = tot_len - dep_c
                dep_pct = (dep_c / tot_len * 100) if tot_len > 0 else 0.0
                
                row_data = {
                    'Skenario': sc_name,
                    'Klien/Set': f'Client {c_idx}',
                    'Latih': tr_len,
                    'Validasi': vl_len,
                    'Total': tot_len,
                    'Depresi': dep_c,
                    'Non-depresi': non_c,
                    'Rasio Depresi': f"{dep_pct:.1f}%"
                }
                
                summary_rows.append(row_data)
                
                # Simpan untuk tab skenario detail
                scenario_details[sc_name].append({
                    'Client': f'Client {c_idx}',
                    'Train Size': tr_len,
                    'Train Depressed': int(df_c_tr['binary_label'].sum()),
                    'Train Non-Depressed': tr_len - int(df_c_tr['binary_label'].sum()),
                    'Val Size': vl_len,
                    'Val Depressed': int(df_c_vl['binary_label'].sum()),
                    'Val Non-Depressed': vl_len - int(df_c_vl['binary_label'].sum()),
                    'Total Size': tot_len,
                    'Total Depressed': dep_c,
                    'Total Non-Depressed': non_c,
                    'Depression Ratio': f"{dep_pct:.1f}%"
                })
                
    df_summary_all = pd.DataFrame(summary_rows)
    summary_xlsx_path = os.path.join(output_dir, 'client_data_summary.xlsx')
    
    with pd.ExcelWriter(summary_xlsx_path, engine='openpyxl') as writer:
        df_summary_all.to_excel(writer, index=False, sheet_name='Consolidated Overview')
        for sc_name, details in scenario_details.items():
            if len(details) > 0:
                pd.DataFrame(details).to_excel(writer, index=False, sheet_name=sc_name)
    print(f"- Laporan konsolidasi disimpan ke: {summary_xlsx_path}")
    
    # ----------------------------------------------------
    # 2. KONSOLIDASI: partition_leakage_check.xlsx
    # ----------------------------------------------------
    audit_rows = []
    
    # Audit 1: Global Split
    if len(train_pool_pids) > 0 and len(val_pool_pids) > 0 and len(test_pids) > 0:
        intersect_tr_vl = train_pool_pids.intersection(val_pool_pids)
        intersect_tr_ts = train_pool_pids.intersection(test_pids)
        intersect_vl_ts = val_pool_pids.intersection(test_pids)
        
        audit_rows.append({
            'Skenario': 'Global Split',
            'Pemeriksaan': 'Overlap Train-Val Pool',
            'Status': 'Clear' if len(intersect_tr_vl) == 0 else 'Overlap',
            'Catatan': f'Jumlah overlap: {len(intersect_tr_vl)}' if len(intersect_tr_vl) > 0 else 'Tidak ada tumpang tindih'
        })
        audit_rows.append({
            'Skenario': 'Global Split',
            'Pemeriksaan': 'Overlap Train Pool-Global Test',
            'Status': 'Clear' if len(intersect_tr_ts) == 0 else 'Overlap',
            'Catatan': f'Jumlah overlap: {len(intersect_tr_ts)}' if len(intersect_tr_ts) > 0 else 'Tidak ada tumpang tindih'
        })
        audit_rows.append({
            'Skenario': 'Global Split',
            'Pemeriksaan': 'Overlap Val Pool-Global Test',
            'Status': 'Clear' if len(intersect_vl_ts) == 0 else 'Overlap',
            'Catatan': f'Jumlah overlap: {len(intersect_vl_ts)}' if len(intersect_vl_ts) > 0 else 'Tidak ada tumpang tindih'
        })
        
    # Audit untuk setiap skenario federated
    for sc_name, sc_folder in scenarios.items():
        sc_path = os.path.join(output_dir, sc_folder)
        if not os.path.exists(sc_path):
            continue
            
        client_train_pids = []
        client_val_pids = []
        client_all_pids = []
        
        for i in range(3):
            c_idx = i + 1
            train_file = os.path.join(sc_path, f'client_{c_idx}_train.csv')
            val_file = os.path.join(sc_path, f'client_{c_idx}_val.csv')
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                df_tr = pd.read_csv(train_file)
                df_vl = pd.read_csv(val_file)
                
                tr_pids = set(df_tr['participant_id'].astype(int))
                vl_pids = set(df_vl['participant_id'].astype(int))
                all_pids = tr_pids.union(vl_pids)
                
                client_train_pids.append(tr_pids)
                client_val_pids.append(vl_pids)
                client_all_pids.append(all_pids)
                
                # Check 1: Train-Val overlap pada klien yang sama
                intersect = tr_pids.intersection(vl_pids)
                audit_rows.append({
                    'Skenario': sc_name,
                    'Pemeriksaan': f'Overlap Train-Val Client {c_idx}',
                    'Status': 'Clear' if len(intersect) == 0 else 'Overlap',
                    'Catatan': f'Jumlah overlap: {len(intersect)}' if len(intersect) > 0 else 'Tidak ada tumpang tindih lokal'
                })
                
                # Check 2: Kebocoran Global Test ke klien
                intersect_test = all_pids.intersection(test_pids)
                audit_rows.append({
                    'Skenario': sc_name,
                    'Pemeriksaan': f'Overlap Client {c_idx}-Global Test',
                    'Status': 'Clear' if len(intersect_test) == 0 else 'Overlap',
                    'Catatan': f'Jumlah overlap: {len(intersect_test)}' if len(intersect_test) > 0 else 'Tidak ada kebocoran global test'
                })
                
        # Check 3: Overlap antar klien (antarklien tidak boleh saling berbagi participant)
        if len(client_all_pids) == 3:
            intersect_c1_c2 = client_all_pids[0].intersection(client_all_pids[1])
            intersect_c1_c3 = client_all_pids[0].intersection(client_all_pids[2])
            intersect_c2_c3 = client_all_pids[1].intersection(client_all_pids[2])
            
            audit_rows.append({
                'Skenario': sc_name,
                'Pemeriksaan': 'Overlap Antar-Klien (C1-C2)',
                'Status': 'Clear' if len(intersect_c1_c2) == 0 else 'Overlap',
                'Catatan': f'Jumlah overlap: {len(intersect_c1_c2)}' if len(intersect_c1_c2) > 0 else 'Tidak ada tumpang tindih antar klien'
            })
            audit_rows.append({
                'Skenario': sc_name,
                'Pemeriksaan': 'Overlap Antar-Klien (C1-C3)',
                'Status': 'Clear' if len(intersect_c1_c3) == 0 else 'Overlap',
                'Catatan': f'Jumlah overlap: {len(intersect_c1_c3)}' if len(intersect_c1_c3) > 0 else 'Tidak ada tumpang tindih antar klien'
            })
            audit_rows.append({
                'Skenario': sc_name,
                'Pemeriksaan': 'Overlap Antar-Klien (C2-C3)',
                'Status': 'Clear' if len(intersect_c2_c3) == 0 else 'Overlap',
                'Catatan': f'Jumlah overlap: {len(intersect_c2_c3)}' if len(intersect_c2_c3) > 0 else 'Tidak ada tumpang tindih antar klien'
            })
            
    df_audit = pd.DataFrame(audit_rows)
    leakage_xlsx_path = os.path.join(output_dir, 'partition_leakage_check.xlsx')
    
    with pd.ExcelWriter(leakage_xlsx_path, engine='openpyxl') as writer:
        df_audit.to_excel(writer, index=False, sheet_name='Leakage Audit Log')
    print(f"- Laporan leakage check disimpan ke: {leakage_xlsx_path}")
    
    # ----------------------------------------------------
    # 3. KONSOLIDASI: noniid_heterogeneity_metrics.xlsx
    # ----------------------------------------------------
    metrics_rows = []
    
    for sc_name, sc_folder in scenarios.items():
        sc_path = os.path.join(output_dir, sc_folder)
        if not os.path.exists(sc_path):
            continue
            
        for i in range(3):
            c_idx = i + 1
            train_file = os.path.join(sc_path, f'client_{c_idx}_train.csv')
            val_file = os.path.join(sc_path, f'client_{c_idx}_val.csv')
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                df_tr = process_client_file(train_file, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
                df_vl = process_client_file(val_file, label_dict, text_dict, vis_ratio_dict, duration_dict, shimmer_dict)
                df_all = pd.concat([df_tr, df_vl])
                
                tot = len(df_all)
                dep = int(df_all['binary_label'].sum())
                dep_ratio = (dep / tot) * 100 if tot > 0 else 0.0
                
                mean_tokens = df_all['token_count'].mean()
                mean_vis_ratio = df_all['valid_frame_ratio'].mean()
                mean_duration = df_all['session_duration_sec'].mean()
                mean_shimmer = df_all['shimmer'].mean()
                
                metrics_rows.append({
                    'Skenario': sc_name,
                    'Klien': f'Client {c_idx}',
                    'Total Sampel': tot,
                    'Depresi (%)': round(dep_ratio, 1),
                    'Rerata Token Teks': round(mean_tokens, 1),
                    'Rerata Face Detection Rate': f"{mean_vis_ratio * 100:.2f}%",
                    'Rerata Durasi Sesi (detik)': round(mean_duration, 1),
                    'Rerata Shimmer Audio': round(mean_shimmer, 4)
                })
                
    df_metrics = pd.DataFrame(metrics_rows)
    
    # Buat tabel ringkasan tingkat heterogenitas skenario (Tabel 7)
    hetero_summary = [
        {'Skenario': 'IID_3Clients', 'Label Skew': 'rendah', 'Quantity Skew': 'rendah', 'Modality Skew': 'rendah', 'Tingkat Heterogenitas': 'rendah'},
        {'Skenario': 'NonIID_LabelSkew', 'Label Skew': 'tinggi', 'Quantity Skew': 'rendah', 'Modality Skew': 'rendah', 'Tingkat Heterogenitas': 'sedang'},
        {'Skenario': 'NonIID_QuantitySkew', 'Label Skew': 'rendah', 'Quantity Skew': 'tinggi', 'Modality Skew': 'rendah', 'Tingkat Heterogenitas': 'sedang'},
        {'Skenario': 'NonIID_ModalitySkew', 'Label Skew': 'rendah', 'Quantity Skew': 'rendah', 'Modality Skew': 'tinggi', 'Tingkat Heterogenitas': 'sedang'},
        {'Skenario': 'NonIID_Combined', 'Label Skew': 'tinggi', 'Quantity Skew': 'tinggi', 'Modality Skew': 'sedang', 'Tingkat Heterogenitas': 'tinggi'},
    ]
    df_hetero_summary = pd.DataFrame(hetero_summary)
    
    hetero_xlsx_path = os.path.join(output_dir, 'noniid_heterogeneity_metrics.xlsx')
    
    with pd.ExcelWriter(hetero_xlsx_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, index=False, sheet_name='Detailed Client Metrics')
        df_hetero_summary.to_excel(writer, index=False, sheet_name='Scenario Level Summary')
    print(f"- Laporan heterogenitas disimpan ke: {hetero_xlsx_path}")
    
    print("\nKonsolidasi Laporan Excel selesai sepenuhnya.")

def main():
    print("==========================================")
    print("MENJALANKAN PIPELINE PARTISI FL UTAMA")
    print("==========================================")
    
    # Jalankan modul-modul partisi secara berurutan
    run_script('federated_learning/script_global_split.py')
    run_script('federated_learning/script_iid_partition.py')
    run_script('federated_learning/script_non_iid_partition.py', ['--scenario', 'all'])
    
    # Setelah pembagian data selesai, lakukan konsolidasi laporan
    generate_consolidated_reports()
    
    print("\nPipeline partisi data Federated Learning telah selesai dijalankan secara penuh.")

if __name__ == '__main__':
    main()
