import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config, MultimodalDepressionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.3 - Verifikasi Input Baseline Terpusat")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML configuration file'
    )
    return parser.parse_args()


def verify_inputs(config_path):
    # Find base workspace directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
    
    output_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("      VERIFIKASI DATA INPUT BASELINE MULTIMODAL TERPUSAT")
    print("=" * 70)
    
    # Load splits
    splits = {
        'train': os.path.join(base_dir, config['data']['train_index']),
        'validation': os.path.join(base_dir, config['data']['validation_index']),
        'test': os.path.join(base_dir, config['data']['test_index'])
    }
    
    split_dfs = {}
    participant_sets = {}
    
    for split_name, split_file in splits.items():
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"File index {split_name} tidak ditemukan: {split_file}")
        df = pd.read_csv(split_file)
        split_dfs[split_name] = df
        participant_sets[split_name] = set(df['participant_id'].astype(int))
        print(f"Loaded {split_name:10s} split: {len(df):3d} subjek.")

    # 1. Overlap Check
    print("\n--- 1. Periksa Overlap Participant ID ---")
    train_val_overlap = participant_sets['train'].intersection(participant_sets['validation'])
    train_test_overlap = participant_sets['train'].intersection(participant_sets['test'])
    val_test_overlap = participant_sets['validation'].intersection(participant_sets['test'])
    
    print(f"Overlap Train <-> Validation: {len(train_val_overlap)} subjek")
    print(f"Overlap Train <-> Test:       {len(train_test_overlap)} subjek")
    print(f"Overlap Val   <-> Test:       {len(val_test_overlap)} subjek")
    
    has_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap) > 0
    if has_leakage:
        print("[ALERT] Terdeteksi overlap participant_id antarsplit!")
    else:
        print("[OK] Tidak ada data leakage antarsplit (0 overlap).")

    # 2. Detailed Per-Participant Verification Log
    print("\n--- 2. Memeriksa Integritas Feature & Label Per Subjek ---")
    prep_dir = os.path.join(base_dir, config['data']['preprocessing_output_dir'])
    
    log_rows = []
    
    for split_name, df in split_dfs.items():
        ds = MultimodalDepressionDataset(df, config, base_dir=base_dir)
        
        for idx in range(len(ds)):
            sample = ds[idx]
            pid = int(sample['participant_id'])
            
            text_tensor = sample['text']
            audio_tensor = sample['audio']
            visual_tensor = sample['visual']
            label_val = float(sample['label'])
            
            text_valid = int((text_tensor.abs().sum() > 0).item())
            audio_valid = int((audio_tensor.abs().sum() > 0).item())
            visual_valid = int((visual_tensor.abs().sum() > 0).item())
            label_valid = int(not np.isnan(label_val))
            
            text_shape_ok = int(text_tensor.shape[0] == config['features']['text_dim'])
            audio_shape_ok = int(audio_tensor.shape[0] == config['features']['audio_dim'])
            visual_shape_ok = int(visual_tensor.shape[0] == config['features']['visual_dim'])
            
            is_valid = text_valid and audio_valid and visual_valid and label_valid and text_shape_ok and audio_shape_ok and visual_shape_ok
            status_str = "valid" if is_valid else "warning/excluded"
            
            log_rows.append({
                'participant_id': pid,
                'split': split_name,
                'text_valid': text_valid,
                'audio_valid': audio_valid,
                'visual_valid': visual_valid,
                'label_valid': label_valid,
                'text_shape_dim': text_tensor.shape[0],
                'audio_shape_dim': audio_tensor.shape[0],
                'visual_shape_dim': visual_tensor.shape[0],
                'binary_label': int(label_val),
                'status': status_str
            })

    log_df = pd.DataFrame(log_rows)

    # 3. Label Distribution Summary
    print("\n--- 3. Distribusi Label Biner per Split ---")
    summary_rows = []
    for split_name in ['train', 'validation', 'test']:
        sub_df = log_df[log_df['split'] == split_name]
        n_total = len(sub_df)
        n_dep = (sub_df['binary_label'] == 1).sum()
        n_nondep = (sub_df['binary_label'] == 0).sum()
        pct_dep = (n_dep / n_total * 100) if n_total > 0 else 0
        
        print(f"Split {split_name:10s}: Total={n_total:3d} | Depressed(1)={n_dep:2d} ({pct_dep:.1f}%) | Non-Depressed(0)={n_nondep:3d}")
        summary_rows.append({
            'split': split_name,
            'total_subjects': n_total,
            'depressed_count': n_dep,
            'non_depressed_count': n_nondep,
            'depressed_percentage': round(pct_dep, 2)
        })
        
    summary_df = pd.DataFrame(summary_rows)

    # 4. Save Log File Excel
    out_xlsx_path = os.path.join(output_dir, 'centralized_input_verification_log.xlsx')
    with pd.ExcelWriter(out_xlsx_path, engine='openpyxl') as writer:
        log_df.to_excel(writer, sheet_name='verification_details', index=False)
        summary_df.to_excel(writer, sheet_name='label_distribution', index=False)

    print("\n" + "=" * 70)
    print(f"HASIL VERIFIKASI Selesai!")
    print(f"Total Subjek Diverifikasi: {len(log_df)} ({len(log_df[log_df['status']=='valid'])} valid)")
    print(f"Log Verifikasi disimpan ke: {out_xlsx_path}")
    print("=" * 70)

    return log_df, summary_df


if __name__ == '__main__':
    args = parse_args()
    verify_inputs(args.config)
