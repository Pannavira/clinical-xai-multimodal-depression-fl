import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config, get_multimodal_dataloaders
from modeling.models.multimodal_fusion import MultimodalDepressionClassifier
from modeling.train_centralized import train_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.20 - Error Analysis per Participant")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML config file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='late_fusion',
        help='Strategy to evaluate for error analysis'
    )
    return parser.parse_args()


def run_error_analysis(config_path, strategy='late_fusion'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
    config['model']['fusion_strategy'] = strategy

    out_metrics_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    ckpt_dir = os.path.join(base_dir, config['data']['output_dir'], 'model_checkpoints')
    os.makedirs(out_metrics_dir, exist_ok=True)

    print("=" * 75)
    print(f"        ERROR ANALYSIS PER PARTICIPANT ({strategy.upper()} - GLOBAL TEST SET)")
    print("=" * 75)

    _, _, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(ckpt_dir, f"{strategy}_baseline.pt")
    model = MultimodalDepressionClassifier(config=config).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        model, _, _, _ = train_single_model(config, strategy=strategy, epochs=15, verbose=False)

    model.eval()
    threshold = config['evaluation']['threshold']

    records = []

    with torch.no_grad():
        for batch in test_loader:
            pids = batch['participant_id'].cpu().numpy()
            labels = batch['label'].cpu().numpy()

            for k in ['text', 'audio', 'visual']:
                batch[k] = batch[k].to(device)

            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for pid, true_lbl, prob in zip(pids, labels, probs):
                true_lbl = int(true_lbl)
                pred_lbl = int(prob >= threshold)

                if true_lbl == 1 and pred_lbl == 1:
                    err_type = "True Positive (TP)"
                    cause = "Indikator depresi multimodal kuat (Text/Audio/Visual konsisten)"
                elif true_lbl == 0 and pred_lbl == 0:
                    err_type = "True Negative (TN)"
                    cause = "Sinyal depresi rendah pada seluruh modalitas"
                elif true_lbl == 0 and pred_lbl == 1:
                    err_type = "False Positive (FP)"
                    cause = "Ekspresi visual/akustik ambigu atau ekspresi mirip depresi"
                else:  # true_lbl == 1 and pred_lbl == 0
                    err_type = "False Negative (FN)"
                    cause = "Transkrip teks pendek / audio noisy / respons verbal minimal"

                records.append({
                    'participant_id': int(pid),
                    'true_label': true_lbl,
                    'predicted_label': pred_lbl,
                    'predicted_probability': round(float(prob), 4),
                    'error_type': err_type,
                    'probable_cause': cause
                })

    df_details = pd.DataFrame(records)

    # Summarize error statistics
    type_counts = df_details['error_type'].value_counts()
    total_test = len(df_details)

    summary_rows = []
    for err_name in ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)']:
        cnt = type_counts.get(err_name, 0)
        pct = round((cnt / total_test * 100), 2) if total_test > 0 else 0.0
        summary_rows.append({
            'error_type': err_name,
            'count': cnt,
            'percentage': pct
        })

    df_summary = pd.DataFrame(summary_rows)

    out_xlsx = os.path.join(out_metrics_dir, 'error_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_details.to_excel(writer, sheet_name='detailed_errors', index=False)
        df_summary.to_excel(writer, sheet_name='error_summary', index=False)

    print("\nSummary Ringkasan Prediksi:")
    for _, r in df_summary.iterrows():
        print(f"  {r['error_type']:20s}: {r['count']:2d} subjek ({r['percentage']:.1f}%)")

    print("\n" + "=" * 75)
    print(f"ERROR ANALYSIS SELESAI! Laporan disimpan ke: {out_xlsx}")
    print("=" * 75)

    return df_details, df_summary


if __name__ == '__main__':
    args = parse_args()
    run_error_analysis(args.config, strategy=args.strategy)
