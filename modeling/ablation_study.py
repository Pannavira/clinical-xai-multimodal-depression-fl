import os
import sys
import argparse
import pandas as pd
import torch

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config, get_multimodal_dataloaders
from modeling.models.multimodal_fusion import MultimodalDepressionClassifier
from modeling.train_centralized import evaluate, train_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.21 - Modality Ablation Study")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML config file'
    )
    return parser.parse_args()


def run_ablation_study(config_path):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)

    out_metrics_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    ckpt_dir = os.path.join(base_dir, config['data']['output_dir'], 'model_checkpoints')
    os.makedirs(out_metrics_dir, exist_ok=True)

    print("=" * 75)
    print("           MODALITY ABLATION STUDY (GLOBAL TEST SET)")
    print("=" * 75)

    _, _, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()

    ablation_setups = [
        {'name': 'Full Multimodal', 'strategy': 'late_fusion', 'text': True, 'audio': True, 'visual': True},
        {'name': 'Without Text', 'strategy': 'audio_visual', 'text': False, 'audio': True, 'visual': True},
        {'name': 'Without Audio', 'strategy': 'text_visual', 'text': True, 'audio': False, 'visual': True},
        {'name': 'Without Visual', 'strategy': 'text_audio', 'text': True, 'audio': True, 'visual': False},
        {'name': 'Text-Only', 'strategy': 'text_only', 'text': True, 'audio': False, 'visual': False},
        {'name': 'Audio-Only', 'strategy': 'audio_only', 'text': False, 'audio': True, 'visual': False},
        {'name': 'Visual-Only', 'strategy': 'visual_only', 'text': False, 'audio': False, 'visual': True},
    ]

    ablation_records = []
    baseline_f1 = 0.0

    for setup in ablation_setups:
        strat = setup['strategy']
        ckpt_path = os.path.join(ckpt_dir, f"{strat}_baseline.pt")
        strat_config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
        strat_config['model']['fusion_strategy'] = strat

        model = MultimodalDepressionClassifier(config=strat_config).to(device)

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            model, _, _, _ = train_single_model(strat_config, strategy=strat, epochs=15, verbose=False)

        metrics, _, _ = evaluate(model, test_loader, criterion, device, threshold=config['evaluation']['threshold'])

        if setup['name'] == 'Full Multimodal':
            baseline_f1 = metrics['f1_score']
            interpretation = "Base Multimodal Reference"
        else:
            diff = metrics['f1_score'] - baseline_f1
            if diff < 0:
                interpretation = f"F1 drops by {abs(diff):.4f} when modality removed"
            else:
                interpretation = f"F1 improves/matches by {diff:.4f}"

        record = {
            'experiment': setup['name'],
            'strategy_code': strat,
            'has_text': "Ya" if setup['text'] else "Tidak",
            'has_audio': "Ya" if setup['audio'] else "Tidak",
            'has_visual': "Ya" if setup['visual'] else "Tidak",
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'interpretation': interpretation
        }
        ablation_records.append(record)
        print(f"Setup: {setup['name']:17s} | Text:{record['has_text']:5s} | Audio:{record['has_audio']:5s} | Vis:{record['has_visual']:5s} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc_roc']:.4f}")

    ablation_df = pd.DataFrame(ablation_records)

    out_xlsx = os.path.join(out_metrics_dir, 'modality_ablation_study.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        ablation_df.to_excel(writer, sheet_name='ablation_study', index=False)

    print("\n" + "=" * 75)
    print(f"ABLATION STUDY SELESAI! Tabel disimpan ke: {out_xlsx}")
    print("=" * 75)

    return ablation_df


if __name__ == '__main__':
    args = parse_args()
    run_ablation_study(args.config)
