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
    parser = argparse.ArgumentParser(description="Langkah 4.16 & 4.17 - Evaluasi Model Baseline Terpusat")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML config file'
    )
    return parser.parse_args()


def evaluate_all_baselines(config_path):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)

    out_metrics_dir = os.path.join(base_dir, config['data']['output_dir'], 'metrics')
    ckpt_dir = os.path.join(base_dir, config['data']['output_dir'], 'model_checkpoints')
    os.makedirs(out_metrics_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 75)
    print("       EVALUASI MODEL BASELINE TERPUSAT (GLOBAL TEST SET)")
    print("=" * 75)

    _, _, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()

    strategies = MultimodalDepressionClassifier.VALID_STRATEGIES
    results = []

    for strat in strategies:
        ckpt_path = os.path.join(ckpt_dir, f"{strat}_baseline.pt")
        
        strat_config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
        strat_config['model']['fusion_strategy'] = strat

        model = MultimodalDepressionClassifier(config=strat_config).to(device)

        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint for {strat:15s} from: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"Checkpoint for {strat:15s} not found. Training model for 15 epochs...")
            model, _, _, _ = train_single_model(strat_config, strategy=strat, epochs=15, verbose=False)

        metrics, _, _ = evaluate(model, test_loader, criterion, device, threshold=config['evaluation']['threshold'])
        
        row = {
            'model_name': strat,
            'modality': strat.replace('_', ' ').title(),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'test_loss': metrics['loss']
        }
        results.append(row)
        print(f"Strategy: {strat:15s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc_roc']:.4f}")

    results_df = pd.DataFrame(results)

    out_xlsx = os.path.join(out_metrics_dir, 'centralized_baseline_metrics.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='baseline_metrics', index=False)

    print("\n" + "=" * 75)
    print(f"EVALUASI SELESAI! Tabel metrik utama disimpan ke: {out_xlsx}")
    print("=" * 75)

    return results_df


if __name__ == '__main__':
    args = parse_args()
    evaluate_all_baselines(args.config)
