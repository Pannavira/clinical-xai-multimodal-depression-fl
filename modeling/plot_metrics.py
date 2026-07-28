import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config, get_multimodal_dataloaders
from modeling.models.multimodal_fusion import MultimodalDepressionClassifier
from modeling.train_centralized import evaluate, train_single_model


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.18 & 4.19 - Plotting Confusion Matrix, ROC Curve, dan PR Curve")
    parser.add_argument(
        '--config',
        type=str,
        default='modeling/configs/centralized_baseline_config.yaml',
        help='Path to centralized baseline YAML config file'
    )
    return parser.parse_args()


def plot_all_metrics(config_path):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)

    plots_dir = os.path.join(base_dir, config['data']['output_dir'], 'plots')
    cm_dir = os.path.join(plots_dir, 'confusion_matrix')
    roc_dir = os.path.join(plots_dir, 'roc_curve')
    pr_dir = os.path.join(plots_dir, 'pr_curve')

    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    print("=" * 75)
    print("      PLOTTING METRICS (CONFUSION MATRIX, ROC CURVE, PR CURVE)")
    print("=" * 75)

    _, _, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()
    ckpt_dir = os.path.join(base_dir, config['data']['output_dir'], 'model_checkpoints')

    strategies = MultimodalDepressionClassifier.VALID_STRATEGIES
    model_data = {}

    for strat in strategies:
        ckpt_path = os.path.join(ckpt_dir, f"{strat}_baseline.pt")
        strat_config = load_config(os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path)
        strat_config['model']['fusion_strategy'] = strat

        model = MultimodalDepressionClassifier(config=strat_config).to(device)

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            model, _, _, _ = train_single_model(strat_config, strategy=strat, epochs=15, verbose=False)

        _, targets, probs = evaluate(model, test_loader, criterion, device, threshold=config['evaluation']['threshold'])
        model_data[strat] = {
            'targets': np.array(targets).astype(int),
            'probs': np.array(probs)
        }

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Helper function for plotting confusion matrix without external dependencies
    def save_cm_plot(cm_matrix, title_str, save_path):
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm_matrix, cmap='Blues')
        ax.set_title(title_str, fontsize=14, pad=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Depressed', 'Depressed'], fontsize=11)
        ax.set_yticklabels(['Non-Depressed', 'Depressed'], fontsize=11)

        # Annotate counts inside cells
        for i in range(2):
            for j in range(2):
                val = cm_matrix[i, j]
                color = "white" if val > cm_matrix.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # 1. Plot Confusion Matrices
    print("\n[1/3] Generating Confusion Matrices...")
    for strat in ['text_only', 'audio_only', 'visual_only', 'late_fusion']:
        data = model_data[strat]
        preds = (data['probs'] >= config['evaluation']['threshold']).astype(int)
        cm = confusion_matrix(data['targets'], preds)

        out_cm_path = os.path.join(cm_dir, f'cm_{strat}.png')
        save_cm_plot(cm, f'Confusion Matrix - {strat.replace("_", " ").title()}', out_cm_path)
        print(f"  - Saved: {out_cm_path}")

    # Save primary confusion matrix png
    primary_cm = os.path.join(base_dir, config['data']['output_dir'], 'plots', 'confusion_matrix_baseline.png')
    data_mf = model_data['late_fusion']
    preds_mf = (data_mf['probs'] >= config['evaluation']['threshold']).astype(int)
    cm_mf = confusion_matrix(data_mf['targets'], preds_mf)
    save_cm_plot(cm_mf, 'Confusion Matrix - Full Multimodal Baseline', primary_cm)

    # 2. Plot ROC Curves
    print("\n[2/3] Generating ROC Curves...")
    plt.figure(figsize=(8, 6))
    for strat in strategies:
        data = model_data[strat]
        fpr, tpr, _ = roc_curve(data['targets'], data['probs'])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{strat.replace("_", " ").title()} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('ROC Curve Comparison - All Baseline Models', fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    out_roc_comp = os.path.join(roc_dir, 'roc_all_models_comparison.png')
    plt.savefig(out_roc_comp, dpi=300)
    plt.close()
    print(f"  - Saved: {out_roc_comp}")

    # Primary ROC curve png
    primary_roc = os.path.join(base_dir, config['data']['output_dir'], 'plots', 'roc_curve_baseline.png')
    data_mf = model_data['late_fusion']
    fpr, tpr, _ = roc_curve(data_mf['targets'], data_mf['probs'])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Late Fusion (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Full Multimodal Baseline', fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(primary_roc, dpi=300)
    plt.close()

    # 3. Plot Precision-Recall Curves
    print("\n[3/3] Generating PR Curves...")
    plt.figure(figsize=(8, 6))
    for strat in strategies:
        data = model_data[strat]
        prec, rec, _ = precision_recall_curve(data['targets'], data['probs'])
        pr_auc = auc(rec, prec)

        plt.plot(rec, prec, lw=2, label=f'{strat.replace("_", " ").title()} (PR-AUC = {pr_auc:.3f})')

    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curve Comparison - All Baseline Models', fontsize=14, pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_pr_comp = os.path.join(pr_dir, 'pr_all_models_comparison.png')
    plt.savefig(out_pr_comp, dpi=300)
    plt.close()
    print(f"  - Saved: {out_pr_comp}")

    # Primary PR curve png
    primary_pr = os.path.join(base_dir, config['data']['output_dir'], 'plots', 'pr_curve_baseline.png')
    data_mf = model_data['late_fusion']
    prec, rec, _ = precision_recall_curve(data_mf['targets'], data_mf['probs'])
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(7, 5.5))
    plt.plot(rec, prec, color='blue', lw=2.5, label=f'Late Fusion (PR-AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Full Multimodal Baseline', fontsize=14, pad=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(primary_pr, dpi=300)
    plt.close()

    print("\n" + "=" * 75)
    print("PLOTTING SELESAI! Seluruh grafik visualisasi berhasil disimpan di modeling_output/plots/")
    print("=" * 75)


if __name__ == '__main__':
    args = parse_args()
    plot_all_metrics(args.config)
