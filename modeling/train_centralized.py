import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure root workspace directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.dataset_loader import load_config, get_multimodal_dataloaders, get_pos_weight
from modeling.models.multimodal_fusion import MultimodalDepressionClassifier


def set_seed(seed=42):
    """Set random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, y_probs):
    """Finds optimal decision threshold on validation predictions that maximizes F1-Score."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        y_pred = (np.array(y_probs) >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh)


def calculate_metrics(y_true, y_probs, threshold=0.5):
    """Calculate classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC."""
    y_true = np.array(y_true).astype(int)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        # AUC-ROC requires both classes present in y_true
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_probs)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': round(float(auc), 4),
        'threshold': round(float(threshold), 4)
    }


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model on a dataloader and return loss + metrics dict."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            # Transfer tensors to device
            for k in ['text', 'audio', 'visual', 'label']:
                batch[k] = batch[k].to(device)

            targets = batch['label'].unsqueeze(1)
            logits = model(batch)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_targets.extend(targets.squeeze(1).cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_probs, threshold=threshold)
    metrics['loss'] = round(float(avg_loss), 4)
    
    return metrics, all_targets, all_probs


def train_single_model(config_path_or_dict, strategy='late_fusion', epochs=None, lr=None, 
                       batch_size=None, weight_decay=None, seed=None, verbose=True):
    """
    Train a single baseline model strategy (unimodal, bimodal, or late fusion).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(base_dir, config_path_or_dict) if isinstance(config_path_or_dict, str) and not os.path.isabs(config_path_or_dict) else config_path_or_dict)

    # Override config params if explicitly passed
    if epochs is not None:
        config['training']['epochs'] = epochs
    if lr is not None:
        config['training']['learning_rate'] = lr
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if weight_decay is not None:
        config['training']['weight_decay'] = weight_decay
    if seed is not None:
        config['training']['random_seed'] = seed

    config['model']['fusion_strategy'] = strategy

    set_seed(config['training']['random_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("=" * 70)
        print(f" TRAINING BASELINE STRATEGY: {strategy.upper()} (Device: {device})")
        print("=" * 70)

    train_loader, val_loader, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)

    # Calculate pos_weight for class imbalance handling
    pos_w_val = get_pos_weight(train_loader.dataset)
    pos_weight = torch.tensor([pos_w_val], dtype=torch.float32).to(device)
    if verbose:
        print(f"Computed Class Balance pos_weight: {pos_w_val:.4f}")

    model = MultimodalDepressionClassifier(config=config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    max_epochs = config['training']['epochs']
    patience = config['training'].get('patience', 15)
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_thresh = 0.5
    patience_counter = 0

    best_model_weights = copy.deepcopy(model.state_dict())

    # Directories for logs and checkpoints
    out_dir = os.path.join(base_dir, config['data']['output_dir'])
    log_dir = os.path.join(out_dir, 'training_logs')
    ckpt_dir = os.path.join(out_dir, 'model_checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_logs = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            for k in ['text', 'audio', 'visual', 'label']:
                batch[k] = batch[k].to(device)

            targets = batch['label'].unsqueeze(1)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Evaluate on validation set using dynamic best threshold search
        raw_val_metrics, val_targets, val_probs = evaluate(model, val_loader, criterion, device, threshold=0.5)
        opt_thresh = find_best_threshold(val_targets, val_probs)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, threshold=opt_thresh)

        log_row = {
            'epoch': epoch,
            'train_loss': round(float(avg_train_loss), 4),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1_score'],
            'val_auc': val_metrics['auc_roc'],
            'opt_threshold': opt_thresh
        }
        epoch_logs.append(log_row)

        if verbose:
            print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f} (Thresh: {opt_thresh:.2f})")

        # Checkpoint criterion: Val F1 primary, Val AUC secondary
        if val_metrics['f1_score'] > best_val_f1 or (val_metrics['f1_score'] == best_val_f1 and val_metrics['auc_roc'] > best_val_auc):
            best_val_f1 = val_metrics['f1_score']
            best_val_auc = val_metrics['auc_roc']
            best_thresh = opt_thresh
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint .pt
            ckpt_path = os.path.join(ckpt_dir, f"{strategy}_baseline.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_thresh,
                'val_f1': best_val_f1,
                'val_auc': best_val_auc
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    # Export training log CSV
    log_df = pd.DataFrame(epoch_logs)
    log_csv_path = os.path.join(log_dir, f"{strategy}_training_log.csv")
    log_df.to_csv(log_csv_path, index=False)

    # Load best model weights for final evaluation
    model.load_state_dict(best_model_weights)
    train_best_metrics, _, _ = evaluate(model, train_loader, criterion, device, threshold=best_thresh)
    val_best_metrics, _, _ = evaluate(model, val_loader, criterion, device, threshold=best_thresh)
    test_metrics, _, _ = evaluate(model, test_loader, criterion, device, threshold=best_thresh)

    f1_gap = round(train_best_metrics['f1_score'] - test_metrics['f1_score'], 4)

    if verbose:
        print("\n" + "-" * 60)
        print(f" BEST CHECKPOINT RESULTS FOR {strategy.upper()} (Optimal Threshold: {best_thresh:.2f}):")
        print(f" Train      -> F1: {train_best_metrics['f1_score']:.4f} | AUC: {train_best_metrics['auc_roc']:.4f} | Acc: {train_best_metrics['accuracy']:.4f} | Rec: {train_best_metrics['recall']:.4f}")
        print(f" Validation -> F1: {val_best_metrics['f1_score']:.4f} | AUC: {val_best_metrics['auc_roc']:.4f} | Acc: {val_best_metrics['accuracy']:.4f} | Rec: {val_best_metrics['recall']:.4f}")
        print(f" Global Test -> F1: {test_metrics['f1_score']:.4f} | AUC: {test_metrics['auc_roc']:.4f} | Acc: {test_metrics['accuracy']:.4f} | Rec: {test_metrics['recall']:.4f}")
        print(f" F1-Score Gap (Train - Test): {f1_gap:.4f}")
        print("-" * 60 + "\n")

    return model, val_best_metrics, test_metrics, log_df


def parse_args():
    parser = argparse.ArgumentParser(description="Langkah 4.15 - Training Centralized Baseline Models")
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
        choices=MultimodalDepressionClassifier.VALID_STRATEGIES + ['all'],
        help='Baseline model strategy to train (or "all" for all 7 strategies)'
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override max training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.strategy == 'all':
        for strat in MultimodalDepressionClassifier.VALID_STRATEGIES:
            train_single_model(args.config, strategy=strat, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    else:
        train_single_model(args.config, strategy=args.strategy, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
