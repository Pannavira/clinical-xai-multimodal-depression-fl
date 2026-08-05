import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification on imbalanced medical datasets.

    Focal Loss adaptively down-weights easy (well-classified) negatives and
    focuses training on hard (misclassified) examples, particularly the
    minority positive class (depressed patients).

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha (float): Weight for the positive class (>0.5 = emphasize positives = boost recall).
        gamma (float): Focusing exponent. gamma=0 recovers standard BCE. gamma=2 is standard Focal.
        pos_weight (torch.Tensor, optional): Additional positive class weight (like BCE pos_weight).
                                             Applied on top of alpha for extreme imbalance.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Raw model output logits of shape (B, 1).
            targets (Tensor): Binary labels of shape (B, 1), values in {0.0, 1.0}.
        Returns:
            Tensor: Scalar focal loss value.
        """
        # Binary cross-entropy per sample (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute probabilities from logits
        probs = torch.sigmoid(logits)

        # p_t = prob of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting per sample
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Combine: alpha * focal_weight * bce
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Threshold Strategies
# ---------------------------------------------------------------------------

def find_best_threshold(y_true, y_probs):
    """Finds optimal decision threshold that maximizes F1-Score on validation predictions."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        y_pred = (np.array(y_probs) >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh)


def find_best_threshold_recall_first(y_true, y_probs, min_precision: float = 0.30):
    """
    Finds threshold that maximizes Recall (Sensitivity) subject to Precision >= min_precision.

    In clinical depression screening, False Negatives (missed depressed patients) are
    far more costly than False Positives. This strategy prioritizes recall while
    maintaining a minimum acceptable precision floor to avoid trivial predictions.

    Args:
        y_true: Ground-truth binary labels.
        y_probs: Model predicted probabilities.
        min_precision (float): Minimum precision constraint. Default = 0.30.

    Returns:
        float: Optimal threshold value.
    """
    best_thresh = 0.5
    best_recall = -1.0

    for thresh in np.arange(0.10, 0.90, 0.02):
        y_pred = (np.array(y_probs) >= thresh).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)

        # Accept threshold only if precision constraint is satisfied
        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_thresh = thresh

    # Fallback: if no threshold satisfies precision constraint, use F1-based threshold
    if best_recall == -1.0:
        best_thresh = find_best_threshold(y_true, y_probs)

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_model(config_path_or_dict, strategy='late_fusion', epochs=None, lr=None,
                       batch_size=None, weight_decay=None, seed=None, verbose=True,
                       use_focal_loss=None, augment_noise_std=None,
                       use_lr_scheduler=None, use_gated_fusion=None,
                       threshold_strategy=None, min_precision_constraint=None):
    """
    Train a single baseline model strategy (unimodal, bimodal, or late fusion).

    Args:
        config_path_or_dict: Path to YAML config or config dict.
        strategy (str): Fusion strategy name.
        epochs (int, optional): Override max training epochs.
        lr (float, optional): Override learning rate.
        batch_size (int, optional): Override batch size.
        weight_decay (float, optional): Override L2 weight decay.
        seed (int, optional): Override random seed.
        verbose (bool): Print training progress.
        use_focal_loss (bool, optional): Override advanced_training.use_focal_loss from config.
        augment_noise_std (float, optional): Override advanced_training.augment_noise_std.
        use_lr_scheduler (bool, optional): Override advanced_training.use_lr_scheduler.
        use_gated_fusion (bool, optional): Override advanced_training.use_gated_fusion.
        threshold_strategy (str, optional): Override advanced_training.threshold_strategy.
        min_precision_constraint (float, optional): Override advanced_training.min_precision_constraint.
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

    # Load advanced training config with argument overrides
    adv_cfg = config.get('advanced_training', {})

    _use_focal_loss = use_focal_loss if use_focal_loss is not None else adv_cfg.get('use_focal_loss', False)
    _focal_alpha = adv_cfg.get('focal_alpha', 0.75)
    _focal_gamma = adv_cfg.get('focal_gamma', 2.0)
    _augment_noise_std = augment_noise_std if augment_noise_std is not None else adv_cfg.get('augment_noise_std', 0.0)
    _use_lr_scheduler = use_lr_scheduler if use_lr_scheduler is not None else adv_cfg.get('use_lr_scheduler', False)
    _lr_scheduler_patience = adv_cfg.get('lr_scheduler_patience', 3)
    _lr_scheduler_factor = adv_cfg.get('lr_scheduler_factor', 0.5)
    _modality_dropout_prob = adv_cfg.get('modality_dropout_prob', 0.0)
    _use_gated_fusion = use_gated_fusion if use_gated_fusion is not None else adv_cfg.get('use_gated_fusion', False)
    _threshold_strategy = threshold_strategy if threshold_strategy is not None else adv_cfg.get('threshold_strategy', 'f1')
    _min_precision = min_precision_constraint if min_precision_constraint is not None else adv_cfg.get('min_precision_constraint', 0.30)

    set_seed(config['training']['random_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("=" * 70)
        print(f" TRAINING STRATEGY: {strategy.upper()} (Device: {device})")
        print(f" Focal Loss: {_use_focal_loss} | Noise Aug: {_augment_noise_std} | Modality Dropout: {_modality_dropout_prob}")
        print(f" Gated Fusion: {_use_gated_fusion} | Threshold Strategy: {_threshold_strategy}")
        print("=" * 70)

    train_loader, val_loader, test_loader = get_multimodal_dataloaders(config, base_dir=base_dir)

    # Calculate pos_weight for class imbalance handling
    pos_w_val = get_pos_weight(train_loader.dataset)
    pos_weight = torch.tensor([pos_w_val], dtype=torch.float32).to(device)
    if verbose:
        print(f"Computed Class Balance pos_weight: {pos_w_val:.4f}")

    model = MultimodalDepressionClassifier(
        config=config,
        use_gated_fusion=_use_gated_fusion
    ).to(device)

    # Loss function selection
    if _use_focal_loss:
        criterion = FocalLoss(alpha=_focal_alpha, gamma=_focal_gamma)
        if verbose:
            print(f"Using FocalLoss (alpha={_focal_alpha}, gamma={_focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if verbose:
            print(f"Using BCEWithLogitsLoss (pos_weight={pos_w_val:.4f})")

    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    # LR Scheduler
    scheduler = None
    if _use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=_lr_scheduler_patience,
            factor=_lr_scheduler_factor,
            min_lr=1e-6
        )
        if verbose:
            print(f"LR Scheduler: ReduceLROnPlateau (patience={_lr_scheduler_patience}, factor={_lr_scheduler_factor})")

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

            # Gaussian Noise Augmentation on all modality features
            if _augment_noise_std > 0.0:
                batch['text'] = batch['text'] + torch.randn_like(batch['text']) * _augment_noise_std
                batch['audio'] = batch['audio'] + torch.randn_like(batch['audio']) * _augment_noise_std
                batch['visual'] = batch['visual'] + torch.randn_like(batch['visual']) * _augment_noise_std

            # Modality Dropout (randomly zero-out individual modalities during training)
            if _modality_dropout_prob > 0.0:
                active_keys = [k for k in ['text', 'audio', 'visual'] if k in strategy or strategy == 'late_fusion']
                if len(active_keys) > 1:
                    for k in active_keys:
                        if torch.rand(1).item() < _modality_dropout_prob:
                            # Ensure not all modalities are dropped simultaneously
                            if sum(1 for key in active_keys if torch.all(batch[key] == 0)) < len(active_keys) - 1:
                                batch[k] = torch.zeros_like(batch[k])

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Evaluate on validation set using selected threshold strategy
        raw_val_metrics, val_targets, val_probs = evaluate(model, val_loader, criterion, device, threshold=0.5)

        if _threshold_strategy == 'recall_first':
            opt_thresh = find_best_threshold_recall_first(val_targets, val_probs, min_precision=_min_precision)
        else:
            opt_thresh = find_best_threshold(val_targets, val_probs)

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, threshold=opt_thresh)

        # Step the LR scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        current_lr = optimizer.param_groups[0]['lr']

        log_row = {
            'epoch': epoch,
            'train_loss': round(float(avg_train_loss), 4),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1_score'],
            'val_auc': val_metrics['auc_roc'],
            'opt_threshold': opt_thresh,
            'learning_rate': current_lr
        }
        epoch_logs.append(log_row)

        if verbose:
            print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val Rec: {val_metrics['recall']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f} (Thresh: {opt_thresh:.2f} | LR: {current_lr:.2e})")

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
