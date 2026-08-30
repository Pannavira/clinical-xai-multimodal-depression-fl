"""
train.py
========
Pipeline Training dan Evaluasi Calibrated Centralized MFDCL Baseline untuk Deteksi Depresi (E-DAIC).

Fitur:
- Fitur Ternormalisasi Z-score (StandardScaler) dari Training Split (Zero Leakage).
- Class-Weighted MSE Loss dengan pos_weight_multiplier dan penalti False Negative.
- Dropout terkalibrasi (0.4 conv, 0.3 regressor) untuk mencegah underfitting pada dataset kecil.
- Optimizer AdamW + CosineAnnealingLR.
- Validation Checkpointing berdasarkan F1 biner tertinggi.
- Threshold Calibration pada Validation Set untuk memaksimalkan Recall & F1 serta menekan False Negatives.
- Evaluasi Akhir pada Test Set E-DAIC (Raw Threshold 10.0 & Calibrated Threshold).
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
)

# Impor modul lokal
try:
    from .model import MultimodalBaselineModel
    from .loss import ClassWeightedMSELoss, MultiTaskClinicalLoss
    from .dataset import build_dataloaders
except ImportError:
    from model import MultimodalBaselineModel
    from loss import ClassWeightedMSELoss, MultiTaskClinicalLoss
    from dataset import build_dataloaders


def setup_logger(output_dir: Path) -> logging.Logger:
    """Mengonfigurasi logging ke konsol dan file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logger = logging.getLogger("MFDCL_Calibrated")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42) -> None:
    """Mengatur random seed untuk memastikan reproduktifitas eksperimen."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_optimal_val_threshold(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Mencari threshold optimal pada Validation Set untuk memaksimalkan F1-Score."""
    bin_targets = (val_targets >= 10.0).astype(int)
    thresholds = np.linspace(4.0, 16.0, 121)

    best_th = 10.0
    best_score = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for th in thresholds:
        bin_preds = (val_preds >= th).astype(int)
        score = f1_score(bin_targets, bin_preds, zero_division=0)
        prec = precision_score(bin_targets, bin_preds, zero_division=0)
        rec = recall_score(bin_targets, bin_preds, zero_division=0)

        if score > best_score:
            best_score = float(score)
            best_th = float(th)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_th, best_score, best_prec, best_rec


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Menjalankan 1 epoch training loop."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        audio = batch["audio"].to(device)    # (B, 512, 80)
        visual = batch["visual"].to(device)  # (B, 512, 72, 3)
        text = batch["text"].to(device)      # (B, 512, 768)
        target_phq = batch["phq8_score"].to(device)  # (B,)

        optimizer.zero_grad()
        pred_phq = model(audio, visual, text)  # (B, 1)

        loss = criterion(pred_phq, target_phq)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * audio.size(0)
        all_preds.extend(pred_phq.detach().cpu().view(-1).numpy())
        all_targets.extend(target_phq.detach().cpu().view(-1).numpy())

    num_samples = len(all_preds)
    avg_loss = total_loss / max(num_samples, 1)

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    mae = mean_absolute_error(targets_np, preds_np)
    mse = mean_squared_error(targets_np, preds_np)

    bin_preds = (preds_np >= 10.0).astype(int)
    bin_targets = (targets_np >= 10.0).astype(int)
    f1 = f1_score(bin_targets, bin_preds, zero_division=0)
    acc = accuracy_score(bin_targets, bin_preds)

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "f1": f1,
        "accuracy": acc,
    }


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 10.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Mengevaluasi model pada dataset dengan threshold tertentu."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            target_phq = batch["phq8_score"].to(device)

            pred_phq = model(audio, visual, text)
            loss = criterion(pred_phq, target_phq)

            total_loss += loss.item() * audio.size(0)
            all_preds.extend(pred_phq.detach().cpu().view(-1).numpy())
            all_targets.extend(target_phq.detach().cpu().view(-1).numpy())

    num_samples = len(all_preds)
    avg_loss = total_loss / max(num_samples, 1)

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    mae = mean_absolute_error(targets_np, preds_np)
    mse = mean_squared_error(targets_np, preds_np)
    rmse = np.sqrt(mse)

    bin_preds = (preds_np >= threshold).astype(int)
    bin_targets = (targets_np >= 10.0).astype(int)

    f1 = f1_score(bin_targets, bin_preds, zero_division=0)
    prec = precision_score(bin_targets, bin_preds, zero_division=0)
    rec = recall_score(bin_targets, bin_preds, zero_division=0)
    acc = accuracy_score(bin_targets, bin_preds)

    cm = confusion_matrix(bin_targets, bin_preds)
    fn = int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    metrics = {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }
    return metrics, preds_np, targets_np


def run_training(args: argparse.Namespace) -> MultimodalBaselineModel:
    """Pipeline training, validation checkpointing, dan test set evaluation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    logger.info("=" * 75)
    logger.info("  CENTRALIZED MULTIMODAL BASELINE TRAINING (CALIBRATED MFDCL)")
    logger.info("=" * 75)
    logger.info(f"Device                 : {device}")
    logger.info(f"Random Seed            : {args.seed}")
    logger.info(f"Epochs                 : {args.epochs}")
    logger.info(f"Batch Size             : {args.batch_size}")
    logger.info(f"Learning Rate          : {args.lr}")
    logger.info(f"Weight Decay           : {args.weight_decay}")
    logger.info(f"Optimizer              : {args.optimizer}")
    logger.info(f"Scheduler              : {args.scheduler}")
    logger.info(f"Dropout (Conv / Head)  : ({args.dropout_conv}, {args.regressor_dropout})")
    logger.info(f"Pos Weight Multiplier  : {args.pos_weight_multiplier}x")
    logger.info(f"FN Penalty             : {args.fn_penalty}x")
    logger.info(f"Feature Normalization  : {args.normalize} (Z-Score on Train Split)")
    logger.info(f"Output Dir             : {output_dir.resolve()}")

    # 1. Bangun DataLoader
    train_loader, val_loader, test_loader = build_dataloaders(
        index_csv=args.index_csv,
        labels_csv=args.labels_csv,
        feature_base_dir=args.data_dir,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    # 2. Inisialisasi Model, Loss, Optimizer, Scheduler
    model = MultimodalBaselineModel(
        dropout_conv=args.dropout_conv,
        regressor_dropout=args.regressor_dropout,
        pooling_type=args.pooling_type,
        use_gating=args.use_gating,
    ).to(device)

    if args.loss_type == "multitask":
        criterion = MultiTaskClinicalLoss(
            binarize_threshold=10.0,
            pos_weight_multiplier=args.pos_weight_multiplier,
            fn_penalty=args.fn_penalty,
            bce_weight=args.bce_weight,
        )
    else:
        criterion = ClassWeightedMSELoss(
            binarize_threshold=10.0,
            pos_weight_multiplier=args.pos_weight_multiplier,
            fn_penalty=args.fn_penalty,
        )

    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

    if args.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_f1 = -1.0
    best_epoch = 0
    best_val_preds = None
    best_val_targets = None
    best_model_path = output_dir / "best_model.pt"

    history = []

    logger.info(f"\nMemulai training loop {args.epochs} epochs (tracking validation F1)...")
    logger.info("-" * 75)
    header = f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val MAE':^8} | {'Val F1':^8} | {'Val Acc':^8} | {'LR':^9}"
    logger.info(header)
    logger.info("-" * 75)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Val
        val_res, v_preds, v_targets = evaluate(model, val_loader, criterion, device, threshold=10.0)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row_str = (
            f"{epoch:^7d} | "
            f"{train_res['loss']:^10.4f} | "
            f"{val_res['loss']:^10.4f} | "
            f"{val_res['mae']:^8.2f} | "
            f"{val_res['f1']:^8.4f} | "
            f"{val_res['accuracy']:^8.4f} | "
            f"{current_lr:^9.2e}"
        )

        is_best = False
        if val_res["f1"] > best_val_f1:
            best_val_f1 = val_res["f1"]
            best_epoch = epoch
            best_val_preds = v_preds.copy()
            best_val_targets = v_targets.copy()
            is_best = True
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": float(best_val_f1),
                    "val_metrics": val_res,
                    "hyperparameters": vars(args),
                },
                best_model_path,
            )
            row_str += "  <-- [BEST F1]"

        logger.info(row_str)

        history.append({
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_mae": train_res["mae"],
            "train_mse": train_res["mse"],
            "train_f1": train_res["f1"],
            "train_acc": train_res["accuracy"],
            "val_loss": val_res["loss"],
            "val_mae": val_res["mae"],
            "val_mse": val_res["mse"],
            "val_rmse": val_res["rmse"],
            "val_f1": val_res["f1"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_acc": val_res["accuracy"],
            "lr": current_lr,
            "is_best": is_best,
        })

    elapsed = time.time() - start_time
    logger.info("-" * 75)
    logger.info(f"Training selesai dalam {elapsed:.2f} detik.")
    logger.info(f"Best Validation F1: {best_val_f1:.4f} pada Epoch {best_epoch}")

    # Simpan history ke CSV
    history_df = pd.DataFrame(history)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"History training tersimpan di: {history_path}")

    # 3. Final Test Evaluation
    logger.info("\n" + "=" * 75)
    logger.info("  FINAL EVALUATION PADA E-DAIC TEST SET (BEST CHECKPOINT)")
    logger.info("=" * 75)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Memuat checkpoint terbaik dari Epoch {checkpoint['epoch']} (Val F1: {checkpoint['best_val_f1']:.4f})")

    # Evaluasi Raw (Threshold 10.0)
    raw_res, test_preds, test_targets = evaluate(model, test_loader, criterion, device, threshold=10.0)

    # Kalibrasi threshold pada Validation set
    opt_th, opt_vf1, opt_vprec, opt_vrec = find_optimal_val_threshold(best_val_preds, best_val_targets)
    cal_res, _, _ = evaluate(model, test_loader, criterion, device, threshold=opt_th)

    bin_test_targets = (test_targets >= 10.0).astype(int)
    bin_test_preds_raw = (test_preds >= 10.0).astype(int)
    bin_test_preds_cal = (test_preds >= opt_th).astype(int)

    logger.info("\n--- REGRESSION METRICS (Continuous PHQ-8) ---")
    logger.info(f"Test MAE   : {raw_res['mae']:.4f}")
    logger.info(f"Test MSE   : {raw_res['mse']:.4f}")
    logger.info(f"Test RMSE  : {raw_res['rmse']:.4f}")
    logger.info(f"Test Loss  : {raw_res['loss']:.4f}")

    logger.info("\n--- BINARY METRICS: RAW THRESHOLD (PHQ-8 >= 10.0) ---")
    logger.info(f"Test Accuracy  : {raw_res['accuracy'] * 100:.2f}% ({raw_res['accuracy']:.4f})")
    logger.info(f"Test Precision : {raw_res['precision']:.4f}")
    logger.info(f"Test Recall    : {raw_res['recall']:.4f}")
    logger.info(f"Test F1-Score  : {raw_res['f1']:.4f}")
    logger.info(f"False Negatives: {raw_res['fn']} (TP: {raw_res['tp']})")

    cm_raw = confusion_matrix(bin_test_targets, bin_test_preds_raw)
    logger.info("Confusion Matrix (Raw 10.0):")
    logger.info(f"                 Pred Normal(0)   Pred Depressed(1)")
    logger.info(f"True Normal(0)        {cm_raw[0, 0]:<15d} {cm_raw[0, 1]:<15d}")
    if cm_raw.shape[0] > 1 and cm_raw.shape[1] > 1:
        logger.info(f"True Depressed(1)     {cm_raw[1, 0]:<15d} {cm_raw[1, 1]:<15d}")

    logger.info(f"\n--- BINARY METRICS: CALIBRATED THRESHOLD (PHQ-8 >= {opt_th:.2f}) ---")
    logger.info(f"Test Accuracy  : {cal_res['accuracy'] * 100:.2f}% ({cal_res['accuracy']:.4f})")
    logger.info(f"Test Precision : {cal_res['precision']:.4f}")
    logger.info(f"Test Recall    : {cal_res['recall']:.4f}")
    logger.info(f"Test F1-Score  : {cal_res['f1']:.4f}")
    logger.info(f"False Negatives: {cal_res['fn']} (TP: {cal_res['tp']})")

    cm_cal = confusion_matrix(bin_test_targets, bin_test_preds_cal)
    logger.info(f"Confusion Matrix (Calibrated {opt_th:.2f}):")
    logger.info(f"                 Pred Normal(0)   Pred Depressed(1)")
    logger.info(f"True Normal(0)        {cm_cal[0, 0]:<15d} {cm_cal[0, 1]:<15d}")
    if cm_cal.shape[0] > 1 and cm_cal.shape[1] > 1:
        logger.info(f"True Depressed(1)     {cm_cal[1, 0]:<15d} {cm_cal[1, 1]:<15d}")

    report = classification_report(
        bin_test_targets,
        bin_test_preds_raw,
        target_names=["Normal (0)", "Depressed (1)"],
        digits=4,
        zero_division=0,
    )
    logger.info(f"\nClassification Report (Raw 10.0):\n{report}")

    # Simpan prediksi test ke CSV
    test_results_df = pd.DataFrame({
        "true_phq8": test_targets,
        "pred_phq8": test_preds,
        "true_label": bin_test_targets,
        "pred_label_raw": bin_test_preds_raw,
        "pred_label_calibrated": bin_test_preds_cal,
    })
    test_results_path = output_dir / "test_predictions.csv"
    test_results_df.to_csv(test_results_path, index=False)

    summary_df = pd.DataFrame([{
        "best_epoch": checkpoint["epoch"],
        "val_best_f1_raw": checkpoint["best_val_f1"],
        "val_opt_threshold": opt_th,
        "val_best_f1_calibrated": opt_vf1,
        "test_mae": raw_res["mae"],
        "test_mse": raw_res["mse"],
        "test_rmse": raw_res["rmse"],
        "test_accuracy_raw": raw_res["accuracy"],
        "test_precision_raw": raw_res["precision"],
        "test_recall_raw": raw_res["recall"],
        "test_f1_raw": raw_res["f1"],
        "test_fn_raw": raw_res["fn"],
        "test_accuracy_calibrated": cal_res["accuracy"],
        "test_precision_calibrated": cal_res["precision"],
        "test_recall_calibrated": cal_res["recall"],
        "test_f1_calibrated": cal_res["f1"],
        "test_fn_calibrated": cal_res["fn"],
    }])
    summary_path = output_dir / "test_summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Hasil prediksi test tersimpan di: {test_results_path}")
    logger.info(f"Ringkasan metrik test tersimpan di: {summary_path}")

    return model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized Multimodal Baseline Training (MFDCL Protocol)")
    parser.add_argument("--data_dir", type=str, default="modeling/01_Input_From_Partition", help="Base directory of features")
    parser.add_argument("--index_csv", type=str, default="modeling/01_Input_From_Partition/multimodal_feature_index.csv", help="Index CSV path")
    parser.add_argument("--labels_csv", type=str, default="data/detailed_lables.csv", help="Labels CSV path")
    parser.add_argument("--output_dir", type=str, default="modeling2/output", help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate (default: 4e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"], help="Scheduler type")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for StepLR")
    parser.add_argument("--dropout_conv", type=float, default=0.4, help="Dropout in Conv layers (default: 0.4)")
    parser.add_argument("--regressor_dropout", type=float, default=0.3, help="Dropout in regressor head (default: 0.3)")
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "attention"], help="BiLSTM pooling type")
    parser.add_argument("--use_gating", action="store_true", help="Enable gated multimodal fusion")
    parser.add_argument("--loss_type", type=str, default="weighted_mse", choices=["weighted_mse", "multitask"], help="Loss function type")
    parser.add_argument("--pos_weight_multiplier", type=float, default=1.5, help="Multiplier for positive class weight (default: 1.5)")
    parser.add_argument("--fn_penalty", type=float, default=1.2, help="Penalty for false negatives (default: 1.2)")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="Weight for auxiliary BCE loss if multitask")
    parser.add_argument("--normalize", action="store_true", default=True, help="Enable Z-score normalization from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_training(args)
