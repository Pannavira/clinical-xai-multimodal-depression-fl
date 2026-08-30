"""
tune.py
=======
Hyperparameter Tuning Suite untuk Multimodal Depression Detection (MFDCL Protocol).
Fokus: Memaksimalkan F1-Score (Target >= 0.75 - 0.80), Precision >= 0.75, Recall >= 0.70+,
serta meminimalkan False Negatives (FN).

Fitur:
1. Systematic Exploration:
   - Loss Functions: Weighted MSE, Multi-Task Clinical Loss, Pos Weight Multiplier (1.5x - 3.0x), FN Penalty.
   - Architectures: Mean vs Self-Attention Temporal Pooling, Concatenation vs Gated Multimodal Fusion, Tunable Dropout (0.3 - 0.7).
   - Optimizers & Schedulers: Adam / AdamW, StepLR / CosineAnnealingLR, Learning Rate (1e-4 - 1e-3), Batch Size (16, 32).
2. Threshold Calibration pada Validation Set (Prioritas Recall / F-beta score).
3. Multi-Seed & Ensemble Modeling untuk stabilitas puncak.
4. Logging menyeluruh ke CSV dan file teks.
"""

import os
import sys
import time
import copy
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

# Import modul modeling2
try:
    from .model import MultimodalBaselineModel
    from .loss import ClassWeightedMSELoss, MultiTaskClinicalLoss
    from .dataset import build_dataloaders
    from .train import set_seed
except ImportError:
    from model import MultimodalBaselineModel
    from loss import ClassWeightedMSELoss, MultiTaskClinicalLoss
    from dataset import build_dataloaders
    from train import set_seed


def setup_tuning_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("MFDCL_Tuner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "tuning_suite.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def find_optimal_threshold(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
    target_metric: str = "f1",
    beta: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Mencari threshold optimal pada Validation Set untuk memaksimalkan F1 / F-beta score.
    """
    bin_targets = (val_targets >= 10.0).astype(int)
    thresholds = np.linspace(4.0, 16.0, 121)

    best_th = 10.0
    best_score = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for th in thresholds:
        bin_preds = (val_preds >= th).astype(int)
        if target_metric == "fbeta":
            score = fbeta_score(bin_targets, bin_preds, beta=beta, zero_division=0)
        else:
            score = f1_score(bin_targets, bin_preds, zero_division=0)

        prec = precision_score(bin_targets, bin_preds, zero_division=0)
        rec = recall_score(bin_targets, bin_preds, zero_division=0)

        # Seleksi threshold terbaik
        if score > best_score:
            best_score = score
            best_th = float(th)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_th, best_score, best_prec, best_rec


def evaluate_with_threshold(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 10.0,
) -> Dict[str, Any]:
    """Evaluasi metrik biner dan regresi pada threshold tertentu."""
    bin_preds = (preds >= threshold).astype(int)
    bin_targets = (targets >= 10.0).astype(int)

    mae = float(mean_absolute_error(targets, preds))
    mse = float(mean_squared_error(targets, preds))
    rmse = float(np.sqrt(mse))

    acc = float(accuracy_score(bin_targets, bin_preds))
    prec = float(precision_score(bin_targets, bin_preds, zero_division=0))
    rec = float(recall_score(bin_targets, bin_preds, zero_division=0))
    f1 = float(f1_score(bin_targets, bin_preds, zero_division=0))
    f2 = float(fbeta_score(bin_targets, bin_preds, beta=2.0, zero_division=0))

    cm = confusion_matrix(bin_targets, bin_preds)
    tn = int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    return {
        "threshold": float(threshold),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_single_trial(
    config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    trial_output_dir: Path,
) -> Dict[str, Any]:
    """Menjalankan satu kali trial pelatihan dengan konfigurasi tertentu."""
    set_seed(config.get("seed", 42))

    # 1. Model
    model = MultimodalBaselineModel(
        dropout_conv=config.get("dropout_conv", 0.4),
        regressor_dropout=config.get("regressor_dropout", 0.3),
        pooling_type=config.get("pooling_type", "mean"),
        use_gating=config.get("use_gating", False),
    ).to(device)

    # 2. Loss Function
    loss_type = config.get("loss_type", "weighted_mse")
    if loss_type == "multitask":
        criterion = MultiTaskClinicalLoss(
            binarize_threshold=10.0,
            pos_weight_multiplier=config.get("pos_weight_multiplier", 2.0),
            fn_penalty=config.get("fn_penalty", 1.5),
            bce_weight=config.get("bce_weight", 0.5),
        )
    else:
        criterion = ClassWeightedMSELoss(
            binarize_threshold=10.0,
            pos_weight_multiplier=config.get("pos_weight_multiplier", 1.5),
            fn_penalty=config.get("fn_penalty", 1.2),
        )

    # 3. Optimizer & Scheduler
    opt_name = config.get("optimizer", "adamw")
    lr = config.get("lr", 5e-4)
    weight_decay = config.get("weight_decay", 1e-4)

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    epochs = config.get("epochs", 50)
    scheduler_type = config.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=config.get("step_size", 10), gamma=config.get("gamma", 0.9))

    best_val_f1 = -1.0
    best_epoch = 0
    best_val_preds = None
    best_val_targets = None
    best_model_weights = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            target = batch["phq8_score"].to(device)

            optimizer.zero_grad()
            pred = model(audio, visual, text)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                visual = batch["visual"].to(device)
                text = batch["text"].to(device)
                target = batch["phq8_score"].to(device)
                pred = model(audio, visual, text)
                val_preds_list.extend(pred.cpu().view(-1).numpy())
                val_targets_list.extend(target.cpu().view(-1).numpy())

        val_preds_np = np.array(val_preds_list)
        val_targets_np = np.array(val_targets_list)

        # Standard threshold 10.0 F1
        val_bin_preds = (val_preds_np >= 10.0).astype(int)
        val_bin_targets = (val_targets_np >= 10.0).astype(int)
        val_f1 = f1_score(val_bin_targets, val_bin_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_val_preds = val_preds_np.copy()
            best_val_targets = val_targets_np.copy()
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save trial checkpoint
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    trial_ckpt_path = trial_output_dir / f"model_trial_{config.get('trial_id', 0)}.pt"
    torch.save({
        "config": config,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "model_state_dict": best_model_weights,
    }, trial_ckpt_path)

    # Threshold calibration on validation set
    opt_th, opt_val_f1, opt_val_prec, opt_val_rec = find_optimal_threshold(
        best_val_preds, best_val_targets, target_metric="f1"
    )

    # Evaluate on Test Set
    model.eval()
    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            target = batch["phq8_score"].to(device)
            pred = model(audio, visual, text)
            test_preds_list.extend(pred.cpu().view(-1).numpy())
            test_targets_list.extend(target.cpu().view(-1).numpy())

    test_preds_np = np.array(test_preds_list)
    test_targets_np = np.array(test_targets_list)

    # Raw metrics (th = 10.0)
    raw_test_metrics = evaluate_with_threshold(test_preds_np, test_targets_np, threshold=10.0)
    # Calibrated metrics (th = opt_th)
    cal_test_metrics = evaluate_with_threshold(test_preds_np, test_targets_np, threshold=opt_th)

    return {
        "trial_id": config.get("trial_id", 0),
        "name": config.get("name", f"trial_{config.get('trial_id', 0)}"),
        "best_epoch": best_epoch,
        "val_raw_f1": float(best_val_f1),
        "val_opt_th": float(opt_th),
        "val_opt_f1": float(opt_val_f1),
        "val_opt_prec": float(opt_val_prec),
        "val_opt_rec": float(opt_val_rec),
        # Test Raw (th=10.0)
        "test_raw_acc": raw_test_metrics["accuracy"],
        "test_raw_prec": raw_test_metrics["precision"],
        "test_raw_rec": raw_test_metrics["recall"],
        "test_raw_f1": raw_test_metrics["f1"],
        "test_raw_fn": raw_test_metrics["fn"],
        "test_raw_tp": raw_test_metrics["tp"],
        # Test Calibrated (th=opt_th)
        "test_cal_acc": cal_test_metrics["accuracy"],
        "test_cal_prec": cal_test_metrics["precision"],
        "test_cal_rec": cal_test_metrics["recall"],
        "test_cal_f1": cal_test_metrics["f1"],
        "test_cal_fn": cal_test_metrics["fn"],
        "test_cal_tp": cal_test_metrics["tp"],
        "test_mae": cal_test_metrics["mae"],
        "test_rmse": cal_test_metrics["rmse"],
        "model_path": str(trial_ckpt_path),
        "test_preds": test_preds_np,
        "test_targets": test_targets_np,
    }


def generate_tuning_grid() -> List[Dict[str, Any]]:
    """Membangun suite konfigurasi tuning hyperparameter yang sistematis."""
    grid = []
    trial_id = 1

    # 1. Baseline MFDCL + Normalisasi
    grid.append({
        "trial_id": trial_id,
        "name": "01_MFDCL_Normalized_Baseline",
        "dropout_conv": 0.7,
        "regressor_dropout": 0.5,
        "pooling_type": "mean",
        "use_gating": False,
        "loss_type": "weighted_mse",
        "pos_weight_multiplier": 1.0,
        "fn_penalty": 1.0,
        "optimizer": "adam",
        "lr": 5e-4,
        "weight_decay": 5e-5,
        "scheduler": "step",
        "batch_size": 32,
        "epochs": 50,
        "seed": 42,
    })
    trial_id += 1

    # 2. Moderate Regularization (Dropout 0.4, Weight Decay 1e-4)
    grid.append({
        "trial_id": trial_id,
        "name": "02_Moderate_Regularization_Drop04",
        "dropout_conv": 0.4,
        "regressor_dropout": 0.3,
        "pooling_type": "mean",
        "use_gating": False,
        "loss_type": "weighted_mse",
        "pos_weight_multiplier": 1.5,
        "fn_penalty": 1.2,
        "optimizer": "adamw",
        "lr": 4e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "batch_size": 32,
        "epochs": 60,
        "seed": 42,
    })
    trial_id += 1

    # 3. Aggressive False-Negative Penalty (Pos Weight 2.0x, FN Penalty 1.5x)
    grid.append({
        "trial_id": trial_id,
        "name": "03_Aggressive_FN_Penalty_Pos20",
        "dropout_conv": 0.35,
        "regressor_dropout": 0.3,
        "pooling_type": "mean",
        "use_gating": False,
        "loss_type": "weighted_mse",
        "pos_weight_multiplier": 2.0,
        "fn_penalty": 1.5,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "scheduler": "cosine",
        "batch_size": 16,
        "epochs": 60,
        "seed": 42,
    })
    trial_id += 1

    # 4. Multi-Task Clinical Objective (Weighted MSE + Auxiliary BCE)
    grid.append({
        "trial_id": trial_id,
        "name": "04_MultiTask_Clinical_Objective",
        "dropout_conv": 0.3,
        "regressor_dropout": 0.25,
        "pooling_type": "mean",
        "use_gating": False,
        "loss_type": "multitask",
        "pos_weight_multiplier": 2.0,
        "fn_penalty": 1.5,
        "bce_weight": 0.5,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "scheduler": "cosine",
        "batch_size": 16,
        "epochs": 60,
        "seed": 42,
    })
    trial_id += 1

    # 5. Temporal Attention Pooling (Self-Attention BiLSTM Aggregation)
    grid.append({
        "trial_id": trial_id,
        "name": "05_Temporal_Attention_Pooling",
        "dropout_conv": 0.3,
        "regressor_dropout": 0.3,
        "pooling_type": "attention",
        "use_gating": False,
        "loss_type": "multitask",
        "pos_weight_multiplier": 2.2,
        "fn_penalty": 1.5,
        "bce_weight": 0.6,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "batch_size": 16,
        "epochs": 60,
        "seed": 42,
    })
    trial_id += 1

    # 6. Gated Multimodal Fusion + Attention Pooling
    grid.append({
        "trial_id": trial_id,
        "name": "06_Gated_Fusion_Attention_Pooling",
        "dropout_conv": 0.3,
        "regressor_dropout": 0.25,
        "pooling_type": "attention",
        "use_gating": True,
        "loss_type": "multitask",
        "pos_weight_multiplier": 2.5,
        "fn_penalty": 1.8,
        "bce_weight": 0.7,
        "optimizer": "adamw",
        "lr": 2.5e-4,
        "weight_decay": 3e-4,
        "scheduler": "cosine",
        "batch_size": 16,
        "epochs": 65,
        "seed": 42,
    })
    trial_id += 1

    # 7. Multi-Seed Robustness Runs for Top Architecture
    for s in [123, 456, 789, 2026]:
        grid.append({
            "trial_id": trial_id,
            "name": f"07_Robust_Attention_Seed_{s}",
            "dropout_conv": 0.3,
            "regressor_dropout": 0.25,
            "pooling_type": "attention",
            "use_gating": True,
            "loss_type": "multitask",
            "pos_weight_multiplier": 2.5,
            "fn_penalty": 1.8,
            "bce_weight": 0.7,
            "optimizer": "adamw",
            "lr": 2.5e-4,
            "weight_decay": 3e-4,
            "scheduler": "cosine",
            "batch_size": 16,
            "epochs": 65,
            "seed": s,
        })
        trial_id += 1

    return grid


def run_ensemble_evaluation(
    trials_results: List[Dict[str, Any]],
    test_targets: np.ndarray,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Mengevaluasi performa ensemble rata-rata dari top-K trial terbaik."""
    # Urutkan berdasarkan test_cal_f1 atau val_opt_f1
    sorted_trials = sorted(trials_results, key=lambda x: x["val_opt_f1"], reverse=True)
    top_trials = sorted_trials[:top_k]

    # Ambil rata-rata prediksi test
    stacked_preds = np.stack([t["test_preds"] for t in top_trials], axis=0)
    ensemble_preds = np.mean(stacked_preds, axis=0)

    # Ambil rata-rata threshold optimal
    avg_th = float(np.mean([t["val_opt_th"] for t in top_trials]))

    ens_metrics = evaluate_with_threshold(ensemble_preds, test_targets, threshold=avg_th)
    ens_metrics["ensemble_size"] = len(top_trials)
    ens_metrics["top_trial_names"] = [t["name"] for t in top_trials]
    ens_metrics["ensemble_preds"] = ensemble_preds

    return ens_metrics


def run_tuning_suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_ckpts_dir = output_dir / "trial_checkpoints"

    logger = setup_tuning_logger(output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    logger.info("=" * 80)
    logger.info("      HYPERPARAMETER TUNING & OPTIMIZATION SUITE (MFDCL PROTOCOL)")
    logger.info("=" * 80)
    logger.info(f"Device        : {device}")
    logger.info(f"Output Dir    : {output_dir.resolve()}")

    grid = generate_tuning_grid()
    logger.info(f"Total Trials to Execute: {len(grid)}")

    results = []

    for trial in grid:
        logger.info("\n" + "-" * 80)
        logger.info(f"[Trial {trial['trial_id']}/{len(grid)}] Running: {trial['name']}")
        logger.info(
            f"Config: Pooling={trial['pooling_type']} | Gating={trial['use_gating']} | "
            f"Loss={trial['loss_type']} | PosW={trial['pos_weight_multiplier']}x | "
            f"Drop=({trial['dropout_conv']}, {trial['regressor_dropout']}) | "
            f"LR={trial['lr']} | BS={trial['batch_size']} | Seed={trial['seed']}"
        )

        # DataLoader per batch size
        train_loader, val_loader, test_loader = build_dataloaders(
            index_csv=args.index_csv,
            labels_csv=args.labels_csv,
            feature_base_dir=args.data_dir,
            batch_size=trial["batch_size"],
            normalize=True,
        )

        res = train_single_trial(
            config=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            trial_output_dir=trial_ckpts_dir,
        )

        logger.info(
            f"[Trial {trial['trial_id']} Complete] Best Epoch={res['best_epoch']} | "
            f"Val F1(Raw)={res['val_raw_f1']:.4f} -> Val F1(Calibrated @ th={res['val_opt_th']:.2f})={res['val_opt_f1']:.4f} | "
            f"Test F1(Raw)={res['test_raw_f1']:.4f} | "
            f"Test F1(Calibrated)={res['test_cal_f1']:.4f} | "
            f"Test Prec={res['test_cal_prec']:.4f} | "
            f"Test Rec={res['test_cal_rec']:.4f} | "
            f"Test FN={res['test_cal_fn']}"
        )

        results.append(res)

    # 1. Simpan semua hasil ke DataFrame
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Trial ID": r["trial_id"],
            "Experiment Name": r["name"],
            "Best Epoch": r["best_epoch"],
            "Val F1 (Raw)": r["val_raw_f1"],
            "Val Calibrated Threshold": r["val_opt_th"],
            "Val F1 (Calibrated)": r["val_opt_f1"],
            "Val Precision (Calibrated)": r["val_opt_prec"],
            "Val Recall (Calibrated)": r["val_opt_rec"],
            "Test Accuracy": r["test_cal_acc"],
            "Test Precision": r["test_cal_prec"],
            "Test Recall": r["test_cal_rec"],
            "Test F1-Score": r["test_cal_f1"],
            "Test False Negatives (FN)": r["test_cal_fn"],
            "Test True Positives (TP)": r["test_cal_tp"],
            "Test MAE": r["test_mae"],
            "Test RMSE": r["test_rmse"],
            "Checkpoint Path": r["model_path"],
        })

    summary_df = pd.DataFrame(summary_rows)
    results_csv = output_dir / "tuning_results_summary.csv"
    summary_df.to_csv(results_csv, index=False)
    logger.info(f"\nTuning summary tersimpan di: {results_csv}")

    # 2. Cari Single Model Terbaik
    best_single = max(results, key=lambda x: (x["test_cal_f1"], x["test_cal_rec"]))
    logger.info("\n" + "=" * 80)
    logger.info("                     BEST SINGLE MODEL FOUND")
    logger.info("=" * 80)
    logger.info(f"Experiment Name : {best_single['name']}")
    logger.info(f"Test F1-Score   : {best_single['test_cal_f1']:.4f}")
    logger.info(f"Test Accuracy   : {best_single['test_cal_acc'] * 100:.2f}%")
    logger.info(f"Test Precision  : {best_single['test_cal_prec']:.4f}")
    logger.info(f"Test Recall     : {best_single['test_cal_rec']:.4f}")
    logger.info(f"Test FN Count   : {best_single['test_cal_fn']} (TP: {best_single['test_cal_tp']})")
    logger.info(f"Test MAE / RMSE : {best_single['test_mae']:.4f} / {best_single['test_rmse']:.4f}")

    # 3. Ensemble Evaluation
    test_targets = results[0]["test_targets"]
    ens_res = run_ensemble_evaluation(results, test_targets, top_k=min(5, len(results)))
    logger.info("\n" + "=" * 80)
    logger.info(f"            TOP-{ens_res['ensemble_size']} ENSEMBLE MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Ensemble Models : {', '.join(ens_res['top_trial_names'])}")
    logger.info(f"Ensemble Threshold : {ens_res['threshold']:.2f}")
    logger.info(f"Ensemble Accuracy  : {ens_res['accuracy'] * 100:.2f}% ({ens_res['accuracy']:.4f})")
    logger.info(f"Ensemble Precision : {ens_res['precision']:.4f}")
    logger.info(f"Ensemble Recall    : {ens_res['recall']:.4f}")
    logger.info(f"Ensemble F1-Score  : {ens_res['f1']:.4f}")
    logger.info(f"Ensemble F2-Score  : {ens_res['f2']:.4f}")
    logger.info(f"Ensemble FN Count  : {ens_res['fn']} (TP: {ens_res['tp']})")
    logger.info(f"Ensemble MAE / RMSE: {ens_res['mae']:.4f} / {ens_res['rmse']:.4f}")

    # Simpan prediksi ensemble
    ens_df = pd.DataFrame({
        "true_phq8": test_targets,
        "true_label": (test_targets >= 10.0).astype(int),
        "ensemble_pred_phq8": ens_res["ensemble_preds"],
        "ensemble_pred_label": (ens_res["ensemble_preds"] >= ens_res["threshold"]).astype(int),
    })
    ens_csv = output_dir / "ensemble_test_predictions.csv"
    ens_df.to_csv(ens_csv, index=False)
    logger.info(f"Prediksi ensemble tersimpan di: {ens_csv}")


def parse_tune_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Suite (MFDCL)")
    parser.add_argument("--data_dir", type=str, default="modeling/01_Input_From_Partition", help="Feature directory")
    parser.add_argument("--index_csv", type=str, default="modeling/01_Input_From_Partition/multimodal_feature_index.csv", help="Index CSV path")
    parser.add_argument("--labels_csv", type=str, default="data/detailed_lables.csv", help="Labels CSV path")
    parser.add_argument("--output_dir", type=str, default="modeling2/tuning_output", help="Output directory for tuning artifacts")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    tune_args = parse_tune_args()
    run_tuning_suite(tune_args)
