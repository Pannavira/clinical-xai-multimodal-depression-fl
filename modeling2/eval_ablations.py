import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Add project root
sys.path.insert(0, str(Path.cwd()))

from modeling2.model import MultimodalBaselineModel
from modeling2.dataset import build_dataloaders
from modeling2.tune import find_optimal_threshold, evaluate_with_threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_loader, val_loader, test_loader = build_dataloaders(
    index_csv='modeling/01_Input_From_Partition/multimodal_feature_index.csv',
    labels_csv='data/detailed_lables.csv',
    feature_base_dir='modeling/01_Input_From_Partition',
    batch_size=32,
    normalize=True
)

ckpt = torch.load('modeling2/output/best_model.pt', map_location=device, weights_only=False)
model = MultimodalBaselineModel(
    dropout_conv=0.4,
    regressor_dropout=0.3,
    pooling_type='mean',
    use_gating=False
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

modalities = [
    ("text_only", False, False, True),
    ("visual_only", False, True, False),
    ("audio_only", True, False, False),
    ("text_audio", True, False, True),
    ("visual_audio", True, True, False),
    ("text_visual", False, True, True),
    ("text_audio_visual", True, True, True),
]

summary_rows = []

# Targets from test loader
test_pids = []
for batch in test_loader:
    test_pids.extend(batch['participant_id'])

# Prepare output directory
out_dir_1 = Path("modeling2/output/output02_Moderate_Regularization_Drop04")
out_dir_2 = Path("output/output02_Moderate_Regularization_Drop04")
out_dir_1.mkdir(parents=True, exist_ok=True)
out_dir_2.mkdir(parents=True, exist_ok=True)

for name, use_a, use_v, use_t in modalities:
    # 1. Validation Set
    v_preds, v_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].to(device) if use_a else torch.zeros_like(batch['audio']).to(device)
            visual = batch['visual'].to(device) if use_v else torch.zeros_like(batch['visual']).to(device)
            text = batch['text'].to(device) if use_t else torch.zeros_like(batch['text']).to(device)
            target = batch['phq8_score'].to(device)
            v_preds.extend(model(audio, visual, text).cpu().view(-1).numpy())
            v_targets.extend(target.cpu().view(-1).numpy())

    v_preds = np.array(v_preds)
    v_targets = np.array(v_targets)
    opt_th, opt_val_f1, opt_val_prec, opt_val_rec = find_optimal_threshold(v_preds, v_targets)

    # 2. Test Set
    t_preds, t_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device) if use_a else torch.zeros_like(batch['audio']).to(device)
            visual = batch['visual'].to(device) if use_v else torch.zeros_like(batch['visual']).to(device)
            text = batch['text'].to(device) if use_t else torch.zeros_like(batch['text']).to(device)
            target = batch['phq8_score'].to(device)
            t_preds.extend(model(audio, visual, text).cpu().view(-1).numpy())
            t_targets.extend(target.cpu().view(-1).numpy())

    t_preds = np.array(t_preds)
    t_targets = np.array(t_targets)

    # Metrics
    raw_m = evaluate_with_threshold(t_preds, t_targets, threshold=10.0)
    cal_m = evaluate_with_threshold(t_preds, t_targets, threshold=opt_th)

    # Predictions CSV
    pred_df = pd.DataFrame({
        "participant_id": test_pids,
        "true_phq8": t_targets,
        "pred_phq8": t_preds,
        "true_label": (t_targets >= 10.0).astype(int),
        "pred_label_raw_th10": (t_preds >= 10.0).astype(int),
        "pred_label_calibrated": (t_preds >= opt_th).astype(int),
    })
    pred_df.to_csv(out_dir_1 / f"predictions_{name}.csv", index=False)
    pred_df.to_csv(out_dir_2 / f"predictions_{name}.csv", index=False)

    summary_rows.append({
        "Modality": name,
        "Val_Opt_Threshold": round(opt_th, 2),
        "Val_F1_Raw": round(evaluate_with_threshold(v_preds, v_targets, threshold=10.0)['f1'], 4),
        "Val_F1_Calibrated": round(opt_val_f1, 4),
        "Test_Accuracy_Raw": round(raw_m["accuracy"], 4),
        "Test_Precision_Raw": round(raw_m["precision"], 4),
        "Test_Recall_Raw": round(raw_m["recall"], 4),
        "Test_F1_Raw": round(raw_m["f1"], 4),
        "Test_FN_Raw": raw_m["fn"],
        "Test_TP_Raw": raw_m["tp"],
        "Test_TN_Raw": raw_m["tn"],
        "Test_FP_Raw": raw_m["fp"],
        "Test_Accuracy_Calibrated": round(cal_m["accuracy"], 4),
        "Test_Precision_Calibrated": round(cal_m["precision"], 4),
        "Test_Recall_Calibrated": round(cal_m["recall"], 4),
        "Test_F1_Calibrated": round(cal_m["f1"], 4),
        "Test_FN_Calibrated": cal_m["fn"],
        "Test_TP_Calibrated": cal_m["tp"],
        "Test_TN_Calibrated": cal_m["tn"],
        "Test_FP_Calibrated": cal_m["fp"],
        "Test_MAE": round(raw_m["mae"], 4),
        "Test_MSE": round(raw_m["mse"], 4),
        "Test_RMSE": round(raw_m["rmse"], 4),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out_dir_1 / "modality_ablation_summary.csv", index=False)
summary_df.to_csv(out_dir_2 / "modality_ablation_summary.csv", index=False)

print("Saved ablation summary successfully!")
print(summary_df[["Modality", "Val_Opt_Threshold", "Test_Accuracy_Raw", "Test_F1_Raw", "Test_Accuracy_Calibrated", "Test_F1_Calibrated", "Test_MAE", "Test_RMSE"]].to_string(index=False))
