import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path.cwd()))

from modeling2.model import AudioEncoder, VisualEncoder, TextEncoder
from modeling2.loss import ClassWeightedMSELoss
from modeling2.dataset import build_dataloaders
from modeling2.train import set_seed
from modeling2.tune import find_optimal_threshold, evaluate_with_threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training ablated models on device: {device}")

train_loader, val_loader, test_loader = build_dataloaders(
    index_csv='modeling/01_Input_From_Partition/multimodal_feature_index.csv',
    labels_csv='data/detailed_lables.csv',
    feature_base_dir='modeling/01_Input_From_Partition',
    batch_size=32,
    normalize=True
)

test_pids = []
for batch in test_loader:
    test_pids.extend(batch['participant_id'])


class FlexibleModalityModel(nn.Module):
    def __init__(self, active_modalities, dropout_conv=0.4, regressor_dropout=0.3):
        super().__init__()
        self.active_modalities = active_modalities
        dim_total = 0
        if "audio" in active_modalities:
            self.audio_encoder = AudioEncoder(dropout_conv=dropout_conv, pooling_type="mean")
            dim_total += 256
        else:
            self.audio_encoder = None

        if "visual" in active_modalities:
            self.visual_encoder = VisualEncoder(dropout_conv=dropout_conv, pooling_type="mean")
            dim_total += 256
        else:
            self.visual_encoder = None

        if "text" in active_modalities:
            self.text_encoder = TextEncoder(dropout_conv=dropout_conv, pooling_type="mean")
            dim_total += 256
        else:
            self.text_encoder = None

        self.regressor = nn.Sequential(
            nn.Linear(dim_total, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=regressor_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, audio, visual, text):
        feats = []
        if self.audio_encoder is not None:
            feats.append(self.audio_encoder(audio))
        if self.visual_encoder is not None:
            feats.append(self.visual_encoder(visual))
        if self.text_encoder is not None:
            feats.append(self.text_encoder(text))

        fused = torch.cat(feats, dim=-1)
        return self.regressor(fused)


experiments = [
    ("text_only", ["text"]),
    ("visual_only", ["visual"]),
    ("audio_only", ["audio"]),
    ("text_audio", ["text", "audio"]),
    ("visual_audio", ["visual", "audio"]),
    ("text_visual", ["text", "visual"]),
]

out_dir_1 = Path("modeling2/output/output02_Moderate_Regularization_Drop04")
out_dir_2 = Path("output/output02_Moderate_Regularization_Drop04")
out_dir_1.mkdir(parents=True, exist_ok=True)
out_dir_2.mkdir(parents=True, exist_ok=True)

trained_summary_rows = []

epochs = 60
lr = 4e-4
weight_decay = 1e-4

for name, active_mods in experiments:
    print(f"\n--- Training Dedicated Model: {name} (Modalities: {active_mods}) ---")
    set_seed(42)
    model = FlexibleModalityModel(active_mods, dropout_conv=0.4, regressor_dropout=0.3).to(device)
    criterion = ClassWeightedMSELoss(binarize_threshold=10.0, pos_weight_multiplier=1.5, fn_penalty=1.2)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    best_weights = None
    best_val_preds, best_val_targets = None, None

    for epoch in range(1, epochs + 1):
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

        # Val eval
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                visual = batch["visual"].to(device)
                text = batch["text"].to(device)
                target = batch["phq8_score"].to(device)
                pred = model(audio, visual, text)
                v_preds.extend(pred.cpu().view(-1).numpy())
                v_targets.extend(target.cpu().view(-1).numpy())

        v_preds = np.array(v_preds)
        v_targets = np.array(v_targets)
        bin_vp = (v_preds >= 10.0).astype(int)
        bin_vt = (v_targets >= 10.0).astype(int)
        val_f1 = f1_score(bin_vt, bin_vp, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_val_preds = v_preds.copy()
            best_val_targets = v_targets.copy()

    # Load best weights
    model.load_state_dict(best_weights)
    opt_th, opt_val_f1, _, _ = find_optimal_threshold(best_val_preds, best_val_targets)

    # Test eval
    model.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            target = batch["phq8_score"].to(device)
            pred = model(audio, visual, text)
            t_preds.extend(pred.cpu().view(-1).numpy())
            t_targets.extend(target.cpu().view(-1).numpy())

    t_preds = np.array(t_preds)
    t_targets = np.array(t_targets)

    raw_m = evaluate_with_threshold(t_preds, t_targets, threshold=10.0)
    cal_m = evaluate_with_threshold(t_preds, t_targets, threshold=opt_th)

    # Save predictions
    pred_df = pd.DataFrame({
        "participant_id": test_pids,
        "true_phq8": t_targets,
        "pred_phq8": t_preds,
        "true_label": (t_targets >= 10.0).astype(int),
        "pred_label_raw_th10": (t_preds >= 10.0).astype(int),
        "pred_label_calibrated": (t_preds >= opt_th).astype(int),
    })
    pred_df.to_csv(out_dir_1 / f"dedicated_trained_predictions_{name}.csv", index=False)
    pred_df.to_csv(out_dir_2 / f"dedicated_trained_predictions_{name}.csv", index=False)

    print(f"[{name}] Best Epoch: {best_epoch} | Val F1: {best_val_f1:.4f} | Opt Th: {opt_th:.2f} | Test Acc: {raw_m['accuracy']:.4f} | Test F1: {raw_m['f1']:.4f} | Cal Test F1: {cal_m['f1']:.4f}")

    trained_summary_rows.append({
        "Modality": name,
        "Best_Epoch": best_epoch,
        "Val_Opt_Threshold": round(opt_th, 2),
        "Val_F1_Raw": round(best_val_f1, 4),
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

# Add text_audio_visual from trial 2
t2_ckpt = torch.load("modeling2/output/best_model.pt", map_location=device, weights_only=False)
trained_summary_rows.append({
    "Modality": "text_audio_visual",
    "Best_Epoch": 7,
    "Val_Opt_Threshold": 10.0,
    "Val_F1_Raw": 0.5714,
    "Val_F1_Calibrated": 0.5714,
    "Test_Accuracy_Raw": 0.7857,
    "Test_Precision_Raw": 0.7143,
    "Test_Recall_Raw": 0.7143,
    "Test_F1_Raw": 0.7143,
    "Test_FN_Raw": 6,
    "Test_TP_Raw": 15,
    "Test_TN_Raw": 29,
    "Test_FP_Raw": 6,
    "Test_Accuracy_Calibrated": 0.7857,
    "Test_Precision_Calibrated": 0.7143,
    "Test_Recall_Calibrated": 0.7143,
    "Test_F1_Calibrated": 0.7143,
    "Test_FN_Calibrated": 6,
    "Test_TP_Calibrated": 15,
    "Test_TN_Calibrated": 29,
    "Test_FP_Calibrated": 6,
    "Test_MAE": 4.6645,
    "Test_MSE": 30.5871,
    "Test_RMSE": 5.5306,
})

df_dedicated = pd.DataFrame(trained_summary_rows)
df_dedicated.to_csv(out_dir_1 / "dedicated_trained_ablation_summary.csv", index=False)
df_dedicated.to_csv(out_dir_2 / "dedicated_trained_ablation_summary.csv", index=False)
print("\nAll dedicated ablated models trained and evaluated successfully!")
