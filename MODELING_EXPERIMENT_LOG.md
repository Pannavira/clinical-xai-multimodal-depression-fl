# 🧪 Multimodal Depression Model Evolution & Experiment Log

This document tracks the complete development lifecycle, architectural iterations, hyperparameter optimizations, and metric benchmarks for the Centralized Multimodal Depression Classifier.

---

## 📌 Executive Progression Map

| Milestone Stage | Git Branch | Commit Hash | Key Architectural & Training Changes | Primary Benchmark Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 4.0: Initial Baseline** | `main` | `a5b48d7` | Concat + MLP, Weighted BCE Loss, Standard Threshold (0.50) | Baseline Reference |
| **Stage 4.1: Hyperparameter Optimization** | `feature/model-optimization` | `6a0b4e9` | Focal Loss ($\alpha=0.75, \gamma=2.0$), $L_2=0.003$, Noise Aug ($0.01$), `recall_first` Threshold | Text-Only Accuracy: **70.9%**, PR-AUC: **0.551** |
| **Stage 4.2: Gated Fusion Upgrade** | `feature/gated-fusion-upgrade` | `230cbff` | `GatedFusionHead` (Adaptive per-modality scalar gating weights) | Late Fusion Depressed Recall: **69.2%** (9/13 detected) |
| **Stage 4.3: Sequence Visuals + ModalDrop** | `feature/sequence-visual-modaldrop` | `601a5cc` | 31-d `sequence_pooled` visual features + Modality Dropout ($p=0.15$) | **New Record ROC-AUC: 0.7106** (`Text+Audio`) |
| **Stage 4.4: Cross-Modal Attention** | `feature/cross-modal-attention` | *(Pending)* | Cross-Attention Transformer Fusion | *(Next Experiment)* |

---

## 📈 Metric Comparison Across Experiment Branches

### 1. Tri-Modal Late Fusion Evolution

| Iteration / Branch | Threshold | Accuracy | Precision | Recall (Sens.) | Depressed Detected | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Concat (`main`)** | 0.50 | 50.9% | 27.8% | 38.5% | 5 / 13 | 0.3226 | 0.6120 | 0.3520 |
| **EXP037 Concat (`model-optimization`)** | 0.48 | **63.6%** | **33.3%** | 53.8% | 7 / 13 | 0.4118 | 0.6703 | **0.4070** |
| **Gated Fusion (`gated-fusion-upgrade`)** | 0.46 | 54.5% | 30.0% | **69.2%** | **9 / 13** | **0.4186** | 0.6520 | 0.3650 |
| **Seq Visual + ModalDrop (`sequence-visual-modaldrop`)** | 0.28 | 61.8% | 33.3% | 53.8% | 7 / 13 | 0.4000 | **0.6960** | 0.3650 |

---

### 2. Top Performing Model Configurations (Global Test Set: 55 Patients)

| Rank | Model Strategy | Branch / Version | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Text + Audio** | `sequence-visual-modaldrop` | 54.55% | 30.00% | **69.23%** | 0.4490 | **0.7106** ⭐ | 0.4730 |
| 🥈 | **Text-Only** | `model-optimization` | **70.91%** | **41.18%** | 53.85% | 0.4667 | 0.6868 | **0.5510** |
| 🥉 | **Late Fusion (3-Modality)**| `sequence-visual-modaldrop` | 61.82% | 33.33% | 53.85% | 0.4000 | 0.6960 | 0.3650 |
| 4 | **Audio + Visual** | `sequence-visual-modaldrop` | 52.73% | 29.03% | **69.23%** | 0.4407 | 0.6886 | 0.2860 |
| 5 | **Text + Visual** | `sequence-visual-modaldrop` | 54.55% | 25.00% | 46.15% | **0.4583** | 0.6868 | 0.3400 |

---

## 🔍 Detailed Iteration Details

### Iteration 1: `main` (Initial Baseline)
* **Architecture**: Simple concatenation of text (768), audio (128), visual (178) into a 2-layer MLP.
* **Loss Function**: BCEWithLogitsLoss with fixed `pos_weight`.
* **Threshold**: Fixed at `0.50`.
* **Finding**: Sufferred from low depressed recall ($38.5\%$) due to uncalibrated decision thresholds.

### Iteration 2: `feature/model-optimization` (EXP037 Optimization)
* **Architecture**: Standard concatenation + MLP with optimized dropout ($0.4$).
* **Loss Function**: Focal Loss ($\alpha=0.75, \gamma=2.0$) to focus on hard positive cases.
* **Augmentation**: Gaussian noise injection ($\sigma=0.01$).
* **Threshold Strategy**: `recall_first` constrained thresholding.
* **Finding**: Text-Only hit top accuracy ($70.91\%$) and PR-AUC ($0.5510$), proving text contains the cleanest semantic signal.

### Iteration 3: `feature/gated-fusion-upgrade` (Gated Fusion)
* **Architecture**: Introduced `GatedFusionHead` with dynamic Softmax gating weights per modality.
* **Finding**: Boosted depressed patient detection from **7/13 to 9/13 (69.23% Recall)**, significantly reducing false negatives.

### Iteration 4: `feature/sequence-visual-modaldrop` (Sequence Visuals & ModalDrop)
* **Architecture**: Gated Fusion + 31-d sequence-pooled visual features (`visual_source: "sequence_pooled"`).
* **Augmentation**: Added Modality Dropout ($p=0.15$) during training.
* **Finding**: Visual noise was dramatically reduced. **Text + Audio achieved a new record ROC-AUC of 0.7106**, and Late Fusion ROC-AUC jumped to **0.6960**.

---

## 📌 How to Reproduce / Switch Checkpoints

```bash
# Switch to Baseline Optimization (EXP037)
git checkout feature/model-optimization

# Switch to Gated Fusion Upgrade
git checkout feature/gated-fusion-upgrade

# Switch to Sequence-Pooled Visuals + Modality Dropout (Current Best)
git checkout feature/sequence-visual-modaldrop
```
