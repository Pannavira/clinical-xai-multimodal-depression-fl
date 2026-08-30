"""
loss.py
=======
Weighted MSE Loss berbasis frekuensi kelas per batch untuk penanganan
class imbalance dan mitigasi False Negative pada regresi skor depresi PHQ-8 (Paper MFDCL Baseline).

Formula:
    1. Binerisasi target PHQ-8 sementara di dalam batch:
       y_bin = (y_true >= binarize_threshold).float()  (default threshold: 10.0)
    2. Hitung bobot per kelas:
       n_1 = jumlah sampel dengan y_bin == 1 dalam batch
       n_0 = jumlah sampel dengan y_bin == 0 dalam batch
       w_1 = (1.0 / n_1) * pos_weight_multiplier (jika n_1 > 0)
       w_0 = 1.0 / n_0 (jika n_0 > 0)
    3. Bobot sampel:
       w_i = w_1 jika y_bin_i == 1 else w_0
    4. False Negative Penalty (Asymmetric Error):
       Jika fn_penalty > 1.0 dan y_bin_i == 1 dan pred < threshold:
          w_i = w_i * fn_penalty
    5. Loss:
       Loss = mean(w_i * (pred_phq8 - y_true)^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedMSELoss(nn.Module):
    """
    Class-Weighted Mean Squared Error Loss berbasis frekuensi kelas per-batch
    dengan dukungan pos_weight_multiplier dan penalti False Negative.

    Parameter:
    ----------
    binarize_threshold : float, default=10.0
        Ambang batas binerisasi skor PHQ-8 (>= 10.0 = Depressed).
    pos_weight_multiplier : float, default=1.0
        Pengali bobot kelas positif (depresi) untuk menaikkan Recall & menekan FN.
    fn_penalty : float, default=1.0
        Penalti tambahan untuk sampel depresi yang diprediksi di bawah threshold (FN).
    eps : float, default=1e-8
        Konstanta stabilitas numerik untuk mencegah pembagian dengan nol.
    """

    def __init__(
        self,
        binarize_threshold: float = 10.0,
        pos_weight_multiplier: float = 1.0,
        fn_penalty: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.binarize_threshold = float(binarize_threshold)
        self.pos_weight_multiplier = float(pos_weight_multiplier)
        self.fn_penalty = float(fn_penalty)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        batch_size = pred_flat.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 1. Binerisasi target PHQ-8 sementara di dalam batch
        y_bin = (target_flat >= self.binarize_threshold).float()

        # 2. Hitung frekuensi per kelas dalam batch
        n_1 = (y_bin == 1.0).sum()
        n_0 = (y_bin == 0.0).sum()

        # 3. Hitung bobot per kelas dengan pengali pos_weight_multiplier
        w_1 = (self.pos_weight_multiplier / (n_1.float() + self.eps)) if n_1 > 0 else torch.tensor(0.0, device=pred.device)
        w_0 = (1.0 / (n_0.float() + self.eps)) if n_0 > 0 else torch.tensor(0.0, device=pred.device)

        if n_1 == 0:
            w_0 = torch.tensor(1.0 / float(batch_size), device=pred.device)
        elif n_0 == 0:
            w_1 = torch.tensor(self.pos_weight_multiplier / float(batch_size), device=pred.device)

        # Sample weight w_i
        w_i = torch.where(y_bin == 1.0, w_1, w_0)

        # 4. Asymmetric False Negative Penalty
        if self.fn_penalty > 1.0:
            fn_mask = (y_bin == 1.0) & (pred_flat < self.binarize_threshold)
            w_i = torch.where(fn_mask, w_i * self.fn_penalty, w_i)

        # 5. Loss = mean(w_i * (pred_phq8 - y_true)^2)
        sq_err = (pred_flat - target_flat) ** 2
        loss = torch.mean(w_i * sq_err)

        return loss

    def __repr__(self) -> str:
        return (
            f"ClassWeightedMSELoss(binarize_threshold={self.binarize_threshold}, "
            f"pos_weight_multiplier={self.pos_weight_multiplier}, "
            f"fn_penalty={self.fn_penalty}, eps={self.eps})"
        )


class MultiTaskClinicalLoss(nn.Module):
    """
    Kombinasi Weighted MSE Loss (Regresi PHQ-8) + Binary Cross Entropy (Klasifikasi Klinis Depresi).
    Sangat efektif meningkatkan discriminability antara Normal dan Depresi.
    """

    def __init__(
        self,
        binarize_threshold: float = 10.0,
        pos_weight_multiplier: float = 1.5,
        fn_penalty: float = 1.2,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.weighted_mse = ClassWeightedMSELoss(
            binarize_threshold=binarize_threshold,
            pos_weight_multiplier=pos_weight_multiplier,
            fn_penalty=fn_penalty,
        )
        self.bce_weight = float(bce_weight)
        self.binarize_threshold = float(binarize_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.weighted_mse(pred, target)

        if self.bce_weight <= 0.0:
            return mse_loss

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        y_bin = (target_flat >= self.binarize_threshold).float()

        # Sigmoid centered at threshold
        logits = pred_flat - self.binarize_threshold
        # Pos weight for BCE
        pos_weight = torch.tensor([2.0], device=pred.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y_bin, pos_weight=pos_weight)

        total_loss = mse_loss + self.bce_weight * bce_loss
        return total_loss
