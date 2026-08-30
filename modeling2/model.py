"""
model.py
========
Arsitektur Multimodal Depression Detection (Paper MFDCL & Enhanced Variants).

Mencakup:
1. AudioEncoder: Conv1d(80->128) + BN + ReLU + MaxPool1d(2) + Dropout + BiLSTM(128, 256) + Linear(512, 256) -> I_a (B, 256)
2. VisualEncoder: Conv2d(3->64, k=(3, 72)) + Squeeze + BN + ReLU + MaxPool1d(2) + Dropout + BiLSTM(64, 256) + Linear(512, 256) -> I_v (B, 256)
3. TextEncoder: Conv1d(768->128) + BN + ReLU + MaxPool1d(2) + Dropout + BiLSTM(128, 256) + Linear(512, 256) -> I_t (B, 256)
4. MultimodalBaselineModel: Concatenation / Gating + Regressor MLP -> pred_phq8 (B, 1)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """Self-Attention Pooling untuk merangkum representasi temporal BiLSTM."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = self.attn(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled


class AudioEncoder(nn.Module):
    """
    Audio Feature Extractor:
    Input: (B, 80, 512) atau (B, 512, 80) log-Mel spectrogram.
    Conv1d(80->128, k=3, p=1) -> BatchNorm1d(128) -> ReLU -> MaxPool1d(2)
    -> Dropout -> BiLSTM(128, 256, num_layers=2) -> Linear(512, 256) -> Output: I_a (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 80,
        conv_out_channels: int = 128,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.7,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Sequence length 512 -> 256
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 512 and x.size(2) == 80:
            x = x.permute(0, 2, 1)

        h = self.conv(x)
        h = h.permute(0, 2, 1)  # (B, 256, 128)
        lstm_out, _ = self.lstm(h)  # (B, 256, 512)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class VisualEncoder(nn.Module):
    """
    Visual Feature Extractor (3D Facial Landmarks + Gaze):
    Input: (B, 3, 512, 72) atau (B, 512, 72, 3).
    Conv2d(3->64, k=(3, 72), p=(1, 0)) -> Squeeze -> BatchNorm1d(64) -> ReLU
    -> MaxPool1d(2) -> Dropout -> BiLSTM(64, 256, num_layers=2) -> Linear(512, 256) -> Output: I_v (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_nodes: int = 72,
        conv_out_channels: int = 64,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.7,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=(3, num_nodes),
            padding=(1, 0),
        )

        self.temporal_block = nn.Sequential(
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Sequence length 512 -> 256
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 512 and x.size(2) == 72 and x.size(3) == 3:
            x = x.permute(0, 3, 1, 2)

        h = self.conv2d(x)
        h = h.squeeze(-1)
        h = self.temporal_block(h)
        h = h.permute(0, 2, 1)  # (B, 256, 64)
        lstm_out, _ = self.lstm(h)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class TextEncoder(nn.Module):
    """
    Text Feature Extractor (Sentence-BERT Embeddings):
    Input: (B, 768, 512) atau (B, 512, 768).
    Conv1d(768->128, k=3, p=1) -> BatchNorm1d(128) -> ReLU -> MaxPool1d(2)
    -> Dropout -> BiLSTM(128, 256, num_layers=2) -> Linear(512, 256) -> Output: I_t (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 768,
        conv_out_channels: int = 128,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.7,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # Sequence length 512 -> 256
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 512 and x.size(2) == 768:
            x = x.permute(0, 2, 1)

        h = self.conv(x)
        h = h.permute(0, 2, 1)
        lstm_out, _ = self.lstm(h)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class GatedMultimodalFusion(nn.Module):
    """Adaptive Gated Fusion untuk menimbang kontribusi Audio, Visual, dan Teks secara dinamis."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.gate_a = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.gate_v = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())

    def forward(self, I_a: torch.Tensor, I_v: torch.Tensor, I_t: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([I_a, I_v, I_t], dim=-1)  # (B, 3*dim)
        g_a = self.gate_a(concat)
        g_v = self.gate_v(concat)
        g_t = self.gate_t(concat)
        fused = torch.cat([I_a * g_a, I_v * g_v, I_t * g_t], dim=-1)
        return fused


class MultimodalBaselineModel(nn.Module):
    """
    Multimodal Depression Detection Baseline Model (MFDCL Architecture & Enhanced Options).
    """

    def __init__(
        self,
        audio_in_channels: int = 80,
        visual_in_channels: int = 3,
        visual_num_nodes: int = 72,
        text_in_channels: int = 768,
        unimodal_output_dim: int = 256,
        regressor_hidden_dim: int = 128,
        dropout_conv: float = 0.7,
        regressor_dropout: float = 0.5,
        pooling_type: str = "mean",
        use_gating: bool = False,
    ) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(
            in_channels=audio_in_channels,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )
        self.visual_encoder = VisualEncoder(
            in_channels=visual_in_channels,
            num_nodes=visual_num_nodes,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )
        self.text_encoder = TextEncoder(
            in_channels=text_in_channels,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )

        self.use_gating = use_gating
        if use_gating:
            self.fusion_module = GatedMultimodalFusion(dim=unimodal_output_dim)
        else:
            self.fusion_module = None

        fusion_dim = unimodal_output_dim * 3  # 256 * 3 = 768

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, regressor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=regressor_dropout),
            nn.Linear(regressor_hidden_dim, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        I_a = self.audio_encoder(audio)    # (B, 256)
        I_v = self.visual_encoder(visual)  # (B, 256)
        I_t = self.text_encoder(text)      # (B, 256)

        if self.fusion_module is not None:
            x_concat = self.fusion_module(I_a, I_v, I_t)
        else:
            x_concat = torch.cat([I_a, I_v, I_t], dim=-1)

        pred_phq8 = self.regressor(x_concat)  # (B, 1)
        return pred_phq8

    def extract_features(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        I_a = self.audio_encoder(audio)
        I_v = self.visual_encoder(visual)
        I_t = self.text_encoder(text)
        if self.fusion_module is not None:
            x_concat = self.fusion_module(I_a, I_v, I_t)
        else:
            x_concat = torch.cat([I_a, I_v, I_t], dim=-1)
        return {
            "audio_latent": I_a,
            "visual_latent": I_v,
            "text_latent": I_t,
            "fused_latent": x_concat,
        }
