import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    PyTorch Encoder for Visual Modality.
    Processes 178-dimensional tabular visual feature vectors (Action Units, Landmarks, Gaze, Pose).
    """

    def __init__(self, input_dim=178, hidden_dim=128, dropout_rate=0.3):
        super(VisualEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual feature tensor of shape (batch_size, 178)
        Returns:
            torch.Tensor: Latent visual representation of shape (batch_size, hidden_dim)
        """
        return self.net(x)
